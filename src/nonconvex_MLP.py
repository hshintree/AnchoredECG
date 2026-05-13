"""
Non-convex MLP residual adapter baselines.

Plain ReLU MLP solving the same residual prediction problem as the GReLU
adapter:
    u = [phi_stress; z_rest]   (384-dim)
    r = z_stress - z_rest      (192-dim)

Grid (single fixed seed):
    sizes  in {tiny, medium, wide}       -- 3 architectures
    lr     in {1e-4, 1e-3, 1e-2}         -- 3 learning rates
    wd     in {0.0, 1e-4}                -- 2 L2 weight decays
    l1     in {0.0, 1e-5, 1e-4}          -- 3 L1 penalties on the first layer
    => 3 x 3 x 2 x 3 = 54 runs

Sizes:
    tiny      384 -> 64        -> 192    (~25k trainable params)
    medium    384 -> 256       -> 192    (~148k)
    wide      384 -> 1024      -> 192    (~590k -- matched to GReLU m=1024)

Writes one JSONL record per run to checkpoints/sweep_results.jsonl so the
box plot in plots.py can compare side-by-side with the GReLU sweep.

Usage:
    python src/nonconvex_MLP.py
    python src/nonconvex_MLP.py --epochs 60
    python src/nonconvex_MLP.py --sizes medium wide
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapter_interface import load_regression_head
from grelu_adapter import (
    build_tensors,
    evaluate_residual_adapter,
    load_split,
    make_loader,
    pick_device,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


# ---------------------------------------------------------------------------
# MLP definitions
# ---------------------------------------------------------------------------

class MLPResidualAdapter(nn.Module):
    """Plain feedforward ReLU MLP. Predicts r = z_stress - z_rest from u."""

    def __init__(self, d_input: int = 384, d_output: int = 192,
                 hidden_dims: list[int] | tuple[int, ...] = (128,),
                 dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        prev = d_input
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, d_output))
        self.net = nn.Sequential(*layers)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.net(u)

    def first_layer_weight(self) -> torch.Tensor:
        """The 384 -> hidden weight matrix (mirrors GReLU.linear1)."""
        for m in self.net:
            if isinstance(m, nn.Linear):
                return m.weight
        raise RuntimeError("No Linear layer found in MLP")

    @torch.no_grad()
    def predict_latent(self, phi: torch.Tensor, z_rest: torch.Tensor) -> torch.Tensor:
        return z_rest + self.forward(torch.cat([phi, z_rest], dim=1))


SIZE_CONFIGS: dict[str, dict] = {
    "tiny":   {"hidden_dims": (64,),    "dropout": 0.0},
    "medium": {"hidden_dims": (256,),   "dropout": 0.0},
    "wide":   {"hidden_dims": (1024,),  "dropout": 0.0},
    "huge":   {"hidden_dims": (2048,),  "dropout": 0.0},
}

DEFAULT_SIZES = ["tiny", "medium", "wide", "huge"]
DEFAULT_LRS = [1e-4, 1e-3, 1e-2]
DEFAULT_WDS = [0.0]
DEFAULT_L1S = [0.0]


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def save_mlp(model: "MLPResidualAdapter", path: str | Path,
             hidden_dims, dropout: float,
             d_input: int, d_output: int,
             config: dict | None = None,
             metrics: dict | None = None) -> None:
    """Persist an MLP model with enough metadata to rebuild it from scratch."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict":  model.state_dict(),
        "hidden_dims": list(hidden_dims),
        "dropout":     float(dropout),
        "d_input":     int(d_input),
        "d_output":    int(d_output),
        "config":      config,
        "metrics":     metrics,
    }, path)


def load_mlp(path: str | Path, device: torch.device | None = None
             ) -> "MLPResidualAdapter":
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = MLPResidualAdapter(
        d_input=int(ckpt["d_input"]), d_output=int(ckpt["d_output"]),
        hidden_dims=ckpt["hidden_dims"], dropout=float(ckpt["dropout"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    if device is not None:
        model = model.to(device)
    model.eval()
    return model


def train_mlp(
    train_split: dict,
    val_split: dict,
    head: nn.Module,
    hidden_dims: list[int] | tuple[int, ...],
    dropout: float = 0.0,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    lambda_l2: float = 0.0,
    lambda_l1: float = 0.0,
    seed: int = 42,
    device: torch.device | None = None,
    verbose: bool = False,
) -> tuple["MLPResidualAdapter", dict]:
    """Train one MLP configuration. Returns (trained_model, metrics_dict)."""
    device = device or pick_device(None)
    torch.manual_seed(seed)
    np.random.seed(seed)

    u_tr, r_tr, _, _ = build_tensors(train_split, device=device)
    u_va, r_va, zr_va, zs_va = build_tensors(val_split, device=device)

    model = MLPResidualAdapter(
        d_input=u_tr.shape[1], d_output=r_tr.shape[1],
        hidden_dims=hidden_dims, dropout=dropout,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lambda_l2)
    loss_fn = nn.MSELoss()
    loader = make_loader(u_tr, r_tr, batch_size, shuffle=True, seed=seed)

    t0 = time.time()
    last_train_mse = float("nan")
    for epoch in range(1, epochs + 1):
        model.train()
        running, n = 0.0, 0
        for ub, rb in loader:
            optim.zero_grad()
            pred = model(ub)
            loss = loss_fn(pred, rb)
            if lambda_l1 > 0:
                loss = loss + lambda_l1 * model.first_layer_weight().abs().sum()
            loss.backward()
            optim.step()
            running += loss.item() * ub.size(0)
            n += ub.size(0)
        last_train_mse = running / max(n, 1)
        if verbose and (epoch == 1 or epoch % max(1, epochs // 5) == 0):
            print(f"    epoch {epoch:4d}  train_mse={last_train_mse:.6f}")
    train_time = time.time() - t0

    model.eval()
    metrics = evaluate_residual_adapter(
        model, u_va, r_va, zr_va, zs_va,
        head=head, y=val_split["ischemia"], bl_st80=val_split["bl_st80"],
    )
    return model, {
        "val_residual_mse": metrics["residual_mse"],
        "val_latent_mse": metrics["latent_mse"],
        "val_auroc": metrics.get("auroc", float("nan")),
        "train_mse": last_train_mse,
        "n_params": count_trainable(model),
        "train_time_s": train_time,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42,
                    help="Single fixed seed for all runs.")
    ap.add_argument("--sizes", type=str, nargs="+", default=DEFAULT_SIZES,
                    choices=list(SIZE_CONFIGS.keys()))
    ap.add_argument("--lrs", type=float, nargs="+", default=DEFAULT_LRS)
    ap.add_argument("--wds", type=float, nargs="+", default=DEFAULT_WDS)
    ap.add_argument("--l1s", type=float, nargs="+", default=DEFAULT_L1S)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", type=str, default=str(CKPT_DIR / "sweep_results.jsonl"),
                    help="JSONL append path for sweep records.")
    ap.add_argument("--ckpt", type=str, default=str(CKPT_DIR / "mlp_best.pt"),
                    help="Path to save the best MLP found during the sweep.")
    ap.add_argument("--ckpt-metric", type=str, default="auroc",
                    choices=["auroc", "latent_mse"],
                    help="Metric used to decide which run is 'best' and gets saved.")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    device = pick_device(args.device)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading data ...   device: {device}")
    train_split = load_split(DATA_DIR, "train")
    val_split = load_split(DATA_DIR, "val")
    head = load_regression_head(DATA_DIR / "regression_head.pt")
    print(f"  train: {len(train_split['ischemia'])}   val: {len(val_split['ischemia'])}")

    grid = list(itertools.product(args.sizes, args.lrs, args.wds, args.l1s))
    print(f"  grid: {len(args.sizes)} sizes x {len(args.lrs)} lrs x "
          f"{len(args.wds)} wds x {len(args.l1s)} l1s = {len(grid)} runs\n")

    def is_better(cand: dict, cur_best: dict | None) -> bool:
        if cur_best is None:
            return True
        if args.ckpt_metric == "auroc":
            return float(cand["val_auroc"]) > float(cur_best["val_auroc"])
        return float(cand["val_latent_mse"]) < float(cur_best["val_latent_mse"])

    best_rec: dict | None = None
    rows = []
    with out_path.open("a") as f:
        for i, (cfg_name, lr, wd, l1) in enumerate(grid, 1):
            cfg = SIZE_CONFIGS[cfg_name]
            model, metrics = train_mlp(
                train_split, val_split, head,
                hidden_dims=cfg["hidden_dims"], dropout=cfg["dropout"],
                epochs=args.epochs, batch_size=args.batch_size,
                lr=lr, lambda_l2=wd, lambda_l1=l1,
                seed=args.seed, device=device, verbose=args.verbose,
            )
            rec = {
                "method": "MLP",
                "config": cfg_name,
                "label": f"MLP/{cfg_name}/lr={lr:g}/wd={wd:g}/l1={l1:g}",
                "seed": int(args.seed),
                "size": cfg_name,
                "hidden_dims": list(cfg["hidden_dims"]),
                "dropout": cfg["dropout"],
                "lr": float(lr),
                "lambda_l2": float(wd),
                "lambda_l1": float(l1),
                "epochs": args.epochs,
                **metrics,
            }
            f.write(json.dumps(rec) + "\n"); f.flush()
            rows.append(rec)
            marker = ""
            if is_better(rec, best_rec):
                best_rec = rec
                save_mlp(
                    model, args.ckpt,
                    hidden_dims=cfg["hidden_dims"], dropout=cfg["dropout"],
                    d_input=384, d_output=192,
                    config={
                        "size": cfg_name, "lr": float(lr),
                        "lambda_l2": float(wd), "lambda_l1": float(l1),
                        "epochs": args.epochs, "seed": int(args.seed),
                        "ckpt_metric": args.ckpt_metric,
                    },
                    metrics=metrics,
                )
                marker = "  <- new best, saved"
            print(f"[{i:2d}/{len(grid)}] MLP {cfg_name:<6s} lr={lr:g} wd={wd:g} l1={l1:g}  "
                  f"params={rec['n_params']:>7d}  "
                  f"val_latent_mse={rec['val_latent_mse']:.4f}  "
                  f"val_auroc={rec['val_auroc']:.4f}  "
                  f"({rec['train_time_s']:.1f}s){marker}")

    # Summary grouped by size
    print("\n--- MLP sweep summary (across hyperparams, per size) ---")
    by_size: dict[str, list[dict]] = {}
    for r in rows:
        by_size.setdefault(r["config"], []).append(r)
    print(f"{'config':<10s} {'n':>3s}  {'latent_mse  best (min)':>26s}  {'auroc  best (max)':>22s}")
    for name in args.sizes:
        items = by_size.get(name, [])
        if not items:
            continue
        lat = np.array([r["val_latent_mse"] for r in items])
        au  = np.array([r["val_auroc"] for r in items])
        best = min(items, key=lambda r: r["val_latent_mse"])
        print(f"{name:<10s} {len(items):>3d}  "
              f"{lat.mean():.4f}+/-{lat.std():.4f}  ({lat.min():.4f})   "
              f"{au.mean():.4f}+/-{au.std():.4f}  ({au.max():.4f})")
        print(f"            best @ lr={best['lr']:g} wd={best['lambda_l2']:g} "
              f"l1={best['lambda_l1']:g} -> latent_mse={best['val_latent_mse']:.4f} "
              f"auroc={best['val_auroc']:.4f}")
    print(f"\nAppended {len(rows)} records to {out_path}")
    if best_rec is not None:
        print(f"\nBest MLP (by {args.ckpt_metric}) saved to {args.ckpt}")
        print(f"  size={best_rec['config']:<6s}  lr={best_rec['lr']:g}  "
              f"wd={best_rec['lambda_l2']:g}  l1={best_rec['lambda_l1']:g}  "
              f"epochs={best_rec['epochs']}")
        print(f"  val_auroc={best_rec['val_auroc']:.4f}  "
              f"val_latent_mse={best_rec['val_latent_mse']:.4f}  "
              f"params={best_rec['n_params']}")


if __name__ == "__main__":
    main()
