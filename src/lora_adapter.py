"""
Low-Rank Adaptation residual adapter for AnchoredECG.

The map u (384) -> r (192) is constrained to factor as

    r_hat = B (A u)         A in R^{rank x 384},  B in R^{192 x rank}

so the implicit weight matrix W = B A has rank <= `rank`. As `rank`
increases this approaches the unconstrained linear adapter; the rank at
which LoRA matches the linear baseline is an empirical estimate of the
intrinsic dimensionality of the stress-induced latent subspace.

Init: `A` uses PyTorch's default Kaiming-uniform for nn.Linear; `B` is
initialized to zero so the adapter starts at the identity-residual
prediction (z_stress_hat = z_rest) before any training. Trained with Adam
and MSE loss using the same loop structure as nonconvex_MLP.py.

Sweep:
    rank      in {2, 4, 8, 16, 32, 64, 128}
    lr        in {1e-4, 1e-3, 1e-2}
    lambda_l2 in {0.0, 1e-4}                  -> 42 runs

Records are appended to checkpoints/sweep_results.jsonl with method = "LoRA".

Usage:
    python src/lora_adapter.py
    python src/lora_adapter.py --ranks 8 32 128 --epochs 200
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from datetime import datetime
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

D_INPUT = 384
D_OUTPUT = 192


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LoRAResidualAdapter(nn.Module):
    """r_hat = B(A(u)).  W = B @ A has rank <= `rank`."""

    def __init__(self, d_input: int = D_INPUT, d_output: int = D_OUTPUT,
                 rank: int = 8):
        super().__init__()
        self.d_input = int(d_input)
        self.d_output = int(d_output)
        self.rank = int(rank)
        self.A = nn.Linear(d_input, rank, bias=False)
        self.B = nn.Linear(rank, d_output, bias=False)
        # B = 0 -> initial prediction is the zero residual, so training
        # starts from the safe baseline z_stress_hat = z_rest.
        with torch.no_grad():
            self.B.weight.zero_()

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.B(self.A(u))

    @torch.no_grad()
    def predict_latent(self, phi: torch.Tensor, z_rest: torch.Tensor) -> torch.Tensor:
        u = torch.cat([phi, z_rest], dim=1)
        return z_rest + self.forward(u)

    def effective_weight(self) -> torch.Tensor:
        """Returns the implicit weight matrix W = B @ A, shape (d_output, d_input)."""
        return self.B.weight @ self.A.weight


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_lora(model: LoRAResidualAdapter, path: str | Path,
              config: dict | None = None, metrics: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "d_input": model.d_input,
        "d_output": model.d_output,
        "rank": model.rank,
        "config": config,
        "metrics": metrics,
    }, path)


def load_lora(path: str | Path, device: torch.device | None = None
              ) -> LoRAResidualAdapter:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    m = LoRAResidualAdapter(
        d_input=int(ckpt["d_input"]), d_output=int(ckpt["d_output"]),
        rank=int(ckpt["rank"]),
    )
    m.load_state_dict(ckpt["state_dict"])
    if device is not None:
        m = m.to(device)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_lora(train_split: dict, val_split: dict, head: nn.Module,
               rank: int, epochs: int = 100, batch_size: int = 256,
               lr: float = 1e-3, lambda_l2: float = 0.0,
               seed: int = 42, device: torch.device | None = None,
               verbose: bool = False
               ) -> tuple[LoRAResidualAdapter, dict]:
    """Train one LoRA configuration. Returns (model, metrics_dict)."""
    device = device or pick_device(None)
    torch.manual_seed(seed)
    np.random.seed(seed)

    u_tr, r_tr, _, _ = build_tensors(train_split, device=device)
    u_va, r_va, zr_va, zs_va = build_tensors(val_split, device=device)

    model = LoRAResidualAdapter(
        d_input=u_tr.shape[1], d_output=r_tr.shape[1], rank=rank,
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
            loss.backward()
            optim.step()
            running += loss.item() * ub.size(0)
            n += ub.size(0)
        last_train_mse = running / max(n, 1)
        if verbose and (epoch == 1 or epoch % max(1, epochs // 5) == 0):
            print(f"    epoch {epoch:4d}  train_mse={last_train_mse:.6f}")
    train_time = time.time() - t0

    model.eval()
    m = evaluate_residual_adapter(
        model, u_va, r_va, zr_va, zs_va,
        head=head, y=val_split["ischemia"], bl_st80=val_split["bl_st80"],
    )
    return model, {
        "val_residual_mse": m["residual_mse"],
        "val_latent_mse": m["latent_mse"],
        "val_auroc": m.get("auroc", float("nan")),
        "train_mse": last_train_mse,
        "n_params": count_trainable(model),
        "train_time_s": train_time,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_RANKS = [2, 4, 8, 16, 32, 64, 128]
DEFAULT_LRS = [1e-4, 1e-3, 1e-2]
DEFAULT_WDS = [0.0, 1e-4]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ranks", type=int, nargs="+", default=DEFAULT_RANKS)
    ap.add_argument("--lrs", type=float, nargs="+", default=DEFAULT_LRS)
    ap.add_argument("--wds", type=float, nargs="+", default=DEFAULT_WDS)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", type=str, default=str(CKPT_DIR / "sweep_results.jsonl"))
    ap.add_argument("--ckpt", type=str, default=str(CKPT_DIR / "lora_best.pt"))
    ap.add_argument("--ckpt-metric", type=str, default="auroc",
                    choices=["auroc", "latent_mse"])
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    device = pick_device(args.device)
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading data ...   device: {device}")
    train_split = load_split(DATA_DIR, "train")
    val_split = load_split(DATA_DIR, "val")
    head = load_regression_head(DATA_DIR / "regression_head.pt")
    print(f"  train: {len(train_split['ischemia'])}   val: {len(val_split['ischemia'])}")

    grid = list(itertools.product(args.ranks, args.lrs, args.wds))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"  grid: {len(args.ranks)} ranks x {len(args.lrs)} lrs x "
          f"{len(args.wds)} wds  =  {len(grid)} runs")
    print(f"  run_id: {run_id}\n")

    def is_better(cand: dict, cur: dict | None) -> bool:
        if cur is None:
            return True
        if args.ckpt_metric == "auroc":
            return float(cand["val_auroc"]) > float(cur["val_auroc"])
        return float(cand["val_latent_mse"]) < float(cur["val_latent_mse"])

    rows, best_rec = [], None

    with out_path.open("a") as f:
        for i, (rank, lr, wd) in enumerate(grid, 1):
            model, metrics = train_lora(
                train_split, val_split, head,
                rank=rank, epochs=args.epochs, batch_size=args.batch_size,
                lr=lr, lambda_l2=wd,
                seed=args.seed, device=device, verbose=args.verbose,
            )
            cfg_name = f"r={rank}"
            rec = {
                "method": "LoRA",
                "config": cfg_name,
                "label": f"LoRA/{cfg_name}/lr={lr:g}/wd={wd:g}",
                "run_id": run_id,
                "seed": int(args.seed),
                "rank": int(rank),
                "lr": float(lr),
                "lambda_l2": float(wd),
                "epochs": args.epochs,
                **metrics,
            }
            f.write(json.dumps(rec) + "\n"); f.flush()
            rows.append(rec)
            marker = ""
            if is_better(rec, best_rec):
                best_rec = rec
                save_lora(model, args.ckpt,
                          config={"rank": int(rank), "lr": float(lr),
                                  "lambda_l2": float(wd),
                                  "epochs": int(args.epochs),
                                  "seed": int(args.seed),
                                  "ckpt_metric": args.ckpt_metric},
                          metrics=metrics)
                marker = "  <- new best, saved"
            print(f"[{i:2d}/{len(grid)}] LoRA r={rank:<4d} lr={lr:g} wd={wd:g}  "
                  f"params={rec['n_params']:>6d}  "
                  f"val_latent_mse={rec['val_latent_mse']:.4f}  "
                  f"val_auroc={rec['val_auroc']:.4f}  "
                  f"({rec['train_time_s']:.1f}s){marker}")

    # Summary grouped by rank
    print("\n--- LoRA sweep summary (per rank) ---")
    by_rank: dict[str, list[dict]] = {}
    for r in rows:
        by_rank.setdefault(r["config"], []).append(r)
    print(f"{'config':<8s} {'n':>3s}  "
          f"{'latent_mse  best (min)':>26s}  {'auroc  best (max)':>22s}")
    for rank in args.ranks:
        items = by_rank.get(f"r={rank}", [])
        if not items:
            continue
        lat = np.array([r["val_latent_mse"] for r in items])
        au  = np.array([r["val_auroc"] for r in items])
        best = min(items, key=lambda r: r["val_latent_mse"])
        print(f"r={rank:<6d} {len(items):>3d}  "
              f"{lat.mean():.4f}+/-{lat.std():.4f}  ({lat.min():.4f})   "
              f"{au.mean():.4f}+/-{au.std():.4f}  ({au.max():.4f})")
        print(f"            best @ lr={best['lr']:g} wd={best['lambda_l2']:g} "
              f"-> latent_mse={best['val_latent_mse']:.4f} "
              f"auroc={best['val_auroc']:.4f}")
    print(f"\nAppended {len(rows)} records to {out_path}")
    if best_rec is not None:
        print(f"\nBest LoRA (by {args.ckpt_metric}) saved to {args.ckpt}")
        print(f"  rank={best_rec['rank']}  lr={best_rec['lr']:g}  "
              f"wd={best_rec['lambda_l2']:g}  epochs={best_rec['epochs']}")
        print(f"  val_auroc={best_rec['val_auroc']:.4f}  "
              f"val_latent_mse={best_rec['val_latent_mse']:.4f}  "
              f"params={best_rec['n_params']}")


if __name__ == "__main__":
    main()
