"""
Multi-head self-attention residual adapter(s) for AnchoredECG.

Treats the input as a 2-token sequence  x = [phi_stress, z_rest]  with each
token in R^{192}, applies one transformer-style block (multihead attention
+ residual + LayerNorm), then maps to the residual r in one of two ways:

    "attn_linear":  proj(attended_phi_token)                          -> r_hat
    "attn_additive": linear(u) + correction(attended_phi_token)       -> r_hat

The additive variant decomposes the prediction into an interpretable global
linear term plus an input-conditional correction; correction_magnitude()
reports the average ratio ||correction||_2 / ||linear_pred||_2.

IMPORTANT: `forward(u)` returns ONLY a tensor (so it plugs cleanly into
evaluate_residual_adapter, which expects model(u) to be a Tensor). The most
recent attention weights are detached and stashed on
`self._last_attn_weights` so they can be inspected after a forward pass.

Sweep:
    variant   in {attn_linear, attn_additive}
    n_heads   in {1, 2, 4}
    dropout   in {0.0, 0.1}
    lr        in {1e-4, 1e-3}
    lambda_l2 in {0.0, 1e-4}                                 -> 48 runs

Records are appended to checkpoints/sweep_results.jsonl with
method = "AttentionLinear" | "AttentionAdditive".

Usage:
    python src/attention_adapter.py
    python src/attention_adapter.py --variants attn_additive --epochs 200
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

D_MODEL = 192  # phi_stress and z_rest are both 192-dim


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class AttentionBlock(nn.Module):
    """One pre-/post-residual multihead attention block on a 2-token sequence."""

    def __init__(self, d_model: int = D_MODEL, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, phi: torch.Tensor, z_rest: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # (B, 2, d_model): token 0 = phi_stress, token 1 = z_rest
        x = torch.stack([phi, z_rest], dim=1)
        out, weights = self.attn(x, x, x, need_weights=True, average_attn_weights=True)
        x = self.norm(x + out)
        # We hand back the phi-attended token + attention weights so the parent
        # adapter can stash them for inspection.
        return x[:, 0, :], weights


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

class AttentionLinearAdapter(nn.Module):
    """attn_linear variant:  attention block -> linear projection to r."""

    variant_name = "attn_linear"

    def __init__(self, d_model: int = D_MODEL, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.dropout = float(dropout)
        self.attn_block = AttentionBlock(d_model, n_heads, dropout)
        self.proj = nn.Linear(d_model, d_model)
        # Set by every forward() so callers can inspect attention.
        self._last_attn_weights: torch.Tensor | None = None

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        phi, z_rest = u[:, :self.d_model], u[:, self.d_model:]
        attended, weights = self.attn_block(phi, z_rest)
        self._last_attn_weights = weights.detach() if weights is not None else None
        return self.proj(attended)

    @torch.no_grad()
    def predict_latent(self, phi: torch.Tensor, z_rest: torch.Tensor) -> torch.Tensor:
        u = torch.cat([phi, z_rest], dim=1)
        return z_rest + self.forward(u)


class AdditiveAttentionAdapter(nn.Module):
    """attn_additive variant:  linear(u)  +  correction(attended)."""

    variant_name = "attn_additive"

    def __init__(self, d_model: int = D_MODEL, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.dropout = float(dropout)
        self.linear = nn.Linear(2 * d_model, d_model)
        self.attn_block = AttentionBlock(d_model, n_heads, dropout)
        self.correction = nn.Linear(d_model, d_model)
        self._last_attn_weights: torch.Tensor | None = None

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # IMPORTANT: must return a plain Tensor for evaluate_residual_adapter.
        # The attention weights are stashed on self._last_attn_weights instead.
        phi, z_rest = u[:, :self.d_model], u[:, self.d_model:]
        linear_pred = self.linear(u)
        attended, weights = self.attn_block(phi, z_rest)
        self._last_attn_weights = weights.detach() if weights is not None else None
        correction = self.correction(attended)
        return linear_pred + correction

    @torch.no_grad()
    def predict_latent(self, phi: torch.Tensor, z_rest: torch.Tensor) -> torch.Tensor:
        u = torch.cat([phi, z_rest], dim=1)
        return z_rest + self.forward(u)

    @torch.no_grad()
    def correction_magnitude(self, u: torch.Tensor) -> float:
        """Mean ||correction(u)||_2 / ||linear(u)||_2 over the batch."""
        phi, z_rest = u[:, :self.d_model], u[:, self.d_model:]
        linear_pred = self.linear(u).detach()
        attended, _ = self.attn_block(phi, z_rest)
        correction = self.correction(attended).detach()
        lin_norm = torch.linalg.norm(linear_pred, dim=1)
        corr_norm = torch.linalg.norm(correction, dim=1)
        return float((corr_norm / (lin_norm + 1e-12)).mean().item())


VARIANT_REGISTRY: dict[str, type[nn.Module]] = {
    "attn_linear": AttentionLinearAdapter,
    "attn_additive": AdditiveAttentionAdapter,
}


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_attention(model: nn.Module, path: str | Path, variant: str,
                   d_model: int, n_heads: int, dropout: float,
                   config: dict | None = None, metrics: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "variant": variant,
        "d_model": int(d_model),
        "n_heads": int(n_heads),
        "dropout": float(dropout),
        "config": config,
        "metrics": metrics,
    }, path)


def load_attention(path: str | Path, device: torch.device | None = None
                   ) -> nn.Module:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    variant = ckpt["variant"]
    if variant not in VARIANT_REGISTRY:
        raise ValueError(f"Unknown variant {variant!r}; "
                         f"expected one of {list(VARIANT_REGISTRY)}")
    cls = VARIANT_REGISTRY[variant]
    m = cls(d_model=int(ckpt["d_model"]), n_heads=int(ckpt["n_heads"]),
            dropout=float(ckpt["dropout"]))
    m.load_state_dict(ckpt["state_dict"])
    if device is not None:
        m = m.to(device)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_attention(train_split: dict, val_split: dict, head: nn.Module,
                    variant: str, n_heads: int, dropout: float,
                    epochs: int = 100, batch_size: int = 256,
                    lr: float = 1e-3, lambda_l2: float = 0.0,
                    seed: int = 42, device: torch.device | None = None,
                    verbose: bool = False
                    ) -> tuple[nn.Module, dict]:
    """Train one attention-adapter configuration. Returns (model, metrics)."""
    device = device or pick_device(None)
    torch.manual_seed(seed)
    np.random.seed(seed)

    u_tr, r_tr, _, _ = build_tensors(train_split, device=device)
    u_va, r_va, zr_va, zs_va = build_tensors(val_split, device=device)

    if variant not in VARIANT_REGISTRY:
        raise ValueError(f"Unknown variant {variant!r}")
    cls = VARIANT_REGISTRY[variant]
    model = cls(d_model=D_MODEL, n_heads=n_heads, dropout=dropout).to(device)
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
    extras: dict[str, float] = {}
    if isinstance(model, AdditiveAttentionAdapter):
        # Use the val tensor so the ratio is reported on held-out inputs.
        extras["correction_magnitude_val"] = model.correction_magnitude(u_va)
    return model, {
        "val_residual_mse": m["residual_mse"],
        "val_latent_mse": m["latent_mse"],
        "val_auroc": m.get("auroc", float("nan")),
        "train_mse": last_train_mse,
        "n_params": count_trainable(model),
        "train_time_s": train_time,
        **extras,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_VARIANTS = ["attn_linear", "attn_additive"]
DEFAULT_N_HEADS = [1, 2, 4]
DEFAULT_DROPOUTS = [0.0, 0.1]
DEFAULT_LRS = [1e-4, 1e-3]
DEFAULT_WDS = [0.0, 1e-4]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variants", type=str, nargs="+", default=DEFAULT_VARIANTS,
                    choices=list(VARIANT_REGISTRY))
    ap.add_argument("--n-heads", type=int, nargs="+", default=DEFAULT_N_HEADS)
    ap.add_argument("--dropouts", type=float, nargs="+", default=DEFAULT_DROPOUTS)
    ap.add_argument("--lrs", type=float, nargs="+", default=DEFAULT_LRS)
    ap.add_argument("--wds", type=float, nargs="+", default=DEFAULT_WDS)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", type=str, default=str(CKPT_DIR / "sweep_results.jsonl"))
    ap.add_argument("--ckpt", type=str, default=str(CKPT_DIR / "attention_best.pt"))
    ap.add_argument("--ckpt-metric", type=str, default="auroc",
                    choices=["auroc", "latent_mse"])
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def _method_label(variant: str) -> str:
    return {"attn_linear": "AttentionLinear",
            "attn_additive": "AttentionAdditive"}[variant]


def main():
    args = parse_args()
    device = pick_device(args.device)
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate heads divide d_model
    for h in args.n_heads:
        if D_MODEL % h != 0:
            raise SystemExit(f"n_heads={h} does not divide d_model={D_MODEL}")

    print(f"Loading data ...   device: {device}")
    train_split = load_split(DATA_DIR, "train")
    val_split = load_split(DATA_DIR, "val")
    head = load_regression_head(DATA_DIR / "regression_head.pt")
    print(f"  train: {len(train_split['ischemia'])}   val: {len(val_split['ischemia'])}")

    grid = list(itertools.product(args.variants, args.n_heads, args.dropouts,
                                  args.lrs, args.wds))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"  grid: {len(args.variants)} variants x {len(args.n_heads)} heads x "
          f"{len(args.dropouts)} dropouts x {len(args.lrs)} lrs x {len(args.wds)} wds "
          f"=  {len(grid)} runs")
    print(f"  run_id: {run_id}\n")

    def is_better(cand: dict, cur: dict | None) -> bool:
        if cur is None:
            return True
        if args.ckpt_metric == "auroc":
            return float(cand["val_auroc"]) > float(cur["val_auroc"])
        return float(cand["val_latent_mse"]) < float(cur["val_latent_mse"])

    rows, best_rec = [], None

    with out_path.open("a") as f:
        for i, (variant, n_heads, dropout, lr, wd) in enumerate(grid, 1):
            model, metrics = train_attention(
                train_split, val_split, head,
                variant=variant, n_heads=n_heads, dropout=dropout,
                epochs=args.epochs, batch_size=args.batch_size,
                lr=lr, lambda_l2=wd,
                seed=args.seed, device=device, verbose=args.verbose,
            )
            cfg_name = f"{variant}/h={n_heads}"
            rec = {
                "method": _method_label(variant),
                "config": cfg_name,
                "label": f"{cfg_name}/drop={dropout:g}/lr={lr:g}/wd={wd:g}",
                "run_id": run_id,
                "seed": int(args.seed),
                "variant": variant,
                "n_heads": int(n_heads),
                "dropout": float(dropout),
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
                save_attention(model, args.ckpt,
                               variant=variant, d_model=D_MODEL,
                               n_heads=int(n_heads), dropout=float(dropout),
                               config={"variant": variant,
                                       "n_heads": int(n_heads),
                                       "dropout": float(dropout),
                                       "lr": float(lr), "lambda_l2": float(wd),
                                       "epochs": int(args.epochs),
                                       "seed": int(args.seed),
                                       "ckpt_metric": args.ckpt_metric},
                               metrics=metrics)
                marker = "  <- new best, saved"
            extra = ""
            if "correction_magnitude_val" in metrics:
                extra = f"  corr/lin={metrics['correction_magnitude_val']:.3f}"
            print(f"[{i:2d}/{len(grid)}] {variant:<14s} h={n_heads} d={dropout:g} "
                  f"lr={lr:g} wd={wd:g}  "
                  f"params={rec['n_params']:>6d}  "
                  f"val_latent_mse={rec['val_latent_mse']:.4f}  "
                  f"val_auroc={rec['val_auroc']:.4f}{extra}  "
                  f"({rec['train_time_s']:.1f}s){marker}")

    # Summary grouped by (variant, n_heads)
    print("\n--- attention sweep summary (per variant/heads) ---")
    by_cfg: dict[str, list[dict]] = {}
    for r in rows:
        by_cfg.setdefault(r["config"], []).append(r)
    print(f"{'config':<22s} {'n':>3s}  "
          f"{'latent_mse  best (min)':>26s}  {'auroc  best (max)':>22s}")
    for variant in args.variants:
        for h in args.n_heads:
            key = f"{variant}/h={h}"
            items = by_cfg.get(key, [])
            if not items:
                continue
            lat = np.array([r["val_latent_mse"] for r in items])
            au  = np.array([r["val_auroc"] for r in items])
            best = min(items, key=lambda r: r["val_latent_mse"])
            print(f"{key:<22s} {len(items):>3d}  "
                  f"{lat.mean():.4f}+/-{lat.std():.4f}  ({lat.min():.4f})   "
                  f"{au.mean():.4f}+/-{au.std():.4f}  ({au.max():.4f})")
            print(f"            best @ drop={best['dropout']:g} lr={best['lr']:g} "
                  f"wd={best['lambda_l2']:g} -> "
                  f"latent_mse={best['val_latent_mse']:.4f} "
                  f"auroc={best['val_auroc']:.4f}")
    print(f"\nAppended {len(rows)} records to {out_path}")
    if best_rec is not None:
        print(f"\nBest attention model (by {args.ckpt_metric}) saved to {args.ckpt}")
        print(f"  variant={best_rec['variant']}  heads={best_rec['n_heads']}  "
              f"dropout={best_rec['dropout']:g}  lr={best_rec['lr']:g}  "
              f"wd={best_rec['lambda_l2']:g}")
        print(f"  val_auroc={best_rec['val_auroc']:.4f}  "
              f"val_latent_mse={best_rec['val_latent_mse']:.4f}  "
              f"params={best_rec['n_params']}")


if __name__ == "__main__":
    main()
