"""
Polish a trained convex GReLU residual adapter.

After convex training with fixed gates, two standard refinement steps are
useful in the Pilanci / Varshney & Pilanci convex-distillation pipeline:

  1. Magnitude pruning + convex refit
     Compute per-neuron importance s_j = ||proj.weight[:, j]||_2 and keep the
     top-k neurons. Zero the rest. The pruned model is STILL convex (gates
     remain frozen), so we refit the surviving (linear1, proj) rows/columns
     by re-running Adam on the MSE objective. This sweep produces an
     accuracy-vs-sparsity curve, mirroring the SCNN VaryingSparsity notebooks.

  2. Non-convex polish (optional)
     Unfreeze the gate weights `linear2` and continue training with Adam at a
     small learning rate for a few epochs. This sacrifices the convexity
     guarantee but typically recovers a small amount of additional accuracy
     by letting the gates adapt to the surviving neurons.

Usage:
    python src/polish.py --ckpt checkpoints/grelu_best.pt
    python src/polish.py --ckpt checkpoints/grelu_best.pt --keep 64 128 256 512 1024
    python src/polish.py --ckpt checkpoints/grelu_best.pt --nonconvex --polish-epochs 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapter_interface import load_regression_head
from grelu_adapter import (
    DEFAULT_CONFIG,
    GReLUResidualAdapter,
    build_tensors,
    evaluate_residual_adapter,
    load_adapter,
    load_split,
    make_loader,
    pick_device,
    save_adapter,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


# ---------------------------------------------------------------------------
# Pruning helpers
# ---------------------------------------------------------------------------

def neuron_importance(adapter: GReLUResidualAdapter) -> torch.Tensor:
    """Per-neuron importance score = ||proj.weight[:, j]||_2 (output L2 norm).

    Returns shape (m,) on CPU.
    """
    return adapter.proj.weight.detach().cpu().norm(dim=0)  # (d_output, m) -> (m,)


def prune_to_topk(adapter: GReLUResidualAdapter, k: int) -> tuple[GReLUResidualAdapter, torch.Tensor]:
    """Zero out all but the top-k neurons by importance. Returns the pruned
    adapter (modified IN PLACE) and a (m,) boolean keep-mask."""
    s = neuron_importance(adapter)
    m = adapter.m
    if k >= m:
        return adapter, torch.ones(m, dtype=torch.bool)

    topk_vals, topk_idx = torch.topk(s, k)
    keep = torch.zeros(m, dtype=torch.bool)
    keep[topk_idx] = True
    keep_dev = keep.to(adapter.linear1.weight.device)

    with torch.no_grad():
        # Zero entire rows of linear1 for dropped neurons and entire columns
        # of proj. linear2 (gate) is left alone -- it's frozen anyway and we
        # only ever read its sign through the gate.
        adapter.linear1.weight[~keep_dev, :] = 0.0
        adapter.proj.weight[:, ~keep_dev] = 0.0

    return adapter, keep


def refit_active(
    adapter: GReLUResidualAdapter,
    keep: torch.Tensor,
    u_tr: torch.Tensor, r_tr: torch.Tensor,
    epochs: int = 30,
    lr: float = 1e-3,
    lambda_l2: float = 1e-4,
    batch_size: int = 256,
    verbose: bool = False,
) -> None:
    """Re-fit only the surviving neurons' parameters via MSE on (U, R).

    Gates linear2 stay frozen, so the pruned problem is still convex.
    """
    keep_dev = keep.to(adapter.linear1.weight.device)
    loader = make_loader(u_tr, r_tr, batch_size, shuffle=True, seed=0)
    optim = torch.optim.Adam(adapter.trainable_parameters(), lr=lr, weight_decay=lambda_l2)
    loss_fn = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        adapter.train()
        running, n = 0.0, 0
        for ub, rb in loader:
            optim.zero_grad()
            pred = adapter(ub)
            loss = loss_fn(pred, rb)
            loss.backward()
            # Mask out dead-neuron gradients each step so they stay at zero.
            with torch.no_grad():
                adapter.linear1.weight.grad[~keep_dev, :] = 0.0
                adapter.proj.weight.grad[:, ~keep_dev] = 0.0
            optim.step()
            with torch.no_grad():
                adapter.linear1.weight[~keep_dev, :] = 0.0
                adapter.proj.weight[:, ~keep_dev] = 0.0
            running += loss.item() * ub.size(0)
            n += ub.size(0)
        if verbose and (epoch == 1 or epoch % max(1, epochs // 5) == 0):
            print(f"    refit epoch {epoch:3d}/{epochs}  train_mse={running/max(n,1):.6f}")


# ---------------------------------------------------------------------------
# Non-convex polish
# ---------------------------------------------------------------------------

def nonconvex_polish(
    adapter: GReLUResidualAdapter,
    u_tr: torch.Tensor, r_tr: torch.Tensor,
    epochs: int = 30,
    lr: float = 1e-4,
    lambda_l2: float = 1e-4,
    batch_size: int = 256,
    verbose: bool = True,
) -> None:
    """Unfreeze the gate weights and continue training with Adam.

    This step gives up convexity but lets the gates adapt to the post-pruning
    neuron set. Use a small lr so the gates don't drift far from their
    convex-optimal patterns.
    """
    adapter.linear2.weight.requires_grad_(True)
    loader = make_loader(u_tr, r_tr, batch_size, shuffle=True, seed=0)
    params = [p for p in adapter.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params, lr=lr, weight_decay=lambda_l2)
    loss_fn = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        adapter.train()
        running, n = 0.0, 0
        for ub, rb in loader:
            optim.zero_grad()
            pred = adapter(ub)
            loss = loss_fn(pred, rb)
            loss.backward()
            optim.step()
            running += loss.item() * ub.size(0)
            n += ub.size(0)
        if verbose and (epoch == 1 or epoch % max(1, epochs // 5) == 0):
            print(f"    polish epoch {epoch:3d}/{epochs}  train_mse={running/max(n,1):.6f}")

    # Re-freeze for downstream use so saved checkpoint behavior is unambiguous.
    adapter.linear2.weight.requires_grad_(False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(CKPT_DIR / "grelu_best.pt"),
                    help="Path to trained convex adapter")
    ap.add_argument("--keep", type=int, nargs="*",
                    default=[16, 32, 64, 128, 256, 512, 1024],
                    help="Top-k neuron counts to sweep over")
    ap.add_argument("--refit-epochs", type=int, default=30)
    ap.add_argument("--refit-lr", type=float, default=1e-3)
    ap.add_argument("--lambda-l2", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--nonconvex", action="store_true",
                    help="After the densest refit, run a non-convex polish")
    ap.add_argument("--polish-epochs", type=int, default=30)
    ap.add_argument("--polish-lr", type=float, default=1e-4)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out", type=str, default=str(CKPT_DIR / "grelu_polished.pt"))
    ap.add_argument("--csv", type=str, default=str(CKPT_DIR / "polish_sweep.csv"))
    return ap.parse_args()


def main():
    args = parse_args()
    device = pick_device(args.device)

    print(f"Loading data ...   device: {device}")
    train_split = load_split(DATA_DIR, "train")
    val_split = load_split(DATA_DIR, "val")
    head = load_regression_head(DATA_DIR / "regression_head.pt")

    u_tr, r_tr, _, _ = build_tensors(train_split, device=device)
    u_va, r_va, zr_va, zs_va = build_tensors(val_split, device=device)

    # Baseline metrics on the converged convex adapter
    print(f"\nLoading convex adapter from {args.ckpt} ...")
    adapter0 = load_adapter(args.ckpt, device=device)
    base = evaluate_residual_adapter(
        adapter0, u_va, r_va, zr_va, zs_va,
        head=head, y=val_split["ischemia"], bl_st80=val_split["bl_st80"],
    )
    print("  baseline:", " ".join(f"{k}={v:.4f}" for k, v in base.items()))

    # Importance histogram summary
    s = neuron_importance(adapter0)
    print(f"  neuron importance: min={s.min():.4f}  median={s.median():.4f}  "
          f"max={s.max():.4f}  >1%-of-max={(s > 0.01 * s.max()).sum().item()}/{adapter0.m}")

    # ---- pruning sweep ------------------------------------------------------
    rows = [("keep", "residual_mse", "latent_mse", "auroc")]
    rows.append(("baseline_full", base["residual_mse"], base["latent_mse"],
                 base.get("auroc", float("nan"))))

    best_metric = float("inf")
    best_keep = None
    best_state = None

    for k in sorted(set(args.keep)):
        if k > adapter0.m:
            continue
        print(f"\n=== keep top-{k} of {adapter0.m} neurons ===")
        adapter = load_adapter(args.ckpt, device=device)
        adapter, keep = prune_to_topk(adapter, k)
        # Metrics RIGHT after pruning (no refit) -- shows the raw expressivity loss
        m_after_prune = evaluate_residual_adapter(
            adapter, u_va, r_va, zr_va, zs_va,
            head=head, y=val_split["ischemia"], bl_st80=val_split["bl_st80"],
        )
        print(f"  after pruning:        residual_mse={m_after_prune['residual_mse']:.6f}  "
              f"latent_mse={m_after_prune['latent_mse']:.6f}  "
              f"auroc={m_after_prune.get('auroc', float('nan')):.4f}")

        refit_active(adapter, keep, u_tr, r_tr,
                     epochs=args.refit_epochs, lr=args.refit_lr,
                     lambda_l2=args.lambda_l2, batch_size=args.batch_size,
                     verbose=True)
        m_after_refit = evaluate_residual_adapter(
            adapter, u_va, r_va, zr_va, zs_va,
            head=head, y=val_split["ischemia"], bl_st80=val_split["bl_st80"],
        )
        print(f"  after refit:          residual_mse={m_after_refit['residual_mse']:.6f}  "
              f"latent_mse={m_after_refit['latent_mse']:.6f}  "
              f"auroc={m_after_refit.get('auroc', float('nan')):.4f}")

        rows.append((k, m_after_refit["residual_mse"], m_after_refit["latent_mse"],
                     m_after_refit.get("auroc", float("nan"))))

        if m_after_refit["latent_mse"] < best_metric:
            best_metric = m_after_refit["latent_mse"]
            best_keep = k
            best_state = {k_: v.detach().cpu().clone() for k_, v in adapter.state_dict().items()}

    # ---- optional non-convex polish on the best pruned model ----------------
    if args.nonconvex and best_state is not None:
        print(f"\n=== non-convex polish on top-{best_keep} model "
              f"({args.polish_epochs} epochs, lr={args.polish_lr}) ===")
        adapter = load_adapter(args.ckpt, device=device)
        adapter.load_state_dict({k_: v.to(device) for k_, v in best_state.items()})
        nonconvex_polish(adapter, u_tr, r_tr,
                         epochs=args.polish_epochs, lr=args.polish_lr,
                         lambda_l2=args.lambda_l2, batch_size=args.batch_size)
        m_polished = evaluate_residual_adapter(
            adapter, u_va, r_va, zr_va, zs_va,
            head=head, y=val_split["ischemia"], bl_st80=val_split["bl_st80"],
        )
        print(f"  after non-convex polish:  residual_mse={m_polished['residual_mse']:.6f}  "
              f"latent_mse={m_polished['latent_mse']:.6f}  "
              f"auroc={m_polished.get('auroc', float('nan')):.4f}")
        rows.append((f"nonconvex@top{best_keep}", m_polished["residual_mse"],
                     m_polished["latent_mse"], m_polished.get("auroc", float("nan"))))
        if m_polished["latent_mse"] < best_metric:
            best_state = {k_: v.detach().cpu().clone() for k_, v in adapter.state_dict().items()}
            best_metric = m_polished["latent_mse"]

    # ---- save best + write CSV ---------------------------------------------
    if best_state is not None:
        adapter = load_adapter(args.ckpt, device=device)
        adapter.load_state_dict({k_: v.to(device) for k_, v in best_state.items()})
        save_adapter(adapter, args.out, config={**DEFAULT_CONFIG, "polished": True})
        print(f"\nBest polished model -> {args.out}  (val_latent_mse={best_metric:.6f})")

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w") as f:
        for row in rows:
            f.write(",".join(str(c) for c in row) + "\n")
    print(f"Sweep CSV  -> {csv_path}")


if __name__ == "__main__":
    main()
