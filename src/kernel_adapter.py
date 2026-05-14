"""
Kernel Ridge residual adapter for AnchoredECG.

Solves the dual problem:

    alpha = (K + lambda I)^{-1} R    K_{ij} = k(u_i, u_j)

at fit time, then predicts r_hat = K_test @ alpha at inference. This tests
whether patient similarity in embedding space predicts residual similarity --
a non-parametric baseline against the linear / GReLU / MLP families.

Inputs are first centered + scaled with a StandardScaler fitted on train.
n_params is reported as the size of the dual coefficient matrix
(n_train x 192), matching the convention in the spec.

Sweep:
    kernel  in {rbf, laplace, polynomial}
    gamma   in {1e-4, 1e-3, 1e-2, 0.1, "scale"}     "scale" = 1/(d * X.var())
    alpha   in {1e-4, 1e-3, 1e-2, 0.1}              -> 60 fits
    For polynomial only: degree = 3 (fixed).

Memory: the kernel matrix is (n_train, n_train). For our dataset (n_train
~ 4.6k) it occupies ~170 MB in float64 -- fine in RAM, but each fit costs
O(n^3) so plan ~10-30 s per config. For n_train > 3000, consider switching
to a Nystrom approximation (sklearn.kernel_approximation.Nystroem).

Records are appended to checkpoints/sweep_results.jsonl with
method = "KernelRidge".

Usage:
    python src/kernel_adapter.py
    python src/kernel_adapter.py --kernels rbf laplace --alphas 1e-3 1e-2
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapter_interface import load_regression_head
from grelu_adapter import (
    build_tensors,
    evaluate_residual_adapter,
    load_split,
    pick_device,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

D_INPUT = 384
D_OUTPUT = 192

# sklearn names its laplacian kernel "laplacian"; we accept "laplace" too.
_KERNEL_ALIAS = {"rbf": "rbf", "laplace": "laplacian", "laplacian": "laplacian",
                 "polynomial": "polynomial", "poly": "polynomial"}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class KernelResidualAdapter(nn.Module):
    """Wraps a fitted sklearn KernelRidge + StandardScaler as an nn.Module."""

    def __init__(self, d_input: int = D_INPUT, d_output: int = D_OUTPUT):
        super().__init__()
        self.d_input = int(d_input)
        self.d_output = int(d_output)
        self.kernel_name: str | None = None
        self.gamma_value: float | None = None
        self.alpha_value: float | None = None
        self.degree: int = 3
        self.scaler = None        # StandardScaler
        self.kernel_model = None  # KernelRidge

    @staticmethod
    def _resolve_gamma(gamma, X_scaled: np.ndarray) -> float:
        if isinstance(gamma, str) and gamma.lower() == "scale":
            var = float(X_scaled.var())
            return 1.0 / (X_scaled.shape[1] * max(var, 1e-12))
        return float(gamma)

    def fit(self, u_tr: np.ndarray, r_tr: np.ndarray,
            kernel: str = "rbf", gamma="scale", alpha: float = 1e-3,
            degree: int = 3) -> None:
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.preprocessing import StandardScaler

        kname = _KERNEL_ALIAS.get(kernel.lower())
        if kname is None:
            raise ValueError(f"Unknown kernel {kernel!r}; "
                             f"choose from {sorted(set(_KERNEL_ALIAS))}")

        self.kernel_name = kname
        self.degree = int(degree)
        self.alpha_value = float(alpha)

        u_np = np.asarray(u_tr, dtype=np.float64)
        r_np = np.asarray(r_tr, dtype=np.float64)
        if u_np.shape[0] > 3000:
            print(f"  [kernel] WARNING: n_train={u_np.shape[0]} -- "
                  f"K is {u_np.shape[0]}x{u_np.shape[0]} entries "
                  f"(~{u_np.shape[0]**2*8/1e9:.2f} GB float64). "
                  f"Consider sklearn.kernel_approximation.Nystroem.")

        self.scaler = StandardScaler().fit(u_np)
        u_scaled = self.scaler.transform(u_np)
        gamma_val = self._resolve_gamma(gamma, u_scaled)
        self.gamma_value = gamma_val

        if kname == "polynomial":
            self.kernel_model = KernelRidge(
                alpha=float(alpha), kernel="polynomial",
                gamma=gamma_val, degree=self.degree, coef0=1.0,
            )
        else:
            self.kernel_model = KernelRidge(
                alpha=float(alpha), kernel=kname, gamma=gamma_val,
            )
        self.kernel_model.fit(u_scaled, r_np)

    @torch.no_grad()
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        if self.kernel_model is None or self.scaler is None:
            raise RuntimeError("KernelResidualAdapter has not been fit yet")
        u_np = u.detach().cpu().numpy().astype(np.float64)
        u_scaled = self.scaler.transform(u_np)
        r_hat = self.kernel_model.predict(u_scaled).astype(np.float32)
        return torch.from_numpy(r_hat).to(u.device).to(u.dtype)

    @torch.no_grad()
    def predict_latent(self, phi: torch.Tensor, z_rest: torch.Tensor) -> torch.Tensor:
        u = torch.cat([phi, z_rest], dim=1)
        return z_rest + self.forward(u)

    @property
    def n_params(self) -> int:
        if self.kernel_model is None or not hasattr(self.kernel_model, "dual_coef_"):
            return 0
        return int(np.asarray(self.kernel_model.dual_coef_).size)


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_kernel(adapter: KernelResidualAdapter, path: str | Path,
                config: dict | None = None, metrics: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "kernel_model": adapter.kernel_model,
        "scaler": adapter.scaler,
        "kernel_name": adapter.kernel_name,
        "gamma_value": adapter.gamma_value,
        "alpha_value": adapter.alpha_value,
        "degree": adapter.degree,
        "d_input": adapter.d_input,
        "d_output": adapter.d_output,
        "config": config,
        "metrics": metrics,
    }, path)


def load_kernel(path: str | Path) -> KernelResidualAdapter:
    obj = joblib.load(path)
    a = KernelResidualAdapter(d_input=obj["d_input"], d_output=obj["d_output"])
    a.kernel_model = obj["kernel_model"]
    a.scaler = obj["scaler"]
    a.kernel_name = obj["kernel_name"]
    a.gamma_value = obj["gamma_value"]
    a.alpha_value = obj["alpha_value"]
    a.degree = obj.get("degree", 3)
    a.eval()
    return a


# ---------------------------------------------------------------------------
# Train + eval one configuration
# ---------------------------------------------------------------------------

def _build_numpy(train_split: dict, val_split: dict, device):
    u_tr, r_tr, _, _ = build_tensors(train_split, device=device)
    u_va, r_va, zr_va, zs_va = build_tensors(val_split, device=device)
    return (u_tr.cpu().numpy(), r_tr.cpu().numpy(),
            u_va, r_va, zr_va, zs_va)


def fit_one(u_tr_np, r_tr_np, u_va, r_va, zr_va, zs_va, head, val_split, device,
            kernel: str, gamma, alpha: float, degree: int = 3
            ) -> tuple[KernelResidualAdapter, dict]:
    adapter = KernelResidualAdapter()
    t0 = time.time()
    adapter.fit(u_tr_np, r_tr_np, kernel=kernel, gamma=gamma,
                alpha=alpha, degree=degree)
    train_time = time.time() - t0
    adapter.to(device).eval()
    m = evaluate_residual_adapter(
        adapter, u_va, r_va, zr_va, zs_va,
        head=head, y=val_split["ischemia"], bl_st80=val_split["bl_st80"],
    )
    metrics = {
        "val_residual_mse": m["residual_mse"],
        "val_latent_mse": m["latent_mse"],
        "val_auroc": m.get("auroc", float("nan")),
        "train_mse": float("nan"),
        "n_params": adapter.n_params,
        "train_time_s": train_time,
    }
    return adapter, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_KERNELS = ["rbf", "laplace", "polynomial"]
DEFAULT_GAMMAS = [1e-4, 1e-3, 1e-2, 0.1, "scale"]
DEFAULT_ALPHAS = [1e-4, 1e-3, 1e-2, 0.1]


def _parse_gamma(v: str):
    if v.lower() == "scale":
        return "scale"
    return float(v)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kernels", type=str, nargs="+", default=DEFAULT_KERNELS,
                    help="One or more of: rbf laplace polynomial")
    ap.add_argument("--gammas", type=_parse_gamma, nargs="+", default=DEFAULT_GAMMAS,
                    help='Bandwidth values, with "scale" allowed (=> 1/(d * X.var()))')
    ap.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    ap.add_argument("--degree", type=int, default=3,
                    help="Polynomial degree (only used for kernel=polynomial).")
    ap.add_argument("--epochs", type=int, default=0,
                    help="Unused (closed-form solver); accepted for CLI symmetry.")
    ap.add_argument("--batch-size", type=int, default=256,
                    help="Unused (closed-form solver); accepted for CLI symmetry.")
    ap.add_argument("--out", type=str, default=str(CKPT_DIR / "sweep_results.jsonl"))
    ap.add_argument("--ckpt", type=str, default=str(CKPT_DIR / "kernel_best.pt"))
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

    u_tr_np, r_tr_np, u_va, r_va, zr_va, zs_va = _build_numpy(
        train_split, val_split, device,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    grid = list(itertools.product(args.kernels, args.gammas, args.alphas))
    print(f"  grid: {len(args.kernels)} kernels x {len(args.gammas)} gammas x "
          f"{len(args.alphas)} alphas  =  {len(grid)} runs")
    print(f"  run_id: {run_id}\n")

    def is_better(cand: dict, cur: dict | None) -> bool:
        if cur is None:
            return True
        if args.ckpt_metric == "auroc":
            return float(cand["val_auroc"]) > float(cur["val_auroc"])
        return float(cand["val_latent_mse"]) < float(cur["val_latent_mse"])

    rows, best_rec = [], None

    with out_path.open("a") as f:
        for i, (kernel, gamma, alpha) in enumerate(grid, 1):
            gamma_label = "scale" if gamma == "scale" else f"{float(gamma):g}"
            cfg_name = f"{kernel}/g={gamma_label}/a={alpha:g}"
            adapter, metrics = fit_one(
                u_tr_np, r_tr_np, u_va, r_va, zr_va, zs_va, head, val_split, device,
                kernel=kernel, gamma=gamma, alpha=alpha, degree=args.degree,
            )
            rec = {
                "method": "KernelRidge",
                "config": kernel,
                "label": f"KernelRidge/{cfg_name}",
                "run_id": run_id,
                "seed": int(args.seed),
                "kernel": kernel,
                "gamma": ("scale" if gamma == "scale" else float(gamma)),
                "gamma_value_used": adapter.gamma_value,
                "alpha": float(alpha),
                "degree": int(args.degree) if kernel.lower() in ("polynomial", "poly") else None,
                **metrics,
            }
            f.write(json.dumps(rec) + "\n"); f.flush()
            rows.append(rec)
            marker = ""
            if is_better(rec, best_rec):
                best_rec = rec
                save_kernel(adapter, args.ckpt,
                            config={"kernel": kernel, "gamma": rec["gamma"],
                                    "alpha": float(alpha), "degree": rec["degree"],
                                    "ckpt_metric": args.ckpt_metric,
                                    "seed": int(args.seed)},
                            metrics=metrics)
                marker = "  <- new best, saved"
            print(f"[{i:2d}/{len(grid)}] KernelRidge {cfg_name:<28s}  "
                  f"params={metrics['n_params']:>7d}  "
                  f"val_latent_mse={metrics['val_latent_mse']:.4f}  "
                  f"val_auroc={metrics['val_auroc']:.4f}  "
                  f"({metrics['train_time_s']:.1f}s){marker}")

    # Summary grouped by kernel
    print("\n--- kernel ridge sweep summary (per kernel) ---")
    by_kern: dict[str, list[dict]] = {}
    for r in rows:
        by_kern.setdefault(r["kernel"], []).append(r)
    print(f"{'kernel':<12s} {'n':>3s}  "
          f"{'latent_mse  best (min)':>26s}  {'auroc  best (max)':>22s}")
    for name in args.kernels:
        items = by_kern.get(name, [])
        if not items:
            continue
        lat = np.array([r["val_latent_mse"] for r in items])
        au  = np.array([r["val_auroc"] for r in items])
        best = min(items, key=lambda r: r["val_latent_mse"])
        print(f"{name:<12s} {len(items):>3d}  "
              f"{lat.mean():.4f}+/-{lat.std():.4f}  ({lat.min():.4f})   "
              f"{au.mean():.4f}+/-{au.std():.4f}  ({au.max():.4f})")
        gamma_str = best["gamma"] if isinstance(best["gamma"], str) else f"{best['gamma']:g}"
        print(f"            best @ gamma={gamma_str} alpha={best['alpha']:g} "
              f"-> latent_mse={best['val_latent_mse']:.4f} "
              f"auroc={best['val_auroc']:.4f}")
    print(f"\nAppended {len(rows)} records to {out_path}")
    if best_rec is not None:
        print(f"\nBest KernelRidge (by {args.ckpt_metric}) saved to {args.ckpt}")
        gamma_str = best_rec["gamma"] if isinstance(best_rec["gamma"], str) else f"{best_rec['gamma']:g}"
        print(f"  kernel={best_rec['kernel']}  gamma={gamma_str}  alpha={best_rec['alpha']:g}")
        print(f"  val_auroc={best_rec['val_auroc']:.4f}  "
              f"val_latent_mse={best_rec['val_latent_mse']:.4f}  "
              f"params={best_rec['n_params']}")


if __name__ == "__main__":
    main()
