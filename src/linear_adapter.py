"""
Elastic-net linear residual adapter for the AnchoredECG latent recovery problem.

Implements the sparse linear residual adapter from the proposal Eq. (1):

    min_W   sum_i ||W u_i - r_i||^2 + lambda_1 ||W||_1 + lambda_2 ||W||_F^2

where u_i = concat(phi_stress, z_rest) and r_i = z_stress - z_rest. We use
sklearn's MultiTaskElasticNet (group-L1 across output dims, plus L2) so the
penalty is applied row-wise across input features, which matches the
proposal's intent of sparse dependence on input dimensions. Pure L1
(l1_ratio == 1) falls back to MultiTaskLasso; pure L2 (l1_ratio == 0) falls
back to Ridge to avoid sklearn's noisy convergence path at the boundary.

Sweep:
    l1_ratio  in {0.0, 0.1, 0.5, 0.9, 1.0}
    alpha     in {1e-4, 1e-3, 1e-2, 0.1, 1.0}        -> 25 ElasticNet fits

Additional ablations as named configs:
    "prior_only"        predict r = 0    (i.e. z_stress_hat = z_rest, no fit)
    "single_lead_only"  zero the z_rest block of u before fitting + predicting

Records are appended to checkpoints/sweep_results.jsonl with
method = "ElasticNet" | "prior_only" | "single_lead_only".

Usage:
    python src/linear_adapter.py
    python src/linear_adapter.py --epochs 2000 --elastic-tol 5e-4
    python src/linear_adapter.py --l1-ratios 0 0.5 1
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import warnings
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


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LinearResidualAdapter(nn.Module):
    """Wraps a fitted sklearn linear estimator (or a no-op ablation) so it
    can be called as `adapter(u: Tensor) -> Tensor` by
    `evaluate_residual_adapter`.

    Ablations supported via `ablation`:
        "none"             standard fit on u = [phi; z_rest]
        "prior_only"       no fit; predicts r = 0  ->  z_stress_hat = z_rest
        "single_lead_only" zero the z_rest block of u at fit and at predict
    """

    def __init__(self, d_input: int = D_INPUT, d_output: int = D_OUTPUT,
                 ablation: str = "none"):
        super().__init__()
        if ablation not in {"none", "prior_only", "single_lead_only"}:
            raise ValueError(f"unknown ablation {ablation!r}")
        self.d_input = int(d_input)
        self.d_output = int(d_output)
        self.ablation = ablation
        self.sklearn_model = None  # filled by fit()

    def _maybe_zero_prior(self, u_np: np.ndarray) -> np.ndarray:
        if self.ablation == "single_lead_only":
            u_np = u_np.copy()
            u_np[:, self.d_output:] = 0.0
        return u_np

    def fit(self, u_tr: np.ndarray, r_tr: np.ndarray,
            alpha: float = 1e-3, l1_ratio: float = 0.5,
            max_iter: int = 1200, tol: float = 1e-3,
            random_state: int | None = None, selection: str = "random") -> None:
        """Fit on numpy arrays. No-op for the prior_only ablation.

        MultiTaskElasticNet / MultiTaskLasso use coordinate descent, which can
        appear to "hang" on ~4k × 384 problems if `max_iter` is huge and `tol`
        is very tight. Defaults are chosen to finish in tens of seconds typical
        laptops; increase `--epochs` / tighten `--elastic-tol` if you need higher
        numerical accuracy.
        """
        if self.ablation == "prior_only":
            self.sklearn_model = None
            return

        u_np = self._maybe_zero_prior(np.asarray(u_tr, dtype=np.float64))
        r_np = np.asarray(r_tr, dtype=np.float64)

        if l1_ratio <= 1e-9:
            from sklearn.linear_model import Ridge
            self.sklearn_model = Ridge(alpha=float(alpha))
        elif l1_ratio >= 1.0 - 1e-9:
            from sklearn.linear_model import MultiTaskLasso
            self.sklearn_model = MultiTaskLasso(
                alpha=float(alpha), max_iter=int(max_iter), tol=float(tol),
                selection=selection,
                random_state=random_state,
            )
        else:
            from sklearn.linear_model import MultiTaskElasticNet
            self.sklearn_model = MultiTaskElasticNet(
                alpha=float(alpha), l1_ratio=float(l1_ratio),
                max_iter=int(max_iter), tol=float(tol),
                selection=selection,
                random_state=random_state,
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # silence ConvergenceWarning at high lambda
            self.sklearn_model.fit(u_np, r_np)

    @torch.no_grad()
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        if self.ablation == "prior_only":
            return torch.zeros(u.shape[0], self.d_output,
                               device=u.device, dtype=u.dtype)
        if self.sklearn_model is None:
            raise RuntimeError("LinearResidualAdapter has not been fit yet")
        u_np = self._maybe_zero_prior(u.detach().cpu().numpy().astype(np.float64))
        r_hat = self.sklearn_model.predict(u_np).astype(np.float32)
        return torch.from_numpy(r_hat).to(u.device).to(u.dtype)

    @torch.no_grad()
    def predict_latent(self, phi: torch.Tensor, z_rest: torch.Tensor) -> torch.Tensor:
        u = torch.cat([phi, z_rest], dim=1)
        return z_rest + self.forward(u)

    @property
    def n_params(self) -> int:
        if self.ablation == "prior_only":
            return 0
        # W is (n_outputs, n_inputs)
        return int(self.d_input * self.d_output)


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_linear(adapter: LinearResidualAdapter, path: str | Path,
                config: dict | None = None, metrics: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "sklearn_model": adapter.sklearn_model,
        "ablation": adapter.ablation,
        "d_input": adapter.d_input,
        "d_output": adapter.d_output,
        "config": config,
        "metrics": metrics,
    }, path)


def load_linear(path: str | Path) -> LinearResidualAdapter:
    obj = joblib.load(path)
    a = LinearResidualAdapter(
        d_input=obj["d_input"], d_output=obj["d_output"], ablation=obj["ablation"],
    )
    a.sklearn_model = obj["sklearn_model"]
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


def _eval_on_val(adapter: LinearResidualAdapter, u_va, r_va, zr_va, zs_va,
                 head, val_split, device) -> dict:
    adapter.to(device).eval()
    return evaluate_residual_adapter(
        adapter, u_va, r_va, zr_va, zs_va,
        head=head, y=val_split["ischemia"], bl_st80=val_split["bl_st80"],
    )


def fit_one(u_tr_np: np.ndarray, r_tr_np: np.ndarray,
            u_va, r_va, zr_va, zs_va, head, val_split, device,
            alpha: float, l1_ratio: float, max_iter: int,
            ablation: str = "none",
            elastic_tol: float = 1e-3,
            random_state: int | None = None,
            selection: str = "random",
            progress: str | None = None) -> tuple[LinearResidualAdapter, dict, float]:
    """Fit a single LinearResidualAdapter and return (adapter, metrics, train_time)."""
    adapter = LinearResidualAdapter(ablation=ablation)
    if progress:
        print(progress, flush=True)
    t0 = time.time()
    adapter.fit(u_tr_np, r_tr_np, alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter,
                tol=elastic_tol, random_state=random_state, selection=selection)
    train_time = time.time() - t0
    m = _eval_on_val(adapter, u_va, r_va, zr_va, zs_va, head, val_split, device)
    metrics = {
        "val_residual_mse": m["residual_mse"],
        "val_latent_mse": m["latent_mse"],
        "val_auroc": m.get("auroc", float("nan")),
        "train_mse": float("nan"),  # closed-form solvers don't expose a train loss curve
        "n_params": adapter.n_params,
        "train_time_s": train_time,
    }
    return adapter, metrics, train_time


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_L1_RATIOS = [0.0, 0.1, 0.5, 0.9, 1.0]
DEFAULT_ALPHAS = [1e-4, 1e-3, 1e-2, 0.1, 1.0]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--l1-ratios", type=float, nargs="+", default=DEFAULT_L1_RATIOS)
    ap.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    ap.add_argument("--epochs", type=int, default=1200,
                    help="max_iter for MultiTaskElasticNet / MultiTaskLasso. "
                         "Very large values slow each fit a lot on this dataset size.")
    ap.add_argument("--elastic-tol", type=float, default=1e-3,
                    help="Stopping tolerance for coordinate descent (ElasticNet/Lasso). "
                         "Smaller = more accurate but much slower.")
    ap.add_argument("--batch-size", type=int, default=256,
                    help="Unused (closed-form solver); accepted for CLI symmetry.")
    ap.add_argument("--out", type=str, default=str(CKPT_DIR / "sweep_results.jsonl"))
    ap.add_argument("--ckpt", type=str, default=str(CKPT_DIR / "linear_best.pt"))
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
    n_grid_fits = len(args.l1_ratios) * len(args.alphas)
    n_total = n_grid_fits + 2  # +2 ablations
    print(f"  grid: {len(args.l1_ratios)} l1_ratios × {len(args.alphas)} alphas = {n_grid_fits} fits "
          f"(ElasticNet/Lasso reuse warm_start across alphas per l1_ratio) + 2 ablations = {n_total} runs")
    print(f"  run_id: {run_id}\n")

    def is_better(cand: dict, cur: dict | None) -> bool:
        if cur is None:
            return True
        if args.ckpt_metric == "auroc":
            return float(cand["val_auroc"]) > float(cur["val_auroc"])
        return float(cand["val_latent_mse"]) < float(cur["val_latent_mse"])

    rows: list[dict] = []
    best_rec: dict | None = None

    with out_path.open("a") as f:

        def _record(method: str, config_name: str, label: str, extra: dict,
                    adapter: LinearResidualAdapter, metrics: dict, idx: int):
            nonlocal best_rec
            rec = {
                "method": method,
                "config": config_name,
                "label": label,
                "run_id": run_id,
                "seed": int(args.seed),
                **extra,
                **metrics,
            }
            f.write(json.dumps(rec) + "\n"); f.flush()
            rows.append(rec)
            marker = ""
            if is_better(rec, best_rec):
                best_rec = rec
                save_linear(adapter, args.ckpt,
                            config={**extra, "method": method,
                                    "ckpt_metric": args.ckpt_metric,
                                    "seed": int(args.seed)},
                            metrics=metrics)
                marker = "  <- new best, saved"
            print(f"[{idx:2d}/{n_total}] {method:<16s} {config_name:<18s} "
                  f"params={metrics['n_params']:>6d}  "
                  f"val_latent_mse={metrics['val_latent_mse']:.4f}  "
                  f"val_auroc={metrics['val_auroc']:.4f}  "
                  f"({metrics['train_time_s']:.2f}s){marker}")

        idx = 0
        selection = "random"
        rs = int(args.seed)

        for l1_ratio in args.l1_ratios:
            if l1_ratio <= 1e-9:
                for alpha in args.alphas:
                    idx += 1
                    prog = None
                    if args.verbose:
                        prog = (f"    [linear] Ridge  alpha={alpha:g}  "
                                f"(solve via Cholesky — fast) ...")
                    adapter, metrics, _ = fit_one(
                        u_tr_np, r_tr_np, u_va, r_va, zr_va, zs_va, head, val_split, device,
                        alpha=alpha, l1_ratio=l1_ratio, max_iter=args.epochs, ablation="none",
                        elastic_tol=args.elastic_tol, random_state=rs,
                        selection=selection, progress=prog,
                    )
                    cfg_name = f"alpha={alpha:g}/l1={l1_ratio:g}"
                    _record("ElasticNet", cfg_name, f"ElasticNet/{cfg_name}",
                            {"alpha": float(alpha), "l1_ratio": float(l1_ratio),
                             "max_iter": int(args.epochs), "elastic_tol": float(args.elastic_tol),
                             "warm_path": False},
                            adapter, metrics, idx)
                continue

            if l1_ratio >= 1.0 - 1e-9:
                from sklearn.linear_model import MultiTaskLasso
                u_fit = np.asarray(u_tr_np, dtype=np.float64)
                r_fit = np.asarray(r_tr_np, dtype=np.float64)
                lasso = MultiTaskLasso(
                    warm_start=True, max_iter=int(args.epochs), tol=float(args.elastic_tol),
                    selection=selection, random_state=rs,
                )
                for alpha in sorted(args.alphas, reverse=True):
                    idx += 1
                    msg = (f"    [linear] MultiTaskLasso  alpha={alpha:g}  "
                           f"(coordinate descent, warm_start) ...")
                    print(msg, flush=True)
                    t0 = time.time()
                    lasso.set_params(alpha=float(alpha))
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        lasso.fit(u_fit, r_fit)
                    train_time = time.time() - t0
                    adapter = LinearResidualAdapter(ablation="none")
                    adapter.sklearn_model = copy.deepcopy(lasso)
                    m = _eval_on_val(adapter, u_va, r_va, zr_va, zs_va, head, val_split, device)
                    metrics = {
                        "val_residual_mse": m["residual_mse"],
                        "val_latent_mse": m["latent_mse"],
                        "val_auroc": m.get("auroc", float("nan")),
                        "train_mse": float("nan"),
                        "n_params": adapter.n_params,
                        "train_time_s": train_time,
                    }
                    cfg_name = f"alpha={alpha:g}/l1={l1_ratio:g}"
                    _record("ElasticNet", cfg_name, f"ElasticNet/{cfg_name}",
                            {"alpha": float(alpha), "l1_ratio": float(l1_ratio),
                             "max_iter": int(args.epochs), "elastic_tol": float(args.elastic_tol),
                             "warm_path": True},
                            adapter, metrics, idx)
                continue

            # General ElasticNet block: amortize coordinate descent across alphas
            from sklearn.linear_model import MultiTaskElasticNet
            u_fit = np.asarray(u_tr_np, dtype=np.float64)
            r_fit = np.asarray(r_tr_np, dtype=np.float64)
            enet = MultiTaskElasticNet(
                l1_ratio=float(l1_ratio),
                warm_start=True, max_iter=int(args.epochs), tol=float(args.elastic_tol),
                selection=selection, random_state=rs,
            )
            for alpha in sorted(args.alphas, reverse=True):
                idx += 1
                print(
                    f"    [linear] MultiTaskElasticNet  l1_ratio={l1_ratio:g}  alpha={alpha:g}  "
                    f"(coordinate descent — can take tens of seconds; warm_start amortizes alphas)",
                    flush=True,
                )
                t0 = time.time()
                enet.set_params(alpha=float(alpha))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    enet.fit(u_fit, r_fit)
                train_time = time.time() - t0
                adapter = LinearResidualAdapter(ablation="none")
                adapter.sklearn_model = copy.deepcopy(enet)
                m = _eval_on_val(adapter, u_va, r_va, zr_va, zs_va, head, val_split, device)
                metrics = {
                    "val_residual_mse": m["residual_mse"],
                    "val_latent_mse": m["latent_mse"],
                    "val_auroc": m.get("auroc", float("nan")),
                    "train_mse": float("nan"),
                    "n_params": adapter.n_params,
                    "train_time_s": train_time,
                }
                cfg_name = f"alpha={alpha:g}/l1={l1_ratio:g}"
                _record("ElasticNet", cfg_name, f"ElasticNet/{cfg_name}",
                        {"alpha": float(alpha), "l1_ratio": float(l1_ratio),
                         "max_iter": int(args.epochs), "elastic_tol": float(args.elastic_tol),
                         "warm_path": True},
                        adapter, metrics, idx)

        # Ablation 1: prior_only (r=0, no fit). Recorded as its own method.
        idx += 1
        adapter = LinearResidualAdapter(ablation="prior_only")
        t0 = time.time()
        # No fit -- still time the (no-op) construction to keep schema consistent
        train_time = time.time() - t0
        m = _eval_on_val(adapter, u_va, r_va, zr_va, zs_va, head, val_split, device)
        metrics_prior = {
            "val_residual_mse": m["residual_mse"],
            "val_latent_mse": m["latent_mse"],
            "val_auroc": m.get("auroc", float("nan")),
            "train_mse": float("nan"),
            "n_params": adapter.n_params,
            "train_time_s": train_time,
        }
        _record("prior_only", "prior_only", "prior_only/no-fit",
                {"alpha": None, "l1_ratio": None}, adapter, metrics_prior, idx)

        # Ablation 2: single_lead_only with the best ElasticNet-grid alpha/l1.
        # Pick the best alpha/l1 by AUROC so the ablation is fair.
        idx += 1
        best_enet = max((r for r in rows if r["method"] == "ElasticNet"),
                        key=lambda r: r["val_auroc"])
        adapter, metrics_single, _ = fit_one(
            u_tr_np, r_tr_np, u_va, r_va, zr_va, zs_va, head, val_split, device,
            alpha=best_enet["alpha"], l1_ratio=best_enet["l1_ratio"],
            max_iter=args.epochs, ablation="single_lead_only",
            elastic_tol=args.elastic_tol, random_state=rs, selection=selection,
            progress="    [linear] single_lead_only ablation (re-fit) ...",
        )
        cfg_name = f"alpha={best_enet['alpha']:g}/l1={best_enet['l1_ratio']:g}"
        _record("single_lead_only", "single_lead_only",
                f"single_lead_only/{cfg_name}",
                {"alpha": float(best_enet["alpha"]),
                 "l1_ratio": float(best_enet["l1_ratio"]),
                 "max_iter": int(args.epochs),
                 "elastic_tol": float(args.elastic_tol)},
                adapter, metrics_single, idx)

    # Summary
    print("\n--- linear sweep summary ---")
    by_method: dict[str, list[dict]] = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)
    print(f"{'method':<18s} {'n':>3s}  "
          f"{'latent_mse  best (min)':>26s}  {'auroc  best (max)':>22s}")
    for name in ["ElasticNet", "prior_only", "single_lead_only"]:
        items = by_method.get(name, [])
        if not items:
            continue
        lat = np.array([r["val_latent_mse"] for r in items])
        au = np.array([r["val_auroc"] for r in items])
        best = min(items, key=lambda r: r["val_latent_mse"])
        print(f"{name:<18s} {len(items):>3d}  "
              f"{lat.mean():.4f}+/-{lat.std():.4f}  ({lat.min():.4f})   "
              f"{au.mean():.4f}+/-{au.std():.4f}  ({au.max():.4f})")
        if name == "ElasticNet":
            print(f"            best @ alpha={best['alpha']:g} l1={best['l1_ratio']:g} "
                  f"-> latent_mse={best['val_latent_mse']:.4f} "
                  f"auroc={best['val_auroc']:.4f}")
    print(f"\nAppended {len(rows)} records to {out_path}")
    if best_rec is not None:
        print(f"\nBest model (by {args.ckpt_metric}) saved to {args.ckpt}")
        print(f"  method={best_rec['method']}  config={best_rec['config']}")
        print(f"  val_auroc={best_rec['val_auroc']:.4f}  "
              f"val_latent_mse={best_rec['val_latent_mse']:.4f}  "
              f"params={best_rec['n_params']}")


if __name__ == "__main__":
    main()
