"""
GReLU residual adapter hyperparameter sweep.

Grid (single fixed seed):
    sizes  m in {256, 1024, 4096}        -- small / medium / large
    lr     in {1e-4, 1e-3, 1e-2}         -- 3 learning rates
    wd     in {0.0, 1e-4}                -- 2 L2 weight decays
    l1     in {0.0, 1e-5, 1e-4}          -- 3 L1 penalties on linear1.weight
    => 3 x 3 x 2 x 3 = 54 runs

Writes one JSONL record per run to checkpoints/sweep_results.jsonl. The box
plot in plots.py groups runs by (method, m) so each box collects the 18
hyperparameter combinations at that size.

Usage:
    python src/grelu_sweep.py
    python src/grelu_sweep.py --epochs 60
    python src/grelu_sweep.py --sizes 256 1024
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapter_interface import load_regression_head
from grelu_adapter import (
    DEFAULT_CONFIG,
    GReLUResidualAdapter,
    build_tensors,
    evaluate_residual_adapter,
    load_split,
    pick_device,
    train_grelu_adapter,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

DEFAULT_SIZES = [256, 1024, 4096]
DEFAULT_LRS = [1e-4, 1e-3, 1e-2]
DEFAULT_WDS = [0.0]
DEFAULT_L1S = [0.0]


def count_trainable(model: GReLUResidualAdapter) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42,
                    help="Single fixed seed for all runs.")
    ap.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES,
                    help="Values of m (hidden width).")
    ap.add_argument("--lrs", type=float, nargs="+", default=DEFAULT_LRS)
    ap.add_argument("--wds", type=float, nargs="+", default=DEFAULT_WDS)
    ap.add_argument("--l1s", type=float, nargs="+", default=DEFAULT_L1S)
    ap.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    ap.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    ap.add_argument("--eval-every", type=int, default=DEFAULT_CONFIG["eval_every"])
    ap.add_argument("--out", type=str, default=str(CKPT_DIR / "sweep_results.jsonl"))
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
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"  grid: {len(args.sizes)} sizes x {len(args.lrs)} lrs x "
          f"{len(args.wds)} wds x {len(args.l1s)} l1s = {len(grid)} runs")
    print(f"  run_id: {run_id}\n")

    rows = []
    with out_path.open("a") as f:
        for i, (m, lr, wd, l1) in enumerate(grid, 1):
            cfg = {
                **DEFAULT_CONFIG,
                "m": int(m),
                "lr": float(lr),
                "lambda_l2": float(wd),
                "lambda_l1": float(l1),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "eval_every": args.eval_every,
                "seed": int(args.seed),
                "device": args.device,
            }

            t0 = time.time()
            adapter, history = train_grelu_adapter(
                train_split, val_split, head=head, config=cfg, verbose=args.verbose,
            )
            train_time = time.time() - t0

            u_va, r_va, zr_va, zs_va = build_tensors(val_split, device=device)
            metrics = evaluate_residual_adapter(
                adapter, u_va, r_va, zr_va, zs_va,
                head=head, y=val_split["ischemia"], bl_st80=val_split["bl_st80"],
            )

            rec = {
                "method": "GReLU",
                "config": f"m={m}",
                "label": f"GReLU/m={m}/lr={lr:g}/wd={wd:g}/l1={l1:g}",
                "run_id": run_id,
                "seed": int(args.seed),
                "m": int(m),
                "lr": float(lr),
                "lambda_l2": float(wd),
                "lambda_l1": float(l1),
                "epochs": args.epochs,
                "val_residual_mse": metrics["residual_mse"],
                "val_latent_mse": metrics["latent_mse"],
                "val_auroc": metrics.get("auroc", float("nan")),
                "train_mse": history.train_mse[-1] if history.train_mse else float("nan"),
                "n_params": count_trainable(adapter),
                "train_time_s": train_time,
            }
            f.write(json.dumps(rec) + "\n"); f.flush()
            rows.append(rec)
            print(f"[{i:2d}/{len(grid)}] GReLU m={m:<5d} lr={lr:g} wd={wd:g} l1={l1:g}  "
                  f"params={rec['n_params']:>7d}  "
                  f"val_latent_mse={rec['val_latent_mse']:.4f}  "
                  f"val_auroc={rec['val_auroc']:.4f}  "
                  f"({rec['train_time_s']:.1f}s)")

    # Summary grouped by m
    print("\n--- GReLU sweep summary (across hyperparams, per size) ---")
    by_m: dict[int, list[dict]] = {}
    for r in rows:
        by_m.setdefault(r["m"], []).append(r)
    print(f"{'config':<10s} {'n':>3s}  {'latent_mse  best (min)':>26s}  {'auroc  best (max)':>22s}")
    for m in args.sizes:
        items = by_m.get(m, [])
        if not items:
            continue
        lat = np.array([r["val_latent_mse"] for r in items])
        au  = np.array([r["val_auroc"] for r in items])
        best = min(items, key=lambda r: r["val_latent_mse"])
        print(f"m={m:<8d} {len(items):>3d}  "
              f"{lat.mean():.4f}+/-{lat.std():.4f}  ({lat.min():.4f})   "
              f"{au.mean():.4f}+/-{au.std():.4f}  ({au.max():.4f})")
        print(f"            best @ lr={best['lr']:g} wd={best['lambda_l2']:g} "
              f"l1={best['lambda_l1']:g} -> latent_mse={best['val_latent_mse']:.4f} "
              f"auroc={best['val_auroc']:.4f}")
    print(f"\nAppended {len(rows)} records to {out_path}")


if __name__ == "__main__":
    main()
