"""
Evaluate your GReLU adapter on the validation set.

Usage:
    python src/evaluate.py
    python src/evaluate.py --weights path/to/grelu_weights.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapter_interface import AdapterBase, load_regression_head, evaluate_adapter
from grelu_adapter import GReLUAdapter

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_split(name):
    d = np.load(DATA_DIR / f"{name}.npz")
    return dict(
        z_rest=d["z_rest"],
        z_stress=d["z_stress"],
        phi_stress=d["phi_stress"],
        bl_st80=d["bl_st80"],
        ischemia=d["ischemia"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to saved adapter weights (skip training)")
    args = parser.parse_args()

    print("Loading data...")
    train = load_split("train")
    val = load_split("val")
    head = load_regression_head(DATA_DIR / "regression_head.pt")

    print(f"  Train: {len(train['ischemia'])} patients  "
          f"({train['ischemia'].mean():.1%} ischemic)")
    print(f"  Val:   {len(val['ischemia'])} patients  "
          f"({val['ischemia'].mean():.1%} ischemic)")

    if args.weights:
        print(f"\nLoading adapter from {args.weights}...")
        adapter = GReLUAdapter.load(args.weights)
    else:
        print("\nTraining GReLU adapter...")
        adapter = GReLUAdapter()
        adapter.fit(train["phi_stress"], train["z_rest"], train["z_stress"])

    print("\nEvaluating on val set...")
    metrics = evaluate_adapter(adapter, val, head)

    print(f"\n{'Metric':<15s} {'Value':>10s}")
    print("-" * 28)
    for k, v in metrics.items():
        print(f"{k:<15s} {v:10.4f}")

    if args.weights is None:
        save_path = DATA_DIR.parent / "grelu_weights.pt"
        adapter.save(save_path)
        print(f"\nSaved weights to {save_path}")


if __name__ == "__main__":
    main()
