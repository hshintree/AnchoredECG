"""
Train the convex GReLU residual adapter on AnchoredECG.

Single seed:
    python src/train.py

With seed variance (3 seeds, std reported):
    python src/train.py --seeds 42 43 44

Override hyperparameters (R-FISTA: lr is the initial step size,
lambda-l1 / lambda-l2 are the regularizers on linear1.weight):
    python src/train.py --m 2048 --epochs 300 --lr 1e-3 \
        --lambda-l2 1e-4 --lambda-l1 1e-5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grelu_adapter import (
    DEFAULT_CONFIG,
    GReLUResidualAdapter,
    build_tensors,
    evaluate_residual_adapter,
    load_split,
    pick_device,
    save_adapter,
    seed_variance,
    train_grelu_adapter,
)
from adapter_interface import load_regression_head

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=DEFAULT_CONFIG["m"])
    ap.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    ap.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    ap.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    ap.add_argument("--lambda-l2", type=float, default=DEFAULT_CONFIG["lambda_l2"],
                    help="L2 weight decay on linear1 and proj weights.")
    ap.add_argument("--lambda-l1", type=float, default=DEFAULT_CONFIG["lambda_l1"],
                    help="L1 penalty on linear1.weight (R-FISTA soft-thresholding prox).")
    ap.add_argument("--ckpt-metric", type=str, default=DEFAULT_CONFIG["ckpt_metric"],
                    choices=["latent_mse", "auroc"],
                    help="Metric used to decide which iterate to checkpoint. "
                         "'latent_mse' = lowest val latent MSE wins (default); "
                         "'auroc' = highest val ischemia AUROC wins.")
    ap.add_argument("--zero-prior", action="store_true",
                    help="Ablation: feed the network u = [phi_stress; zeros(192)] "
                         "instead of [phi_stress; z_rest]. The residual target "
                         "r = z_stress - z_rest and inference-time anchoring "
                         "z_hat = z_rest + r_hat are unchanged.")
    ap.add_argument("--eval-every", type=int, default=DEFAULT_CONFIG["eval_every"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[42],
                    help="One seed -> single run. Multiple -> seed variance.")
    ap.add_argument("--device", type=str, default=None,
                    help="cuda, mps, cpu. Auto-detected if omitted.")
    ap.add_argument("--ckpt", type=str, default=str(CKPT_DIR / "grelu_best.pt"))
    return ap.parse_args()


def cfg_from_args(args) -> dict:
    return {
        **DEFAULT_CONFIG,
        "m": args.m,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "lambda_l2": args.lambda_l2,
        "lambda_l1": args.lambda_l1,
        "ckpt_metric": args.ckpt_metric,
        "zero_prior": args.zero_prior,
        "eval_every": args.eval_every,
        "device": args.device,
    }


def main():
    args = parse_args()
    cfg = cfg_from_args(args)

    print("Loading data ...")
    train_split = load_split(DATA_DIR, "train")
    val_split = load_split(DATA_DIR, "val")
    head = load_regression_head(DATA_DIR / "regression_head.pt")
    device = pick_device(cfg["device"])

    n_train = len(train_split["ischemia"])
    n_val = len(val_split["ischemia"])
    print(f"  train: {n_train} patients   val: {n_val} patients   device: {device}")
    print(f"  d_input={cfg['d_input']}   d_output={cfg['d_output']}   m={cfg['m']}")

    # ---- baselines ----------------------------------------------------------
    _, _, zr_va, zs_va = build_tensors(val_split, device=device)
    rest_only_mse = ((zr_va - zs_va) ** 2).sum(dim=1).mean().item()
    print(f"\nRest-only baseline (no adapter): val_latent_mse = {rest_only_mse:.6f}")

    # ---- training -----------------------------------------------------------
    if len(args.seeds) == 1:
        cfg["seed"] = int(args.seeds[0])
        print(f"\nTraining single seed {cfg['seed']} ...")
        adapter, history = train_grelu_adapter(
            train_split, val_split, head=head, config=cfg, ckpt_path=args.ckpt,
        )
        print("\n--- Final ---")
        print(f"  ckpt metric:          {history.ckpt_metric}")
        print(f"  best epoch:           {history.best_epoch}")
        print(f"  best val_latent_mse:  {history.best_val_latent_mse:.6f}")
        print(f"  best val_auroc:       {history.best_val_auroc:.6f}")
        print(f"  final train_mse:      {history.train_mse[-1]:.6f}")
        if history.val_residual_mse:
            print(f"  final val_resid_mse:  {history.val_residual_mse[-1]:.6f}")
            print(f"  final val_latent_mse: {history.val_latent_mse[-1]:.6f}")
        if history.val_auroc:
            print(f"  final val_auroc:      {history.val_auroc[-1]:.4f}")
        print(f"\nCheckpoint -> {args.ckpt}")

        # Final-on-val report (single-seed). Match the training-time u-build
        # so zero_prior models are evaluated on zero-prior inputs.
        u_va, r_va, zr_va, zs_va = build_tensors(
            val_split, device=device, zero_prior=cfg.get("zero_prior", False),
        )
        final = evaluate_residual_adapter(
            adapter, u_va, r_va, zr_va, zs_va,
            head=head, y=val_split["ischemia"], bl_st80=val_split["bl_st80"],
        )
        print("\n--- Val metrics (final adapter weights) ---")
        for k, v in final.items():
            print(f"  {k:<14s} {v:.6f}")

    else:
        print(f"\nSeed variance across {len(args.seeds)} runs: {args.seeds}")
        summary = seed_variance(
            train_split, val_split, head=head,
            seeds=tuple(args.seeds), config=cfg, verbose=True,
        )
        print("\n--- Seed variance summary (last-epoch metrics) ---")
        for k, v in summary.items():
            if isinstance(v, list):
                print(f"  {k:<22s} {v}")
            else:
                print(f"  {k:<22s} {v:.6f}")


if __name__ == "__main__":
    main()
