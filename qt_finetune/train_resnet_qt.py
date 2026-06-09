"""
Train a 1-D ResNet-18 on QT interval regression (PTB-XL, 500 Hz) — ablation.

This is a from-scratch supervised baseline (no pretraining, no graph) used to
check whether the electrode-graph GNN's lead-I result is meaningfully better
than a vanilla CNN.

Two configurations (selected with --leads):
    --leads lead1   -> input = lead I only            (in_channels=1)
    --leads all12   -> input = all 12 leads           (in_channels=12)

Endpoint  : QT_Int_Global (ms), single scalar.
Loss      : MSE.  Metrics: MSE, MAE, Pearson R (per epoch).
Splits    : strat_fold <= 8 train, == 9 val.

Usage:
    python qt_finetune/train_resnet_qt.py --leads lead1 --epochs 40
    python qt_finetune/train_resnet_qt.py --leads all12 --epochs 40
    python qt_finetune/train_resnet_qt.py --leads lead1 --smoke-test
"""
from __future__ import annotations

import os
import sys
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# W&B is optional — gracefully disabled if not installed or no API key
try:
    import wandb as _wandb
    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_MODELS_DIR = _REPO_ROOT / "models"
for _p in (_MODELS_DIR, _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ecg_resnet1d import ECGResNet1d
from resnet_dataset import ResNetQTDataset

DEFAULT_LABELS = _SCRIPT_DIR / "qt_labels.csv"
DEFAULT_OUTPUT = _SCRIPT_DIR / "checkpoints"
_DATA_DIR = _REPO_ROOT / "data"
DEFAULT_SIGNALS_DIR = (
    _DATA_DIR / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
)

# Each config -> (lead_indices, derived, in_channels).
# derived is a key into resnet_dataset.DERIVED_LEADS (single augmented channel).
LEAD_CONFIGS = {
    "lead1": (
        [0],                 # Lead I only
        None, 1),
    "all12": (
        list(range(12)),     # all 12 leads
        None, 12),
    "icm": (
        None,                # derived: V3 - V2 (ICM-style single lead)
        "v3_v2", 1),
}


def resolve_leads(name: str):
    """Return (lead_indices, derived, in_channels) for a config name."""
    lead_indices, derived, in_channels = LEAD_CONFIGS[name]
    return lead_indices, derived, in_channels


def pearson_r(pred, target):
    if len(pred) < 2:
        return 0.0
    p = pred - pred.mean()
    t = target - target.mean()
    num = (p * t).sum()
    den = (p.norm() * t.norm()).clamp(min=1e-8)
    return (num / den).item()


def run_epoch(model, loader, device, optimizer=None, desc=""):
    train = optimizer is not None
    model.train(train)
    mse_acc, mae_acc, n = 0.0, 0.0, 0
    all_pred, all_tgt = [], []
    pbar = tqdm(loader, desc=desc, unit="batch", leave=True, dynamic_ncols=True)
    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for x, qt in pbar:
            x = x.to(device, non_blocking=True)
            qt = qt.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            pred = model(x)["logits"]
            loss = nn.functional.mse_loss(pred, qt)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            mse_acc += loss.item()
            mae_acc += (pred.detach() - qt).abs().mean().item()
            all_pred.append(pred.detach().flatten())
            all_tgt.append(qt.flatten())
            n += 1
            pbar.set_postfix(mse=f"{mse_acc/n:.1f}", mae=f"{mae_acc/n:.1f}",
                             refresh=False)
    all_pred = torch.cat(all_pred)
    all_tgt = torch.cat(all_tgt)
    return mse_acc / max(n, 1), mae_acc / max(n, 1), pearson_r(all_pred, all_tgt)


def smoke_test(args, device, lead_indices, derived, in_channels):
    print("=== Smoke test ===")
    ds = ResNetQTDataset(args.ptbxl_dir, args.labels_csv, "train",
                         lead_indices=lead_indices, derived=derived)
    print(f"  Dataset: {len(ds)} samples,  in_channels={in_channels}")
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    x, qt = next(iter(loader))
    print(f"  Batch: x={tuple(x.shape)}  qt={tuple(qt.shape)}  qt={qt.flatten().tolist()}")
    model = ECGResNet1d(in_channels=in_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")
    out = model(x.to(device))
    print(f"  logits: {tuple(out['logits'].shape)}  embed: {tuple(out['embed'].shape)}")
    loss = nn.functional.mse_loss(out["logits"], qt.to(device))
    loss.backward()
    print(f"  Loss (MSE): {loss.item():.2f}")
    print("=== Smoke test passed ===\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train 1-D ResNet-18 on QT regression (PTB-XL, 500 Hz)")
    parser.add_argument("--leads", choices=list(LEAD_CONFIGS), required=True,
                        help="lead1 (Lead I), all12 (all 12 leads), "
                             "or icm (derived V3-V2 single lead)")
    parser.add_argument("--ptbxl-dir", default=str(DEFAULT_SIGNALS_DIR))
    parser.add_argument("--labels-csv", default=str(DEFAULT_LABELS))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--ckpt-name", default=None,
                        help="checkpoint filename (default best_resnet_<leads>.pt)")
    parser.add_argument("--patience", type=int, default=5,
                        help="early-stop after this many epochs without "
                             "val-MAE improvement of at least --min-delta")
    parser.add_argument("--min-delta", type=float, default=0.1,
                        help="minimum val-MAE decrease (ms) to count as improvement")
    parser.add_argument("--wandb-project", default="anchoredecg-qt-resnet")
    parser.add_argument("--run-name", default=None,
                        help="W&B run name (default resnet_<leads>)")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    lead_indices, derived, in_channels = resolve_leads(args.leads)

    device = (torch.device("cuda") if torch.cuda.is_available() else
              torch.device("mps") if torch.backends.mps.is_available() else
              torch.device("cpu"))
    print(f"Device: {device}   leads={args.leads} "
          f"(indices={lead_indices}, derived={derived}, in_channels={in_channels})")

    if args.smoke_test:
        smoke_test(args, device, lead_indices, derived, in_channels)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = args.ckpt_name or f"best_resnet_{args.leads}.pt"

    print("Building datasets ...")
    train_ds = ResNetQTDataset(args.ptbxl_dir, args.labels_csv, "train",
                               lead_indices=lead_indices, derived=derived)
    val_ds = ResNetQTDataset(args.ptbxl_dir, args.labels_csv, "val",
                             lead_indices=lead_indices, derived=derived)
    print(f"  Train: {len(train_ds)}   Val: {len(val_ds)}")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin,
                            persistent_workers=args.num_workers > 0)

    model = ECGResNet1d(in_channels=in_channels, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ResNet-18-1d  in_channels={in_channels}  params={n_params:,}")

    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── W&B init ──────────────────────────────────────────────────────────
    run_name = args.run_name or f"resnet_{args.leads}"
    _wb = None
    if not args.no_wandb and _WANDB_OK and os.environ.get("WANDB_API_KEY"):
        import wandb
        _wb = wandb.init(
            project=args.wandb_project,
            name=run_name,
            id=run_name.replace(".", "_"),
            config={
                "model": "resnet18_1d",
                "leads": args.leads,
                "in_channels": in_channels,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "dropout": args.dropout,
                "patience": args.patience,
                "min_delta": args.min_delta,
                "n_params": n_params,
            },
            resume="allow",
        )
        print(f"  W&B run: {_wb.url}")

    best_val_mae = float("inf")
    epochs_no_improve = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_mse, tr_mae, tr_r = run_epoch(
            model, train_loader, device, optimizer,
            desc=f"Train {epoch:>2}/{args.epochs}")
        va_mse, va_mae, va_r = run_epoch(
            model, val_loader, device, None,
            desc=f"Val   {epoch:>2}/{args.epochs}")
        scheduler.step()
        dt = time.time() - t0

        improved = va_mae < best_val_mae - args.min_delta
        if improved:
            best_val_mae = va_mae
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mae": va_mae, "val_mse": va_mse, "val_r": va_r,
                "leads": args.leads, "lead_indices": lead_indices,
                "derived": derived, "in_channels": in_channels,
                "args": vars(args),
            }, output_dir / ckpt_name)
        else:
            epochs_no_improve += 1

        star = " *best*" if improved else ""
        tqdm.write(
            f"Epoch {epoch:>2}/{args.epochs}  {dt:5.1f}s  "
            f"train MSE={tr_mse:.1f} MAE={tr_mae:.1f} R={tr_r:.3f}  |  "
            f"val MSE={va_mse:.1f} MAE={va_mae:.1f} R={va_r:.3f}{star}"
            f"  (no-improve {epochs_no_improve}/{args.patience})")

        if _wb is not None:
            _wb.log({
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train/mse": tr_mse, "train/mae": tr_mae, "train/r": tr_r,
                "val/mse": va_mse, "val/mae": va_mae, "val/r": va_r,
                "val/best_mae": best_val_mae,
                "epochs_no_improve": epochs_no_improve,
            }, step=epoch)

        if epochs_no_improve >= args.patience:
            tqdm.write(
                f"\nEarly stopping at epoch {epoch}: val MAE has not improved "
                f"by >={args.min_delta} ms for {args.patience} epochs.")
            break

    print(f"\nBest val MAE: {best_val_mae:.2f} ms")
    print(f"Saved to {output_dir / ckpt_name}")
    if _wb is not None:
        _wb.summary["best_val_mae"] = best_val_mae
        _wb.finish()


if __name__ == "__main__":
    main()
