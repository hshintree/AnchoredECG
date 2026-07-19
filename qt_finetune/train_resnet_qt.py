"""
Train a 1-D ResNet-18 on QT interval regression (PTB-XL) — ablation.

This is a from-scratch supervised baseline (no pretraining, no graph) used to
check whether the electrode-graph GNN's lead-I result is meaningfully better
than a vanilla CNN.

Two configurations (selected with --leads):
    --leads lead1   -> input = lead I only            (in_channels=1)
    --leads all12   -> input = all 12 leads           (in_channels=12)

Sample rate (--target-fs) — we are optimising for Apple Watch input:
    512  Apple Watch native; user data needs no resampling at all  (primary)
    128  exact /4 of the watch rate; 4x shorter sequences          (ablation)
    500  PTB-XL native, no resampling; the original baseline

The label is in milliseconds and is therefore rate-independent — changing
--target-fs changes the input length, never the target. See rates.py.

Endpoint  : QT_Int_Global (ms), single scalar.
Loss      : MSE.  Metrics: MSE, MAE, Pearson R (per epoch).
Splits    : strat_fold <= 8 train, == 9 val.

Usage:
    python qt_finetune/train_resnet_qt.py --leads lead1 --target-fs 512
    python qt_finetune/train_resnet_qt.py --leads lead1 --target-fs 128
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
from torch.utils.data import DataLoader, WeightedRandomSampler
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
from rates import DEFAULT_RATE, SUPPORTED_RATES, describe, window_len
from qt_objective import (DEFAULT_QUANTILES, Objective, TailWeighter,
                          init_quantile_head)

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


def run_epoch(model, loader, device, objective, optimizer=None, desc=""):
    """One pass. Metrics are always computed on the POINT estimate (the median
    quantile in pinball mode) so they stay comparable with the MSE baselines."""
    train = optimizer is not None
    model.train(train)
    loss_acc, mae_acc, cov_acc, n = 0.0, 0.0, 0.0, 0
    all_pred, all_tgt = [], []
    pbar = tqdm(loader, desc=desc, unit="batch", leave=True, dynamic_ncols=True)
    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for x, qt in pbar:
            x = x.to(device, non_blocking=True)
            qt = qt.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)["logits"]
            loss = objective.loss(logits, qt)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            with torch.no_grad():
                point = objective.point(logits).detach()
                cov = objective.coverage(logits.detach(), qt)
            loss_acc += loss.item()
            mae_acc += (point - qt).abs().mean().item()
            cov_acc += 0.0 if cov != cov else cov          # NaN-safe
            all_pred.append(point.flatten())
            all_tgt.append(qt.flatten())
            n += 1
            pbar.set_postfix(loss=f"{loss_acc/n:.1f}", mae=f"{mae_acc/n:.1f}",
                             refresh=False)
    all_pred = torch.cat(all_pred)
    all_tgt = torch.cat(all_tgt)
    return (loss_acc / max(n, 1), mae_acc / max(n, 1),
            pearson_r(all_pred, all_tgt), cov_acc / max(n, 1))


def smoke_test(args, device, lead_indices, derived, in_channels):
    print("=== Smoke test ===")
    print(f"  {describe(args.target_fs)}")
    ds = ResNetQTDataset(args.ptbxl_dir, args.labels_csv, "train",
                         lead_indices=lead_indices, derived=derived,
                         target_fs=args.target_fs)
    print(f"  Dataset: {len(ds)} samples,  in_channels={in_channels},  "
          f"expected L={window_len(args.target_fs)}")
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    x, qt = next(iter(loader))
    print(f"  Batch: x={tuple(x.shape)}  qt={tuple(qt.shape)}  qt={qt.flatten().tolist()}")

    weighter = None
    if args.tail_weighting != "none":
        weighter = TailWeighter([it["qt_ms"] for it in ds.index],
                                bin_width=args.tail_bin_width,
                                alpha=args.tail_alpha, clip=args.tail_clip)
        print(weighter.describe())
    quantiles = tuple(float(q) for q in args.quantiles.split(",")) \
        if args.loss == "pinball" else (0.5,)
    objective = Objective(args.loss, quantiles,
                          weighter=weighter if args.tail_weighting == "loss" else None)
    objective.to(device)

    model = ECGResNet1d(in_channels=in_channels,
                        num_outputs=objective.num_outputs).to(device)
    if args.loss == "pinball":
        init_quantile_head(model.classifier, objective.quantiles)
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}  "
          f"num_outputs={objective.num_outputs}")
    out = model(x.to(device))
    print(f"  logits: {tuple(out['logits'].shape)}  embed: {tuple(out['embed'].shape)}")

    qtd = qt.to(device)
    pred = objective.transform(out["logits"])
    print(f"  quantiles (row 0): "
          f"{[round(v, 1) for v in pred[0].tolist()]}   target={qtd[0].item():.0f} ms")
    mono = bool((pred[:, 1:] >= pred[:, :-1]).all()) if pred.shape[1] > 1 else True
    print(f"  monotonic across quantiles: {mono}")
    loss = objective.loss(out["logits"], qtd)
    loss.backward()
    print(f"  Loss ({args.loss}): {loss.item():.2f}   "
          f"point={objective.point(out['logits'])[0].item():.1f} ms   "
          f"coverage={objective.coverage(out['logits'], qtd)*100:.0f}%")
    assert mono, "quantile monotonicity violated"
    print("=== Smoke test passed ===\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train 1-D ResNet-18 on QT regression (PTB-XL, 500 Hz)")
    parser.add_argument("--leads", choices=list(LEAD_CONFIGS), required=True,
                        help="lead1 (Lead I), all12 (all 12 leads), "
                             "or icm (derived V3-V2 single lead)")
    parser.add_argument("--target-fs", type=int, default=DEFAULT_RATE,
                        choices=list(SUPPORTED_RATES),
                        help="input sample rate (Hz). 512 = Apple Watch native, "
                             "128 = exact /4 of it, 500 = PTB-XL native baseline")
    # --- tail-robustness: pinball loss + inverse-frequency weighting ---
    parser.add_argument("--loss", choices=("mse", "pinball"), default="mse",
                        help="mse = original scalar baseline; pinball = quantile "
                             "regression (median point estimate + interval)")
    parser.add_argument("--quantiles", default=",".join(map(str, DEFAULT_QUANTILES)),
                        help="comma-separated quantiles for --loss pinball")
    parser.add_argument("--tail-weighting", choices=("none", "loss", "sampler"),
                        default="none",
                        help="none = uniform; loss = per-sample inverse-frequency "
                             "loss weights (every record still seen once per epoch); "
                             "sampler = WeightedRandomSampler oversampling. "
                             "Neither discards data.")
    parser.add_argument("--tail-alpha", type=float, default=0.5,
                        help="0 = uniform, 0.5 = sqrt-inverse freq, 1 = full inverse")
    parser.add_argument("--tail-clip", type=float, default=10.0,
                        help="max/min sample weight ratio")
    parser.add_argument("--tail-bin-width", type=float, default=25.0,
                        help="QT histogram bin width (ms) for the weighting")
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
    # Defaults raised from (5, 0.1): val MAE swung ~5 ms between epochs in the
    # baseline runs, so a 0.1 ms improvement threshold was firing on noise and
    # stopping at epoch 17 of 50 — before the cosine schedule had annealed.
    parser.add_argument("--patience", type=int, default=10,
                        help="early-stop after this many epochs without "
                             "val-MAE improvement of at least --min-delta")
    parser.add_argument("--min-delta", type=float, default=0.25,
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
    print(f"Rate  : {describe(args.target_fs)}")

    if args.smoke_test:
        smoke_test(args, device, lead_indices, derived, in_channels)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Rate is always in the filename so runs at different rates never collide
    # (and never overwrite the original 500 Hz checkpoints).
    tag = f"{args.leads}_{args.target_fs}hz"
    if args.loss != "mse":
        tag += f"_{args.loss}"
    if args.tail_weighting != "none":
        tag += f"_tail{args.tail_weighting}"
    ckpt_name = args.ckpt_name or f"best_resnet_{tag}.pt"

    print("Building datasets ...")
    train_ds = ResNetQTDataset(args.ptbxl_dir, args.labels_csv, "train",
                               lead_indices=lead_indices, derived=derived,
                               target_fs=args.target_fs)
    val_ds = ResNetQTDataset(args.ptbxl_dir, args.labels_csv, "val",
                             lead_indices=lead_indices, derived=derived,
                             target_fs=args.target_fs)
    print(f"  Train: {len(train_ds)}   Val: {len(val_ds)}   "
          f"input L={window_len(args.target_fs)}")

    # --- tail weighting (fitted on the TRAIN split only) ---------------------
    weighter = None
    if args.tail_weighting != "none":
        train_qts = [it["qt_ms"] for it in train_ds.index]
        weighter = TailWeighter(train_qts, bin_width=args.tail_bin_width,
                                alpha=args.tail_alpha, clip=args.tail_clip)
        print(weighter.describe())

    quantiles = tuple(float(q) for q in args.quantiles.split(",")) \
        if args.loss == "pinball" else (0.5,)
    objective = Objective(args.loss, quantiles,
                          weighter=weighter if args.tail_weighting == "loss" else None)
    objective.to(device)
    print(f"  objective: {args.loss}  quantiles={objective.quantiles}  "
          f"tail_weighting={args.tail_weighting}")

    pin = device.type == "cuda"
    if args.tail_weighting == "sampler":
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weighter.sample_weights, dtype=torch.double),
            num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=sampler, num_workers=args.num_workers,
                                  pin_memory=pin,
                                  persistent_workers=args.num_workers > 0)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=pin,
                                  persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin,
                            persistent_workers=args.num_workers > 0)

    model = ECGResNet1d(in_channels=in_channels, num_outputs=objective.num_outputs,
                        dropout=args.dropout).to(device)
    if args.loss == "pinball":
        init_quantile_head(model.classifier, objective.quantiles)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ResNet-18-1d  in_channels={in_channels}  "
          f"num_outputs={objective.num_outputs}  params={n_params:,}")

    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── W&B init ──────────────────────────────────────────────────────────
    run_name = args.run_name or f"resnet_{args.leads}_{args.target_fs}hz"
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
                "target_fs": args.target_fs,
                "input_len": window_len(args.target_fs),
                "target": "QT_Int_Global_ms",
                "loss": args.loss,
                "quantiles": list(objective.quantiles),
                "tail_weighting": args.tail_weighting,
                "tail_alpha": args.tail_alpha,
                "tail_clip": args.tail_clip,
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
        tr_mse, tr_mae, tr_r, tr_cov = run_epoch(
            model, train_loader, device, objective, optimizer,
            desc=f"Train {epoch:>2}/{args.epochs}")
        va_mse, va_mae, va_r, va_cov = run_epoch(
            model, val_loader, device, objective, None,
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
                "target_fs": args.target_fs,
                "input_len": window_len(args.target_fs),
                # needed to reconstruct the head and decode predictions
                "loss": args.loss,
                "quantiles": list(objective.quantiles),
                "num_outputs": objective.num_outputs,
                "tail_weighting": args.tail_weighting,
                "val_coverage": va_cov,
                "args": vars(args),
            }, output_dir / ckpt_name)
        else:
            epochs_no_improve += 1

        star = " *best*" if improved else ""
        cov = f" cov={va_cov*100:.0f}%" if args.loss == "pinball" else ""
        tqdm.write(
            f"Epoch {epoch:>2}/{args.epochs}  {dt:5.1f}s  "
            f"train loss={tr_mse:.1f} MAE={tr_mae:.1f} R={tr_r:.3f}  |  "
            f"val loss={va_mse:.1f} MAE={va_mae:.1f} R={va_r:.3f}{cov}{star}"
            f"  (no-improve {epochs_no_improve}/{args.patience})")

        if _wb is not None:
            _wb.log({
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train/loss": tr_mse, "train/mae": tr_mae, "train/r": tr_r,
                "val/loss": va_mse, "val/mae": va_mae, "val/r": va_r,
                "val/coverage": va_cov, "train/coverage": tr_cov,
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
