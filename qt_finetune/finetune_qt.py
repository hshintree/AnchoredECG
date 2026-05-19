"""
Fine-tune GraphECG encoder on QT interval regression (PTB-XL).

Endpoint: QT_Int_Global (ms) -- single scalar per recording.
Resampling: 500 Hz -> 250 Hz (5000 -> 2500 samples).
Lead dropout: 70% of training batches use random 1-12 lead subsets.
Loss: MSE.
Metrics: MSE, MAE, Pearson R (all per epoch).

Usage:
    python qt_finetune/finetune_qt.py --ptbxl-dir /path/to/ptbxl --smoke-test
    python qt_finetune/finetune_qt.py --ptbxl-dir /path/to/ptbxl --epochs 30 --batch-size 32
"""
from __future__ import annotations

import sys
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_MODELS_DIR = _REPO_ROOT / "models"
if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from ecg_gnn_simple import ECGGNNSimple
from qt_dataset import QTFinetuneDataset, TARGET_LEN
from torch_geometric.loader import DataLoader as PyGDataLoader

DEFAULT_CHECKPOINT = _REPO_ROOT / "checkpoints" / "shd_pretrained.pt"
DEFAULT_LABELS = _SCRIPT_DIR / "qt_labels.csv"
DEFAULT_OUTPUT = _SCRIPT_DIR / "checkpoints"


def create_model(pretrained_path, device):
    model = ECGGNNSimple(
        input_length=TARGET_LEN,
        sh_max_degree=4,
        node_dim=128,
        edge_dim=192,
        hidden_dim=192,
        num_gnn_layers=3,
        tabular_dim=7,
        dropout=0.3,
    )

    if pretrained_path and Path(pretrained_path).exists():
        ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"]
        state = {k: v for k, v in state.items() if not k.startswith("classifier")}
        missing, _ = model.load_state_dict(state, strict=False)
        print(f"Loaded pretrained encoder  (skipped: {missing})")
    else:
        print("WARNING: no pretrained checkpoint -- training from scratch")

    # 199 = 192 hidden + 7 zero-padded tabular
    # Single scalar output for QT interval
    model.classifier = nn.Sequential(
        nn.Linear(199, 64),
        nn.Mish(inplace=True),
        nn.Linear(64, 1),
    )
    return model.to(device)


def pearson_r(pred, target):
    if len(pred) < 2:
        return 0.0
    p = pred - pred.mean()
    t = target - target.mean()
    num = (p * t).sum()
    den = (p.norm() * t.norm()).clamp(min=1e-8)
    return (num / den).item()


def train_epoch(model, loader, optimizer, device, epoch, total_epochs):
    model.train()
    mse_acc, mae_acc, r_acc, n = 0.0, 0.0, 0.0, 0
    all_pred, all_tgt = [], []
    pbar = tqdm(loader, desc=f"Train {epoch}/{total_epochs}", leave=False)
    for batch_data, qt in pbar:
        batch_data = batch_data.to(device)
        qt = qt.to(device)

        optimizer.zero_grad()
        pred = model(batch_data, tabular=None)["logits"]  # (B, 1)
        loss = nn.functional.mse_loss(pred, qt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        mse_acc += loss.item()
        mae_acc += (pred.detach() - qt).abs().mean().item()
        all_pred.append(pred.detach().flatten())
        all_tgt.append(qt.flatten())
        n += 1
        pbar.set_postfix(mse=f"{mse_acc/n:.2f}", mae=f"{mae_acc/n:.2f}")

    all_pred = torch.cat(all_pred)
    all_tgt = torch.cat(all_tgt)
    r = pearson_r(all_pred, all_tgt)
    return mse_acc / max(n, 1), mae_acc / max(n, 1), r


@torch.no_grad()
def eval_epoch(model, loader, device, epoch, total_epochs):
    model.eval()
    mse_acc, mae_acc, n = 0.0, 0.0, 0
    all_pred, all_tgt = [], []
    pbar = tqdm(loader, desc=f"Val   {epoch}/{total_epochs}", leave=False)
    for batch_data, qt in pbar:
        batch_data = batch_data.to(device)
        qt = qt.to(device)

        pred = model(batch_data, tabular=None)["logits"]
        mse_acc += nn.functional.mse_loss(pred, qt).item()
        mae_acc += (pred - qt).abs().mean().item()
        all_pred.append(pred.flatten())
        all_tgt.append(qt.flatten())
        n += 1
        pbar.set_postfix(mse=f"{mse_acc/n:.2f}", mae=f"{mae_acc/n:.2f}")

    all_pred = torch.cat(all_pred)
    all_tgt = torch.cat(all_tgt)
    r = pearson_r(all_pred, all_tgt)
    return mse_acc / max(n, 1), mae_acc / max(n, 1), r


def smoke_test(args, device):
    print("=== Smoke test ===")
    ds = QTFinetuneDataset(args.ptbxl_dir, args.labels_csv, "train",
                           lead_dropout_prob=0.7)
    print(f"  Dataset: {len(ds)} samples")

    loader = PyGDataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    batch_data, qt = next(iter(loader))
    print(f"  Batch graph : nodes={batch_data.x.shape}, edges={batch_data.edge_attr.shape}")
    print(f"  QT targets  : {qt.shape}, values={qt.flatten().tolist()}")

    model = create_model(args.checkpoint, device)
    model.train()

    out = model(batch_data.to(device), tabular=None)
    print(f"  Logits      : {out['logits'].shape}")
    print(f"  Graph embed : {out['graph_embed'].shape}")

    loss = nn.functional.mse_loss(out["logits"], qt.to(device))
    loss.backward()
    print(f"  Loss (MSE)  : {loss.item():.2f}")

    grad_norm = sum(
        p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None
    ) ** 0.5
    print(f"  Grad norm   : {grad_norm:.4f}")
    print("=== Smoke test passed ===\n")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune GraphECG on QT interval regression (PTB-XL)"
    )
    parser.add_argument("--ptbxl-dir", type=str, required=True,
                        help="Path to PTB-XL root (contains records500/)")
    parser.add_argument("--labels-csv", type=str, default=str(DEFAULT_LABELS))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--encoder-lr", type=float, default=5e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--lead-dropout", type=float, default=0.7)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.smoke_test:
        smoke_test(args, device)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building datasets ...")
    train_ds = QTFinetuneDataset(
        args.ptbxl_dir, args.labels_csv, "train",
        lead_dropout_prob=args.lead_dropout,
    )
    val_ds = QTFinetuneDataset(args.ptbxl_dir, args.labels_csv, "val")
    print(f"  Train: {len(train_ds)}   Val: {len(val_ds)}")

    train_loader = PyGDataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    val_loader = PyGDataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    model = create_model(args.checkpoint, device)
    n_enc = sum(p.numel() for n, p in model.named_parameters()
                if not n.startswith("classifier") and p.requires_grad)
    n_head = sum(p.numel() for n, p in model.named_parameters()
                 if n.startswith("classifier") and p.requires_grad)
    print(f"  Encoder params : {n_enc:,}")
    print(f"  Head params    : {n_head:,}")

    enc_params = [p for n, p in model.named_parameters()
                  if not n.startswith("classifier")]
    head_params = [p for n, p in model.named_parameters()
                   if n.startswith("classifier")]
    optimizer = AdamW([
        {"params": enc_params, "lr": args.encoder_lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_mae = float("inf")
    hdr = (f"{'Ep':>3} {'Time':>5}  "
           f"{'TrMSE':>8} {'TrMAE':>7} {'TrR':>6}  "
           f"{'VaMSE':>8} {'VaMAE':>7} {'VaR':>6}")
    print(f"\n{hdr}")
    print("-" * len(hdr))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_mse, tr_mae, tr_r = train_epoch(
            model, train_loader, optimizer, device, epoch, args.epochs)
        va_mse, va_mae, va_r = eval_epoch(
            model, val_loader, device, epoch, args.epochs)
        scheduler.step()
        dt = time.time() - t0

        improved = va_mae < best_val_mae
        if improved:
            best_val_mae = va_mae
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mae": va_mae, "val_mse": va_mse, "val_r": va_r,
                "args": vars(args),
            }, output_dir / "best_qt_model.pt")

        print(
            f"{epoch:3d} {dt:5.1f}s  "
            f"{tr_mse:8.2f} {tr_mae:7.2f} {tr_r:6.3f}  "
            f"{va_mse:8.2f} {va_mae:7.2f} {va_r:6.3f}"
            f"{'  *' if improved else ''}"
        )

    print(f"\nBest val MAE: {best_val_mae:.2f} ms")
    print(f"Saved to {output_dir / 'best_qt_model.pt'}")


if __name__ == "__main__":
    main()
