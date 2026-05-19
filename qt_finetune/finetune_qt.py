"""
Fine-tune GraphECG regression head on QT interval regression (PTB-XL).

The pretrained encoder (GNN + waveform encoders) is FROZEN by default.
Only the QT regression head (classifier) is trained.

Endpoint: QT_Int_Global (ms) -- single scalar per recording.
Resampling: 500 Hz -> 250 Hz (5000 -> 2500 samples).
Lead dropout: 70% of training batches use random 1-12 lead subsets.
Loss: MSE.
Metrics: MSE, MAE, Pearson R (all per epoch).

Usage:
    python qt_finetune/finetune_qt.py --smoke-test
    python qt_finetune/finetune_qt.py --epochs 30 --batch-size 32
    python qt_finetune/finetune_qt.py --epochs 30 --unfreeze-encoder  # optional full FT
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
from qt_dataset import QTFinetuneDataset, TARGET_LEN, TABULAR_DIM
from torch_geometric.loader import DataLoader as PyGDataLoader

DEFAULT_CHECKPOINT = _REPO_ROOT / "checkpoints" / "shd_pretrained.pt"
DEFAULT_LABELS = _SCRIPT_DIR / "qt_labels.csv"
DEFAULT_OUTPUT = _SCRIPT_DIR / "checkpoints"
# Convenience defaults pointing to the two PTB-XL datasets under data/
_DATA_DIR = _REPO_ROOT / "data"
DEFAULT_SIGNALS_DIR = (
    _DATA_DIR / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
)
DEFAULT_FEATURES_DIR = (
    _DATA_DIR / "ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1"
)


def create_model(pretrained_path, device):
    model = ECGGNNSimple(
        input_length=TARGET_LEN,
        sh_max_degree=4,
        node_dim=128,
        edge_dim=192,
        hidden_dim=192,
        num_gnn_layers=3,
        tabular_dim=TABULAR_DIM,
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


def freeze_encoder(model: nn.Module) -> tuple[int, int]:
    """Freeze all parameters except ``classifier`` (regression head).

    Frozen trunk modules stay in eval() during training so BatchNorm uses
    running statistics and does not update.
    """
    n_frozen, n_trainable = 0, 0
    for name, param in model.named_parameters():
        if name.startswith("classifier"):
            param.requires_grad = True
            n_trainable += param.numel()
        else:
            param.requires_grad = False
            n_frozen += param.numel()
    return n_frozen, n_trainable


def set_train_mode_head_only(model: nn.Module) -> None:
    """Train the head; keep frozen encoder submodules in eval mode."""
    model.classifier.train()
    for name, child in model.named_children():
        if name != "classifier":
            child.eval()


def head_parameters(model: nn.Module):
    return [p for n, p in model.named_parameters()
            if n.startswith("classifier") and p.requires_grad]


def encoder_parameters(model: nn.Module):
    return [p for n, p in model.named_parameters()
            if not n.startswith("classifier") and p.requires_grad]


def pearson_r(pred, target):
    if len(pred) < 2:
        return 0.0
    p = pred - pred.mean()
    t = target - target.mean()
    num = (p * t).sum()
    den = (p.norm() * t.norm()).clamp(min=1e-8)
    return (num / den).item()


def train_epoch(model, loader, optimizer, device, epoch, total_epochs, head_only: bool):
    if head_only:
        set_train_mode_head_only(model)
    else:
        model.train()
    mse_acc, mae_acc, n = 0.0, 0.0, 0
    all_pred, all_tgt = [], []
    trainable = [p for p in model.parameters() if p.requires_grad]
    pbar = tqdm(
        loader,
        desc=f"Train {epoch:>2}/{total_epochs}",
        unit="batch",
        leave=True,
        dynamic_ncols=True,
    )
    for batch_data, qt in pbar:
        batch_data = batch_data.to(device)
        qt = qt.to(device)
        tabular = getattr(batch_data, "tabular", None)
        if tabular is not None:
            tabular = tabular.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(batch_data, tabular=tabular)["logits"]  # (B, 1)
        loss = nn.functional.mse_loss(pred, qt)
        loss.backward()
        if trainable:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        mse_acc += loss.item()
        mae_acc += (pred.detach() - qt).abs().mean().item()
        all_pred.append(pred.detach().flatten())
        all_tgt.append(qt.flatten())
        n += 1
        pbar.set_postfix(mse=f"{mse_acc/n:.1f}", mae=f"{mae_acc/n:.1f}", refresh=False)

    all_pred = torch.cat(all_pred)
    all_tgt = torch.cat(all_tgt)
    r = pearson_r(all_pred, all_tgt)
    return mse_acc / max(n, 1), mae_acc / max(n, 1), r


@torch.no_grad()
def eval_epoch(model, loader, device, epoch, total_epochs):
    model.eval()
    mse_acc, mae_acc, n = 0.0, 0.0, 0
    all_pred, all_tgt = [], []
    pbar = tqdm(
        loader,
        desc=f"Val   {epoch:>2}/{total_epochs}",
        unit="batch",
        leave=True,
        dynamic_ncols=True,
    )
    for batch_data, qt in pbar:
        batch_data = batch_data.to(device)
        qt = qt.to(device)
        tabular = getattr(batch_data, "tabular", None)
        if tabular is not None:
            tabular = tabular.to(device)

        pred = model(batch_data, tabular=tabular)["logits"]
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
    ds = QTFinetuneDataset(
        args.ptbxl_dir, args.labels_csv, "train",
        ptbxl_features_dir=args.ptbxl_features_dir,
        lead_dropout_prob=0.7,
    )
    print(f"  Dataset: {len(ds)} samples")

    loader = PyGDataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    batch_data, qt = next(iter(loader))
    print(f"  Batch graph : nodes={batch_data.x.shape}, edges={batch_data.edge_attr.shape}")
    print(f"  QT targets  : {qt.shape}, values={qt.flatten().tolist()}")

    model = create_model(args.checkpoint, device)
    n_frozen, n_head = freeze_encoder(model)
    print(f"  Frozen encoder params : {n_frozen:,}")
    print(f"  Trainable head params   : {n_head:,}")
    set_train_mode_head_only(model)

    tabular = getattr(batch_data, "tabular", None)
    if tabular is not None:
        tabular = tabular.to(device)
    out = model(batch_data.to(device), tabular=tabular)
    print(f"  Logits      : {out['logits'].shape}")
    print(f"  Graph embed : {out['graph_embed'].shape}")

    loss = nn.functional.mse_loss(out["logits"], qt.to(device))
    loss.backward()
    print(f"  Loss (MSE)  : {loss.item():.2f}")

    enc_grad = sum(
        p.grad.norm().item() ** 2 for p in encoder_parameters(model) if p.grad is not None
    ) ** 0.5
    head_grad = sum(
        p.grad.norm().item() ** 2 for p in head_parameters(model) if p.grad is not None
    ) ** 0.5
    print(f"  Encoder grad norm (expect 0): {enc_grad:.6f}")
    print(f"  Head grad norm              : {head_grad:.4f}")
    assert enc_grad == 0.0, "Encoder received gradients -- freeze_encoder failed"
    print("=== Smoke test passed ===\n")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune GraphECG on QT interval regression (PTB-XL)"
    )
    parser.add_argument(
        "--ptbxl-dir", type=str,
        default=str(DEFAULT_SIGNALS_DIR),
        help="Path to PTB-XL *signals* dataset root "
             "(ptb-xl-a-large-publicly-available-...). Must contain records500/.",
    )
    parser.add_argument(
        "--ptbxl-features-dir", type=str,
        default=str(DEFAULT_FEATURES_DIR) if DEFAULT_FEATURES_DIR.exists() else None,
        help="Optional path to PTB-XL *feature* dataset root "
             "(ptb-xl-a-comprehensive-...). Supplies ecgdeli tabular features. "
             "Defaults to the sibling directory under data/ if it exists.",
    )
    parser.add_argument("--labels-csv", type=str, default=str(DEFAULT_LABELS))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--unfreeze-encoder", action="store_true",
        help="Also fine-tune the GNN encoder (default: train regression head only).",
    )
    parser.add_argument("--encoder-lr", type=float, default=5e-5,
                        help="LR for encoder when --unfreeze-encoder is set.")
    parser.add_argument("--head-lr", type=float, default=1e-3,
                        help="LR for the QT regression head (always trained).")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--lead-dropout", type=float, default=0.7)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    if args.smoke_test:
        smoke_test(args, device)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building datasets ...")
    print(f"  Signals  : {args.ptbxl_dir}")
    print(f"  Features : {args.ptbxl_features_dir or '(not provided -- tabular=zeros)'}")
    train_ds = QTFinetuneDataset(
        args.ptbxl_dir, args.labels_csv, "train",
        ptbxl_features_dir=args.ptbxl_features_dir,
        lead_dropout_prob=args.lead_dropout,
    )
    val_ds = QTFinetuneDataset(
        args.ptbxl_dir, args.labels_csv, "val",
        ptbxl_features_dir=args.ptbxl_features_dir,
    )
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
    head_only = not args.unfreeze_encoder
    if head_only:
        n_frozen, n_head = freeze_encoder(model)
        print(f"  Mode: HEAD ONLY (encoder frozen, {n_frozen:,} params)")
        print(f"  Trainable regression head: {n_head:,} params")
        optimizer = AdamW(
            head_parameters(model),
            lr=args.head_lr,
            weight_decay=args.weight_decay,
        )
    else:
        for p in model.parameters():
            p.requires_grad = True
        n_enc = sum(p.numel() for p in encoder_parameters(model))
        n_head = sum(p.numel() for p in head_parameters(model))
        print(f"  Mode: FULL fine-tune (encoder + head)")
        print(f"  Encoder params : {n_enc:,}  @ lr={args.encoder_lr:g}")
        print(f"  Head params    : {n_head:,}  @ lr={args.head_lr:g}")
        optimizer = AdamW([
            {"params": encoder_parameters(model), "lr": args.encoder_lr},
            {"params": head_parameters(model), "lr": args.head_lr},
        ], weight_decay=args.weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_mae = float("inf")
    epoch_pbar = tqdm(
        range(1, args.epochs + 1),
        desc="Epochs",
        unit="epoch",
        dynamic_ncols=True,
    )

    for epoch in epoch_pbar:
        t0 = time.time()
        tr_mse, tr_mae, tr_r = train_epoch(
            model, train_loader, optimizer, device, epoch, args.epochs, head_only,
        )
        va_mse, va_mae, va_r = eval_epoch(
            model, val_loader, device, epoch, args.epochs,
        )
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
                "head_only": head_only,
                "args": vars(args),
            }, output_dir / "best_qt_model.pt")

        star = " *best*" if improved else ""
        epoch_pbar.set_postfix(
            time=f"{dt:.0f}s",
            tr_mae=f"{tr_mae:.1f}",
            va_mae=f"{va_mae:.1f}",
            va_r=f"{va_r:.3f}",
            refresh=True,
        )
        tqdm.write(
            f"Epoch {epoch:>2}/{args.epochs}  {dt:5.1f}s  "
            f"train MSE={tr_mse:.1f} MAE={tr_mae:.1f} R={tr_r:.3f}  |  "
            f"val MSE={va_mse:.1f} MAE={va_mae:.1f} R={va_r:.3f}{star}"
        )

    print(f"\nBest val MAE: {best_val_mae:.2f} ms")
    print(f"Saved to {output_dir / 'best_qt_model.pt'}")
    if head_only:
        print("(checkpoint: frozen encoder + trained QT regression head)")


if __name__ == "__main__":
    main()
