"""
End-to-end JEPA QT training script.

Trains a Lead-I → 12-lead JEPA predictor jointly with a fine-tuned context
encoder using the following architecture:

    Context branch (gradients flow):
        Lead I ECG → E_context (ECGGNNSimple, fine-tuned) → z_c ∈ ℝ^192

    Target branch (stop_grad, EMA encoder):
        12-lead ECG → E_target (EMA of E_context) → z_t ∈ ℝ^192

    Predictor (residual, all existing nn.Module backbones):
        ẑ = z_c + f_θ(z_c)

    QT head:
        QT̂ = h(ẑ)   (Linear → Mish → Linear)

    Combined loss:
        L = λ_JEPA · MSE(ẑ, stop_grad(z_t)) + λ_QT · MSE(QT̂, QT_label)

Five differentiable predictor backbones (matching the 6 frozen adapters):
    linear    — nn.Linear(192, 192) residual
    mlp       — _MLPNet (512-hidden, GELU+Dropout)
    attention — _AttentionNet (12 tokens × 16-dim, 2× TransformerEncoder)
    lora      — _LoRANet (rank=32, alpha=32, B-zeros init)
    grelu     — _GReLUNet (fixed gate, AdamW — not R-FISTA for e2e compat)

Note: KRR (sklearn, non-differentiable) cannot participate in end-to-end
      gradient-based training and is not listed here.

Two-phase training schedule:
    Phase 1 — warm-up (--warmup-epochs, default 50):
        Context encoder frozen. Only predictor + QT head are optimised.
        EMA encoder = frozen copy (no EMA updates yet).
        Equivalent to the existing offline adapter training, but online.

    Phase 2 — end-to-end fine-tune (--finetune-epochs, default 150):
        Context encoder unfrozen, lr = --lr-encoder (default 5e-5).
        EMA encoder updated: E_t ← τ·E_t + (1-τ)·E_c  (each batch).
        Predictor + QT head lr = --lr-predictor (default 1e-3).

Checkpoints are saved to:
    qt_finetune/checkpoints/jepa/jepa_{predictor_name}.pt

Usage:
    # Train one backbone
    python qt_finetune/train_jepa.py --predictor mlp

    # Train multiple backbones sequentially
    python qt_finetune/train_jepa.py --predictor mlp attention lora linear grelu

    # Faster smoke test
    python qt_finetune/train_jepa.py --predictor mlp --warmup-epochs 2 --finetune-epochs 3

    # Re-train even if checkpoint exists
    python qt_finetune/train_jepa.py --predictor mlp --force-retrain

    # Evaluate only (no training)
    python qt_finetune/train_jepa.py --predictor mlp --eval-only
"""
from __future__ import annotations

import sys
import argparse
import copy
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT   = _SCRIPT_DIR.parent
_MODELS_DIR  = _REPO_ROOT / "models"

# W&B is optional — gracefully disabled if not installed or no API key
try:
    import wandb as _wandb
    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False
for _p in (_MODELS_DIR, _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ecg_gnn_simple import ECGGNNSimple
from qt_dataset import QTFinetuneDataset, TARGET_LEN
from validate_qt import ForcedLeadDataset, load_model
from jepa_dataset import PairedLeadDataset, collate_paired, JEPAInferenceModel
from qt_adapter import (
    _MLPNet, _AttentionNet, _LoRANet, _GReLUNet, _RWNANet,
    MLP_DEFAULT_CONFIG, ATTENTION_DEFAULT_CONFIG,
    LORA_DEFAULT_CONFIG, GRELU_DEFAULT_CONFIG, RWNA_DEFAULT_CONFIG,
)
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader as TorchDataLoader

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DATA = _REPO_ROOT / "data"
DEFAULT_CHECKPOINT  = _SCRIPT_DIR / "checkpoints" / "best_qt_model.pt"
DEFAULT_LABELS      = _SCRIPT_DIR / "qt_labels.csv"
DEFAULT_SIGNALS_DIR = _DATA / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
DEFAULT_JEPA_DIR    = _SCRIPT_DIR / "checkpoints" / "jepa"

JEPA_PREDICTOR_NAMES = ["linear", "mlp", "attention", "lora", "grelu", "lora16", "rwna",
                        "cgrelu",     # m=1024 (full-size, frozen-adapter parity)
                        "cgrelu512",  # m=512  (half neurons, ablation)
                        "cgrelu256",  # m=256  (quarter neurons, ablation)
                        ]


# ---------------------------------------------------------------------------
# Model construction helpers
# ---------------------------------------------------------------------------

def _make_qt_head() -> nn.Sequential:
    """Regression head: 192-dim graph embed → 64 → 1 (ms)."""
    return nn.Sequential(
        nn.Linear(192, 64),
        nn.Mish(inplace=True),
        nn.Linear(64, 1),
    )


def build_predictor(name: str, d: int = 192) -> nn.Module:
    """
    Return the predictor nn.Module for the given JEPA variant name.

    The returned module maps z_c (B, d) → residual r (B, d).
    Applied as:  ẑ = z_c + predictor(z_c)

    Reuses existing nn.Modules from qt_adapter.py — no new architecture code.
    For GReLU: uses the same _GReLUNet (fixed gate, linear1+proj trainable)
               but trained with AdamW rather than R-FISTA to support backprop
               through the fine-tuned context encoder.
    """
    name = name.lower()
    if name == "linear":
        # Single linear residual: ẑ = z_c + W·z_c + b  ≡  (I+W)z_c + b
        m = nn.Linear(d, d, bias=True)
        nn.init.zeros_(m.weight)   # start from identity residual
        nn.init.zeros_(m.bias)
        return m
    elif name == "mlp":
        cfg = MLP_DEFAULT_CONFIG
        return _MLPNet(d, d, hidden=cfg["hidden"], dropout=cfg["dropout"])
    elif name == "attention":
        cfg = ATTENTION_DEFAULT_CONFIG
        return _AttentionNet(
            d_input=d,
            n_tokens=cfg["n_tokens"],
            token_dim=cfg["token_dim"],
            nhead=cfg["nhead"],
            n_layers=cfg["n_layers"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"],
        )
    elif name == "lora":
        cfg = LORA_DEFAULT_CONFIG
        return _LoRANet(d, d, rank=cfg["rank"], alpha=cfg["lora_alpha"])
    elif name == "lora16":
        # Rank-16 LoRA — ablation partner to RWNA (k=16); same bottleneck rank,
        # linear vs nonlinear in the projected subspace.
        return _LoRANet(d, d, rank=16, alpha=16.0)
    elif name == "grelu":
        cfg = GRELU_DEFAULT_CONFIG
        return _GReLUNet(d_input=d, d_output=d, m=cfg["m"], seed=cfg["seed"])
    elif name.startswith("cgrelu"):
        # Convex GReLU JEPA family — same _GReLUNet architecture as "grelu",
        # but trained with two structural constraints that restore convexity:
        #   1. proj.weight clamped ≥ 0 after every optimizer step (non-negative
        #      output weights over ReLU-gated features = convex function of z_c)
        #   2. L1 penalty on linear1.weight added to the loss (sparsity prior,
        #      mimicking R-FISTA's proximal soft-threshold in the frozen adapter)
        # Constraints enforced in _run_train_epoch when predictor_name.startswith("cgrelu").
        #
        # Variants (m = GReLU neuron count):
        #   cgrelu     → m=1024  (frozen-adapter parity)
        #   cgrelu512  → m=512   (half neurons)
        #   cgrelu256  → m=256   (quarter neurons)
        _m = {"cgrelu": 1024, "cgrelu512": 512, "cgrelu256": 256}.get(name, 1024)
        return _GReLUNet(d_input=d, d_output=d, m=_m, seed=GRELU_DEFAULT_CONFIG["seed"])
    elif name == "rwna":
        cfg = RWNA_DEFAULT_CONFIG
        # k=16 for JEPA E2E: wide enough gradient signal to potentially break the
        # rank-2 collapse observed in JEPA-Linear (where the linear predictor's
        # rank-2 gradient kept z_c near-degenerate even after 120 epochs of E2E training).
        return _RWNANet(d, d, k=cfg["k"], hidden=cfg["hidden"])
    else:
        raise ValueError(
            f"Unknown predictor: {name!r}. "
            f"Choose from: {JEPA_PREDICTOR_NAMES}"
        )


def build_context_encoder(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """
    Load ECGGNNSimple with the QT waveform-only head from a saved checkpoint.

    Identical to validate_qt.load_model() — kept here for clarity.
    """
    return load_model(checkpoint_path, device)


# ---------------------------------------------------------------------------
# EMA utility
# ---------------------------------------------------------------------------

@torch.no_grad()
def ema_update(online: nn.Module, ema: nn.Module, tau: float) -> None:
    """
    Update EMA encoder in-place: ema ← τ·ema + (1-τ)·online.

    No gradients are tracked.  Only .parameters() are updated (not buffers
    such as BatchNorm running stats, which we don't have in ECGGNNSimple).
    """
    for p_online, p_ema in zip(online.parameters(), ema.parameters()):
        p_ema.data.mul_(tau).add_(p_online.data, alpha=1.0 - tau)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_jepa_ckpt(
    path: str | Path,
    context_encoder: nn.Module,
    predictor: nn.Module,
    qt_head: nn.Module,
    predictor_type: str,
    config: dict,
    val_mae: float,
    val_jepa_mse: float | None,
    epoch: int,
) -> None:
    """Save a full JEPA checkpoint (context encoder + predictor + QT head)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "context_encoder_state": context_encoder.state_dict(),
            "predictor_state":       predictor.state_dict(),
            "qt_head_state":         qt_head.state_dict(),
            "predictor_type":        predictor_type,
            "config":                config,
            "val_mae":               val_mae,
            "val_jepa_mse":          val_jepa_mse,
            "epoch":                 epoch,
        },
        path,
    )
    print(f"    JEPA-{predictor_type} checkpoint saved → {path}")


def load_jepa_ckpt(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, nn.Module, nn.Module, dict]:
    """
    Load a JEPA checkpoint and reconstruct all three modules.

    Returns:
        context_encoder : ECGGNNSimple (with QT head replaced, eval mode)
        predictor       : nn.Module predictor (eval mode)
        qt_head         : nn.Sequential QT head (eval mode)
        config          : dict of training hyper-parameters
    """
    device  = torch.device(device) if isinstance(device, str) else device
    ckpt    = torch.load(path, map_location="cpu", weights_only=False)
    ptype   = ckpt["predictor_type"]
    cfg     = ckpt.get("config", {})

    # Context encoder
    context_encoder = ECGGNNSimple(
        input_length=TARGET_LEN,
        sh_max_degree=4, node_dim=128, edge_dim=192,
        hidden_dim=192, num_gnn_layers=3, tabular_dim=7, dropout=0.3,
    )
    context_encoder.classifier = _make_qt_head()
    context_encoder.load_state_dict(ckpt["context_encoder_state"])
    context_encoder = context_encoder.to(device).eval()

    # Predictor
    predictor = build_predictor(ptype)
    predictor.load_state_dict(ckpt["predictor_state"])
    predictor = predictor.to(device).eval()

    # QT head
    qt_head = _make_qt_head()
    qt_head.load_state_dict(ckpt["qt_head_state"])
    qt_head = qt_head.to(device).eval()

    print(
        f"    JEPA-{ptype} loaded ← {Path(path).name}  "
        f"(epoch={ckpt.get('epoch','?')}  val_MAE={ckpt.get('val_mae', float('nan')):.2f} ms)"
    )
    return context_encoder, predictor, qt_head, cfg


def load_jepa_inference_model(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> JEPAInferenceModel:
    """
    Load a JEPA checkpoint and return a JEPAInferenceModel ready for inference.

    The returned model is drop-in compatible with run_inference() and
    extract_latents() in analyze_qtc.py.
    """
    context_encoder, predictor, qt_head, _ = load_jepa_ckpt(path, device)
    model = JEPAInferenceModel(context_encoder, predictor, qt_head)
    return model.to(torch.device(device) if isinstance(device, str) else device).eval()


# ---------------------------------------------------------------------------
# Resume-checkpoint helpers (preemption recovery)
# ---------------------------------------------------------------------------

def _save_resume(
    path: Path,
    *,
    phase: str,           # "warmup" | "finetune"
    next_epoch: int,      # first epoch to run on the next restart
    context_enc: nn.Module,
    ema_enc: nn.Module,
    predictor: nn.Module,
    qt_head: nn.Module,
    optimizer,
    scheduler,
    best_warmup_mae: float,
    best_val_mae: float,
    best_val_r: float,
    best_val_epoch: int,
) -> None:
    """Persist full training state to volume so a preempted run can resume.

    Writes atomically via a temp file + os.replace() so a mid-write preemption
    never leaves a corrupt resume checkpoint.
    """
    tmp = Path(str(path) + ".tmp")
    torch.save(
        {
            "phase":           phase,
            "next_epoch":      next_epoch,
            "context_enc":     context_enc.state_dict(),
            "ema_enc":         ema_enc.state_dict(),
            "predictor":       predictor.state_dict(),
            "qt_head":         qt_head.state_dict(),
            "optimizer":       optimizer.state_dict(),
            "scheduler":       scheduler.state_dict(),
            "best_warmup_mae": best_warmup_mae,
            "best_val_mae":    best_val_mae,
            "best_val_r":      best_val_r,
            "best_val_epoch":  best_val_epoch,
        },
        tmp,
    )
    os.replace(tmp, path)   # atomic on POSIX — safe against mid-write preemption


def _load_resume(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device, weights_only=False)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_jepa(
    context_encoder: nn.Module,
    predictor: nn.Module,
    qt_head: nn.Module,
    val_ds: QTFinetuneDataset,
    device: torch.device,
    batch_size: int = 32,
) -> dict:
    """
    Evaluate JEPA model on a ForcedLeadDataset (Lead I only at inference —
    no 12-lead ECG needed).

    Returns dict with: mae, rmse, r, bias, w10, w20, pred, tgt.
    """
    context_encoder.eval(); predictor.eval(); qt_head.eval()
    loader = PyGDataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    preds, tgts = [], []
    for bd, qt in loader:
        bd = bd.to(device)
        B  = bd.num_graphs
        z_c   = context_encoder(bd, tabular=torch.zeros(B, 0, device=device))["graph_embed"]
        z_hat = z_c + predictor(z_c)
        qt_pred = qt_head(z_hat).flatten()
        preds.append(qt_pred.cpu().numpy())
        tgts.append(qt.flatten().numpy())

    pred = np.concatenate(preds)
    tgt  = np.concatenate(tgts)
    err  = pred - tgt
    return dict(
        mae  = float(np.abs(err).mean()),
        rmse = float(np.sqrt((err ** 2).mean())),
        r    = float(np.corrcoef(pred, tgt)[0, 1]),
        bias = float(err.mean()),
        w10  = float((np.abs(err) <= 10).mean() * 100),
        w20  = float((np.abs(err) <= 20).mean() * 100),
        pred = pred, tgt = tgt,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _run_train_epoch(
    context_encoder: nn.Module,
    ema_encoder: nn.Module,
    predictor: nn.Module,
    qt_head: nn.Module,
    loader,
    optimizer,
    device: torch.device,
    lambda_jepa: float,
    lambda_qt: float,
    do_ema: bool,
    ema_tau: float,
    lambda_l1_net: float = 0.0,
    clamp_predictor_proj: bool = False,
) -> dict:
    """Single training epoch.  Returns dict with avg losses.

    Optional convex-GReLU constraints (cgrelu predictor only):
      lambda_l1_net        : L1 penalty on predictor.linear1.weight added to the
                             combined loss.  Mimics the R-FISTA proximal soft-threshold
                             step used in the frozen GReLU adapter.
      clamp_predictor_proj : After each optimizer.step(), clamp predictor.proj.weight
                             to >= 0.  Non-negative output weights over ReLU-gated
                             features guarantees the network is a convex function of z_c.
    """
    context_encoder.train(); predictor.train(); qt_head.train()
    ema_encoder.eval()

    sum_jepa = sum_qt = sum_total = 0.0
    n_batches = 0

    for ctx_batch, tgt_batch, qt_labels in loader:
        ctx_batch  = ctx_batch.to(device)
        tgt_batch  = tgt_batch.to(device)
        qt_labels  = qt_labels.to(device)          # (B, 1)
        B = ctx_batch.num_graphs

        # Context embedding (gradients flow)
        z_c = context_encoder(
            ctx_batch, tabular=torch.zeros(B, 0, device=device)
        )["graph_embed"]                              # (B, 192)

        # Target embedding (stop gradient, EMA encoder)
        with torch.no_grad():
            z_t = ema_encoder(
                tgt_batch, tabular=torch.zeros(B, 0, device=device)
            )["graph_embed"]                          # (B, 192)

        # Residual predictor
        z_hat  = z_c + predictor(z_c)               # (B, 192)
        qt_hat = qt_head(z_hat)                      # (B, 1)

        # Losses
        L_jepa = F.mse_loss(z_hat, z_t)
        L_qt   = F.mse_loss(qt_hat, qt_labels)
        loss   = lambda_jepa * L_jepa + lambda_qt * L_qt

        # Optional L1 sparsity on GReLU linear1 weights (convex variant only)
        if lambda_l1_net > 0.0 and hasattr(predictor, "linear1"):
            loss = loss + lambda_l1_net * predictor.linear1.weight.abs().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Convex projection: clamp proj.weight >= 0 after each gradient step.
        # Non-negative output weights over ReLU-gated features = convex function of z_c.
        if clamp_predictor_proj and hasattr(predictor, "proj"):
            with torch.no_grad():
                predictor.proj.weight.clamp_(min=0.0)

        # EMA update (only during Phase 2)
        if do_ema:
            ema_update(context_encoder, ema_encoder, tau=ema_tau)

        sum_jepa  += L_jepa.item()
        sum_qt    += L_qt.item()
        sum_total += loss.item()
        n_batches += 1

    n = max(n_batches, 1)
    return dict(
        jepa  = sum_jepa  / n,
        qt    = sum_qt    / n,
        total = sum_total / n,
    )


def run_jepa_training(args, predictor_name: str) -> None:
    """
    Train one JEPA predictor variant end-to-end.

    Saves a resume checkpoint to the JEPA dir every eval_every epochs so that
    Modal worker preemption (or any other crash) can be recovered automatically:
    the next run picks up from the last saved state rather than epoch 1.

    If the FINAL checkpoint already exists and --force-retrain is not set, skips.
    """
    ckpt_stem   = args.run_name if args.run_name else predictor_name
    ckpt_path   = Path(args.jepa_dir) / f"jepa_{ckpt_stem}.pt"
    resume_path = Path(args.jepa_dir) / f"jepa_{ckpt_stem}_resume.pt"

    # Training is complete only when the final checkpoint exists AND the resume
    # file is gone (it is deleted on clean exit).  If both exist, a previous
    # run was interrupted mid-training — resume from where it left off.
    if ckpt_path.exists() and not resume_path.exists() and not args.force_retrain:
        print(f"\n  JEPA-{predictor_name}: training complete → {ckpt_path}")
        print("  Pass --force-retrain to re-train from scratch.")
        return
    if ckpt_path.exists() and resume_path.exists() and not args.force_retrain:
        print(f"\n  JEPA-{predictor_name}: mid-run checkpoint found, resuming…")

    # Clear stale resume if user explicitly wants a fresh run
    if args.force_retrain and resume_path.exists():
        resume_path.unlink()
        print(f"  Deleted stale resume checkpoint ({resume_path.name})")

    print(f"\n{'='*68}")
    print(f"  JEPA-{predictor_name.upper()}  end-to-end training")
    print(f"{'='*68}")
    t0_total = time.time()

    # Convex-GReLU flags — active for all cgrelu* variants (cgrelu, cgrelu512, cgrelu256)
    _is_cgrelu         = predictor_name.startswith("cgrelu")
    _lambda_l1_net     = 1e-4 if _is_cgrelu else 0.0
    _clamp_proj        = _is_cgrelu
    if _is_cgrelu:
        print(
            f"  ConvexGReLU mode: L1 on linear1.weight (λ_l1={_lambda_l1_net})  "
            f"+ proj.weight clamped ≥ 0 after each step"
        )

    # ── W&B init ──────────────────────────────────────────────────────────
    # W&B run IDs must be alphanumeric + hyphens/underscores only — no dots.
    # Sanitize here so resume="allow" reliably reconnects across restarts.
    # The human-readable display name (name=ckpt_stem) keeps the original form.
    _wb = None
    if _WANDB_OK and os.environ.get("WANDB_API_KEY"):
        import wandb
        wandb_id = ckpt_stem.replace(".", "_")   # "linear_lqt2.0" → "linear_lqt2_0"
        _wb = wandb.init(
            project="anchoredecg-jepa",
            name=ckpt_stem,      # display name: "linear_lqt2.0"  (dots fine here)
            id=wandb_id,         # run ID:       "linear_lqt2_0"   (no dots)
            config={
                "predictor":       predictor_name,
                "warmup_epochs":   args.warmup_epochs,
                "finetune_epochs": args.finetune_epochs,
                "lambda_jepa":     args.lambda_jepa,
                "lambda_qt":       args.lambda_qt,
                "ema_tau":         args.ema_tau,
                "lr_encoder":      args.lr_encoder,
                "lr_predictor":    args.lr_predictor,
                "batch_size":      args.batch_size,
                "run_name":        ckpt_stem,
            },
            resume="allow",
        )
        print(f"  W&B run: {_wb.url}")

    # ── Device ────────────────────────────────────────────────────────────
    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"  Device: {device}")

    # ── Model init (always from base checkpoint — resume overwrites below) ─
    ckpt_path_base  = Path(args.checkpoint)
    context_encoder = build_context_encoder(ckpt_path_base, device)
    ema_encoder     = copy.deepcopy(context_encoder).to(device)
    for p in ema_encoder.parameters():
        p.requires_grad_(False)

    predictor = build_predictor(predictor_name).to(device)
    qt_head   = _make_qt_head().to(device)

    print(
        f"  Context encoder params: {sum(p.numel() for p in context_encoder.parameters()):,}\n"
        f"  Predictor    ({predictor_name:<10}) params: "
        f"{sum(p.numel() for p in predictor.parameters()):,}\n"
        f"  QT head      params: {sum(p.numel() for p in qt_head.parameters()):,}"
    )

    # ── Resume detection ──────────────────────────────────────────────────
    rs = None   # resume state dict
    if resume_path.exists():
        rs = _load_resume(resume_path, device)
        context_encoder.load_state_dict(rs["context_enc"])
        ema_encoder.load_state_dict(rs["ema_enc"])
        predictor.load_state_dict(rs["predictor"])
        qt_head.load_state_dict(rs["qt_head"])
        print(
            f"  ↺ Resuming from {resume_path.name}  "
            f"phase={rs['phase']}  next_epoch={rs['next_epoch']}"
        )

    # ── Datasets ──────────────────────────────────────────────────────────
    ds_kwargs = dict(
        ptbxl_signals_dir=args.ptbxl_dir,
        labels_csv=args.labels_csv,
    )
    train_pair_ds = PairedLeadDataset(split="train", **ds_kwargs)
    val_leadI_ds  = ForcedLeadDataset(lead_indices=[0], split="val", **ds_kwargs)

    from torch.utils.data import DataLoader as TorchLoader
    train_loader = TorchLoader(
        train_pair_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_paired,
        num_workers=0,
        drop_last=False,
    )
    print(
        f"  Train: {len(train_pair_ds):,} paired samples  "
        f"({len(train_loader)} batches @ bs={args.batch_size})\n"
        f"  Val  : {len(val_leadI_ds):,} Lead-I samples"
    )

    best_warmup_mae = rs["best_warmup_mae"] if rs else float("inf")
    best_val_mae    = rs["best_val_mae"]    if rs else float("inf")
    best_val_r      = rs["best_val_r"]      if rs else 0.0
    best_val_epoch  = rs["best_val_epoch"]  if rs else 0

    # ── Phase 1: warm-up (frozen encoder) ─────────────────────────────────
    skip_warmup = rs is not None and rs["phase"] == "finetune"
    if args.warmup_epochs > 0 and not skip_warmup:
        print(f"\n── Phase 1 (warm-up, {args.warmup_epochs} epochs) ────────────────")
        for p in context_encoder.parameters():
            p.requires_grad_(False)

        optimizer = AdamW(
            list(predictor.parameters()) + list(qt_head.parameters()),
            lr=args.lr_predictor, weight_decay=1e-4,
        )
        scheduler = CosineAnnealingLR(
            optimizer, T_max=args.warmup_epochs,
            eta_min=args.lr_predictor * 1e-3,
        )

        # Restore optimizer/scheduler if resuming mid-warmup
        start_warmup = 1
        if rs is not None and rs["phase"] == "warmup":
            optimizer.load_state_dict(rs["optimizer"])
            scheduler.load_state_dict(rs["scheduler"])
            start_warmup = rs["next_epoch"]
            print(f"  Resuming warm-up from epoch {start_warmup}")

        for epoch in range(start_warmup, args.warmup_epochs + 1):
            t0 = time.time()
            tr = _run_train_epoch(
                context_encoder, ema_encoder, predictor, qt_head,
                train_loader, optimizer, device,
                lambda_jepa=args.lambda_jepa, lambda_qt=args.lambda_qt,
                do_ema=False, ema_tau=args.ema_tau,
                lambda_l1_net=_lambda_l1_net,
                clamp_predictor_proj=_clamp_proj,
            )
            scheduler.step()
            # Save after EVERY epoch — worst-case preemption loss is 1 epoch, not 10
            _save_resume(
                resume_path,
                phase="warmup",
                next_epoch=epoch + 1,
                context_enc=context_encoder, ema_enc=ema_encoder,
                predictor=predictor, qt_head=qt_head,
                optimizer=optimizer, scheduler=scheduler,
                best_warmup_mae=best_warmup_mae,
                best_val_mae=best_val_mae,
                best_val_r=best_val_r,
                best_val_epoch=best_val_epoch,
            )
            if epoch % args.eval_every == 0 or epoch == args.warmup_epochs:
                vm = evaluate_jepa(context_encoder, predictor, qt_head,
                                   val_leadI_ds, device, args.batch_size)
                best_warmup_mae = min(best_warmup_mae, vm["mae"])
                print(
                    f"  W{epoch:4d}/{args.warmup_epochs}  "
                    f"train[jepa={tr['jepa']:.5f} qt={tr['qt']:.3f}]  "
                    f"val_MAE={vm['mae']:.2f} ms  R={vm['r']:.3f}  "
                    f"({time.time()-t0:.1f}s)"
                )
                if _wb is not None:
                    _wb.log({
                        "phase": 1,
                        "epoch": epoch,
                        "train/jepa_mse": tr["jepa"],
                        "train/qt_mse":   tr["qt"],
                        "train/total":    tr["total"],
                        "val/mae":        vm["mae"],
                        "val/rmse":       vm["rmse"],
                        "val/r":          vm["r"],
                    }, step=epoch)
                # Re-save with updated eval metrics
                _save_resume(
                    resume_path,
                    phase="warmup",
                    next_epoch=epoch + 1,
                    context_enc=context_encoder, ema_enc=ema_encoder,
                    predictor=predictor, qt_head=qt_head,
                    optimizer=optimizer, scheduler=scheduler,
                    best_warmup_mae=best_warmup_mae,
                    best_val_mae=best_val_mae,
                    best_val_r=best_val_r,
                    best_val_epoch=best_val_epoch,
                )
        print(f"  Warm-up done. Best val MAE = {best_warmup_mae:.2f} ms")

    # ── Phase 2: end-to-end fine-tune (unfrozen encoder + EMA) ───────────
    print(f"\n── Phase 2 (end-to-end, {args.finetune_epochs} epochs) ──────────────")
    for p in context_encoder.parameters():
        p.requires_grad_(True)

    optimizer = AdamW(
        [
            {"params": context_encoder.parameters(), "lr": args.lr_encoder,   "weight_decay": 0.01},
            {"params": predictor.parameters(),        "lr": args.lr_predictor, "weight_decay": 1e-4},
            {"params": qt_head.parameters(),          "lr": args.lr_predictor, "weight_decay": 1e-4},
        ]
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=args.finetune_epochs,
        eta_min=args.lr_encoder * 1e-2,
    )

    # Restore optimizer/scheduler if resuming mid-finetune
    start_finetune = 1
    if rs is not None and rs["phase"] == "finetune":
        optimizer.load_state_dict(rs["optimizer"])
        scheduler.load_state_dict(rs["scheduler"])
        start_finetune = rs["next_epoch"]
        print(f"  Resuming end-to-end from epoch {start_finetune}")

    for epoch in range(start_finetune, args.finetune_epochs + 1):
        t0 = time.time()
        tr = _run_train_epoch(
            context_encoder, ema_encoder, predictor, qt_head,
            train_loader, optimizer, device,
            lambda_jepa=args.lambda_jepa, lambda_qt=args.lambda_qt,
            do_ema=True, ema_tau=args.ema_tau,
            lambda_l1_net=_lambda_l1_net,
            clamp_predictor_proj=_clamp_proj,
        )
        scheduler.step()
        # Save after EVERY epoch — worst-case preemption loss is 1 epoch, not 10
        _save_resume(
            resume_path,
            phase="finetune",
            next_epoch=epoch + 1,
            context_enc=context_encoder, ema_enc=ema_encoder,
            predictor=predictor, qt_head=qt_head,
            optimizer=optimizer, scheduler=scheduler,
            best_warmup_mae=best_warmup_mae,
            best_val_mae=best_val_mae,
            best_val_r=best_val_r,
            best_val_epoch=best_val_epoch,
        )

        if epoch % args.eval_every == 0 or epoch == args.finetune_epochs:
            vm = evaluate_jepa(context_encoder, predictor, qt_head,
                               val_leadI_ds, device, args.batch_size)
            improved = vm["mae"] < best_val_mae
            if improved:
                best_val_mae   = vm["mae"]
                best_val_r     = vm["r"]
                best_val_epoch = epoch
                save_jepa_ckpt(
                    ckpt_path,
                    context_encoder, predictor, qt_head,
                    predictor_type=predictor_name,
                    config=vars(args),
                    val_mae=vm["mae"],
                    val_jepa_mse=tr["jepa"],
                    epoch=epoch,
                )
            star = " ★" if improved else ""
            print(
                f"  E{epoch:4d}/{args.finetune_epochs}  "
                f"train[jepa={tr['jepa']:.5f} qt={tr['qt']:.3f}]  "
                f"val_MAE={vm['mae']:.2f} ms  R={vm['r']:.3f}  "
                f"({time.time()-t0:.1f}s){star}"
            )
            if _wb is not None:
                _wb.log({
                    "phase": 2,
                    "epoch": args.warmup_epochs + epoch,
                    "train/jepa_mse":  tr["jepa"],
                    "train/qt_mse":    tr["qt"],
                    "train/total":     tr["total"],
                    "val/mae":         vm["mae"],
                    "val/rmse":        vm["rmse"],
                    "val/r":           vm["r"],
                    "val/best_mae":    best_val_mae,
                }, step=args.warmup_epochs + epoch)
            # Save resume state so preemption can pick up here
            _save_resume(
                resume_path,
                phase="finetune",
                next_epoch=epoch + 1,
                context_enc=context_encoder, ema_enc=ema_encoder,
                predictor=predictor, qt_head=qt_head,
                optimizer=optimizer, scheduler=scheduler,
                best_warmup_mae=best_warmup_mae,
                best_val_mae=best_val_mae,
                best_val_r=best_val_r,
                best_val_epoch=best_val_epoch,
            )

    # Training complete — clean up resume file
    if resume_path.exists():
        resume_path.unlink()

    if _wb is not None:
        _wb.summary["best_val_mae"]   = best_val_mae
        _wb.summary["best_val_r"]     = best_val_r
        _wb.summary["best_val_epoch"] = best_val_epoch
        _wb.finish()

    elapsed = time.time() - t0_total
    print(
        f"\n  JEPA-{predictor_name} training complete in {elapsed/60:.1f} min\n"
        f"  Best val MAE = {best_val_mae:.2f} ms  R = {best_val_r:.3f}  "
        f"(epoch {best_val_epoch})  → {ckpt_path}"
    )


# ---------------------------------------------------------------------------
# Eval-only mode
# ---------------------------------------------------------------------------

def eval_only(args, predictor_name: str) -> None:
    """Load existing checkpoint and print val metrics without training."""
    ckpt_stem = args.run_name if args.run_name else predictor_name
    ckpt_path = Path(args.jepa_dir) / f"jepa_{ckpt_stem}.pt"
    if not ckpt_path.exists():
        print(f"  JEPA-{predictor_name}: no checkpoint at {ckpt_path}")
        return

    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    context_encoder, predictor, qt_head, cfg = load_jepa_ckpt(ckpt_path, device)

    ds_kwargs = dict(
        ptbxl_signals_dir=args.ptbxl_dir,
        labels_csv=args.labels_csv,
    )
    for split in [args.split]:
        val_ds = ForcedLeadDataset(lead_indices=[0], split=split, **ds_kwargs)
        vm = evaluate_jepa(context_encoder, predictor, qt_head,
                           val_ds, device, args.batch_size)
        print(
            f"  JEPA-{predictor_name}  [{split}]  "
            f"MAE={vm['mae']:.2f} ms  RMSE={vm['rmse']:.2f} ms  "
            f"R={vm['r']:.3f}  bias={vm['bias']:+.2f}  "
            f"±10ms={vm['w10']:.1f}%  ±20ms={vm['w20']:.1f}%"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="JEPA end-to-end QT training (fine-tunes Lead-I encoder + predictor)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Paths
    parser.add_argument("--checkpoint",  default=str(DEFAULT_CHECKPOINT),
                        help="Pre-trained QT checkpoint to initialise context encoder.")
    parser.add_argument("--ptbxl-dir",   default=str(DEFAULT_SIGNALS_DIR))
    parser.add_argument("--labels-csv",  default=str(DEFAULT_LABELS))
    parser.add_argument("--jepa-dir",    default=str(DEFAULT_JEPA_DIR),
                        help="Directory for JEPA checkpoints.")

    # Adapter selection
    parser.add_argument("--predictor", nargs="+",
                        choices=JEPA_PREDICTOR_NAMES, default=["mlp"],
                        help="Predictor backbone(s) to train. Multiple → sequential run.")

    # Training phases
    parser.add_argument("--warmup-epochs",   type=int, default=20,
                        help="Phase 1: frozen-encoder warm-up epochs (0 to skip). "
                             "Default reduced from 50→20 based on convergence analysis "
                             "(JEPA-Linear variants converged 30-70 epochs into fine-tune).")
    parser.add_argument("--finetune-epochs", type=int, default=80,
                        help="Phase 2: end-to-end fine-tune epochs. "
                             "Default reduced from 150→80 based on best-checkpoint analysis.")
    parser.add_argument("--eval-every",      type=int, default=10,
                        help="Evaluate on val every N epochs.")

    # Loss weights
    # λ_QT >> λ_JEPA: QT accuracy is the primary objective.
    # Natural MSE scale difference (QT: ~400–900 ms²; JEPA embedding: ~0.1–1.0)
    # means QT gradients already dominate; the 10:1 ratio makes that intent
    # explicit in the code and ensures the JEPA term acts as regularisation only.
    parser.add_argument("--lambda-jepa", type=float, default=1.0,
                        help="Weight for JEPA latent-prediction loss (auxiliary regulariser).")
    parser.add_argument("--lambda-qt",   type=float, default=1.0,
                        help="Weight for QT regression loss.")

    # EMA
    parser.add_argument("--ema-tau", type=float, default=0.999,
                        help="EMA momentum for target encoder (higher = slower update).")

    # Learning rates
    parser.add_argument("--lr-encoder",    type=float, default=5e-5,
                        help="LR for context encoder in Phase 2.")
    parser.add_argument("--lr-predictor",  type=float, default=1e-3,
                        help="LR for predictor + QT head (both phases).")

    # Misc
    parser.add_argument("--batch-size",    type=int, default=32)
    parser.add_argument("--split",         default="val", choices=["val", "test"],
                        help="Split for eval-only mode.")
    parser.add_argument("--force-retrain", action="store_true", default=False,
                        help="Re-train even if checkpoint already exists.")
    parser.add_argument("--eval-only",     action="store_true", default=False,
                        help="Load existing checkpoints and print metrics only.")
    parser.add_argument("--run-name",      default=None,
                        help="Override checkpoint filename: saves as jepa_{run-name}.pt "
                             "instead of jepa_{predictor}.pt.  Useful for hyperparameter "
                             "sweeps (e.g. --run-name linear_lqt0.5).")

    args = parser.parse_args()
    Path(args.jepa_dir).mkdir(parents=True, exist_ok=True)

    print(f"\nJEPA QT Training — predictor(s): {args.predictor}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  JEPA dir   : {args.jepa_dir}")
    print(f"  Warm-up: {args.warmup_epochs}  Fine-tune: {args.finetune_epochs}  "
          f"λ_jepa={args.lambda_jepa}  λ_qt={args.lambda_qt}  τ={args.ema_tau}")

    for pname in args.predictor:
        if args.eval_only:
            eval_only(args, pname)
        else:
            run_jepa_training(args, pname)

    # ── Summary table ─────────────────────────────────────────────────────
    jepa_dir = Path(args.jepa_dir)
    found = [(p, jepa_dir / f"jepa_{p}.pt") for p in args.predictor
             if (jepa_dir / f"jepa_{p}.pt").exists()]
    if found:
        print(f"\n── JEPA checkpoint summary ({args.jepa_dir}/) ─────────────────")
        for pname, ckpt in found:
            d = torch.load(ckpt, map_location="cpu", weights_only=False)
            print(
                f"  jepa_{pname:<12}.pt  "
                f"epoch={d.get('epoch','?'):>4}  "
                f"val_MAE={d.get('val_mae', float('nan')):>7.2f} ms  "
                f"({ckpt.stat().st_size / 1e6:.1f} MB)"
            )


if __name__ == "__main__":
    main()
