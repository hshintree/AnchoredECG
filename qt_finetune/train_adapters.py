"""
Standalone adapter training + evaluation script for the QT fine-tuning experiment.

Trains all six latent-space adapters that map the Lead-I graph_embed (192-dim)
to the 12-lead graph_embed (192-dim) using a residual formulation:

    r_i = z_12lead_i - z_lead_i
    z_adapted = z_lead_i + adapter(z_lead_i)

Adapters
--------
    Linear    : Ridge regression (sklearn)
    KRR       : KernelRidge + RBF kernel + StandardScaler (sklearn)
    GReLU     : Convex fixed-gate GReLU network trained with R-FISTA (PyTorch)
    MLP       : 3-layer MLP (GELU + Dropout) with AdamW + CosineAnnealingLR
    Attention : Token-based self-attention (reshape→TransformerEncoder→project)
    LoRA      : Low-rank adapter B(A(z))·(α/r); zero-residual initialisation

Latent caching
--------------
Encoder forward passes are the most expensive part of this pipeline (~5-10 min on
CPU for 17 k training samples).  Latents are cached as compressed .npz files so
subsequent runs (e.g. re-training adapters with different hyper-params) skip
re-extraction entirely.

Cache location (all under the latent cache dir):
    train_12lead.npz      — 12-lead train latents
    train_leadI.npz       — Lead I train latents
    {split}_leadI.npz     — Lead I val / test latents

The checkpoint's modification time is embedded in each cache file.  A mismatch
triggers a warning; use --force-reextract to rebuild.

Adapter checkpoint caching
--------------------------
Each adapter is saved to qt_finetune/checkpoints/adapters/ after training.
On subsequent runs the checkpoint is loaded and training is skipped unless
--force-retrain is passed.

    linear_adapter.pkl      — joblib pickle
    krr_adapter.pkl         — joblib pickle
    grelu_adapter.pt        — torch checkpoint (state_dict + config)
    mlp_adapter.pt          — torch checkpoint (state_dict + config)
    attention_adapter.pt    — torch checkpoint (state_dict + config)
    lora_adapter.pt         — torch checkpoint (state_dict + config)

Usage
-----
    # Train all adapters (skips any already cached), evaluate on val split
    python qt_finetune/train_adapters.py

    # Re-train everything from scratch (ignore adapter checkpoints)
    python qt_finetune/train_adapters.py --force-retrain

    # Force re-extraction of latents (e.g. after updating the model checkpoint)
    python qt_finetune/train_adapters.py --force-reextract

    # Evaluate on test split
    python qt_finetune/train_adapters.py --split test

    # Train only specific adapters
    python qt_finetune/train_adapters.py --adapters linear grelu mlp lora

    # GReLU hyper-parameters
    python qt_finetune/train_adapters.py --grelu-epochs 1000 --grelu-m 2048

    # MLP / Attention / LoRA hyper-parameters
    python qt_finetune/train_adapters.py --mlp-epochs 400 --mlp-hidden 1024
    python qt_finetune/train_adapters.py --attn-epochs 400 --attn-n-layers 4
    python qt_finetune/train_adapters.py --lora-rank 64 --lora-epochs 400

    # Pull the model checkpoint first if not already done:
    #   modal volume get ptbxl-data /best_qt_model.pt \\
    #       qt_finetune/checkpoints/best_qt_model.pt
"""
from __future__ import annotations

import sys
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT  = _SCRIPT_DIR.parent
_MODELS_DIR = _REPO_ROOT / "models"
for _p in (_MODELS_DIR, _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from validate_qt import (
    ForcedLeadDataset, load_model, LIMB_LEAD_INDICES,
)
from qt_adapter import (
    get_latents, predict_from_latents,
    LinearResidualAdapter, KernelResidualAdapter, GReLUResidualAdapter,
    MLPResidualAdapter, AttentionResidualAdapter, LoRAResidualAdapter,
    save_linear, load_linear,
    save_krr,    load_krr,
    save_grelu,  load_grelu,
    save_mlp,    load_mlp,
    save_attention, load_attention,
    save_lora,   load_lora,
    GRELU_DEFAULT_CONFIG, MLP_DEFAULT_CONFIG,
    ATTENTION_DEFAULT_CONFIG, LORA_DEFAULT_CONFIG,
)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_DATA = _REPO_ROOT / "data"
DEFAULT_CHECKPOINT   = _SCRIPT_DIR / "checkpoints" / "best_qt_model.pt"
DEFAULT_LABELS       = _SCRIPT_DIR / "qt_labels.csv"
DEFAULT_SIGNALS_DIR  = (
    _DATA / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
)
DEFAULT_LATENT_DIR   = _SCRIPT_DIR / "checkpoints" / "latents"
DEFAULT_ADAPTER_DIR  = _SCRIPT_DIR / "checkpoints" / "adapters"


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def _metrics(pred_qt: np.ndarray, tgt_qt: np.ndarray, label: str) -> dict:
    err  = pred_qt - tgt_qt
    mae  = float(np.abs(err).mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    r    = float(np.corrcoef(pred_qt, tgt_qt)[0, 1])
    bias = float(err.mean())
    w10  = float((np.abs(err) <= 10).mean() * 100)
    w20  = float((np.abs(err) <= 20).mean() * 100)
    print(
        f"  {label:<26}  MAE={mae:.2f} ms  RMSE={rmse:.2f} ms  "
        f"R={r:.3f}  bias={bias:+.2f}  "
        f"±10ms={w10:.1f}%  ±20ms={w20:.1f}%"
    )
    return dict(label=label, mae=mae, rmse=rmse, r=r, bias=bias, w10=w10, w20=w20)


# ---------------------------------------------------------------------------
# Per-adapter train or load
# ---------------------------------------------------------------------------

def _train_or_load_linear(
    zi_tr, z12_tr, ckpt_path, alpha, force,
) -> LinearResidualAdapter:
    if ckpt_path.exists() and not force:
        print(f"  Linear: loading checkpoint ({ckpt_path.name}) …")
        return load_linear(ckpt_path)
    print(f"  Linear: fitting  alpha={alpha}")
    t0 = time.time()
    adapter = LinearResidualAdapter(alpha=alpha).fit(zi_tr, z12_tr)
    print(f"    done in {time.time()-t0:.1f}s")
    save_linear(adapter, ckpt_path)
    return adapter


def _train_or_load_krr(
    zi_tr, z12_tr, ckpt_path, alpha, n_subsample, force,
) -> KernelResidualAdapter:
    if ckpt_path.exists() and not force:
        print(f"  KRR: loading checkpoint ({ckpt_path.name}) …")
        return load_krr(ckpt_path)
    print(f"  KRR: fitting  alpha={alpha}  n_sub={n_subsample}")
    t0 = time.time()
    adapter = KernelResidualAdapter(alpha=alpha, n_subsample=n_subsample).fit(zi_tr, z12_tr)
    print(f"    done in {time.time()-t0:.1f}s")
    save_krr(adapter, ckpt_path)
    return adapter


def _train_or_load_grelu(
    zi_tr, z12_tr,
    zi_val, z12_val,
    ckpt_path, force,
    m, seed, lr, epochs, eval_every, lambda_l2, lambda_l1,
    device,
) -> GReLUResidualAdapter:
    if ckpt_path.exists() and not force:
        print(f"  GReLU: loading checkpoint ({ckpt_path.name}) …")
        return load_grelu(ckpt_path)

    has_val = (zi_val is not None) and (z12_val is not None)
    print(
        f"  GReLU: training  m={m}  epochs={epochs}  eval_every={eval_every}  "
        f"lr={lr:.0e}  λ_l2={lambda_l2:.0e}  λ_l1={lambda_l1:.0e}  "
        f"val={'yes' if has_val else 'no'}"
    )
    t0 = time.time()
    adapter = GReLUResidualAdapter(
        m=m, seed=seed, lr=lr, epochs=epochs,
        eval_every=eval_every, lambda_l2=lambda_l2, lambda_l1=lambda_l1,
        device=str(device),
    ).fit(
        zi_tr, z12_tr,
        z_src_val=zi_val if has_val else None,
        z_tgt_val=z12_val if has_val else None,
        verbose=True,
    )
    print(f"    done in {time.time()-t0:.1f}s")
    save_grelu(adapter, ckpt_path)
    return adapter


def _train_or_load_mlp(
    zi_tr, z12_tr,
    zi_val, z12_val,
    ckpt_path, force,
    hidden, dropout, lr, weight_decay, epochs, eval_every, batch_size, seed,
    device,
) -> MLPResidualAdapter:
    if ckpt_path.exists() and not force:
        print(f"  MLP: loading checkpoint ({ckpt_path.name}) …")
        return load_mlp(ckpt_path)

    has_val = (zi_val is not None) and (z12_val is not None)
    print(
        f"  MLP: training  hidden={hidden}  epochs={epochs}  "
        f"lr={lr:.0e}  wd={weight_decay:.0e}  dropout={dropout}  "
        f"batch={batch_size}  val={'yes' if has_val else 'no'}"
    )
    t0 = time.time()
    adapter = MLPResidualAdapter(
        hidden=hidden, dropout=dropout,
        lr=lr, weight_decay=weight_decay,
        epochs=epochs, eval_every=eval_every,
        batch_size=batch_size, seed=seed,
        device=str(device),
    ).fit(
        zi_tr, z12_tr,
        z_src_val=zi_val if has_val else None,
        z_tgt_val=z12_val if has_val else None,
        verbose=True,
    )
    print(f"    done in {time.time()-t0:.1f}s")
    save_mlp(adapter, ckpt_path)
    return adapter


def _train_or_load_attention(
    zi_tr, z12_tr,
    zi_val, z12_val,
    ckpt_path, force,
    n_tokens, token_dim, nhead, n_layers, dim_feedforward, dropout,
    lr, weight_decay, epochs, eval_every, batch_size, seed,
    device,
) -> AttentionResidualAdapter:
    if ckpt_path.exists() and not force:
        print(f"  Attention: loading checkpoint ({ckpt_path.name}) …")
        return load_attention(ckpt_path)

    has_val = (zi_val is not None) and (z12_val is not None)
    print(
        f"  Attention: training  n_tokens={n_tokens}  token_dim={token_dim}  "
        f"nhead={nhead}  n_layers={n_layers}  epochs={epochs}  "
        f"lr={lr:.0e}  wd={weight_decay:.0e}  val={'yes' if has_val else 'no'}"
    )
    t0 = time.time()
    adapter = AttentionResidualAdapter(
        n_tokens=n_tokens, token_dim=token_dim,
        nhead=nhead, n_layers=n_layers, dim_feedforward=dim_feedforward,
        dropout=dropout, lr=lr, weight_decay=weight_decay,
        epochs=epochs, eval_every=eval_every,
        batch_size=batch_size, seed=seed,
        device=str(device),
    ).fit(
        zi_tr, z12_tr,
        z_src_val=zi_val if has_val else None,
        z_tgt_val=z12_val if has_val else None,
        verbose=True,
    )
    print(f"    done in {time.time()-t0:.1f}s")
    save_attention(adapter, ckpt_path)
    return adapter


def _train_or_load_lora(
    zi_tr, z12_tr,
    zi_val, z12_val,
    ckpt_path, force,
    rank, lora_alpha, lr, weight_decay, epochs, eval_every, batch_size, seed,
    device,
) -> LoRAResidualAdapter:
    if ckpt_path.exists() and not force:
        print(f"  LoRA: loading checkpoint ({ckpt_path.name}) …")
        return load_lora(ckpt_path)

    has_val = (zi_val is not None) and (z12_val is not None)
    print(
        f"  LoRA: training  rank={rank}  alpha={lora_alpha}  epochs={epochs}  "
        f"lr={lr:.0e}  wd={weight_decay:.0e}  batch={batch_size}  "
        f"val={'yes' if has_val else 'no'}"
    )
    t0 = time.time()
    adapter = LoRAResidualAdapter(
        rank=rank, lora_alpha=lora_alpha,
        lr=lr, weight_decay=weight_decay,
        epochs=epochs, eval_every=eval_every,
        batch_size=batch_size, seed=seed,
        device=str(device),
    ).fit(
        zi_tr, z12_tr,
        z_src_val=zi_val if has_val else None,
        z_tgt_val=z12_val if has_val else None,
        verbose=True,
    )
    print(f"    done in {time.time()-t0:.1f}s")
    save_lora(adapter, ckpt_path)
    return adapter


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate Lead-I → 12-lead latent adapters (QT experiment)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Paths
    parser.add_argument("--checkpoint",   default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--ptbxl-dir",    default=str(DEFAULT_SIGNALS_DIR))
    parser.add_argument("--labels-csv",   default=str(DEFAULT_LABELS))
    parser.add_argument("--latent-dir",   default=str(DEFAULT_LATENT_DIR),
                        help="Cache dir for extracted latents (.npz). "
                             "Shared with analyze_qtc.py --adapter-latent-dir.")
    parser.add_argument("--adapter-dir",  default=str(DEFAULT_ADAPTER_DIR),
                        help="Cache dir for trained adapter checkpoints.")
    parser.add_argument("--split",        default="val", choices=["val", "test"],
                        help="Split to evaluate adapters on.")
    parser.add_argument("--batch-size",   type=int, default=64)

    # Cache control
    parser.add_argument("--force-reextract", action="store_true", default=False,
                        help="Re-run encoder forward passes even if latent caches exist.")
    parser.add_argument("--force-retrain",   action="store_true", default=False,
                        help="Re-train adapters even if adapter checkpoints exist.")

    # Adapter selection
    parser.add_argument("--adapters", nargs="+",
                        choices=["linear", "krr", "grelu", "mlp", "attention", "lora"],
                        default=None,
                        help="Subset of adapters to run (default: all six).")

    # Linear hyper-parameters
    parser.add_argument("--linear-alpha",  type=float, default=1.0)

    # KRR hyper-parameters
    parser.add_argument("--krr-alpha",     type=float, default=0.1)
    parser.add_argument("--krr-nsub",      type=int,   default=3000)

    # GReLU hyper-parameters
    parser.add_argument("--grelu-m",          type=int,   default=GRELU_DEFAULT_CONFIG["m"])
    parser.add_argument("--grelu-seed",       type=int,   default=GRELU_DEFAULT_CONFIG["seed"])
    parser.add_argument("--grelu-lr",         type=float, default=GRELU_DEFAULT_CONFIG["lr"])
    parser.add_argument("--grelu-epochs",     type=int,   default=GRELU_DEFAULT_CONFIG["epochs"])
    parser.add_argument("--grelu-eval-every", type=int,   default=GRELU_DEFAULT_CONFIG["eval_every"])
    parser.add_argument("--grelu-lambda-l2",  type=float, default=GRELU_DEFAULT_CONFIG["lambda_l2"])
    parser.add_argument("--grelu-lambda-l1",  type=float, default=GRELU_DEFAULT_CONFIG["lambda_l1"])
    parser.add_argument("--grelu-val",        action="store_true", default=False,
                        help="Use Lead-I val latents for GReLU best-checkpoint tracking "
                             "(restores state with lowest val latent MSE).")
    # MLP hyper-parameters
    parser.add_argument("--mlp-hidden",       type=int,   default=MLP_DEFAULT_CONFIG["hidden"])
    parser.add_argument("--mlp-dropout",      type=float, default=MLP_DEFAULT_CONFIG["dropout"])
    parser.add_argument("--mlp-lr",           type=float, default=MLP_DEFAULT_CONFIG["lr"])
    parser.add_argument("--mlp-weight-decay", type=float, default=MLP_DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--mlp-epochs",       type=int,   default=MLP_DEFAULT_CONFIG["epochs"])
    parser.add_argument("--mlp-eval-every",   type=int,   default=MLP_DEFAULT_CONFIG["eval_every"])
    parser.add_argument("--mlp-batch-size",   type=int,   default=MLP_DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--mlp-seed",         type=int,   default=MLP_DEFAULT_CONFIG["seed"])
    parser.add_argument("--mlp-val",          action="store_true", default=False,
                        help="Use val latents for MLP best-checkpoint tracking.")
    # Attention hyper-parameters
    parser.add_argument("--attn-n-tokens",        type=int,
                        default=ATTENTION_DEFAULT_CONFIG["n_tokens"])
    parser.add_argument("--attn-token-dim",       type=int,
                        default=ATTENTION_DEFAULT_CONFIG["token_dim"])
    parser.add_argument("--attn-nhead",           type=int,
                        default=ATTENTION_DEFAULT_CONFIG["nhead"])
    parser.add_argument("--attn-n-layers",        type=int,
                        default=ATTENTION_DEFAULT_CONFIG["n_layers"])
    parser.add_argument("--attn-dim-feedforward", type=int,
                        default=ATTENTION_DEFAULT_CONFIG["dim_feedforward"])
    parser.add_argument("--attn-dropout",         type=float,
                        default=ATTENTION_DEFAULT_CONFIG["dropout"])
    parser.add_argument("--attn-lr",              type=float,
                        default=ATTENTION_DEFAULT_CONFIG["lr"])
    parser.add_argument("--attn-weight-decay",    type=float,
                        default=ATTENTION_DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--attn-epochs",          type=int,
                        default=ATTENTION_DEFAULT_CONFIG["epochs"])
    parser.add_argument("--attn-eval-every",      type=int,
                        default=ATTENTION_DEFAULT_CONFIG["eval_every"])
    parser.add_argument("--attn-batch-size",      type=int,
                        default=ATTENTION_DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--attn-seed",            type=int,
                        default=ATTENTION_DEFAULT_CONFIG["seed"])
    parser.add_argument("--attn-val",             action="store_true", default=False,
                        help="Use val latents for Attention best-checkpoint tracking.")
    # LoRA hyper-parameters
    parser.add_argument("--lora-rank",         type=int,   default=LORA_DEFAULT_CONFIG["rank"])
    parser.add_argument("--lora-alpha",        type=float, default=LORA_DEFAULT_CONFIG["lora_alpha"])
    parser.add_argument("--lora-lr",           type=float, default=LORA_DEFAULT_CONFIG["lr"])
    parser.add_argument("--lora-weight-decay", type=float, default=LORA_DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--lora-epochs",       type=int,   default=LORA_DEFAULT_CONFIG["epochs"])
    parser.add_argument("--lora-eval-every",   type=int,   default=LORA_DEFAULT_CONFIG["eval_every"])
    parser.add_argument("--lora-batch-size",   type=int,   default=LORA_DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lora-seed",         type=int,   default=LORA_DEFAULT_CONFIG["seed"])
    parser.add_argument("--lora-val",          action="store_true", default=False,
                        help="Use val latents for LoRA best-checkpoint tracking.")

    args = parser.parse_args()

    run_adapters = set(args.adapters) if args.adapters else {
        "linear", "krr", "grelu", "mlp", "attention", "lora"
    }

    # ── Device ────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Model checkpoint ──────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: model checkpoint not found at {ckpt_path}")
        sys.exit(1)
    ckpt_mtime = os.path.getmtime(ckpt_path)
    meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(
        f"Model checkpoint: epoch={meta.get('epoch','?')}  "
        f"val_MAE={meta.get('val_mae', float('nan')):.2f} ms  "
        f"val_R={meta.get('val_r', float('nan')):.3f}"
    )
    model = load_model(ckpt_path, device)

    # ── Cache directories ─────────────────────────────────────────────────
    latent_dir  = Path(args.latent_dir);  latent_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = Path(args.adapter_dir); adapter_dir.mkdir(parents=True, exist_ok=True)

    linear_ckpt    = adapter_dir / "linear_adapter.pkl"
    krr_ckpt       = adapter_dir / "krr_adapter.pkl"
    grelu_ckpt     = adapter_dir / "grelu_adapter.pt"
    mlp_ckpt       = adapter_dir / "mlp_adapter.pt"
    attention_ckpt = adapter_dir / "attention_adapter.pt"
    lora_ckpt      = adapter_dir / "lora_adapter.pt"

    # ── Dataset kwargs ────────────────────────────────────────────────────
    def _ds(split, leads):
        return ForcedLeadDataset(
            lead_indices=leads,
            ptbxl_signals_dir=args.ptbxl_dir,
            labels_csv=args.labels_csv,
            split=split,
        )

    # ── Latent extraction (cached) ────────────────────────────────────────
    print("\n── Latents ───────────────────────────────────────────────────")

    z12_tr, _, _, _ = get_latents(
        model, _ds("train", list(range(12))), device, args.batch_size,
        cache_path=latent_dir / "train_12lead.npz",
        ckpt_mtime=ckpt_mtime, force=args.force_reextract,
        desc="12-lead train",
    )
    zi_tr, _, _, _ = get_latents(
        model, _ds("train", [0]), device, args.batch_size,
        cache_path=latent_dir / "train_leadI.npz",
        ckpt_mtime=ckpt_mtime, force=args.force_reextract,
        desc="Lead I train",
    )
    zi_eval, _, tgt_eval, ids_eval = get_latents(
        model, _ds(args.split, [0]), device, args.batch_size,
        cache_path=latent_dir / f"{args.split}_leadI.npz",
        ckpt_mtime=ckpt_mtime, force=args.force_reextract,
        desc=f"Lead I {args.split}",
    )

    # Optional: val latents for best-state tracking (GReLU / MLP / Attention / LoRA)
    _need_val_latents = (
        ("grelu"     in run_adapters and args.grelu_val) or
        ("mlp"       in run_adapters and args.mlp_val) or
        ("attention" in run_adapters and args.attn_val) or
        ("lora"      in run_adapters and args.lora_val)
    )
    zi_val_shared  = None
    z12_val_shared = None
    if _need_val_latents:
        zi_val_shared, _, _, _ = get_latents(
            model, _ds("val", [0]), device, args.batch_size,
            cache_path=latent_dir / "val_leadI.npz",
            ckpt_mtime=ckpt_mtime, force=args.force_reextract,
            desc="Lead I val",
        )
        z12_val_shared, _, _, _ = get_latents(
            model, _ds("val", list(range(12))), device, args.batch_size,
            cache_path=latent_dir / "val_12lead.npz",
            ckpt_mtime=ckpt_mtime, force=args.force_reextract,
            desc="12-lead val",
        )

    # Convenience aliases (keep grelu names for backward compat)
    zi_val_for_grelu  = zi_val_shared  if args.grelu_val else None
    z12_val_for_grelu = z12_val_shared if args.grelu_val else None

    # ── Baseline ──────────────────────────────────────────────────────────
    print(f"\n── Baseline ({args.split}) ───────────────────────────────────")
    pred_raw = predict_from_latents(model, zi_eval, device)
    all_results = {}
    all_results["Lead I (raw)"] = _metrics(pred_raw, tgt_eval, "Lead I (raw)")

    # ── Train / load adapters ─────────────────────────────────────────────
    print("\n── Adapters ──────────────────────────────────────────────────")
    adapters: dict[str, object] = {}

    if "linear" in run_adapters:
        adapters["Lead I + Linear"] = _train_or_load_linear(
            zi_tr, z12_tr, linear_ckpt,
            alpha=args.linear_alpha, force=args.force_retrain,
        )

    if "krr" in run_adapters:
        adapters["Lead I + KRR"] = _train_or_load_krr(
            zi_tr, z12_tr, krr_ckpt,
            alpha=args.krr_alpha, n_subsample=args.krr_nsub,
            force=args.force_retrain,
        )

    if "grelu" in run_adapters:
        adapters["Lead I + GReLU"] = _train_or_load_grelu(
            zi_tr, z12_tr,
            zi_val=zi_val_for_grelu,
            z12_val=z12_val_for_grelu,
            ckpt_path=grelu_ckpt, force=args.force_retrain,
            m=args.grelu_m, seed=args.grelu_seed,
            lr=args.grelu_lr, epochs=args.grelu_epochs,
            eval_every=args.grelu_eval_every,
            lambda_l2=args.grelu_lambda_l2,
            lambda_l1=args.grelu_lambda_l1,
            device=device,
        )

    if "mlp" in run_adapters:
        adapters["Lead I + MLP"] = _train_or_load_mlp(
            zi_tr, z12_tr,
            zi_val=zi_val_shared  if args.mlp_val else None,
            z12_val=z12_val_shared if args.mlp_val else None,
            ckpt_path=mlp_ckpt, force=args.force_retrain,
            hidden=args.mlp_hidden, dropout=args.mlp_dropout,
            lr=args.mlp_lr, weight_decay=args.mlp_weight_decay,
            epochs=args.mlp_epochs, eval_every=args.mlp_eval_every,
            batch_size=args.mlp_batch_size, seed=args.mlp_seed,
            device=device,
        )

    if "attention" in run_adapters:
        adapters["Lead I + Attention"] = _train_or_load_attention(
            zi_tr, z12_tr,
            zi_val=zi_val_shared  if args.attn_val else None,
            z12_val=z12_val_shared if args.attn_val else None,
            ckpt_path=attention_ckpt, force=args.force_retrain,
            n_tokens=args.attn_n_tokens, token_dim=args.attn_token_dim,
            nhead=args.attn_nhead, n_layers=args.attn_n_layers,
            dim_feedforward=args.attn_dim_feedforward, dropout=args.attn_dropout,
            lr=args.attn_lr, weight_decay=args.attn_weight_decay,
            epochs=args.attn_epochs, eval_every=args.attn_eval_every,
            batch_size=args.attn_batch_size, seed=args.attn_seed,
            device=device,
        )

    if "lora" in run_adapters:
        adapters["Lead I + LoRA"] = _train_or_load_lora(
            zi_tr, z12_tr,
            zi_val=zi_val_shared  if args.lora_val else None,
            z12_val=z12_val_shared if args.lora_val else None,
            ckpt_path=lora_ckpt, force=args.force_retrain,
            rank=args.lora_rank, lora_alpha=args.lora_alpha,
            lr=args.lora_lr, weight_decay=args.lora_weight_decay,
            epochs=args.lora_epochs, eval_every=args.lora_eval_every,
            batch_size=args.lora_batch_size, seed=args.lora_seed,
            device=device,
        )

    # ── Evaluate adapters ─────────────────────────────────────────────────
    if adapters:
        print(f"\n── Evaluation ({args.split}) ─────────────────────────────")
        for name, adapter in adapters.items():
            z_adapted    = adapter.transform(zi_eval)
            pred_adapted = predict_from_latents(model, z_adapted, device)
            all_results[name] = _metrics(pred_adapted, tgt_eval, name)

    # ── Summary table ─────────────────────────────────────────────────────
    W = 90
    print("\n" + "─" * W)
    print(f"  {'Config':<28}  {'MAE':>8}  {'RMSE':>8}  {'R':>7}  "
          f"{'Bias':>8}  {'±10ms':>7}  {'±20ms':>7}")
    print("  " + "─" * (W - 2))
    for name, r in all_results.items():
        print(
            f"  {name:<28}  {r['mae']:>8.2f}  {r['rmse']:>8.2f}  "
            f"{r['r']:>7.3f}  {r['bias']:>+8.2f}  "
            f"{r['w10']:>6.1f}%  {r['w20']:>6.1f}%"
        )
    print("─" * W)

    # ── Cache summary ─────────────────────────────────────────────────────
    print(f"\nLatent cache  : {latent_dir}/")
    for f in sorted(latent_dir.glob("*.npz")):
        print(f"  {f.name:<28}  {f.stat().st_size/1e6:.1f} MB")
    print(f"\nAdapter cache : {adapter_dir}/")
    for p in [linear_ckpt, krr_ckpt, grelu_ckpt, mlp_ckpt, attention_ckpt, lora_ckpt]:
        if p.exists():
            print(f"  {p.name:<28}  {p.stat().st_size/1e6:.1f} MB")
    print("\nDone.")


if __name__ == "__main__":
    main()
