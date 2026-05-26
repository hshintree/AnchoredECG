"""
Standalone adapter training + evaluation script for the QT fine-tuning experiment.

Trains all three latent-space adapters that map the Lead-I graph_embed (192-dim)
to the 12-lead graph_embed (192-dim) using a residual formulation:

    r_i = z_12lead_i - z_lead_i
    z_adapted = z_lead_i + adapter(z_lead_i)

Adapters
--------
    Linear  : Ridge regression (sklearn)
    KRR     : KernelRidge + RBF kernel + StandardScaler (sklearn)
    GReLU   : Convex fixed-gate GReLU network trained with R-FISTA (PyTorch)

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

    linear_adapter.pkl    — joblib pickle
    krr_adapter.pkl       — joblib pickle
    grelu_adapter.pt      — torch checkpoint (state_dict + config)

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
    python qt_finetune/train_adapters.py --adapters linear grelu

    # GReLU hyper-parameters
    python qt_finetune/train_adapters.py --grelu-epochs 1000 --grelu-m 2048

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
    save_linear, load_linear,
    save_krr,    load_krr,
    save_grelu,  load_grelu,
    GRELU_DEFAULT_CONFIG,
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
                        choices=["linear", "krr", "grelu"], default=None,
                        help="Subset of adapters to run (default: all three).")

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

    args = parser.parse_args()

    run_adapters = set(args.adapters) if args.adapters else {"linear", "krr", "grelu"}

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

    linear_ckpt = adapter_dir / "linear_adapter.pkl"
    krr_ckpt    = adapter_dir / "krr_adapter.pkl"
    grelu_ckpt  = adapter_dir / "grelu_adapter.pt"

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

    # Optional: val latents for GReLU best-state tracking
    zi_val_for_grelu  = None
    z12_val_for_grelu = None
    if "grelu" in run_adapters and args.grelu_val:
        zi_val_for_grelu, _, _, _ = get_latents(
            model, _ds("val", [0]), device, args.batch_size,
            cache_path=latent_dir / "val_leadI.npz",
            ckpt_mtime=ckpt_mtime, force=args.force_reextract,
            desc="Lead I val",
        )
        z12_val_for_grelu, _, _, _ = get_latents(
            model, _ds("val", list(range(12))), device, args.batch_size,
            cache_path=latent_dir / "val_12lead.npz",
            ckpt_mtime=ckpt_mtime, force=args.force_reextract,
            desc="12-lead val",
        )

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
    for p, tag in [(linear_ckpt, "linear"), (krr_ckpt, "krr"), (grelu_ckpt, "grelu")]:
        if p.exists():
            print(f"  {p.name:<28}  {p.stat().st_size/1e6:.1f} MB")
    print("\nDone.")


if __name__ == "__main__":
    main()
