#!/usr/bin/env python3
"""
Paper-ready / poster-ready figures for QTc prediction.

Evaluates exactly 6 model configurations on PTB-XL:
  1. 12-lead          — base ECGGNNSimple with all 12 leads
  2. Lead I           — base model, Lead I only
  3. Lead I + Linear  — frozen linear residual adapter
  4. Lead I + MLP     — frozen MLP residual adapter
  5. Lead I + JEPA-Linear (λ=1.0)  — end-to-end JEPA, linear predictor
  6. Lead I + JEPA-CvxGReLU-256    — end-to-end JEPA, convex GReLU (m=256)

Figures saved to --output-dir (default: the good_figs/ directory):
  roc.png / roc.pdf          — combined ROC overlay
  prc.png / prc.pdf          — combined PRC overlay
  bland_altman.png / .pdf    — 2×3 per-config Bland-Altman panels
  pearson.png / .pdf         — combined Pearson R regression overlay

Classification labels:
  Positive = ref_qtc > --threshold (default 450 ms, single universal threshold).
  This gives ~25% prevalence and much more interpretable AUPRC than the clinical
  500/550 ms cut-off which has ~1.7% prevalence.

Usage:
    python good_figs/make_paper_figs.py
    python good_figs/make_paper_figs.py --threshold 450 --split val
    python good_figs/make_paper_figs.py --threshold 500 --split test --batch-size 64
    python good_figs/make_paper_figs.py --output-dir /tmp/figs
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
)
from torch_geometric.loader import DataLoader as PyGDataLoader

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_SCRIPT_DIR  = Path(__file__).resolve().parent           # good_figs/
_REPO_ROOT   = _SCRIPT_DIR.parent                        # AnchoredECG/
_QT_DIR      = _REPO_ROOT / "qt_finetune"
_MODELS_DIR  = _REPO_ROOT / "models"

for _p in (_MODELS_DIR, _QT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from validate_qt import ForcedLeadDataset, load_model        # noqa: E402
from qt_adapter import get_latents, predict_from_latents, load_linear, load_mlp  # noqa: E402

DEFAULT_CHECKPOINT   = _QT_DIR / "checkpoints" / "best_qt_model.pt"
DEFAULT_LABELS       = _QT_DIR / "qt_labels.csv"
DEFAULT_SIGNALS_DIR  = _REPO_ROOT / "data" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
DEFAULT_FEATURES_DIR = _REPO_ROOT / "data" / "ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1"
DEFAULT_ADAPTER_DIR  = _QT_DIR / "checkpoints" / "adapters"
DEFAULT_LATENT_DIR   = _QT_DIR / "checkpoints" / "latents"
DEFAULT_JEPA_DIR     = _QT_DIR / "checkpoints" / "jepa"

# ---------------------------------------------------------------------------
# Visual identity
# ---------------------------------------------------------------------------

# Canonical display names in plot order
CONFIGS_ORDERED = [
    "12-lead",
    "Lead I",
    "Lead I + Linear",
    "Lead I + MLP",
    "Lead I + JEPA-Linear (λ=1.0)",
    "Lead I + JEPA-CvxGReLU-256",
]

PALETTE = {
    "12-lead":                       "#1565C0",   # deep blue
    "Lead I":                        "#E65100",   # deep orange
    "Lead I + Linear":               "#2E7D32",   # dark green
    "Lead I + MLP":                  "#AD1457",   # dark pink/magenta
    "Lead I + JEPA-Linear (λ=1.0)":  "#C62828",   # dark red
    "Lead I + JEPA-CvxGReLU-256":    "#00695C",   # dark teal
}

# All solid lines — color alone distinguishes the configs
LS = {
    "12-lead":                       "-",
    "Lead I":                        "-",
    "Lead I + Linear":               "-",
    "Lead I + MLP":                  "-",
    "Lead I + JEPA-Linear (λ=1.0)":  "-",
    "Lead I + JEPA-CvxGReLU-256":    "-",
}

LW = 2.2   # default line width for curves

# ---------------------------------------------------------------------------
# Global matplotlib style
# ---------------------------------------------------------------------------

def _apply_pub_style() -> None:
    """Apply publication-quality defaults to all subsequent figures."""
    plt.rcParams.update({
        # Fonts
        "font.family":           "DejaVu Sans",
        "font.size":             13,
        "axes.labelsize":        14,
        "axes.titlesize":        14,
        "axes.titleweight":      "bold",
        "xtick.labelsize":       12,
        "ytick.labelsize":       12,
        "legend.fontsize":       11,
        "legend.title_fontsize": 11,
        "legend.framealpha":     0.93,
        "legend.edgecolor":      "#cccccc",
        # Spines
        "axes.spines.top":   False,
        "axes.spines.right": False,
        # Grid
        "axes.grid":         True,
        "grid.alpha":        0.22,
        "grid.linestyle":    "--",
        "grid.linewidth":    0.6,
        # Save
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
    })


def _save(fig: plt.Figure, stem: str, out_dir: Path) -> None:
    """Save figure as PNG (300 dpi)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{stem}.png"
    fig.savefig(p, bbox_inches="tight")
    print(f"  saved → {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, ds, device: torch.device, batch_size: int = 32):
    """Run a forward pass; return (pred_qt, tgt_qt, ecg_ids)."""
    loader = PyGDataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    preds, tgts = [], []
    for bd, qt in loader:
        bd = bd.to(device)
        B  = bd.num_graphs
        p  = model(bd, tabular=torch.zeros(B, 0, device=device))["logits"].flatten()
        preds.append(p.cpu().numpy())
        tgts.append(qt.flatten().numpy())
    return (
        np.concatenate(preds),
        np.concatenate(tgts),
        [i["ecg_id"] for i in ds.index],
    )


# ---------------------------------------------------------------------------
# Metadata + QTc (Bazett)
# ---------------------------------------------------------------------------

def load_metadata(labels_csv: Path, features_dir: Path, split: str) -> pd.DataFrame:
    fold_map = {"train": (1, 8), "val": (9, 9), "test": (10, 10)}
    lo, hi = fold_map[split]
    df = pd.read_csv(labels_csv)
    df = df[(df["strat_fold"] >= lo) & (df["strat_fold"] <= hi)]
    df = df.dropna(subset=["QT_Int_Global"]).copy()
    sl12 = pd.read_csv(
        features_dir / "features" / "12sl_features.csv",
        usecols=["ecg_id", "QRS_Dur_Global", "RR_Mean_Global"],
    )
    df = df.merge(sl12, on="ecg_id", how="left")
    return df.set_index("ecg_id")


def add_qtc(pred_qt: np.ndarray, tgt_qt: np.ndarray,
            ecg_ids: list, meta: pd.DataFrame) -> dict:
    """Apply Bazett QTc correction using 12SL RR_Mean_Global."""
    ids    = np.array(ecg_ids, dtype=int)
    rr_ms  = meta.reindex(ids)["RR_Mean_Global"].values.astype(float)
    qrs_ms = meta.reindex(ids)["QRS_Dur_Global"].values.astype(float)
    rr_sec = np.clip(rr_ms / 1000.0, 0.30, 1.50)
    pred_qtc = pred_qt / np.sqrt(rr_sec)
    ref_qtc  = tgt_qt  / np.sqrt(rr_sec)
    valid    = ~(np.isnan(rr_ms) | np.isnan(qrs_ms))
    return dict(
        ids=ids, pred_qt=pred_qt, tgt_qt=tgt_qt,
        rr_ms=rr_ms, rr_sec=rr_sec, qrs_ms=qrs_ms,
        pred_qtc=pred_qtc, ref_qtc=ref_qtc, valid=valid,
    )


# ---------------------------------------------------------------------------
# Figure 1 — ROC curves
# ---------------------------------------------------------------------------

def plot_roc(all_data: dict, threshold_ms: float, out_dir: Path,
             split: str = "val") -> None:
    """Combined ROC overlay at a fixed QTc threshold."""
    fig, ax = plt.subplots(figsize=(6.5, 5.8))

    for name in CONFIGS_ORDERED:
        if name not in all_data:
            continue
        d       = all_data[name]
        v       = d["valid"]
        y_true  = (d["ref_qtc"][v] > threshold_ms).astype(int)
        y_score = d["pred_qtc"][v]
        n_pos   = int(y_true.sum())
        n       = int(v.sum())

        if n_pos < 5:
            print(f"  ROC [{name}]: only {n_pos} positives at QTc>{threshold_ms:.0f} — skipping")
            continue

        auc           = roc_auc_score(y_true, y_score)
        fpr, tpr, _   = roc_curve(y_true, y_score)
        prev          = n_pos / n

        ax.plot(fpr, tpr,
                color=PALETTE[name], lw=LW, ls=LS[name],
                label=f"{name}  (AUC = {auc:.3f})")
        print(f"  ROC  {name:<35}  AUC={auc:.4f}  n+={n_pos}  prev={prev:.1%}")

    ax.plot([0, 1], [0, 1], "--", color="#999", lw=1.2, label="No-skill (AUC = 0.50)")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.02)
    ax.set_xlabel("False Positive Rate  (1 − Specificity)")
    ax.set_ylabel("True Positive Rate  (Sensitivity)")
    ax.set_title(f"ROC Curves  ·  QTc > {threshold_ms:.0f} ms  ·  {split} split")
    ax.legend(loc="lower right", handlelength=2.2)
    fig.tight_layout()
    _save(fig, f"roc_{split}_thr{int(threshold_ms)}", out_dir)


# ---------------------------------------------------------------------------
# Figure 2 — PRC curves
# ---------------------------------------------------------------------------

def plot_prc(all_data: dict, threshold_ms: float, out_dir: Path,
             split: str = "val") -> None:
    """Combined PRC overlay at a fixed QTc threshold."""
    fig, ax = plt.subplots(figsize=(6.5, 5.8))

    baseline_prev: float | None = None
    for name in CONFIGS_ORDERED:
        if name not in all_data:
            continue
        d       = all_data[name]
        v       = d["valid"]
        y_true  = (d["ref_qtc"][v] > threshold_ms).astype(int)
        y_score = d["pred_qtc"][v]
        n_pos   = int(y_true.sum())
        n       = int(v.sum())

        if n_pos < 5:
            print(f"  PRC [{name}]: only {n_pos} positives at QTc>{threshold_ms:.0f} — skipping")
            continue

        prev  = n_pos / n
        if baseline_prev is None:
            baseline_prev = prev

        auprc        = average_precision_score(y_true, y_score)
        prec, rec, _ = precision_recall_curve(y_true, y_score)

        ax.plot(rec, prec,
                color=PALETTE[name], lw=LW, ls=LS[name],
                label=f"{name}  (AUPRC = {auprc:.3f})")
        print(f"  PRC  {name:<35}  AUPRC={auprc:.4f}  n+={n_pos}  prev={prev:.1%}")

    if baseline_prev is not None:
        ax.axhline(baseline_prev, color="#999", lw=1.2, ls="--",
                   label=f"No-skill  (prev = {baseline_prev:.1%})")

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curves  ·  QTc > {threshold_ms:.0f} ms  ·  {split} split")
    ax.legend(loc="upper right", handlelength=2.2)
    fig.tight_layout()
    _save(fig, f"prc_{split}_thr{int(threshold_ms)}", out_dir)


# ---------------------------------------------------------------------------
# Figure 3 — Bland-Altman (2×3 grid, one panel per config)
# ---------------------------------------------------------------------------

def plot_bland_altman(all_data: dict, threshold_ms: float,
                      out_dir: Path, split: str = "val") -> None:
    """2×3 grid Bland-Altman (predicted QTc vs reference QTc)."""
    present = [n for n in CONFIGS_ORDERED if n in all_data]
    n_panel = len(present)
    ncols   = min(3, n_panel)
    import math
    nrows   = math.ceil(n_panel / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 4.8 * nrows),
        squeeze=False,
    )
    flat = axes.flatten()

    for ax, name in zip(flat, present):
        d        = all_data[name]
        v        = d["valid"]
        pred_qtc = d["pred_qtc"][v]
        ref_qtc  = d["ref_qtc"][v]
        color    = PALETTE[name]

        diff = pred_qtc - ref_qtc
        mean = (pred_qtc + ref_qtc) / 2.0

        bias   = float(diff.mean())
        sd     = float(diff.std(ddof=1))
        loa_lo = bias - 1.96 * sd
        loa_hi = bias + 1.96 * sd

        ax.scatter(mean, diff, s=4, alpha=0.22, color=color, linewidths=0, rasterized=True)
        ax.axhline(bias,   color="black",   lw=1.8, ls="-",
                   label=f"Bias = {bias:+.1f} ms")
        ax.axhline(loa_hi, color="#D32F2F", lw=1.4, ls="--",
                   label=f"+1.96 SD = {loa_hi:+.1f} ms")
        ax.axhline(loa_lo, color="#D32F2F", lw=1.4, ls="--",
                   label=f"−1.96 SD = {loa_lo:+.1f} ms")
        ax.axhline(0, color="#bbb", lw=0.8, ls=":")

        ax.set_xlabel("Mean of Predicted & Reference QTc (ms)", fontsize=11)
        ax.set_ylabel("Predicted − Reference QTc (ms)",         fontsize=11)
        ax.set_title(name, color=color)

        # Annotation box
        ax.text(
            0.03, 0.97,
            f"n = {int(v.sum()):,}\nbias = {bias:+.1f} ms\nLoA  [{loa_lo:+.1f}, {loa_hi:+.1f}]",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=color, alpha=0.88),
        )
        ax.legend(fontsize=9, loc="upper right")

    # Hide unused panels
    for ax in flat[n_panel:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Bland-Altman: Predicted QTc vs Reference QTc (Bazett, 12SL RR)\n"
        f"{split} split  ·  PTB-XL 500 Hz  ·  QTc > {threshold_ms:.0f} ms threshold",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    _save(fig, f"bland_altman_{split}_thr{int(threshold_ms)}", out_dir)


# ---------------------------------------------------------------------------
# Figure 4 — Pearson R (combined regression overlay)
# ---------------------------------------------------------------------------

def plot_pearson(all_data: dict, threshold_ms: float,
                 out_dir: Path, split: str = "val") -> None:
    """
    Combined Pearson overlay: subsampled scatter + regression line per config.
    """
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    rng = np.random.RandomState(42)

    # Global axis limits
    lo, hi = float("inf"), float("-inf")
    for name in CONFIGS_ORDERED:
        if name not in all_data:
            continue
        d = all_data[name]
        v = d["valid"]
        lo = min(lo, d["ref_qtc"][v].min(), d["pred_qtc"][v].min())
        hi = max(hi, d["ref_qtc"][v].max(), d["pred_qtc"][v].max())
    lo -= 12; hi += 12

    for name in CONFIGS_ORDERED:
        if name not in all_data:
            continue
        d        = all_data[name]
        v        = d["valid"]
        ref_qtc  = d["ref_qtc"][v]
        pred_qtc = d["pred_qtc"][v]
        color    = PALETTE[name]
        ls_style = LS[name]

        slope, intercept, r, _, _ = stats.linregress(ref_qtc, pred_qtc)
        mae  = float(np.abs(pred_qtc - ref_qtc).mean())

        # Sparse scatter for density impression (180 points, very transparent)
        n   = len(ref_qtc)
        idx = rng.choice(n, min(180, n), replace=False)
        ax.scatter(ref_qtc[idx], pred_qtc[idx],
                   s=6, alpha=0.12, color=color, linewidths=0, zorder=2,
                   rasterized=True)

        # Regression line
        xs = np.linspace(lo, hi, 400)
        ax.plot(xs, slope * xs + intercept,
                color=color, lw=LW, ls=ls_style, zorder=3,
                label=f"{name}   R={r:.3f}   MAE={mae:.1f} ms")

    # Reference lines
    ax.plot([lo, hi], [lo, hi], "--", color="#777", lw=1.3, alpha=0.65,
            zorder=1, label="Identity  (y = x)")
    ax.axvline(threshold_ms, color="#555", lw=0.85, ls=":", alpha=0.5, zorder=1)
    ax.axhline(threshold_ms, color="#555", lw=0.85, ls=":", alpha=0.5, zorder=1,
               label=f"QTc = {threshold_ms:.0f} ms threshold")

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Reference QTc (Bazett, 12SL RR)  [ms]")
    ax.set_ylabel("Predicted QTc  [ms]")
    ax.set_title(
        f"Pearson R Correlation — Predicted vs Reference QTc\n"
        f"{split} split  ·  PTB-XL 500 Hz",
    )
    ax.legend(fontsize=10, loc="lower right", handlelength=2.4, framealpha=0.93)
    fig.tight_layout()
    _save(fig, f"pearson_{split}_thr{int(threshold_ms)}", out_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures for the QTc prediction paper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint",   default=str(DEFAULT_CHECKPOINT),
                        help="ECGGNNSimple base-model checkpoint.")
    parser.add_argument("--ptbxl-dir",    default=str(DEFAULT_SIGNALS_DIR))
    parser.add_argument("--features-dir", default=str(DEFAULT_FEATURES_DIR))
    parser.add_argument("--labels-csv",   default=str(DEFAULT_LABELS))
    parser.add_argument("--adapter-dir",  default=str(DEFAULT_ADAPTER_DIR))
    parser.add_argument("--latent-dir",   default=str(DEFAULT_LATENT_DIR))
    parser.add_argument("--jepa-dir",     default=str(DEFAULT_JEPA_DIR))
    parser.add_argument("--output-dir",   default=str(_SCRIPT_DIR),
                        help="Directory to write figures into.")
    parser.add_argument("--split",   default="val", choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=float, default=450.0,
                        help="QTc threshold (ms) for prolongation classification. "
                             "Applied universally (single threshold, not narrow/wide QRS rule).")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    _apply_pub_style()

    ckpt_path    = Path(args.checkpoint)
    adapter_dir  = Path(args.adapter_dir)
    latent_dir   = Path(args.latent_dir)
    jepa_dir     = Path(args.jepa_dir)
    out_dir      = Path(args.output_dir)
    thr          = args.threshold
    sp           = args.split

    # ── Checkpoint guard ──────────────────────────────────────────────────
    if not ckpt_path.exists():
        print(f"ERROR: base model checkpoint not found: {ckpt_path}")
        sys.exit(1)

    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"Device: {device}")

    _ckpt_meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"Base model: epoch={_ckpt_meta.get('epoch','?')}  "
          f"val_MAE={_ckpt_meta.get('val_mae', float('nan')):.2f} ms  "
          f"val_R={_ckpt_meta.get('val_r', float('nan')):.3f}")

    model = load_model(ckpt_path, device)

    # ── Metadata ──────────────────────────────────────────────────────────
    meta = load_metadata(Path(args.labels_csv), Path(args.features_dir), sp)

    ds_kwargs = dict(
        ptbxl_signals_dir=args.ptbxl_dir,
        labels_csv=args.labels_csv,
        split=sp,
    )

    all_data: dict[str, dict] = {}

    # ── Config 1: 12-lead ─────────────────────────────────────────────────
    print(f"\n── Inference ({sp}) ──────────────────────────────────────────")
    print("  12-lead ...", end=" ", flush=True)
    ds12   = ForcedLeadDataset(lead_indices=list(range(12)), **ds_kwargs)
    p, t, ids = run_inference(model, ds12, device, args.batch_size)
    all_data["12-lead"] = add_qtc(p, t, ids, meta)
    err = p - t
    print(f"MAE={np.abs(err).mean():.2f} ms  R={np.corrcoef(p, t)[0,1]:.3f}")

    # ── Config 2: Lead I ──────────────────────────────────────────────────
    print("  Lead I  ...", end=" ", flush=True)
    dsI    = ForcedLeadDataset(lead_indices=[0], **ds_kwargs)
    p, t, ids = run_inference(model, dsI, device, args.batch_size)
    all_data["Lead I"] = add_qtc(p, t, ids, meta)
    err = p - t
    print(f"MAE={np.abs(err).mean():.2f} ms  R={np.corrcoef(p, t)[0,1]:.3f}")

    # ── Latents for frozen adapters ───────────────────────────────────────
    latent_dir.mkdir(parents=True, exist_ok=True)
    _ckpt_mtime = ckpt_path.stat().st_mtime

    zi_val, _, tgt_qt_val, ecg_ids_val = get_latents(
        model,
        ForcedLeadDataset(lead_indices=[0], **ds_kwargs),
        device, args.batch_size,
        cache_path=latent_dir / f"{sp}_leadI.npz",
        ckpt_mtime=_ckpt_mtime,
        force=False,
        desc=f"Lead I {sp}",
    )

    # ── Config 3: Lead I + Linear (frozen adapter) ────────────────────────
    _linear_ckpt = adapter_dir / "linear_adapter.pkl"
    if _linear_ckpt.exists():
        print("  Lead I + Linear ...", end=" ", flush=True)
        lin_ada  = load_linear(_linear_ckpt)
        z_lin    = lin_ada.transform(zi_val)
        p_lin    = predict_from_latents(model, z_lin, device)
        all_data["Lead I + Linear"] = add_qtc(p_lin, tgt_qt_val, ecg_ids_val, meta)
        err = p_lin - tgt_qt_val
        print(f"MAE={np.abs(err).mean():.2f} ms  R={np.corrcoef(p_lin, tgt_qt_val)[0,1]:.3f}")
    else:
        print(f"  Lead I + Linear: checkpoint not found at {_linear_ckpt} — skipping")

    # ── Config 4: Lead I + MLP (frozen adapter) ───────────────────────────
    _mlp_ckpt = adapter_dir / "mlp_adapter.pt"
    if _mlp_ckpt.exists():
        print("  Lead I + MLP ...", end=" ", flush=True)
        mlp_ada  = load_mlp(_mlp_ckpt)
        z_mlp    = mlp_ada.transform(zi_val)
        p_mlp    = predict_from_latents(model, z_mlp, device)
        all_data["Lead I + MLP"] = add_qtc(p_mlp, tgt_qt_val, ecg_ids_val, meta)
        err = p_mlp - tgt_qt_val
        print(f"MAE={np.abs(err).mean():.2f} ms  R={np.corrcoef(p_mlp, tgt_qt_val)[0,1]:.3f}")
    else:
        print(f"  Lead I + MLP: checkpoint not found at {_mlp_ckpt} — skipping")

    # ── JEPA models (Load I only at inference) ────────────────────────────
    _jepa_configs = [
        ("Lead I + JEPA-Linear (λ=1.0)", jepa_dir / "jepa_linear_lqt1.0.pt"),
        ("Lead I + JEPA-CvxGReLU-256",   jepa_dir / "jepa_cgrelu256.pt"),
    ]

    _jepa_any = any(p.exists() for _, p in _jepa_configs)
    if _jepa_any:
        from train_jepa import load_jepa_inference_model   # lazy import

        _leadI_ds = ForcedLeadDataset(lead_indices=[0], **ds_kwargs)
        for label, jpath in _jepa_configs:
            if not jpath.exists():
                print(f"  {label}: checkpoint not found at {jpath} — skipping")
                continue
            print(f"  {label} ...", end=" ", flush=True)
            jepa_model = load_jepa_inference_model(jpath, device)
            p_j, t_j, ids_j = run_inference(jepa_model, _leadI_ds, device, args.batch_size)
            all_data[label] = add_qtc(p_j, t_j, ids_j, meta)
            err_j = p_j - t_j
            print(f"MAE={np.abs(err_j).mean():.2f} ms  R={np.corrcoef(p_j, t_j)[0,1]:.3f}")

    # ── Print prevalence at the chosen threshold ──────────────────────────
    print(f"\n── QTc prevalence (first valid config) at threshold = {thr:.0f} ms ──")
    first = next(iter(all_data.values()))
    v     = first["valid"]
    qtc   = first["ref_qtc"][v]
    cnt   = int((qtc > thr).sum())
    print(f"  N={int(v.sum()):,}  n_pos={cnt}  prevalence={cnt / len(qtc) * 100:.1f}%\n")

    # ── Figures ───────────────────────────────────────────────────────────
    print("── Generating figures ────────────────────────────────────────")
    plot_roc(all_data, thr, out_dir, sp)
    plot_prc(all_data, thr, out_dir, sp)
    plot_bland_altman(all_data, thr, out_dir, sp)
    plot_pearson(all_data, thr, out_dir, sp)
    print(f"\nAll figures saved to:  {out_dir}/")


if __name__ == "__main__":
    main()
