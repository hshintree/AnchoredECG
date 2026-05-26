"""
Post-training analysis for the waveform-only QT interval regression model.

Produces three outputs:
  1. qt_pearson_grid.png  — 2×2 predicted-vs-GT scatter panels (one per lead config)
  2. qt_roc_curves.png    — AUROC curves for all 4 lead configs overlaid
  3. qt_prc_curves.png    — AUPRC/PRC curves for all 4 lead configs overlaid

QTc Bazett correction:
    QTc (ms) = predicted_QT / sqrt(RR_Mean_Global_ms / 1000)

Prolongation thresholds (narrow/wide QRS rule):
    Narrow QRS (QRS_Dur_Global ≤ 100 ms) → QTc > 500 ms = prolonged
    Wide   QRS (QRS_Dur_Global > 100 ms) → QTc > 550 ms = prolonged

Ground truth QTc computed from 12SL QT_Int_Global + RR_Mean_Global.
Predicted QTc computed from model prediction + same RR_Mean_Global.

Usage (local):
    python qt_finetune/plot_qt_results.py
    python qt_finetune/plot_qt_results.py --checkpoint path/to/best_qt_model.pt

Usage (Modal — data + checkpoint already in the volume):
    modal run modal_finetune.py::plot_results_entrypoint
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
)
from torch_geometric.loader import DataLoader as PyGDataLoader

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT   = _SCRIPT_DIR.parent
_MODELS_DIR  = _REPO_ROOT / "models"
for _p in (_MODELS_DIR, _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from validate_qt import (
    ForcedLeadDataset, ICMLeadDataset, load_model, LIMB_LEAD_INDICES,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DATA = _REPO_ROOT / "data"
DEFAULT_CHECKPOINT   = _REPO_ROOT / "qt_finetune" / "checkpoints" / "best_qt_model.pt"
DEFAULT_LABELS       = _SCRIPT_DIR / "qt_labels.csv"
DEFAULT_SIGNALS_DIR  = _DATA / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
DEFAULT_FEATURES_DIR = _DATA / "ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1"
DEFAULT_OUTPUT_DIR   = _SCRIPT_DIR / "plots"

# ---------------------------------------------------------------------------
# Colour palette (perceptually distinct, prints OK in greyscale)
# ---------------------------------------------------------------------------

PALETTE = {
    "12-lead":      "#1565C0",   # deep blue
    "6 limb leads": "#2E7D32",   # deep green
    "Lead I":       "#E65100",   # deep orange
    "ICM (V3−V2)":  "#AD1457",   # deep pink
}

LINESTYLES = {
    "12-lead":      "-",
    "6 limb leads": "--",
    "Lead I":       "-.",
    "ICM (V3−V2)":  ":",
}

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model: nn.Module,
    ds,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Run the model on *ds* with shuffle=False.

    Returns:
        pred_qt  : (N,) float32 — predicted QT in ms
        tgt_qt   : (N,) float32 — ground-truth QT_Int_Global in ms
        ecg_ids  : [N] int      — ECG IDs in dataset order (for metadata join)
    """
    loader = PyGDataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    preds, tgts = [], []
    for batch_data, qt in loader:
        batch_data = batch_data.to(device)
        B   = batch_data.num_graphs
        tab = torch.zeros(B, 0, device=device)
        p   = model(batch_data, tabular=tab)["logits"].flatten()
        preds.append(p.cpu().numpy())
        tgts.append(qt.flatten().numpy())
    pred_qt = np.concatenate(preds)
    tgt_qt  = np.concatenate(tgts)
    ecg_ids = [item["ecg_id"] for item in ds.index]
    return pred_qt, tgt_qt, ecg_ids

# ---------------------------------------------------------------------------
# Metadata — QRS_Dur_Global & RR_Mean_Global from 12SL
# ---------------------------------------------------------------------------

def load_test_metadata(labels_csv: Path, features_dir: Path | None) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by ecg_id with columns:
        QT_Int_Global, HR, strat_fold,
        QRS_Dur_Global (ms),  RR_Mean_Global (ms)

    If features_dir is None or the 12SL CSV is missing, RR is derived from HR
    and QRS_Dur_Global is NaN (classification plots will be skipped).
    """
    df = pd.read_csv(labels_csv)
    df = df[df["strat_fold"] == 10].dropna(subset=["QT_Int_Global"]).copy()

    if features_dir is not None:
        sl12_path = Path(features_dir) / "features" / "12sl_features.csv"
        if sl12_path.exists():
            sl12 = pd.read_csv(
                sl12_path,
                usecols=["ecg_id", "QRS_Dur_Global", "RR_Mean_Global"],
            )
            df = df.merge(sl12, on="ecg_id", how="left")
            print(f"  Merged 12SL features: QRS available={df['QRS_Dur_Global'].notna().sum()}, "
                  f"RR available={df['RR_Mean_Global'].notna().sum()}")
        else:
            print(f"  WARNING: {sl12_path} not found — using HR-derived RR, QRS=NaN")
            df["QRS_Dur_Global"] = np.nan
            df["RR_Mean_Global"] = 60_000.0 / df["HR"].clip(lower=30)
    else:
        print("  No features_dir — using HR-derived RR, QRS=NaN")
        df["QRS_Dur_Global"] = np.nan
        df["RR_Mean_Global"] = 60_000.0 / df["HR"].clip(lower=30)

    return df.set_index("ecg_id")

# ---------------------------------------------------------------------------
# Plot 1 — Pearson correlation 2×2 grid
# ---------------------------------------------------------------------------

def plot_pearson_grid(all_results: dict, output_path: Path) -> None:
    """
    2×2 grid — one hexbin scatter panel per lead configuration.
    x = ground truth QT_Int_Global (ms), y = predicted QT (ms).
    """
    names = list(all_results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    axes_flat = axes.flatten()

    for ax, name in zip(axes_flat, names):
        res   = all_results[name]
        pred  = res["pred_qt"]
        tgt   = res["tgt_qt"]
        color = PALETTE[name]

        slope, intercept, r, _, _ = stats.linregress(tgt, pred)
        mae  = float(np.abs(pred - tgt).mean())
        rmse = float(np.sqrt(((pred - tgt) ** 2).mean()))
        bias = float((pred - tgt).mean())

        lo = min(tgt.min(), pred.min()) - 8
        hi = max(tgt.max(), pred.max()) + 8

        # Hexbin density
        hb = ax.hexbin(tgt, pred, gridsize=55, cmap="Blues",
                       mincnt=1, linewidths=0.15)
        cb = fig.colorbar(hb, ax=ax, pad=0.02, shrink=0.85)
        cb.set_label("Count", fontsize=8)
        cb.ax.tick_params(labelsize=7)

        # Identity y = x
        ax.plot([lo, hi], [lo, hi], "--", color="#666", lw=1.2,
                alpha=0.65, label="Identity (y = x)", zorder=3)

        # Regression line
        xs = np.linspace(lo, hi, 300)
        ax.plot(xs, slope * xs + intercept, "-", color=color, lw=2.2,
                alpha=0.9, label="Regression", zorder=4)

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Ground Truth QT$_{Int,Global}$ (ms)", fontsize=11)
        ax.set_ylabel("Predicted QT (ms)", fontsize=11)
        ax.set_title(name, fontsize=13, fontweight="bold", color=color)
        ax.legend(fontsize=9, loc="lower right")
        ax.set_aspect("equal", adjustable="box")

        # Stats annotation
        txt = (
            f"R  = {r:.3f}\n"
            f"MAE  = {mae:.1f} ms\n"
            f"RMSE = {rmse:.1f} ms\n"
            f"Bias = {bias:+.1f} ms\n"
            f"n  = {len(tgt):,}"
        )
        ax.text(
            0.04, 0.97, txt, transform=ax.transAxes,
            fontsize=9.5, va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=color, alpha=0.92),
        )

    fig.suptitle(
        "QT Interval Regression — Test Set (PTB-XL strat_fold=10)\n"
        "Waveform-only model  ·  500 Hz  ·  QT$_{Int,Global}$ (12SL)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Pearson grid  → {output_path}")

# ---------------------------------------------------------------------------
# QTc prolongation classification
# ---------------------------------------------------------------------------

def compute_qtc_classification(
    pred_qt: np.ndarray,
    tgt_qt:  np.ndarray,
    ecg_ids: list[int],
    meta_df: pd.DataFrame,
) -> dict | None:
    """
    Compute Bazett QTc and binary prolongation labels for one lead config.

    Bazett:  QTc = QT_ms / sqrt(RR_Mean_Global_ms / 1000)
    Threshold:
        Narrow QRS (QRS_Dur_Global ≤ 100 ms) → QTc > 500 ms
        Wide   QRS (QRS_Dur_Global  > 100 ms) → QTc > 550 ms

    Returns None if <5 positive cases.
    """
    ids = np.array(ecg_ids, dtype=int)

    # Align metadata (preserve dataset order)
    rr_ms  = meta_df.reindex(ids)["RR_Mean_Global"].values.astype(float)
    qrs_ms = meta_df.reindex(ids)["QRS_Dur_Global"].values.astype(float)

    # Clamp RR to physiological range [300, 1500] ms
    rr_sec = np.clip(rr_ms / 1000.0, 0.30, 1.50)

    # Bazett QTc (ms)
    pred_qtc = pred_qt / np.sqrt(rr_sec)
    tgt_qtc  = tgt_qt  / np.sqrt(rr_sec)

    # Narrow/wide QRS threshold
    threshold = np.where(qrs_ms <= 100, 500.0, 550.0)

    # Exclude samples missing RR or QRS data
    valid = ~(np.isnan(rr_ms) | np.isnan(qrs_ms))
    if valid.sum() < 10:
        print("    Insufficient metadata for classification — skipping.")
        return None

    y_true  = (tgt_qtc[valid] > threshold[valid]).astype(int)
    y_score = pred_qtc[valid]   # continuous predicted QTc as the ranking score

    n_pos = int(y_true.sum())
    if n_pos < 5:
        print(f"    Too few positives (n_pos={n_pos}) — skipping.")
        return None

    return dict(
        y_true=y_true, y_score=y_score,
        n=int(valid.sum()), n_pos=n_pos,
        prevalence=n_pos / valid.sum(),
        pred_qtc=pred_qtc[valid], tgt_qtc=tgt_qtc[valid],
        qrs_dur=qrs_ms[valid], threshold=threshold[valid],
    )

# ---------------------------------------------------------------------------
# Plot 2 — ROC curves (all configs overlaid)
# ---------------------------------------------------------------------------

def plot_roc_curves(all_cls: dict, output_path: Path) -> None:
    """Overlaid ROC curves for all 4 lead configs."""
    fig, ax = plt.subplots(figsize=(8, 7))

    for name, cls in all_cls.items():
        if cls is None:
            continue
        color = PALETTE[name]
        ls    = LINESTYLES[name]
        auroc = roc_auc_score(cls["y_true"], cls["y_score"])
        fpr, tpr, _ = roc_curve(cls["y_true"], cls["y_score"])
        ax.plot(
            fpr, tpr, color=color, lw=2.5, ls=ls,
            label=f"{name}   AUROC = {auroc:.3f}  (n={cls['n']:,}, pos={cls['n_pos']})",
        )
        print(f"  {name:<22}  AUROC={auroc:.4f}  prev={cls['prevalence']:.1%}")

    ax.plot([0, 1], [0, 1], "--", color="#aaa", lw=1.2, label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate",  fontsize=13)
    ax.set_title(
        "ROC — QTc Prolongation Detection\n"
        "Narrow QRS: QTc > 500 ms  ·  Wide QRS: QTc > 550 ms (Bazett)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="lower right", framealpha=0.92)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  ROC curves    → {output_path}")

# ---------------------------------------------------------------------------
# Plot 3 — PRC curves (all configs overlaid)
# ---------------------------------------------------------------------------

def plot_prc_curves(all_cls: dict, output_path: Path) -> None:
    """Overlaid Precision-Recall curves for all 4 lead configs."""
    fig, ax = plt.subplots(figsize=(8, 7))

    # Use the first valid config's prevalence for the baseline line
    prevalence = next(
        (c["prevalence"] for c in all_cls.values() if c is not None), 0.0
    )

    for name, cls in all_cls.items():
        if cls is None:
            continue
        color = PALETTE[name]
        ls    = LINESTYLES[name]
        auprc = average_precision_score(cls["y_true"], cls["y_score"])
        prec, rec, _ = precision_recall_curve(cls["y_true"], cls["y_score"])
        ax.plot(
            rec, prec, color=color, lw=2.5, ls=ls,
            label=f"{name}   AUPRC = {auprc:.3f}",
        )
        print(f"  {name:<22}  AUPRC={auprc:.4f}")

    ax.axhline(
        prevalence, color="#aaa", lw=1.2, ls="--",
        label=f"No-skill baseline (prev = {prevalence:.1%})",
    )
    ax.set_xlabel("Recall",    fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title(
        "Precision-Recall — QTc Prolongation Detection\n"
        "Narrow QRS: QTc > 500 ms  ·  Wide QRS: QTc > 550 ms (Bazett)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper right", framealpha=0.92)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  PRC curves    → {output_path}")

# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_all(
    checkpoint:   Path,
    ptbxl_dir:    Path,
    features_dir: Path | None,
    labels_csv:   Path,
    output_dir:   Path,
    batch_size:   int = 32,
) -> dict[str, Path]:
    """
    Run full analysis pipeline.  Returns dict of {plot_name: output_path}.
    Called by both the local CLI and the Modal function.
    """
    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"Device     : {device}")
    print(f"Checkpoint : {checkpoint}")

    model = load_model(Path(checkpoint), device)

    ds_kwargs = dict(ptbxl_signals_dir=ptbxl_dir, labels_csv=labels_csv)

    configs = [
        ("12-lead",      ForcedLeadDataset(lead_indices=list(range(12)), **ds_kwargs)),
        ("6 limb leads", ForcedLeadDataset(lead_indices=LIMB_LEAD_INDICES, **ds_kwargs)),
        ("Lead I",       ForcedLeadDataset(lead_indices=[0], **ds_kwargs)),
        ("ICM (V3−V2)",  ICMLeadDataset(**ds_kwargs)),
    ]

    # ── Inference ─────────────────────────────────────────────────────────
    print("\n── Inference ─────────────────────────────────────────────────")
    all_results = {}
    for name, ds in configs:
        print(f"\n{name}  ({len(ds)} test samples)")
        pred_qt, tgt_qt, ecg_ids = run_inference(model, ds, device, batch_size)
        err  = pred_qt - tgt_qt
        mae  = float(np.abs(err).mean())
        rmse = float(np.sqrt((err ** 2).mean()))
        r    = float(np.corrcoef(pred_qt, tgt_qt)[0, 1])
        bias = float(err.mean())
        w10  = float((np.abs(err) <= 10).mean() * 100)
        w20  = float((np.abs(err) <= 20).mean() * 100)
        print(f"  MAE={mae:.2f} ms  RMSE={rmse:.2f} ms  R={r:.3f}  "
              f"bias={bias:+.2f}  ≤10ms={w10:.1f}%  ≤20ms={w20:.1f}%")
        all_results[name] = dict(pred_qt=pred_qt, tgt_qt=tgt_qt, ecg_ids=ecg_ids)

    # ── Metadata for QTc classification ───────────────────────────────────
    print("\n── Test metadata ────────────────────────────────────────────")
    meta_df = load_test_metadata(Path(labels_csv), Path(features_dir) if features_dir else None)

    # ── Plot 1: Pearson grid ──────────────────────────────────────────────
    print("\n── Pearson correlation grid ─────────────────────────────────")
    p1 = Path(output_dir) / "qt_pearson_grid.png"
    plot_pearson_grid(all_results, p1)

    # ── QTc classification ────────────────────────────────────────────────
    print("\n── QTc classification (Bazett, narrow/wide QRS thresholds) ──")
    all_cls = {}
    for name, res in all_results.items():
        print(f"\n  {name}")
        cls = compute_qtc_classification(
            res["pred_qt"], res["tgt_qt"], res["ecg_ids"], meta_df
        )
        if cls is not None:
            print(f"    n={cls['n']:,}  n_pos={cls['n_pos']} ({cls['prevalence']:.1%})")
        all_cls[name] = cls

    # ── Plot 2: ROC ───────────────────────────────────────────────────────
    print("\n── ROC curves ───────────────────────────────────────────────")
    p2 = Path(output_dir) / "qt_roc_curves.png"
    plot_roc_curves(all_cls, p2)

    # ── Plot 3: PRC ───────────────────────────────────────────────────────
    print("\n── PRC curves ───────────────────────────────────────────────")
    p3 = Path(output_dir) / "qt_prc_curves.png"
    plot_prc_curves(all_cls, p3)

    print(f"\nDone. All plots in {output_dir}/")
    return {"pearson_grid": p1, "roc_curves": p2, "prc_curves": p3}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot QT regression results + QTc prolongation curves"
    )
    parser.add_argument("--checkpoint",   default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--ptbxl-dir",    default=str(DEFAULT_SIGNALS_DIR))
    parser.add_argument(
        "--features-dir",
        default=str(DEFAULT_FEATURES_DIR) if DEFAULT_FEATURES_DIR.exists() else None,
        help="Path to PTB-XL comprehensive feature dataset root "
             "(for QRS_Dur_Global and RR_Mean_Global from 12SL).",
    )
    parser.add_argument("--labels-csv",  default=str(DEFAULT_LABELS))
    parser.add_argument("--output-dir",  default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch-size",  type=int, default=32)
    args = parser.parse_args()

    run_all(
        checkpoint   = Path(args.checkpoint),
        ptbxl_dir    = Path(args.ptbxl_dir),
        features_dir = Path(args.features_dir) if args.features_dir else None,
        labels_csv   = Path(args.labels_csv),
        output_dir   = Path(args.output_dir),
        batch_size   = args.batch_size,
    )


if __name__ == "__main__":
    main()
