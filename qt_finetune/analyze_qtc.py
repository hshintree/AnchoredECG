"""
QTc Diagnostic Analysis — Validation Set  (no Modal required)

Runs locally against the checkpoint at qt_finetune/checkpoints/best_qt_model.pt
and the PTB-XL data under data/.

Steps
-----
1. Inference on validation split (strat_fold=9) for all 4 lead configs.
2. Compute Bazett QTc from BOTH predicted and ground-truth QT using the SAME
   12SL RR_Mean_Global for both → eliminates heart-rate confound.
3. Print a sample table (random + highest-QTc rows) to verify the calculation.
4. Report QTc prolongation prevalence, AUROC, and AUPRC for each config.
   Includes a note explaining why AUPRC is low at strict thresholds.
5. Bland-Altman: 2×2 grid — predicted QTc vs reference QTc, one panel per config.
6. Pearson QTc: 2×2 grid — predicted QTc vs reference QTc.

Bazett correction:
    QTc_ms = QT_ms / sqrt(RR_Mean_Global_ms / 1000)

Prolongation thresholds (narrow / wide QRS):
    Narrow QRS (QRS_Dur_Global ≤ 100 ms) → QTc > 500 ms
    Wide   QRS (QRS_Dur_Global  > 100 ms) → QTc > 550 ms

Usage:
    python qt_finetune/analyze_qtc.py
    python qt_finetune/analyze_qtc.py --checkpoint qt_finetune/checkpoints/best_qt_model.pt
    python qt_finetune/analyze_qtc.py --split test   # switch to test set

Pull the trained checkpoint first if not already done:
    modal volume get ptbxl-data /best_qt_model.pt \\
        qt_finetune/checkpoints/best_qt_model.pt
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
from qt_adapter import (
    get_latents, predict_from_latents,
    LinearResidualAdapter, KernelResidualAdapter, GReLUResidualAdapter,
    save_linear, load_linear,
    save_krr,    load_krr,
    save_grelu,  load_grelu,
    GRELU_DEFAULT_CONFIG,
)

# ---------------------------------------------------------------------------
# Default local paths
# ---------------------------------------------------------------------------

_DATA = _REPO_ROOT / "data"
DEFAULT_CHECKPOINT   = _REPO_ROOT / "qt_finetune" / "checkpoints" / "best_qt_model.pt"
DEFAULT_LABELS       = _SCRIPT_DIR / "qt_labels.csv"
DEFAULT_SIGNALS_DIR  = _DATA / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
DEFAULT_FEATURES_DIR = _DATA / "ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1"
DEFAULT_OUTPUT_DIR   = _SCRIPT_DIR / "plots"

PALETTE = {
    "12-lead":           "#1565C0",   # deep blue
    "6 limb leads":      "#2E7D32",   # deep green
    "Lead I":            "#E65100",   # deep orange
    "ICM (V3−V2)":       "#AD1457",   # deep pink
    "Lead I + Linear":   "#FF8F00",   # amber  (adapter of Lead I)
    "Lead I + KRR":      "#6A1B9A",   # deep purple
    "Lead I + GReLU":    "#00838F",   # teal
}
LS = {
    "12-lead":           "-",
    "6 limb leads":      "--",
    "Lead I":            "-.",
    "ICM (V3−V2)":       ":",
    "Lead I + Linear":   (0, (3, 1, 1, 1)),   # dash-dot-dot
    "Lead I + KRR":      (0, (5, 1)),          # densely dashed
    "Lead I + GReLU":    (0, (1, 1)),          # densely dotted
}

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, ds, device, batch_size=32):
    loader = PyGDataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    preds, tgts = [], []
    for bd, qt in loader:
        bd = bd.to(device)
        B  = bd.num_graphs
        p  = model(bd, tabular=torch.zeros(B, 0, device=device))["logits"].flatten()
        preds.append(p.cpu().numpy())
        tgts.append(qt.flatten().numpy())
    return np.concatenate(preds), np.concatenate(tgts), [i["ecg_id"] for i in ds.index]

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def load_metadata(labels_csv, features_dir, split):
    fold_map = {"train": (1, 8), "val": (9, 9), "test": (10, 10)}
    lo, hi = fold_map[split]

    df = pd.read_csv(labels_csv)
    df = df[(df["strat_fold"] >= lo) & (df["strat_fold"] <= hi)]
    df = df.dropna(subset=["QT_Int_Global"]).copy()

    sl12 = pd.read_csv(
        Path(features_dir) / "features" / "12sl_features.csv",
        usecols=["ecg_id", "QRS_Dur_Global", "RR_Mean_Global"],
    )
    df = df.merge(sl12, on="ecg_id", how="left")
    return df.set_index("ecg_id")

# ---------------------------------------------------------------------------
# QTc computation
# ---------------------------------------------------------------------------

def add_qtc(pred_qt, tgt_qt, ecg_ids, meta):
    ids      = np.array(ecg_ids, dtype=int)
    rr_ms    = meta.reindex(ids)["RR_Mean_Global"].values.astype(float)
    qrs_ms   = meta.reindex(ids)["QRS_Dur_Global"].values.astype(float)
    qtc_csv  = meta.reindex(ids)["QTc_Bazett"].values.astype(float)   # cross-check
    hr       = meta.reindex(ids)["HR"].values.astype(float)

    rr_sec    = np.clip(rr_ms / 1000.0, 0.30, 1.50)
    pred_qtc  = pred_qt / np.sqrt(rr_sec)
    ref_qtc   = tgt_qt  / np.sqrt(rr_sec)          # same RR as pred
    threshold = np.where(qrs_ms <= 100, 500.0, 550.0)

    valid = ~(np.isnan(rr_ms) | np.isnan(qrs_ms))

    return dict(
        ids=ids, pred_qt=pred_qt, tgt_qt=tgt_qt,
        rr_ms=rr_ms, rr_sec=rr_sec,
        qrs_ms=qrs_ms, hr=hr,
        pred_qtc=pred_qtc, ref_qtc=ref_qtc,
        qtc_csv=qtc_csv,
        threshold=threshold, valid=valid,
    )

# ---------------------------------------------------------------------------
# 1. Sample calculation table
# ---------------------------------------------------------------------------

def print_sample_table(d: dict, n_random=15, n_top=10) -> None:
    """Print rows showing the full QTc calculation chain."""
    n = len(d["ids"])
    np.random.seed(42)
    rand_idx = np.random.choice(n, size=min(n_random, n), replace=False)

    # Top n_top by reference QTc (to include borderline/prolonged cases)
    top_idx = np.argsort(d["ref_qtc"])[-n_top:][::-1]

    all_idx = np.unique(np.concatenate([rand_idx, top_idx]))

    hdr = (f"{'ecg_id':>8}  {'HR':>5}  {'RR_ms':>6}  {'QRS_ms':>7}  "
           f"{'GT_QT':>7}  {'Pred_QT':>8}  "
           f"{'GT_QTc':>7}  {'Pred_QTc':>9}  {'QTc_CSV':>8}  "
           f"{'Thresh':>7}  {'Prolonged?':>10}")
    print("\n" + "─" * len(hdr))
    print("  QTc Calculation Sample Table  (GT_QTc = QT_Int_Global / √(RR_ms/1000))")
    print("─" * len(hdr))
    print(hdr)
    print("─" * len(hdr))
    for i in sorted(all_idx):
        tag = " ← TOP" if i in top_idx else ""
        prol = "YES ⚠" if d["ref_qtc"][i] > d["threshold"][i] else "no"
        print(
            f"  {d['ids'][i]:>8}  {d['hr'][i]:>5.0f}  {d['rr_ms'][i]:>6.0f}  "
            f"{d['qrs_ms'][i]:>7.0f}  "
            f"{d['tgt_qt'][i]:>7.1f}  {d['pred_qt'][i]:>8.1f}  "
            f"{d['ref_qtc'][i]:>7.1f}  {d['pred_qtc'][i]:>9.1f}  "
            f"{d['qtc_csv'][i]:>8.1f}  "
            f"{d['threshold'][i]:>7.0f}  {prol:>10}{tag}"
        )
    print("─" * len(hdr))
    print(
        "  Note: GT_QTc and QTc_CSV should be near-identical.\n"
        "  Any mismatch means QTc_CSV used HR-derived RR vs 12SL RR_Mean_Global.\n"
    )

# ---------------------------------------------------------------------------
# 2. Prevalence + AUROC / AUPRC report
# ---------------------------------------------------------------------------

def _threshold_metrics(y_true: np.ndarray, y_score: np.ndarray,
                        decision_threshold: np.ndarray) -> dict:
    """
    Compute confusion-matrix metrics at the clinical decision threshold.

    decision_threshold is per-sample (500 or 550 ms depending on QRS width),
    so we apply it directly to y_score rather than picking a single cut-off.
    """
    y_pred = (y_score > decision_threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    sens      = tp / (tp + fn) if (tp + fn) > 0 else float("nan")   # = recall
    spec      = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")   # = PPV
    f1        = (2 * precision * sens / (precision + sens)
                 if (precision + sens) > 0 else float("nan"))
    return dict(tp=tp, fp=fp, tn=tn, fn=fn,
                sens=sens, spec=spec, precision=precision, f1=f1)


def print_classification_report(all_data: dict) -> None:
    W = 90
    print("\n" + "╔" + "═" * W + "╗")
    print("║  QTc Prolongation Classification Report" + " " * (W - 40) + "║")
    print("║  Applied threshold: Narrow QRS (≤100ms)→QTc>500ms  /  Wide QRS (>100ms)→QTc>550ms"
          + " " * (W - 86) + "║")
    print("╠" + "═" * W + "╣")

    # ── Ranking metrics (AUROC / AUPRC) ──────────────────────────────────
    hdr1 = (f"  {'Config':<22} {'N':>5} {'N+':>5} {'Prev':>6} "
            f"{'AUROC':>7} {'AUPRC':>7}")
    print(hdr1)
    print("  " + "─" * (len(hdr1) - 2))
    rows = {}
    for name, d in all_data.items():
        v        = d["valid"]
        y_true   = (d["ref_qtc"][v] > d["threshold"][v]).astype(int)
        y_score  = d["pred_qtc"][v]
        thresh_v = d["threshold"][v]
        n, n_pos = int(v.sum()), int(y_true.sum())
        prev     = n_pos / n
        if n_pos < 5:
            print(f"  {name:<22} {n:>5} {n_pos:>5} {prev:>5.1%}  ← too few positives")
            rows[name] = None
            continue
        auroc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)
        cm    = _threshold_metrics(y_true, y_score, thresh_v)
        rows[name] = dict(n=n, n_pos=n_pos, prev=prev,
                          auroc=auroc, auprc=auprc, **cm)
        print(f"  {name:<22} {n:>5} {n_pos:>5} {prev:>5.1%} {auroc:>7.3f} {auprc:>7.3f}")

    # ── At-threshold confusion-matrix metrics ─────────────────────────────
    print("╠" + "─" * W + "╣")
    hdr2 = (f"  {'Config':<22} {'TP':>5} {'FP':>5} {'TN':>6} {'FN':>5} "
            f"{'Sens/Rec':>9} {'Spec':>7} {'Prec/PPV':>9} {'F1':>7}")
    print(hdr2)
    print("  " + "─" * (len(hdr2) - 2))
    for name, r in rows.items():
        if r is None:
            continue
        print(f"  {name:<22} {r['tp']:>5} {r['fp']:>5} {r['tn']:>6} {r['fn']:>5} "
              f"{r['sens']:>8.1%} {r['spec']:>7.1%} "
              f"{r['precision']:>8.1%} {r['f1']:>7.3f}")

    print("╠" + "═" * W + "╣")
    print("║  ⚠  AUPRC ceiling ≈ prevalence (1.7%).  Even perfect ranking gives low AUPRC." +
          " " * (W - 80) + "║")
    print("║     For a more informative metric, consider QTc > 450 ms (25% prevalence)." +
          " " * (W - 79) + "║")
    print("╚" + "═" * W + "╝\n")

    # ── QTc distribution across thresholds ───────────────────────────────
    first = next(iter(all_data.values()))
    v     = first["valid"]
    qtc   = first["ref_qtc"][v]
    print(f"  Reference QTc (GT) distribution  (N={v.sum():,}):")
    for thr in [450, 460, 480, 500, 550]:
        cnt = int((qtc > thr).sum())
        print(f"    QTc > {thr} ms :  {cnt:4d} samples  ({cnt / len(qtc) * 100:.1f}%)")
    print()

# ---------------------------------------------------------------------------
# 3. Bland-Altman  (2×2 grid)
# ---------------------------------------------------------------------------

def _grid(n: int):
    """Return (nrows, ncols, figsize) for n panels, filling row-major."""
    import math
    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)
    return nrows, ncols, (6.5 * ncols, 5.5 * nrows)


def plot_bland_altman(all_data: dict, output_path: Path) -> None:
    names = list(all_data.keys())
    nrows, ncols, figsize = _grid(len(names))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    flat_axes = axes.flatten()
    for ax, name in zip(flat_axes, names):
        d = all_data[name]
        v = d["valid"]
        pred_qtc = d["pred_qtc"][v]
        ref_qtc  = d["ref_qtc"][v]

        diff = pred_qtc - ref_qtc
        mean = (pred_qtc + ref_qtc) / 2.0

        bias   = float(diff.mean())
        sd     = float(diff.std(ddof=1))
        loa_lo = bias - 1.96 * sd
        loa_hi = bias + 1.96 * sd

        color = PALETTE.get(name, "#555")

        ax.scatter(mean, diff, s=3, alpha=0.25, color=color, linewidths=0)
        ax.axhline(bias,   color="black",   lw=1.8, ls="-",  label=f"Bias = {bias:+.1f} ms")
        ax.axhline(loa_hi, color="crimson", lw=1.4, ls="--", label=f"+1.96 SD = {loa_hi:+.1f} ms")
        ax.axhline(loa_lo, color="crimson", lw=1.4, ls="--", label=f"−1.96 SD = {loa_lo:+.1f} ms")
        ax.axhline(0, color="#aaa", lw=0.8, ls=":")

        ax.set_xlabel("Mean of Predicted & Reference QTc (ms)", fontsize=10)
        ax.set_ylabel("Predicted QTc − Reference QTc (ms)",     fontsize=10)
        ax.set_title(name, fontsize=12, fontweight="bold", color=color)
        ax.legend(fontsize=8.5, loc="upper right")
        ax.text(
            0.02, 0.04,
            f"n={v.sum():,}  bias={bias:+.1f}  LOA=[{loa_lo:+.1f}, {loa_hi:+.1f}]",
            transform=ax.transAxes, fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.85),
        )

    # Hide unused axes (if n not a perfect grid)
    for ax in flat_axes[len(names):]:
        ax.set_visible(False)

    fig.suptitle(
        "Bland-Altman: Predicted QTc vs Reference QTc (Bazett, 12SL RR)\n"
        "Validation Set (PTB-XL strat_fold=9) — Waveform-only model, 500 Hz",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Bland-Altman → {output_path}")

# ---------------------------------------------------------------------------
# 4. Pearson QTc  (2×2 grid)
# ---------------------------------------------------------------------------

def plot_pearson_qtc(all_data: dict, output_path: Path) -> None:
    names = list(all_data.keys())
    nrows, ncols, figsize = _grid(len(names))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat_axes = axes.flatten()

    for ax, name in zip(flat_axes, names):
        d = all_data[name]
        v = d["valid"]
        pred_qtc = d["pred_qtc"][v]
        ref_qtc  = d["ref_qtc"][v]
        color    = PALETTE.get(name, "#555")

        slope, intercept, r, _, _ = stats.linregress(ref_qtc, pred_qtc)
        mae  = float(np.abs(pred_qtc - ref_qtc).mean())
        rmse = float(np.sqrt(((pred_qtc - ref_qtc) ** 2).mean()))
        bias = float((pred_qtc - ref_qtc).mean())

        lo = min(ref_qtc.min(), pred_qtc.min()) - 8
        hi = max(ref_qtc.max(), pred_qtc.max()) + 8

        hb = ax.hexbin(ref_qtc, pred_qtc, gridsize=55, cmap="Blues",
                       mincnt=1, linewidths=0.15)
        fig.colorbar(hb, ax=ax, pad=0.02, shrink=0.85).set_label("Count", fontsize=8)

        ax.plot([lo, hi], [lo, hi], "--", color="#666", lw=1.2, alpha=0.6,
                label="Identity (y=x)")
        xs = np.linspace(lo, hi, 300)
        ax.plot(xs, slope * xs + intercept, "-", color=color, lw=2.2,
                alpha=0.9, label="Regression")

        # Mark the clinical prolongation threshold region
        thr_narrow, thr_wide = 500, 550
        ax.axvline(thr_narrow, color="#FF6F00", lw=0.8, ls=":", alpha=0.6,
                   label=f"QTc={thr_narrow} (narrow)")
        ax.axhline(thr_narrow, color="#FF6F00", lw=0.8, ls=":", alpha=0.6)

        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Reference QTc (Bazett, 12SL RR) [ms]",  fontsize=10)
        ax.set_ylabel("Predicted QTc (model QT → Bazett) [ms]", fontsize=10)
        ax.set_title(name, fontsize=12, fontweight="bold", color=color)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_aspect("equal", adjustable="box")

        ax.text(
            0.04, 0.97,
            f"R  = {r:.3f}\nMAE  = {mae:.1f} ms\nRMSE = {rmse:.1f} ms\n"
            f"Bias = {bias:+.1f} ms\nn  = {v.sum():,}",
            transform=ax.transAxes, fontsize=9.5, va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=color, alpha=0.92),
        )

    for ax in flat_axes[len(names):]:
        ax.set_visible(False)

    fig.suptitle(
        "Pearson Correlation: Predicted QTc vs Reference QTc (Bazett, 12SL RR)\n"
        "Validation Set (PTB-XL strat_fold=9) — Waveform-only model, 500 Hz",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Pearson QTc  → {output_path}")

# ---------------------------------------------------------------------------
# 5. One-off PRC at a fixed single threshold (e.g. QTc > 450 ms)
#    NOT run by default — pass --prc-450 to enable.
# ---------------------------------------------------------------------------

def plot_prc_single_threshold(
    all_data: dict,
    threshold_ms: float,
    output_path: Path,
) -> None:
    """
    PRC curves for all configs using a SINGLE universal QTc threshold.

    Unlike the narrow/wide QRS rule, every sample is labelled positive if
    ref_qtc > threshold_ms regardless of QRS duration.  This gives far
    higher prevalence (~25% at 450 ms) and thus a much more interpretable
    AUPRC.

    Not the primary analysis — call explicitly with --prc-450.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    for name, d in all_data.items():
        v        = d["valid"]
        ref_qtc  = d["ref_qtc"][v]
        pred_qtc = d["pred_qtc"][v]

        y_true  = (ref_qtc  > threshold_ms).astype(int)
        y_score = pred_qtc

        n_pos = int(y_true.sum())
        n     = int(v.sum())
        prev  = n_pos / n

        if n_pos < 5:
            print(f"  {name}: only {n_pos} positives at QTc>{threshold_ms:.0f} — skipping")
            continue

        auprc        = average_precision_score(y_true, y_score)
        prec, rec, _ = precision_recall_curve(y_true, y_score)

        ax.plot(rec, prec,
                color=PALETTE[name], lw=2.5, ls=LS[name],
                label=f"{name}   AUPRC={auprc:.3f}  (n+={n_pos}, {prev:.1%})")
        print(f"  {name:<22}  QTc>{threshold_ms:.0f}ms  n+={n_pos}  prev={prev:.1%}  AUPRC={auprc:.4f}")

    prevalence = next(
        (d["ref_qtc"][d["valid"]] > threshold_ms).mean()
        for d in all_data.values()
    )
    ax.axhline(prevalence, color="#aaa", lw=1.2, ls="--",
               label=f"No-skill baseline (prev={prevalence:.1%})")

    ax.set_xlabel("Recall",    fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title(
        f"Precision-Recall Curve — QTc > {threshold_ms:.0f} ms (Bazett, universal threshold)\n"
        "One-off analysis · NOT the primary narrow/wide-QRS threshold",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper right", framealpha=0.92)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  PRC (QTc>{threshold_ms:.0f}ms) → {output_path}")


# ---------------------------------------------------------------------------
# 6. Combined Pearson QTc  — all configs overlaid in one plot
# ---------------------------------------------------------------------------

def plot_pearson_combined(all_data: dict, output_path: Path,
                          split: str = "val") -> None:
    """
    Single-axes scatter + regression overlay for every config in all_data.

    Each config gets:
      - A random subsample of 350 points (alpha=0.12) for density impression
      - A regression line computed on the full split (lw=2.2)
      - A legend entry with R and MAE
    Identity line and QTc=500ms guide lines are shown in neutral colours.
    """
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    rng = np.random.RandomState(42)

    # First pass: global axis limits
    lo, hi = float("inf"), float("-inf")
    for d in all_data.values():
        v = d["valid"]
        lo = min(lo, d["ref_qtc"][v].min(), d["pred_qtc"][v].min())
        hi = max(hi, d["ref_qtc"][v].max(), d["pred_qtc"][v].max())
    lo -= 10; hi += 10

    # Second pass: scatter + regression per config
    for name, d in all_data.items():
        v        = d["valid"]
        ref_qtc  = d["ref_qtc"][v]
        pred_qtc = d["pred_qtc"][v]
        color    = PALETTE.get(name, "#555")
        ls_style = LS.get(name, "-")

        slope, intercept, r, _, _ = stats.linregress(ref_qtc, pred_qtc)
        mae  = float(np.abs(pred_qtc - ref_qtc).mean())
        rmse = float(np.sqrt(((pred_qtc - ref_qtc) ** 2).mean()))

        # Subsample scatter — enough to show density without clutter
        n   = len(ref_qtc)
        idx = rng.choice(n, min(350, n), replace=False)
        ax.scatter(ref_qtc[idx], pred_qtc[idx],
                   s=5, alpha=0.13, color=color, linewidths=0, zorder=2)

        xs = np.linspace(lo, hi, 300)
        ax.plot(xs, slope * xs + intercept,
                color=color, lw=2.3, ls=ls_style, zorder=3,
                label=f"{name}  R={r:.3f}  MAE={mae:.1f} ms  RMSE={rmse:.1f} ms")

    # Reference lines
    ax.plot([lo, hi], [lo, hi], "--", color="#777", lw=1.3, alpha=0.7,
            zorder=1, label="Identity (y = x)")
    ax.axvline(500, color="#FF6F00", lw=0.9, ls=":", alpha=0.55, zorder=1)
    ax.axhline(500, color="#FF6F00", lw=0.9, ls=":", alpha=0.55, zorder=1,
               label="QTc = 500 ms (narrow QRS threshold)")

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Reference QTc (Bazett, 12SL RR) [ms]", fontsize=13)
    ax.set_ylabel("Predicted QTc [ms]", fontsize=13)
    ax.set_title(
        f"Combined Pearson Correlation: Predicted QTc vs Reference QTc\n"
        f"Split = {split}  (PTB-XL, waveform-only model, 500 Hz)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="lower right", framealpha=0.93,
              handlelength=2.5)
    ax.grid(True, alpha=0.22, zorder=0)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Combined Pearson → {output_path}")


# ---------------------------------------------------------------------------
# 7. Combined ROC curves  — all configs overlaid
# ---------------------------------------------------------------------------

def plot_roc_curves(all_data: dict, output_path: Path,
                    split: str = "val") -> None:
    """
    Overlaid ROC curves for every config using the narrow/wide QRS
    prolongation labels (Narrow QRS → QTc>500ms, Wide QRS → QTc>550ms).
    """
    from sklearn.metrics import roc_curve as _roc_curve

    fig, ax = plt.subplots(figsize=(8, 7))
    any_plotted = False

    for name, d in all_data.items():
        v       = d["valid"]
        y_true  = (d["ref_qtc"][v] > d["threshold"][v]).astype(int)
        y_score = d["pred_qtc"][v]
        n_pos   = int(y_true.sum())

        if n_pos < 5:
            print(f"  ROC [{name}]: only {n_pos} positives — skipping")
            continue

        auc           = roc_auc_score(y_true, y_score)
        fpr, tpr, _   = _roc_curve(y_true, y_score)
        ax.plot(fpr, tpr,
                color=PALETTE.get(name, "#555"), lw=2.5, ls=LS.get(name, "-"),
                label=f"{name}   AUC = {auc:.3f}  (n+ = {n_pos})")
        any_plotted = True

    ax.plot([0, 1], [0, 1], "--", color="#aaa", lw=1.2,
            label="No-skill baseline (AUC = 0.50)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=13)
    ax.set_ylabel("True Positive Rate (Sensitivity)",      fontsize=13)
    ax.set_title(
        "ROC Curves — QTc Prolongation\n"
        "Narrow QRS (≤100 ms) → QTc > 500 ms  |  Wide QRS (>100 ms) → QTc > 550 ms\n"
        f"Split = {split}",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9.5, loc="lower right", framealpha=0.92)
    ax.grid(True, alpha=0.28)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  ROC curves    → {output_path}")


# ---------------------------------------------------------------------------
# 8. Combined PRC curves  — narrow/wide threshold, all configs overlaid
# ---------------------------------------------------------------------------

def plot_prc_curves(all_data: dict, output_path: Path,
                    split: str = "val") -> None:
    """
    Overlaid Precision-Recall curves using the narrow/wide QRS prolongation
    rule (same labels as the classification report).  Shows AUPRC and n+.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    baseline_prev: float | None = None
    any_plotted = False

    for name, d in all_data.items():
        v       = d["valid"]
        y_true  = (d["ref_qtc"][v] > d["threshold"][v]).astype(int)
        y_score = d["pred_qtc"][v]
        n_pos   = int(y_true.sum())
        n       = int(v.sum())

        if n_pos < 5:
            print(f"  PRC [{name}]: only {n_pos} positives — skipping")
            continue

        prev  = n_pos / n
        if baseline_prev is None:
            baseline_prev = prev

        auprc        = average_precision_score(y_true, y_score)
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        ax.plot(rec, prec,
                color=PALETTE.get(name, "#555"), lw=2.5, ls=LS.get(name, "-"),
                label=f"{name}   AUPRC = {auprc:.3f}  (n+ = {n_pos}, {prev:.1%})")
        any_plotted = True

    if baseline_prev is not None:
        ax.axhline(baseline_prev, color="#aaa", lw=1.3, ls="--",
                   label=f"No-skill baseline (prevalence = {baseline_prev:.1%})")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall",    fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title(
        "Precision-Recall Curves — QTc Prolongation\n"
        "Narrow QRS (≤100 ms) → QTc > 500 ms  |  Wide QRS (>100 ms) → QTc > 550 ms\n"
        f"Split = {split}",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper right", framealpha=0.92)
    ax.grid(True, alpha=0.28)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  PRC curves    → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="QTc diagnostic analysis (local, no Modal)")
    parser.add_argument("--checkpoint",   default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--ptbxl-dir",    default=str(DEFAULT_SIGNALS_DIR))
    parser.add_argument("--features-dir", default=str(DEFAULT_FEATURES_DIR))
    parser.add_argument("--labels-csv",   default=str(DEFAULT_LABELS))
    parser.add_argument("--output-dir",   default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--split",        default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch-size",   type=int, default=32)
    parser.add_argument("--n-examples",   type=int, default=15,
                        help="Random rows to show in sample table")
    parser.add_argument("--prc-450",      action="store_true", default=False,
                        help="One-off: plot PRC at universal QTc>450ms threshold "
                             "(higher prevalence, more interpretable AUPRC). Off by default.")
    parser.add_argument("--adapters",     action="store_true", default=False,
                        help="Train any adapter whose checkpoint is missing, then include "
                             "all three in every plot.  If all checkpoints already exist "
                             "they are loaded automatically even without this flag — "
                             "use --adapter-force-retrain to force a full retrain.")
    # Adapter/latent cache dirs
    parser.add_argument("--adapter-dir",        default=None,
                        help="Dir for adapter checkpoints "
                             "(default: qt_finetune/checkpoints/adapters/).")
    parser.add_argument("--adapter-latent-dir", default=None,
                        help="Dir for cached latent .npz files "
                             "(default: qt_finetune/checkpoints/latents/). "
                             "Shared with train_adapters.py --latent-dir.")
    parser.add_argument("--adapter-force-retrain",   action="store_true", default=False,
                        help="Re-train adapters even if checkpoints already exist.")
    parser.add_argument("--force-reextract",         action="store_true", default=False,
                        help="Re-extract encoder latents even if .npz caches exist "
                             "(use after updating the model checkpoint).")
    # Linear
    parser.add_argument("--adapter-alpha-linear", type=float, default=1.0,
                        help="Ridge alpha for the linear adapter.")
    # KRR
    parser.add_argument("--adapter-alpha-krr",    type=float, default=0.1,
                        help="KernelRidge alpha for the KRR adapter.")
    parser.add_argument("--adapter-nsub",         type=int,   default=3000,
                        help="KRR training subsample size.")
    # GReLU — defaults come from GRELU_DEFAULT_CONFIG so they stay in sync
    parser.add_argument("--grelu-m",          type=int,
                        default=GRELU_DEFAULT_CONFIG["m"],
                        help="Number of GReLU neurons.")
    parser.add_argument("--grelu-epochs",     type=int,
                        default=GRELU_DEFAULT_CONFIG["epochs"],
                        help="R-FISTA iterations (default: %(default)s).")
    parser.add_argument("--grelu-eval-every", type=int,
                        default=GRELU_DEFAULT_CONFIG["eval_every"],
                        help="Print progress every N iterations (default: %(default)s).")
    parser.add_argument("--grelu-lr",         type=float,
                        default=GRELU_DEFAULT_CONFIG["lr"],
                        help="Initial R-FISTA step size.")
    parser.add_argument("--grelu-lambda-l2",  type=float,
                        default=GRELU_DEFAULT_CONFIG["lambda_l2"],
                        help="L2 regularisation for GReLU.")
    parser.add_argument("--grelu-lambda-l1",  type=float,
                        default=GRELU_DEFAULT_CONFIG["lambda_l1"],
                        help="L1 penalty on feature weights (0 = off).")
    args = parser.parse_args()

    # ── Checkpoint guard ──────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found at {ckpt_path}")
        print("Pull it from Modal first:")
        print("  modal volume get ptbxl-data /best_qt_model.pt "
              "qt_finetune/checkpoints/best_qt_model.pt")
        sys.exit(1)

    # Quick sanity check on the checkpoint quality
    _ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _vmae = _ckpt.get("val_mae", float("nan"))
    _vr   = _ckpt.get("val_r",   float("nan"))
    _ep   = _ckpt.get("epoch",   "?")
    print(f"\nCheckpoint: epoch={_ep}  val_MAE={_vmae:.2f} ms  val_R={_vr:.3f}")
    if _vr < 0.3:
        print("⚠  WARNING: val_R < 0.3 — this looks like an undertrained checkpoint.")
        print("   If the Modal run is still going, wait for it to finish and re-pull.\n")

    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"Device : {device}")

    model = load_model(ckpt_path, device)

    # ── Datasets (validation split by default) ────────────────────────────
    ds_kwargs = dict(
        ptbxl_signals_dir=args.ptbxl_dir,
        labels_csv=args.labels_csv,
        split=args.split,
    )
    configs = [
        ("12-lead",      ForcedLeadDataset(lead_indices=list(range(12)), **ds_kwargs)),
        ("6 limb leads", ForcedLeadDataset(lead_indices=LIMB_LEAD_INDICES, **ds_kwargs)),
        ("Lead I",       ForcedLeadDataset(lead_indices=[0], **ds_kwargs)),
        ("ICM (V3−V2)",  ICMLeadDataset(**ds_kwargs)),
    ]
    print(f"\nSplit: {args.split}  ({len(configs[0][1])} samples per config)\n")

    # ── Metadata ──────────────────────────────────────────────────────────
    meta = load_metadata(args.labels_csv, args.features_dir, args.split)

    # ── Inference + QTc ───────────────────────────────────────────────────
    print("── Inference ─────────────────────────────────────────────────")
    all_data = {}
    for name, ds in configs:
        print(f"  {name} ...", end=" ", flush=True)
        pred_qt, tgt_qt, ecg_ids = run_inference(model, ds, device, args.batch_size)
        d = add_qtc(pred_qt, tgt_qt, ecg_ids, meta)
        err = pred_qt - tgt_qt
        print(f"MAE={np.abs(err).mean():.2f} ms  R={np.corrcoef(pred_qt,tgt_qt)[0,1]:.3f}")
        all_data[name] = d

    # ── Latent adapters ───────────────────────────────────────────────────
    # Auto-detect: if any adapter checkpoint already exists on disk, load and
    # include it automatically — no flag needed.
    # Pass --adapters to (re-)train any missing adapter from scratch.
    # Pass --adapter-force-retrain to force retraining all three.
    import os as _os

    _ckpt_dir   = (Path(args.adapter_dir)        if args.adapter_dir
                   else _SCRIPT_DIR / "checkpoints" / "adapters")
    _latent_dir = (Path(args.adapter_latent_dir) if args.adapter_latent_dir
                   else _SCRIPT_DIR / "checkpoints" / "latents")
    _linear_ckpt = _ckpt_dir / "linear_adapter.pkl"
    _krr_ckpt    = _ckpt_dir / "krr_adapter.pkl"
    _grelu_ckpt  = _ckpt_dir / "grelu_adapter.pt"
    _force_ada   = args.adapter_force_retrain
    _force_lat   = args.force_reextract

    # Which adapters exist on disk right now?
    _cached = {
        "Lead I + Linear": _linear_ckpt.exists() and not _force_ada,
        "Lead I + KRR":    _krr_ckpt.exists()    and not _force_ada,
        "Lead I + GReLU":  _grelu_ckpt.exists()  and not _force_ada,
    }
    _any_cached   = any(_cached.values())
    _need_to_train = args.adapters  # only train when explicitly requested

    # Run the adapter block when: at least one checkpoint exists OR --adapters passed
    if _any_cached or _need_to_train:
        _ckpt_dir.mkdir(parents=True, exist_ok=True)
        _latent_dir.mkdir(parents=True, exist_ok=True)
        _ckpt_mtime = _os.path.getmtime(ckpt_path)

        _cached_label  = ", ".join(k.replace("Lead I + ", "") for k, v in _cached.items() if v)
        _missing_label = ", ".join(k.replace("Lead I + ", "") for k, v in _cached.items() if not v)
        print(f"\n── Latent Adapters ───────────────────────────────────────────")
        if _cached_label:
            print(f"  cached : {_cached_label}")
        if _missing_label and _need_to_train:
            print(f"  to train: {_missing_label}")
        elif _missing_label:
            print(f"  skipping (no checkpoint): {_missing_label}  → pass --adapters to train")

        # Training latents: only needed if we will train at least one adapter
        _will_train_any = _need_to_train and not all(_cached.values())
        z12_tr = zi_tr = None
        if _will_train_any:
            def _ds_train(leads):
                return ForcedLeadDataset(
                    lead_indices=leads,
                    ptbxl_signals_dir=args.ptbxl_dir,
                    labels_csv=args.labels_csv,
                    split="train",
                )
            z12_tr, _, _, _ = get_latents(
                model, _ds_train(list(range(12))), device, args.batch_size,
                cache_path=_latent_dir / "train_12lead.npz",
                ckpt_mtime=_ckpt_mtime, force=_force_lat,
                desc="12-lead train",
            )
            zi_tr, _, _, _ = get_latents(
                model, _ds_train([0]), device, args.batch_size,
                cache_path=_latent_dir / "train_leadI.npz",
                ckpt_mtime=_ckpt_mtime, force=_force_lat,
                desc="Lead I train",
            )

        # Val/test latents for adapter inference (always needed here)
        zi_val, _, tgt_qt_val, ecg_ids_val = get_latents(
            model,
            ForcedLeadDataset(lead_indices=[0], **ds_kwargs),
            device, args.batch_size,
            cache_path=_latent_dir / f"{args.split}_leadI.npz",
            ckpt_mtime=_ckpt_mtime, force=_force_lat,
            desc=f"Lead I {args.split}",
        )

        # ── Load or train each adapter ────────────────────────────────────
        def _resolve_linear():
            if _cached["Lead I + Linear"]:
                return load_linear(_linear_ckpt)
            if _need_to_train:
                print(f"  Linear: fitting  alpha={args.adapter_alpha_linear}")
                a = LinearResidualAdapter(alpha=args.adapter_alpha_linear).fit(zi_tr, z12_tr)
                save_linear(a, _linear_ckpt)
                return a
            return None

        def _resolve_krr():
            if _cached["Lead I + KRR"]:
                return load_krr(_krr_ckpt)
            if _need_to_train:
                print(f"  KRR: fitting  alpha={args.adapter_alpha_krr}  n_sub={args.adapter_nsub}")
                a = KernelResidualAdapter(
                    alpha=args.adapter_alpha_krr, n_subsample=args.adapter_nsub,
                ).fit(zi_tr, z12_tr)
                save_krr(a, _krr_ckpt)
                return a
            return None

        def _resolve_grelu():
            if _cached["Lead I + GReLU"]:
                return load_grelu(_grelu_ckpt)
            if _need_to_train:
                print(
                    f"  GReLU: training  m={args.grelu_m}  epochs={args.grelu_epochs}  "
                    f"eval_every={args.grelu_eval_every}  lr={args.grelu_lr:.0e}  "
                    f"λ_l2={args.grelu_lambda_l2:.0e}  λ_l1={args.grelu_lambda_l1:.0e}"
                )
                a = GReLUResidualAdapter(
                    m=args.grelu_m, lr=args.grelu_lr, epochs=args.grelu_epochs,
                    eval_every=args.grelu_eval_every,
                    lambda_l2=args.grelu_lambda_l2, lambda_l1=args.grelu_lambda_l1,
                    device=str(device),
                ).fit(zi_tr, z12_tr, verbose=True)
                save_grelu(a, _grelu_ckpt)
                return a
            return None

        # ── Apply each resolved adapter ───────────────────────────────────
        for aname, resolver in [
            ("Lead I + Linear", _resolve_linear),
            ("Lead I + KRR",    _resolve_krr),
            ("Lead I + GReLU",  _resolve_grelu),
        ]:
            adapter = resolver()
            if adapter is None:
                continue   # checkpoint missing and --adapters not passed → skip
            z_adapted   = adapter.transform(zi_val)
            pred_qt_ada = predict_from_latents(model, z_adapted, device)
            d_ada       = add_qtc(pred_qt_ada, tgt_qt_val, ecg_ids_val, meta)
            err         = pred_qt_ada - tgt_qt_val
            r           = float(np.corrcoef(pred_qt_ada, tgt_qt_val)[0, 1])
            print(f"  {aname}: MAE={np.abs(err).mean():.2f} ms  R={r:.3f}")
            all_data[aname] = d_ada

    # ── Sample calculation table (12-lead only, clearest diagnostic) ──────
    print("\n── Sample Calculation Table (12-lead) ───────────────────────")
    print_sample_table(all_data["12-lead"], n_random=args.n_examples)

    # ── Classification report ─────────────────────────────────────────────
    print_classification_report(all_data)

    # ── Plots ─────────────────────────────────────────────────────────────
    out = Path(args.output_dir)
    sp  = args.split
    print("── Generating plots ──────────────────────────────────────────")

    # Per-config diagnostic grids (unchanged from before)
    plot_bland_altman(all_data, out / f"qtc_bland_altman_{sp}.png")
    plot_pearson_qtc( all_data, out / f"qtc_pearson_grid_{sp}.png")

    # ── Combined / overlaid plots (new) ──────────────────────────────────
    plot_pearson_combined(
        all_data,
        out / f"qtc_pearson_combined_{sp}.png",
        split=sp,
    )
    plot_roc_curves(
        all_data,
        out / f"qtc_roc_{sp}.png",
        split=sp,
    )
    plot_prc_curves(
        all_data,
        out / f"qtc_prc_{sp}.png",
        split=sp,
    )

    # ── 450 ms ablation PRC (opt-in only) ────────────────────────────────
    if args.prc_450:
        print("\n── PRC ablation: QTc > 450 ms (universal threshold) ──────────")
        plot_prc_single_threshold(
            all_data,
            threshold_ms=450.0,
            output_path=out / f"qtc_prc_450ms_{sp}.png",
        )

    print(f"\nOutputs ({out}/):")
    for f in sorted(out.glob(f"qtc_*_{sp}.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
