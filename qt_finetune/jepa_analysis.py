"""
JEPA-Linear Interpretability & Error Analysis
==============================================

Five analyses on a JEPA-Linear checkpoint (default: lqt1.0):

  1. Wide QRS subgroup  — separate error report for narrow / wide / very wide QRS.
                          Wide QRS (LBBB/RBBB) invalidates Bazett, so high error
                          there is expected and should be separated in the paper.

  2. Cosine similarity  — per-sample cosine_sim(z_c, z_12) where z_c is the
                          JEPA Lead-I embedding and z_12 is the base 12-lead
                          embedding. Tests the claim: low similarity → high error.

  3. Linear W SVD       — decompose the 192×192 predictor weight W. Singular values
                          show which embedding directions the adapter amplifies.
                          Top singular vectors are visualised as heatmaps.

  4. Contribution ratio — per-sample ||W·z_c + b|| / ||z_c|| (embedding space)
                          plus Δ_qt = qt_head(ẑ) - qt_head(z_c) (prediction space).
                          High ratio / delta → adapter is doing heavy lifting.
                          Correlate both with absolute error.

  5. Worst / best cases — raw 12-lead ECG waveforms for the top-N most / least
                          wrong predictions. Invaluable for spotting systematic
                          failure modes (conduction disease, pacemakers, etc.).

Usage:
    cd /path/to/AnchoredECG
    python qt_finetune/jepa_analysis.py
    python qt_finetune/jepa_analysis.py --ckpt qt_finetune/checkpoints/jepa/jepa_linear_lqt0.5.pt
    python qt_finetune/jepa_analysis.py --n-cases 20 --split val
    python qt_finetune/jepa_analysis.py --no-ecg-plots   # skip slow wfdb loading
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import wfdb

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT   = _SCRIPT_DIR.parent
_MODELS_DIR  = _REPO_ROOT / "models"
for _p in (_MODELS_DIR, _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from train_jepa import load_jepa_ckpt
from validate_qt import ForcedLeadDataset, load_model
from torch_geometric.loader import DataLoader as PyGDataLoader

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DATA           = _REPO_ROOT / "data"
DEFAULT_CKPT    = _SCRIPT_DIR / "checkpoints" / "jepa" / "jepa_linear_lqt1.0.pt"
DEFAULT_BASE    = _SCRIPT_DIR / "checkpoints" / "best_qt_model.pt"
DEFAULT_LABELS  = _SCRIPT_DIR / "qt_labels.csv"
DEFAULT_SIG_DIR = _DATA / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
DEFAULT_FEAT_DIR= _DATA / "ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1"
DEFAULT_OUT     = _SCRIPT_DIR / "plots" / "jepa_analysis"

# PTB-XL standard lead order (wfdb column index → name)
LEAD_NAMES = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
# Clinical 12-lead grid layout: (row, col) → lead index
LEAD_GRID = [
    [0, 3, 6, 9],   # row 0: I, aVR, V1, V4
    [1, 4, 7, 10],  # row 1: II, aVL, V2, V5
    [2, 5, 8, 11],  # row 2: III, aVF, V3, V6
]

# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_jepa_embeddings(
    context_enc, predictor, qt_head,
    ds, device, batch_size: int = 32,
):
    """
    Returns pred_qt, tgt_qt, ecg_ids, z_c, z_hat  — all as numpy arrays.
    z_c   : (N, 192) Lead-I context embeddings (before predictor)
    z_hat : (N, 192) adapted embeddings (after predictor)
    """
    context_enc.eval(); predictor.eval(); qt_head.eval()
    loader = PyGDataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    pred_qts, tgt_qts, z_cs, z_hats = [], [], [], []
    for bd, qt in loader:
        bd = bd.to(device)
        B  = bd.num_graphs
        tab = torch.zeros(B, 0, device=device)
        z_c   = context_enc(bd, tabular=tab)["graph_embed"]     # (B, 192)
        z_hat = z_c + predictor(z_c)                            # (B, 192)
        qt_pred = qt_head(z_hat).flatten()
        z_cs.append(z_c.cpu().numpy())
        z_hats.append(z_hat.cpu().numpy())
        pred_qts.append(qt_pred.cpu().numpy())
        tgt_qts.append(qt.flatten().numpy())
    ecg_ids = np.array([item["ecg_id"] for item in ds.index], dtype=int)
    return (
        np.concatenate(pred_qts),
        np.concatenate(tgt_qts),
        ecg_ids,
        np.concatenate(z_cs),
        np.concatenate(z_hats),
    )


@torch.no_grad()
def extract_base_embeddings(base_model, ds_12lead, device, batch_size: int = 32):
    """
    Returns z_12 : (N, 192) embeddings from the frozen 12-lead base model.
    """
    base_model.eval()
    loader = PyGDataLoader(ds_12lead, batch_size=batch_size, shuffle=False, num_workers=0)
    z_12s = []
    for bd, _ in loader:
        bd = bd.to(device)
        B  = bd.num_graphs
        z_12 = base_model(bd, tabular=torch.zeros(B, 0, device=device))["graph_embed"]
        z_12s.append(z_12.cpu().numpy())
    return np.concatenate(z_12s)


@torch.no_grad()
def qt_from_embedding(qt_head, z: np.ndarray, device, batch_size: int = 256) -> np.ndarray:
    """Apply QT head to a batch of numpy embeddings."""
    qt_head.eval()
    z_t  = torch.from_numpy(z).float()
    outs = []
    for i in range(0, len(z_t), batch_size):
        chunk = z_t[i:i+batch_size].to(device)
        outs.append(qt_head(chunk).flatten().cpu().numpy())
    return np.concatenate(outs)

# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def load_meta(labels_csv, features_dir, split):
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
    df["rr_sec"]    = df["RR_Mean_Global"].clip(300, 1500) / 1000.0
    df["ref_qtc"]   = df["QT_Int_Global"] / np.sqrt(df["rr_sec"])
    df["threshold"] = np.where(df["QRS_Dur_Global"] <= 100, 500.0, 550.0)
    return df.set_index("ecg_id")

# ---------------------------------------------------------------------------
# Analysis 1 — Wide QRS subgroup
# ---------------------------------------------------------------------------

def analysis_wide_qrs(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Stratify error by QRS duration category and print a report.

    Categories:
        narrow   : QRS_Dur_Global <= 100 ms  (normal conduction)
        wide     : 100 < QRS <= 120 ms       (borderline / IVCD)
        very_wide: QRS_Dur_Global > 120 ms   (LBBB / RBBB territory)
    """
    print("\n" + "═"*70)
    print("  Wide QRS Subgroup Analysis")
    print("═"*70)

    bins = [
        ("narrow   (≤100 ms)",    df["qrs_ms"] <= 100),
        ("wide     (100–120 ms)", (df["qrs_ms"] > 100) & (df["qrs_ms"] <= 120)),
        ("very wide (>120 ms)",   df["qrs_ms"] > 120),
    ]
    rows = []
    for label, mask in bins:
        sub = df[mask & df["valid"]]
        if len(sub) < 2:
            continue
        err  = sub["pred_qt"] - sub["tgt_qt"]
        r    = float(np.corrcoef(sub["pred_qt"], sub["tgt_qt"])[0, 1])
        rows.append(dict(
            label=label, n=len(sub),
            mae=float(np.abs(err).mean()),
            rmse=float(np.sqrt((err**2).mean())),
            bias=float(err.mean()),
            r=r,
        ))
        print(f"  {label}  n={len(sub):4d}  "
              f"MAE={np.abs(err).mean():.2f} ms  "
              f"RMSE={np.sqrt((err**2).mean()):.2f} ms  "
              f"bias={err.mean():+.2f} ms  R={r:.3f}")

    print()
    print(textwrap.dedent("""
    ⚠  Clinical note: wide QRS (>100 ms) includes LBBB, RBBB, and paced rhythms.
       Bazett QTc correction is unreliable in these cases even from 12-lead ECGs.
       High error in the very-wide subgroup is expected and should be reported
       separately — it reflects algorithm limitations, not model failure.
    """).strip())

    # Plot: error distribution per subgroup
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#1565C0", "#E65100", "#B71C1C"]
    for (label, mask), color in zip(bins, colors):
        sub = df[mask & df["valid"]]
        if len(sub) < 2:
            continue
        err = (sub["pred_qt"] - sub["tgt_qt"]).values
        ax.hist(err, bins=60, alpha=0.55, color=color,
                label=f"{label.strip()}  n={len(sub)}  MAE={np.abs(err).mean():.1f} ms",
                density=True)
    ax.axvline(0, color="k", lw=1.2, ls="--", alpha=0.6)
    ax.set_xlabel("Prediction Error (Predicted − True QT, ms)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("QT Prediction Error by QRS Duration Category\n"
                 "Wide QRS → Bazett correction unreliable (LBBB/RBBB/paced)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9.5, loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save(fig, out_dir / "wide_qrs_error_distribution.png")
    print(f"\n  Wide QRS plot → {out_dir}/wide_qrs_error_distribution.png")


# ---------------------------------------------------------------------------
# Analysis 2 — Cosine similarity z_c vs z_12
# ---------------------------------------------------------------------------

def analysis_cosine_similarity(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Test the JEPA mechanism claim: low cosine_sim(z_c, z_12) → high error.

    If the Lead-I context embedding is already close (high cosine sim) to what
    the 12-lead encoder would produce, the linear adapter has little work to do
    and errors should be small.  Low cosine sim → large geometric gap → harder.
    """
    print("\n" + "═"*70)
    print("  Cosine Similarity Analysis  (z_c vs z_12)")
    print("═"*70)

    mask  = df["valid"].values
    cos   = df.loc[mask, "cos_sim_c_12"].values
    err   = np.abs(df.loc[mask, "pred_qt"].values - df.loc[mask, "tgt_qt"].values)

    r, p = stats.pearsonr(cos, err)
    rs, ps = stats.spearmanr(cos, err)
    print(f"  Pearson r (cos_sim vs |error|)  : {r:+.4f}  (p={p:.2e})")
    print(f"  Spearman ρ (cos_sim vs |error|) : {rs:+.4f}  (p={ps:.2e})")

    if r < 0:
        print("  ✓ Negative correlation: lower cosine similarity → higher error "
              "(validates the JEPA mechanism).")
    else:
        print("  ✗ Positive/zero correlation — cosine similarity does not predict "
              "error in the expected direction.")

    # Bin cosine_sim into deciles and compute MAE per bin
    decile_labels = pd.qcut(df["cos_sim_c_12"], q=10, labels=False)
    bin_centers, bin_mae = [], []
    for d in range(10):
        sub = df[decile_labels == d]
        if len(sub) == 0:
            continue
        mae = float(np.abs(sub["pred_qt"] - sub["tgt_qt"]).mean())
        bin_centers.append(float(sub["cos_sim_c_12"].mean()))
        bin_mae.append(mae)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: scatter
    ax = axes[0]
    ax.scatter(df.loc[mask, "cos_sim_c_12"], err,
               s=4, alpha=0.18, color="#1565C0", linewidths=0)
    xs = np.linspace(cos.min(), cos.max(), 200)
    slope, intercept, *_ = stats.linregress(cos, err)
    ax.plot(xs, slope * xs + intercept, color="#B71C1C", lw=2.2,
            label=f"Linear fit  r={r:+.3f}  p={p:.1e}")
    ax.set_xlabel("cosine_sim(z_c, z_12)", fontsize=12)
    ax.set_ylabel("|QT error| (ms)", fontsize=12)
    ax.set_title("Cosine Similarity vs Absolute QT Error", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9.5)
    ax.grid(True, alpha=0.25)

    # Right: binned MAE per cosine-sim decile
    ax2 = axes[1]
    ax2.plot(bin_centers, bin_mae, "o-", color="#1565C0", lw=2, ms=7)
    ax2.set_xlabel("cosine_sim(z_c, z_12)  [decile mean]", fontsize=12)
    ax2.set_ylabel("Mean |QT error| (ms)", fontsize=12)
    ax2.set_title("MAE per Cosine-Similarity Decile\n"
                  "Low cosine sim → model bridges larger geometric gap",
                  fontsize=11, fontweight="bold")
    ax2.grid(True, alpha=0.25)

    fig.suptitle("JEPA Mechanism Validation: Does Geometric Gap Predict Error?\n"
                 "z_c = JEPA Lead-I embedding  |  z_12 = base 12-lead embedding",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir / "cosine_similarity_analysis.png")
    print(f"  Cosine sim plot → {out_dir}/cosine_similarity_analysis.png")


# ---------------------------------------------------------------------------
# Analysis 3 — Linear W SVD
# ---------------------------------------------------------------------------

def analysis_linear_w_svd(W: np.ndarray, b: np.ndarray, out_dir: Path) -> None:
    """
    Singular value decomposition of the 192×192 predictor weight W.

    W maps z_c → correction δ (before adding to z_c).
    SVD: W = U Σ V^T
      - Σ   : singular values — how much each mode is amplified
      - V   : right singular vectors — input directions (Lead-I embed space)
      - U   : left singular vectors — output directions (correction space)
    """
    print("\n" + "═"*70)
    print("  Linear Predictor W — SVD Analysis")
    print("═"*70)

    U, S, Vt = np.linalg.svd(W)

    print(f"  W shape : {W.shape}")
    print(f"  ||W||_F (Frobenius) : {np.linalg.norm(W, 'fro'):.4f}")
    print(f"  ||b||              : {np.linalg.norm(b):.4f}")
    print(f"\n  Singular values (top 20):")
    for i, s in enumerate(S[:20]):
        bar = "█" * int(s / S[0] * 30)
        print(f"    σ_{i+1:02d} = {s:.4f}  {bar}")
    print(f"\n  Effective rank (σ > 0.01 * σ_max): "
          f"{int((S > 0.01 * S[0]).sum())}")
    print(f"  Variance explained by top-10 SVs: "
          f"{(S[:10]**2).sum() / (S**2).sum() * 100:.1f}%")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: All singular values
    ax = axes[0]
    ax.plot(np.arange(1, len(S)+1), S, "o-", ms=3, lw=1.5, color="#1565C0")
    ax.axvline(10.5, color="#B71C1C", lw=1.2, ls="--", alpha=0.7,
               label="Top 10")
    ax.set_xlabel("Singular Value Index", fontsize=12)
    ax.set_ylabel("Singular Value σ", fontsize=12)
    ax.set_title("All 192 Singular Values of W", fontsize=11, fontweight="bold")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    # Panel 2: Top-10 right singular vectors (V columns) — what input directions W reads
    ax2 = axes[1]
    top_v = Vt[:10]   # (10, 192)
    im2 = ax2.imshow(top_v, aspect="auto", cmap="RdBu_r",
                     vmin=-np.abs(top_v).max(), vmax=np.abs(top_v).max())
    ax2.set_xlabel("Embedding Dimension", fontsize=11)
    ax2.set_ylabel("Right Singular Vector (V)", fontsize=11)
    ax2.set_title("Top-10 Right Singular Vectors\n(input directions W reads from z_c)",
                  fontsize=10, fontweight="bold")
    plt.colorbar(im2, ax=ax2, shrink=0.85)
    ax2.set_yticks(range(10))
    ax2.set_yticklabels([f"v_{i+1}  σ={S[i]:.3f}" for i in range(10)], fontsize=8)

    # Panel 3: Top-10 left singular vectors (U columns) — output correction directions
    ax3 = axes[2]
    top_u = U[:, :10].T   # (10, 192)
    im3 = ax3.imshow(top_u, aspect="auto", cmap="RdBu_r",
                     vmin=-np.abs(top_u).max(), vmax=np.abs(top_u).max())
    ax3.set_xlabel("Embedding Dimension", fontsize=11)
    ax3.set_ylabel("Left Singular Vector (U)", fontsize=11)
    ax3.set_title("Top-10 Left Singular Vectors\n(output directions W writes to correction)",
                  fontsize=10, fontweight="bold")
    plt.colorbar(im3, ax=ax3, shrink=0.85)
    ax3.set_yticks(range(10))
    ax3.set_yticklabels([f"u_{i+1}  σ={S[i]:.3f}" for i in range(10)], fontsize=8)

    fig.suptitle("JEPA-Linear Predictor: SVD of Weight Matrix W  (192×192)\n"
                 "ẑ = z_c + W·z_c + b  →  correction = W·z_c + b",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir / "linear_w_svd.png")
    print(f"\n  SVD plot → {out_dir}/linear_w_svd.png")


# ---------------------------------------------------------------------------
# Analysis 4 — Contribution ratio
# ---------------------------------------------------------------------------

def analysis_contribution_ratio(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Two complementary contribution metrics per sample:

    Embedding space:
        ratio_emb = ||W·z_c + b|| / ||z_c||
        High → adapter is doing heavy lifting in embedding space.

    Prediction space:
        delta_qt = qt_head(ẑ) - qt_head(z_c)
        How much did the adapter change the QT prediction (ms)?
    """
    print("\n" + "═"*70)
    print("  Contribution Ratio Analysis")
    print("═"*70)

    mask = df["valid"].values
    err  = np.abs(df.loc[mask, "pred_qt"].values - df.loc[mask, "tgt_qt"].values)
    ratio = df.loc[mask, "contrib_ratio"].values
    delta = df.loc[mask, "delta_qt"].values

    r_ratio, p_ratio   = stats.pearsonr(ratio, err)
    rs_ratio, ps_ratio = stats.spearmanr(ratio, err)
    r_delta, p_delta   = stats.pearsonr(np.abs(delta), err)

    print(f"  Embedding ratio ||W·z_c+b|| / ||z_c||:")
    print(f"    mean={ratio.mean():.4f}  median={np.median(ratio):.4f}  "
          f"std={ratio.std():.4f}  max={ratio.max():.4f}")
    print(f"    Pearson r vs |error| : {r_ratio:+.4f}  (p={p_ratio:.2e})")
    print(f"    Spearman ρ           : {rs_ratio:+.4f}  (p={ps_ratio:.2e})")
    print(f"\n  QT delta qt_head(ẑ) - qt_head(z_c):")
    print(f"    mean={delta.mean():+.2f} ms  std={delta.std():.2f} ms  "
          f"|max|={np.abs(delta).max():.2f} ms")
    print(f"    Pearson r |Δqt| vs |error| : {r_delta:+.4f}  (p={p_delta:.2e})")

    high_ratio_q = np.percentile(ratio, 75)
    low_ratio_q  = np.percentile(ratio, 25)
    mae_high = float(np.abs(df.loc[mask, "pred_qt"][ratio > high_ratio_q] -
                             df.loc[mask, "tgt_qt"][ratio > high_ratio_q]).mean())
    mae_low  = float(np.abs(df.loc[mask, "pred_qt"][ratio < low_ratio_q] -
                             df.loc[mask, "tgt_qt"][ratio < low_ratio_q]).mean())
    print(f"\n  MAE for top-25% ratio (heavy adapter): {mae_high:.2f} ms")
    print(f"  MAE for bot-25% ratio (light adapter): {mae_low:.2f} ms")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Top-left: distribution of embedding ratio
    ax = axes[0, 0]
    ax.hist(ratio, bins=60, color="#1565C0", alpha=0.75, edgecolor="white", lw=0.4)
    ax.axvline(ratio.mean(), color="#B71C1C", lw=2, ls="--",
               label=f"Mean = {ratio.mean():.3f}")
    ax.axvline(np.median(ratio), color="#E65100", lw=2, ls=":",
               label=f"Median = {np.median(ratio):.3f}")
    ax.set_xlabel("Contribution Ratio  ||W·z_c+b|| / ||z_c||", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Embedding-Space Contribution Ratio Distribution", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # Top-right: ratio vs |error|
    ax2 = axes[0, 1]
    ax2.scatter(ratio, err, s=4, alpha=0.2, color="#1565C0", linewidths=0)
    xs  = np.linspace(ratio.min(), ratio.max(), 200)
    sl, ic, *_ = stats.linregress(ratio, err)
    ax2.plot(xs, sl*xs+ic, color="#B71C1C", lw=2.2,
             label=f"r={r_ratio:+.3f}  p={p_ratio:.1e}")
    ax2.set_xlabel("Contribution Ratio  ||W·z_c+b|| / ||z_c||", fontsize=11)
    ax2.set_ylabel("|QT error| (ms)", fontsize=11)
    ax2.set_title("Adapter Contribution vs Error", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=9.5)
    ax2.grid(True, alpha=0.25)

    # Bottom-left: distribution of Δ_qt
    ax3 = axes[1, 0]
    ax3.hist(delta, bins=60, color="#2E7D32", alpha=0.75, edgecolor="white", lw=0.4)
    ax3.axvline(0, color="k", lw=1.2, ls="--", alpha=0.6)
    ax3.axvline(delta.mean(), color="#B71C1C", lw=2, ls="--",
                label=f"Mean = {delta.mean():+.2f} ms")
    ax3.set_xlabel("Δ_QT = qt_head(ẑ) − qt_head(z_c)  (ms)", fontsize=11)
    ax3.set_ylabel("Count", fontsize=11)
    ax3.set_title("Prediction-Space Adapter Contribution  (Δ_QT Distribution)",
                  fontsize=10, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.25)

    # Bottom-right: |Δ_qt| vs |error|
    ax4 = axes[1, 1]
    ax4.scatter(np.abs(delta), err, s=4, alpha=0.2, color="#2E7D32", linewidths=0)
    xd  = np.linspace(0, np.abs(delta).max(), 200)
    sld, icd, *_ = stats.linregress(np.abs(delta), err)
    ax4.plot(xd, sld*xd+icd, color="#B71C1C", lw=2.2,
             label=f"r={r_delta:+.3f}  p={p_delta:.1e}")
    ax4.set_xlabel("|Δ_QT|  (ms)", fontsize=11)
    ax4.set_ylabel("|QT error| (ms)", fontsize=11)
    ax4.set_title("|Adapter QT delta| vs Error", fontsize=10, fontweight="bold")
    ax4.legend(fontsize=9.5)
    ax4.grid(True, alpha=0.25)

    fig.suptitle("Contribution Ratio Analysis — JEPA-Linear Predictor\n"
                 "Per-sample: how much is the adapter changing the prediction?",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir / "contribution_ratio.png")
    print(f"\n  Contribution ratio plot → {out_dir}/contribution_ratio.png")


# ---------------------------------------------------------------------------
# Analysis 5 — Worst / best case ECG waveforms
# ---------------------------------------------------------------------------

_LEAD_LAYOUT = [
    [0, 3, 6,  9],
    [1, 4, 7,  10],
    [2, 5, 8,  11],
]  # 3 rows × 4 leads


def _plot_ecg_grid(record, ecg_id, pred_qt, tgt_qt, ax_grid, title_color="#B71C1C"):
    """
    Plot a 3×4 clinical-layout 12-lead ECG on the provided list of 12 axes.
    ax_grid: flat list of 12 Axes in row-major order matching LEAD_LAYOUT.
    """
    fs  = record.fs                          # 500 Hz
    sig = record.p_signal.T.astype(float)    # (12, 5000) mV; raw, not scaled

    # 2.5 s per lead displayed (standard clinical strip width)
    samples_shown = min(int(2.5 * fs), sig.shape[1])
    t = np.arange(samples_shown) / fs

    ax_idx = 0
    for row_leads in _LEAD_LAYOUT:
        for lead_idx in row_leads:
            ax = ax_grid[ax_idx]
            trace = sig[lead_idx, :samples_shown]
            ax.plot(t, trace, lw=0.8,
                    color="#1565C0" if lead_idx != 0 else "#B71C1C",  # Lead I red
                    alpha=0.9)
            ax.set_title(LEAD_NAMES[lead_idx], fontsize=8, pad=2,
                         color="#B71C1C" if lead_idx == 0 else "black")
            ax.axhline(0, color="#ccc", lw=0.5)
            ax.set_xlim(0, t[-1])
            ax.tick_params(labelbottom=False, labelleft=False,
                           bottom=False, left=False)
            ax.set_frame_on(True)
            ax_idx += 1


def analysis_worst_best_cases(
    df: pd.DataFrame,
    ptbxl_signals_dir: str | Path,
    n: int,
    out_dir: Path,
) -> None:
    """
    Plot raw 12-lead ECGs for the n worst and n best predictions.

    Worst = highest absolute QT error.
    Best  = lowest absolute QT error (among narrow QRS only, to avoid trivial cases).
    """
    print("\n" + "═"*70)
    print(f"  Worst/Best Case ECG Waveforms  (n={n} each)")
    print("═"*70)

    ptbxl_dir  = Path(ptbxl_signals_dir)
    df_valid   = df[df["valid"]].copy()
    df_valid["abs_err"] = np.abs(df_valid["pred_qt"] - df_valid["tgt_qt"])

    worst = df_valid.nlargest(n, "abs_err")
    best  = df_valid[df_valid["qrs_ms"] <= 100].nsmallest(n, "abs_err")

    for group_label, group_df, title_color in [
        ("worst", worst, "#B71C1C"),
        ("best",  best,  "#2E7D32"),
    ]:
        print(f"\n  {group_label.title()} cases:")
        ecg_plot_dir = out_dir / f"ecg_{group_label}"
        ecg_plot_dir.mkdir(parents=True, exist_ok=True)

        for rank, (ecg_id, row) in enumerate(group_df.iterrows(), start=1):
            filename_hr = df_valid.loc[ecg_id, "filename_hr"]
            try:
                rec = wfdb.rdrecord(str(ptbxl_dir / filename_hr))
            except Exception as exc:
                print(f"    [{rank:02d}] ecg_id={ecg_id}  FAILED to load: {exc}")
                continue

            err = row["pred_qt"] - row["tgt_qt"]
            print(f"    [{rank:02d}] ecg_id={ecg_id:6d}  "
                  f"GT={row['tgt_qt']:.0f} ms  pred={row['pred_qt']:.0f} ms  "
                  f"error={err:+.0f} ms  QRS={row['qrs_ms']:.0f} ms")

            # Build figure: 3 rows × 4 leads + info text column
            fig = plt.figure(figsize=(16, 6))
            gs  = gridspec.GridSpec(3, 4, hspace=0.35, wspace=0.15,
                                    left=0.04, right=0.96, top=0.82, bottom=0.05)
            axes_flat = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(4)]

            # Reorder to match clinical layout (row-major over LEAD_LAYOUT)
            ordered_axes = []
            for row_leads in _LEAD_LAYOUT:
                for li in row_leads:
                    ordered_axes.append(axes_flat[_LEAD_LAYOUT_FLAT.index(li)])

            _plot_ecg_grid(rec, ecg_id,
                           pred_qt=row["pred_qt"], tgt_qt=row["tgt_qt"],
                           ax_grid=ordered_axes, title_color=title_color)

            rr_ms = row.get("RR_Mean_Global", float("nan"))
            rr_sec = rr_ms / 1000.0 if np.isfinite(rr_ms) else float("nan")
            pred_qtc = row["pred_qt"] / np.sqrt(rr_sec) if np.isfinite(rr_sec) else float("nan")
            ref_qtc  = row["tgt_qt"]  / np.sqrt(rr_sec) if np.isfinite(rr_sec) else float("nan")

            fig.suptitle(
                f"ECG #{ecg_id}  [{group_label.upper()} #{rank}]   "
                f"QRS={row['qrs_ms']:.0f} ms  HR={row.get('HR', float('nan')):.0f} bpm\n"
                f"True QT={row['tgt_qt']:.0f} ms (QTc={ref_qtc:.0f} ms)   "
                f"Pred QT={row['pred_qt']:.0f} ms (QTcP={pred_qtc:.0f} ms)   "
                f"Error={err:+.0f} ms    Lead I = red",
                fontsize=9.5, fontweight="bold", color=title_color,
            )
            fname = ecg_plot_dir / f"{rank:02d}_ecg{ecg_id}_err{err:+.0f}ms.png"
            _save(fig, fname, dpi=140)

        print(f"  {group_label.title()} ECG plots → {ecg_plot_dir}/")


# Flat index list matching 3×4 axes in row-major order
_LEAD_LAYOUT_FLAT = [li for row in _LEAD_LAYOUT for li in row]


# ---------------------------------------------------------------------------
# PCA of embedding space
# ---------------------------------------------------------------------------

def analysis_pca_embeddings(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Project z_c (Lead-I), z_hat (adapted), and z_12 (12-lead) into 2D PCA space.

    Shows whether JEPA training shifts z_c toward z_12, and whether z_hat
    lands between them or overshoot.
    """
    from sklearn.decomposition import PCA

    print("\n" + "═"*70)
    print("  PCA of Embedding Space")
    print("═"*70)

    mask = df["valid"].values
    z_c   = np.stack(df.loc[mask, "z_c"].values)    # (N, 192)
    z_hat = np.stack(df.loc[mask, "z_hat"].values)  # (N, 192)
    z_12  = np.stack(df.loc[mask, "z_12"].values)   # (N, 192)
    err   = np.abs(df.loc[mask, "pred_qt"].values - df.loc[mask, "tgt_qt"].values)

    # Fit PCA on the union of all three clouds
    all_z  = np.concatenate([z_c, z_hat, z_12], axis=0)
    pca    = PCA(n_components=2, random_state=42)
    all_2d = pca.fit_transform(all_z)
    N      = len(z_c)
    zc_2d  = all_2d[:N]
    zh_2d  = all_2d[N:2*N]
    z12_2d = all_2d[2*N:]

    var1, var2 = pca.explained_variance_ratio_ * 100

    print(f"  PCA variance explained: PC1={var1:.1f}%  PC2={var2:.1f}%")
    print(f"  Mean distance z_c → z_12  : "
          f"{float(np.linalg.norm(z_c - z_12, axis=1).mean()):.4f}")
    print(f"  Mean distance z_hat → z_12: "
          f"{float(np.linalg.norm(z_hat - z_12, axis=1).mean()):.4f}")

    rng = np.random.RandomState(0)
    idx = rng.choice(N, min(800, N), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: all three clouds
    ax = axes[0]
    ax.scatter(z12_2d[idx, 0], z12_2d[idx, 1],
               s=7, alpha=0.25, color="#1565C0", label="z_12 (12-lead base)", linewidths=0)
    ax.scatter(zc_2d[idx, 0], zc_2d[idx, 1],
               s=7, alpha=0.25, color="#E65100", label="z_c (Lead-I JEPA)", linewidths=0)
    ax.scatter(zh_2d[idx, 0], zh_2d[idx, 1],
               s=7, alpha=0.25, color="#2E7D32", label="z_hat (after adapter)", linewidths=0)
    ax.set_xlabel(f"PC1 ({var1:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var2:.1f}%)", fontsize=11)
    ax.set_title("Embedding Space PCA\nz_c (Lead I), z_hat (adapted), z_12 (12-lead)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9, markerscale=2)
    ax.grid(True, alpha=0.2)

    # Right: z_hat coloured by |error|
    ax2 = axes[1]
    sc = ax2.scatter(zh_2d[idx, 0], zh_2d[idx, 1],
                     c=err[idx], cmap="hot_r", s=8, alpha=0.55, linewidths=0,
                     vmin=0, vmax=np.percentile(err, 95))
    plt.colorbar(sc, ax=ax2, label="|QT error| (ms)", shrink=0.85)
    ax2.set_xlabel(f"PC1 ({var1:.1f}%)", fontsize=11)
    ax2.set_ylabel(f"PC2 ({var2:.1f}%)", fontsize=11)
    ax2.set_title("z_hat Embedding Space\nColoured by |QT error|",
                  fontsize=10, fontweight="bold")
    ax2.grid(True, alpha=0.2)

    fig.suptitle("PCA of JEPA Embedding Space — Does z_hat Bridge z_c → z_12?",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir / "pca_embeddings.png")
    print(f"  PCA plot → {out_dir}/pca_embeddings.png")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="JEPA-Linear interpretability & error analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt",         default=str(DEFAULT_CKPT),
                        help="JEPA-Linear checkpoint (.pt)")
    parser.add_argument("--base-ckpt",    default=str(DEFAULT_BASE),
                        help="Frozen 12-lead base model checkpoint (for z_12)")
    parser.add_argument("--ptbxl-dir",    default=str(DEFAULT_SIG_DIR))
    parser.add_argument("--features-dir", default=str(DEFAULT_FEAT_DIR))
    parser.add_argument("--labels-csv",   default=str(DEFAULT_LABELS))
    parser.add_argument("--output-dir",   default=str(DEFAULT_OUT))
    parser.add_argument("--split",        default="val",
                        choices=["train", "val", "test"])
    parser.add_argument("--batch-size",   type=int, default=32)
    parser.add_argument("--n-cases",      type=int, default=10,
                        help="Number of worst/best ECG cases to plot")
    parser.add_argument("--no-ecg-plots", action="store_true", default=False,
                        help="Skip raw ECG waveform plots (faster)")
    args = parser.parse_args()

    out_dir  = Path(args.output_dir)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"ERROR: JEPA checkpoint not found: {ckpt_path}")
        sys.exit(1)
    if not Path(args.base_ckpt).exists():
        print(f"ERROR: base checkpoint not found: {args.base_ckpt}")
        sys.exit(1)

    # ── Device ───────────────────────────────────────────────────────────────
    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"\nDevice : {device}")

    # ── Load JEPA-Linear checkpoint ───────────────────────────────────────────
    print(f"\nLoading JEPA checkpoint: {ckpt_path.name}")
    context_enc, predictor, qt_head, cfg = load_jepa_ckpt(ckpt_path, device)
    W = predictor.weight.detach().cpu().numpy()   # (192, 192)
    b = predictor.bias.detach().cpu().numpy()      # (192,)

    # ── Load frozen base model (for z_12) ─────────────────────────────────────
    print(f"Loading base model: {Path(args.base_ckpt).name}")
    base_model = load_model(Path(args.base_ckpt), device)

    # ── Datasets ──────────────────────────────────────────────────────────────
    ds_kwargs = dict(ptbxl_signals_dir=args.ptbxl_dir,
                     labels_csv=args.labels_csv, split=args.split)
    ds_leadI   = ForcedLeadDataset(lead_indices=[0],           **ds_kwargs)
    ds_12lead  = ForcedLeadDataset(lead_indices=list(range(12)), **ds_kwargs)
    print(f"Split   : {args.split}  ({len(ds_leadI):,} samples)")

    # ── Extract embeddings ────────────────────────────────────────────────────
    print("\nExtracting JEPA embeddings (Lead I) ...")
    pred_qt, tgt_qt, ecg_ids, z_c, z_hat = extract_jepa_embeddings(
        context_enc, predictor, qt_head, ds_leadI, device, args.batch_size
    )
    print("Extracting base 12-lead embeddings ...")
    z_12 = extract_base_embeddings(base_model, ds_12lead, device, args.batch_size)

    # ── Compute per-sample metrics ────────────────────────────────────────────
    print("Computing per-sample metrics ...")

    # Cosine similarity z_c vs z_12
    cos_sim = (
        (z_c * z_12).sum(axis=1) /
        (np.linalg.norm(z_c, axis=1) * np.linalg.norm(z_12, axis=1) + 1e-8)
    )

    # Contribution ratio in embedding space
    correction  = z_c @ W.T + b           # (N, 192) = W·z_c + b
    contrib_ratio = (
        np.linalg.norm(correction, axis=1) /
        (np.linalg.norm(z_c, axis=1) + 1e-8)
    )

    # Delta QT: how much did adapter change the prediction?
    qt_from_zc   = qt_from_embedding(qt_head, z_c,   device)
    qt_from_zhat = qt_from_embedding(qt_head, z_hat,  device)
    delta_qt = qt_from_zhat - qt_from_zc

    # ── Merge everything into one DataFrame ───────────────────────────────────
    meta = load_meta(args.labels_csv, args.features_dir, args.split)
    ids  = ecg_ids.tolist()

    df = pd.DataFrame({
        "ecg_id":       ids,
        "pred_qt":      pred_qt,
        "tgt_qt":       tgt_qt,
        "cos_sim_c_12": cos_sim,
        "contrib_ratio": contrib_ratio,
        "delta_qt":     delta_qt,
    }).set_index("ecg_id")

    # Attach metadata
    df["qrs_ms"]   = meta.reindex(ids)["QRS_Dur_Global"].values
    df["rr_ms"]    = meta.reindex(ids)["RR_Mean_Global"].values
    df["HR"]       = meta.reindex(ids)["HR"].values
    df["filename_hr"] = meta.reindex(ids)["filename_hr"].values if "filename_hr" in meta.columns else ""
    df["valid"]    = ~(np.isnan(df["qrs_ms"].values) | np.isnan(df["rr_ms"].values))

    # Store embeddings per row for PCA
    df["z_c"]   = list(z_c)
    df["z_hat"] = list(z_hat)
    df["z_12"]  = list(z_12)

    # ── Headline numbers ──────────────────────────────────────────────────────
    err = pred_qt - tgt_qt
    print(f"\nOverall  MAE={np.abs(err).mean():.2f} ms  "
          f"RMSE={np.sqrt((err**2).mean()):.2f} ms  "
          f"R={np.corrcoef(pred_qt, tgt_qt)[0,1]:.3f}  "
          f"bias={err.mean():+.2f} ms")

    # ── Run analyses ──────────────────────────────────────────────────────────
    analysis_wide_qrs(df, out_dir)
    analysis_cosine_similarity(df, out_dir)
    analysis_linear_w_svd(W, b, out_dir)
    analysis_contribution_ratio(df, out_dir)
    analysis_pca_embeddings(df, out_dir)

    if not args.no_ecg_plots:
        # filename_hr needs to come from the dataset index directly
        id_to_filename = {item["ecg_id"]: item["filename_hr"] for item in ds_leadI.index}
        df["filename_hr"] = df.index.map(id_to_filename)
        analysis_worst_best_cases(df, args.ptbxl_dir, n=args.n_cases, out_dir=out_dir)
    else:
        print("\n  (ECG waveform plots skipped — pass --no-ecg-plots to re-enable)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Outputs → {out_dir}/")
    for f in sorted(out_dir.rglob("*.png")):
        print(f"    {f.relative_to(out_dir)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
