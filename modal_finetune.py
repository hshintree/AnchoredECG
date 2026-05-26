"""
Modal app for QT interval fine-tuning on a GPU cloud container.

One-time setup (run from repo root):
    pip install modal
    modal setup                          # authenticates your account

    # --- Option A: populate volume from Google Drive (recommended) ----------
    # 1. Zip the two PTB-XL datasets on your Mac:
    #      cd "/Users/hakeemshindy/Downloads/ICML Supplementary/AnchoredECG/data"
    #      caffeinate -i zip -r ptbxl_signals.zip \
    #          "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
    #      caffeinate -i zip -r ptbxl_features.zip \
    #          "ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1"
    #
    # 2. Upload both zips to Google Drive and copy their share links.
    #    Share → "Anyone with the link" → Viewer.
    #    File ID is the long token between /d/ and /view in the share URL.
    #
    # 3. Run the volume-population step (server-to-server download, no 504):
    #      modal run modal_finetune.py::populate_volume_entrypoint \
    #          --signals-id  <GDRIVE_FILE_ID_FOR_ptbxl_signals.zip> \
    #          --features-id <GDRIVE_FILE_ID_FOR_ptbxl_features.zip>
    #
    # 4. Verify the volume looks right:
    #      modal volume ls ptbxl-data
    #
    # --- Option B: modal volume put (only works for small datasets) ----------
    #    modal volume put ptbxl-data <local_dir> /ptbxl_signals   # often 504s

Then run training:
    modal run modal_finetune.py

Monitor live:
    (logs stream automatically in the terminal while the job runs)

Download the best checkpoint after training completes:
    It is saved automatically to qt_finetune/checkpoints/best_qt_model.pt
    in this repo.
"""

from __future__ import annotations

from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Container image — PyTorch 2.3 + CUDA 12.1 + PyG
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    # PyTorch with CUDA 12.1 (matches A10G / L4)
    .pip_install(
        "torch==2.3.0",
        "torchvision==0.18.0",
        "torchaudio==2.3.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    # PyTorch Geometric core
    .pip_install("torch_geometric==2.5.3")
    # PyG optional C++ extensions (scatter / sparse / cluster)
    .pip_install(
        "torch_scatter",
        "torch_sparse",
        "torch_cluster",
        "torch_spline_conv",
        find_links="https://data.pyg.org/whl/torch-2.3.0+cu121.html",
    )
    # Other dependencies
    .pip_install("wfdb", "pandas", "scipy", "tqdm", "numpy", "scikit-learn")
    # Google Drive downloader (used by populate_volume)
    .pip_install("gdown")
    # Plotting
    .pip_install("matplotlib", "seaborn")
)

# ---------------------------------------------------------------------------
# Persistent volume for the PTB-XL datasets (uploaded once, reused every run)
# ---------------------------------------------------------------------------

ptbxl_vol = modal.Volume.from_name("ptbxl-data", create_if_missing=True)

# ---------------------------------------------------------------------------
# Training image — base image + repo code baked in at build time
# (modal.Mount was removed in newer Modal; copy_local_dir is the modern way)
# ---------------------------------------------------------------------------

training_image = (
    image
    .add_local_dir(str(REPO_ROOT / "models"),      "/repo/models")
    .add_local_dir(str(REPO_ROOT / "qt_finetune"), "/repo/qt_finetune")
    .add_local_dir(str(REPO_ROOT / "checkpoints"), "/repo/checkpoints")
)

# ---------------------------------------------------------------------------
# Modal App
# ---------------------------------------------------------------------------

app = modal.App("anchoredecg-qt-finetune")

# ---------------------------------------------------------------------------
# Volume-population function
# Downloads two zips from Google Drive into the persistent volume.
# Run once before training:
#   modal run modal_finetune.py::populate_volume_entrypoint \
#       --signals-id  <FILE_ID> \
#       --features-id <FILE_ID>
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={"/ptbxl": ptbxl_vol},
    timeout=3600,   # 1 h — enough for a 6 GB download + unzip
    memory=8192,
    cpu=4,
)
def populate_volume(signals_gdrive_id: str, features_gdrive_id: str) -> None:
    """Download PTB-XL zips from Google Drive and extract into the volume."""
    import subprocess
    import tempfile
    import gdown

    def _download_and_extract(file_id: str, dest_dir: str, label: str) -> None:
        print(f"[populate_volume] Downloading {label} (file_id={file_id}) ...")
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            zip_path = tmp.name

        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, zip_path, quiet=False)

        print(f"[populate_volume] Extracting {label} -> {dest_dir} ...")
        import zipfile, os
        os.makedirs(dest_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)

        os.unlink(zip_path)
        print(f"[populate_volume] {label} done.")

    _download_and_extract(signals_gdrive_id,  "/ptbxl/ptbxl_signals",  "signals")
    _download_and_extract(features_gdrive_id, "/ptbxl/ptbxl_features", "features")

    # Persist changes so subsequent function invocations see the new files.
    ptbxl_vol.commit()
    print("[populate_volume] Volume committed. Run 'modal volume ls ptbxl-data' to verify.")


@app.local_entrypoint()
def populate_volume_entrypoint(
    signals_id: str,
    features_id: str,
):
    """Local entry point: modal run modal_finetune.py::populate_volume_entrypoint \\
           --signals-id <ID> --features-id <ID>
    """
    print(f"Populating ptbxl-data volume from Google Drive ...")
    print(f"  signals  file_id: {signals_id}")
    print(f"  features file_id: {features_id}")
    populate_volume.remote(signals_id, features_id)
    print("Done. Volume is ready for training.")


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------


@app.function(
    image=training_image,          # base image + repo code baked in
    gpu="A10G",                    # 24 GB VRAM; swap for "L4" to save cost
    volumes={"/ptbxl": ptbxl_vol},
    timeout=4 * 3600,              # 4 h — plenty for 30 epochs
    memory=32768,                  # 32 GB RAM
    cpu=8,
)
def finetune_qt(epochs: int = 30, batch_size: int = 32) -> None:
    """Run QT fine-tuning inside the container.

    Saves the best checkpoint to the persistent volume at
    /ptbxl/best_qt_model.pt so it survives even when run with --detach.
    """
    import subprocess
    import sys
    import shutil

    output_dir = "/tmp/qt_output"

    cmd = [
        sys.executable,
        "/repo/qt_finetune/finetune_qt.py",
        "--epochs",      str(epochs),
        "--batch-size",  str(batch_size),
        "--num-workers", "4",
        "--ptbxl-dir",   "/ptbxl/ptbxl_signals/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1",
        "--checkpoint",  "/repo/checkpoints/shd_pretrained.pt",
        "--labels-csv",  "/repo/qt_finetune/qt_labels.csv",
        "--output-dir",  output_dir,
    ]

    subprocess.run(cmd, check=True)

    # Copy checkpoint into the persistent volume so it survives container exit
    # and can be retrieved with: modal volume get ptbxl-data /best_qt_model.pt
    ckpt_src = f"{output_dir}/best_qt_model.pt"
    ckpt_dst = "/ptbxl/best_qt_model.pt"
    shutil.copy2(ckpt_src, ckpt_dst)
    ptbxl_vol.commit()
    print(f"\nCheckpoint committed to volume at {ckpt_dst}")
    print("Download locally with:")
    print("  modal volume get ptbxl-data /best_qt_model.pt ./qt_finetune/checkpoints/best_qt_model.pt")


# ---------------------------------------------------------------------------
# Pearson correlation plot — val set, 12-lead and 1-lead side by side
# ---------------------------------------------------------------------------

@app.function(
    image=training_image,
    volumes={"/ptbxl": ptbxl_vol},
    timeout=3600,
    memory=16384,
    cpu=4,
)
def plot_qt_correlation() -> bytes:
    """Run model on val set, produce predicted vs ground-truth scatter plot.

    Returns PNG bytes; also saves to /ptbxl/qt_correlation.png in the volume.
    """
    import sys
    sys.path.insert(0, "/repo/models")
    sys.path.insert(0, "/repo/qt_finetune")

    import numpy as np
    import torch
    import torch.nn as nn
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy import stats
    from torch_geometric.loader import DataLoader as PyGDataLoader

    from ecg_gnn_simple import ECGGNNSimple
    from qt_dataset import TARGET_LEN
    from validate_qt import ForcedLeadDataset, load_model, evaluate

    CHECKPOINT   = "/ptbxl/best_qt_model.pt"
    SIGNALS_DIR  = "/ptbxl/ptbxl_signals/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
    FEATURES_DIR = "/ptbxl/ptbxl_features/ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1"
    LABELS_CSV   = "/repo/qt_finetune/qt_labels.csv"

    device = torch.device("cpu")
    model  = load_model(CHECKPOINT, device)

    configs = [
        ("12-lead",       list(range(12)), "#2563EB"),
        ("1-lead (II)",   [1],             "#DC2626"),
    ]

    fig = plt.figure(figsize=(13, 6))
    fig.patch.set_facecolor("#0F172A")
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    for col, (label, leads, color) in enumerate(configs):
        ds = ForcedLeadDataset(
            lead_indices=leads,
            split="val",
            ptbxl_signals_dir=SIGNALS_DIR,
            labels_csv=LABELS_CSV,
        )
        loader = PyGDataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
        res    = evaluate(model, loader, device, label)

        pred = res["pred"]
        tgt  = res["tgt"]
        slope, intercept, r, p, _ = stats.linregress(tgt, pred)

        ax = fig.add_subplot(gs[col])
        ax.set_facecolor("#1E293B")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

        # Hexbin density
        hb = ax.hexbin(tgt, pred, gridsize=45, cmap="YlOrRd",
                       mincnt=1, linewidths=0.2)
        cb = fig.colorbar(hb, ax=ax, pad=0.02)
        cb.set_label("Count", color="#94A3B8", fontsize=9)
        cb.ax.yaxis.set_tick_params(color="#94A3B8")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="#94A3B8")

        # Identity line
        lo = min(tgt.min(), pred.min()) - 5
        hi = max(tgt.max(), pred.max()) + 5
        ax.plot([lo, hi], [lo, hi], "--", color="#94A3B8",
                linewidth=1.2, alpha=0.7, label="y = x")

        # Regression line
        xs = np.linspace(lo, hi, 200)
        ax.plot(xs, slope * xs + intercept, "-", color=color,
                linewidth=1.8, alpha=0.9, label="Regression")

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Ground Truth QT (ms)", color="#94A3B8", fontsize=11)
        ax.set_ylabel("Predicted QT (ms)",    color="#94A3B8", fontsize=11)
        ax.tick_params(colors="#94A3B8")
        ax.set_title(label, color="white", fontsize=13, fontweight="bold", pad=10)

        # Stats annotation
        stats_txt = (
            f"R  = {r:.3f}\n"
            f"MAE = {res['mae']:.1f} ms\n"
            f"RMSE = {res['rmse']:.1f} ms\n"
            f"n  = {res['n']:,}"
        )
        ax.text(0.04, 0.96, stats_txt,
                transform=ax.transAxes, fontsize=10,
                verticalalignment="top", color="white",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#0F172A",
                          edgecolor="#334155", alpha=0.9))

        ax.legend(fontsize=9, facecolor="#1E293B", edgecolor="#334155",
                  labelcolor="#94A3B8", loc="lower right")

    fig.suptitle(
        "QT Interval Prediction — Validation Set (PTB-XL, n=2,179)",
        color="white", fontsize=14, fontweight="bold", y=1.01
    )

    out_path = "/ptbxl/qt_correlation.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    ptbxl_vol.commit()
    print(f"Saved to {out_path}")

    return open(out_path, "rb").read()


@app.local_entrypoint()
def plot_correlation_entrypoint():
    """Generate QT correlation plot and save locally.

    Usage:
        modal run modal_finetune.py::plot_correlation_entrypoint
    """
    import pathlib
    print("Generating QT correlation plot on Modal ...")
    png_bytes = plot_qt_correlation.remote()

    out = pathlib.Path("qt_finetune/qt_correlation.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(png_bytes)
    print(f"Saved → {out}  ({len(png_bytes) / 1024:.0f} KB)")


# ---------------------------------------------------------------------------
# Per-lead bar chart — Pearson r per single lead vs 12-lead reference
# ---------------------------------------------------------------------------

@app.function(
    image=training_image,
    volumes={"/ptbxl": ptbxl_vol},
    timeout=3600,
    memory=16384,
    cpu=4,
)
def plot_per_lead_qt() -> bytes:
    """
    Both models (Lead-II only and 12-lead) each produce ONE global QT prediction
    per sample.  We then compare those predictions against the per-lead ground-truth
    QT from ECGdeli (QT_Int_I … QT_Int_V6) for every lead independently.

    X-axis : 12 ECG leads
    Y-axis : Pearson r between model prediction and that lead's GT QT
    2 bars : Lead-II model  vs  12-lead model
    """
    import sys
    sys.path.insert(0, "/repo/models")
    sys.path.insert(0, "/repo/qt_finetune")

    import numpy as np
    import torch
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from validate_qt import ForcedLeadDataset, load_model

    CHECKPOINT   = "/ptbxl/best_qt_model.pt"
    SIGNALS_DIR  = "/ptbxl/ptbxl_signals/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
    FEATURES_DIR = "/ptbxl/ptbxl_features/ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1"
    LABELS_CSV   = "/repo/qt_finetune/qt_labels.csv"

    LEAD_NAMES     = ["I", "II", "III", "aVR", "aVL", "aVF",
                      "V1", "V2", "V3", "V4", "V5", "V6"]
    PER_LEAD_COLS  = [f"QT_Int_{n}" for n in LEAD_NAMES]

    REGIONS = [
        ("Inferior", [0, 1, 2],              "#FECACA"),
        ("Anterior", [3, 4, 5],              "#BFDBFE"),
        ("Lateral",  [6, 7, 8, 9, 10, 11],  "#BBF7D0"),
    ]

    device = torch.device("cpu")
    model  = load_model(CHECKPOINT, device)

    # ── helper: run inference, return (predictions ndarray, dataset) ──────
    def _infer(lead_indices, label):
        ds = ForcedLeadDataset(
            lead_indices=lead_indices, split="val",
            ptbxl_signals_dir=SIGNALS_DIR,
            labels_csv=LABELS_CSV,
        )
        loader = PyGDataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
        preds  = []
        model.eval()
        with torch.no_grad():
            for batch_data, _ in loader:
                batch_data = batch_data.to(device)
                B   = batch_data.num_graphs
                tab = torch.zeros(B, 0, device=device)
                preds.append(model(batch_data, tabular=tab)["logits"].flatten().cpu().numpy())
        print(f"  {label}: {len(ds)} samples inferred")
        return np.concatenate(preds), ds

    print("Inferring with Lead II only...")
    preds_leadII, ds_ref = _infer([1], "Lead II")

    print("Inferring with 12 leads...")
    preds_12lead, _      = _infer(list(range(12)), "12-lead")

    # ── load per-lead GT QT from ECGdeli ──────────────────────────────────
    print("Loading per-lead ground-truth QT from ECGdeli...")
    ecgdeli = pd.read_csv(
        f"{FEATURES_DIR}/features/ecgdeli_features.csv",
        usecols=["ecg_id"] + PER_LEAD_COLS,
        low_memory=False,
    ).set_index("ecg_id")

    ecg_ids = [item["ecg_id"] for item in ds_ref.index]   # ordered val samples
    gt = np.full((len(ecg_ids), 12), np.nan, dtype=np.float32)
    for i, eid in enumerate(ecg_ids):
        if eid in ecgdeli.index:
            for j, col in enumerate(PER_LEAD_COLS):
                v = ecgdeli.loc[eid, col]
                if v == v and v is not None:   # NaN check
                    gt[i, j] = float(v)

    # ── Pearson r per lead ────────────────────────────────────────────────
    r_II, r_12 = [], []
    for j, lead in enumerate(LEAD_NAMES):
        mask = ~np.isnan(gt[:, j])
        n    = mask.sum()
        if n < 10:
            r_II.append(np.nan);  r_12.append(np.nan)
            continue
        r_II.append(pearsonr(preds_leadII[mask],  gt[mask, j])[0])
        r_12.append(pearsonr(preds_12lead[mask],  gt[mask, j])[0])
        print(f"  {lead}: n={n}  r(II)={r_II[-1]:.3f}  r(12)={r_12[-1]:.3f}")

    r_II = np.array(r_II)
    r_12 = np.array(r_12)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x, w = np.arange(12), 0.35

    for region_name, idxs, bg in REGIONS:
        lo, hi = min(idxs) - 0.5, max(idxs) + 0.5
        ax.axvspan(lo, hi, color=bg, alpha=0.35, zorder=0)
        ax.text((lo + hi) / 2, 1.02, region_name, ha="center", va="bottom",
                fontsize=10, color="#444", transform=ax.get_xaxis_transform())

    ax.bar(x - w/2, r_II, w, color="#E87040", zorder=3,
           edgecolor="white", linewidth=0.4,
           label=f"1-lead (II)   mean r = {np.nanmean(r_II):.3f}")
    ax.bar(x + w/2, r_12, w, color="#4C72B0", zorder=3,
           edgecolor="white", linewidth=0.4, alpha=0.85,
           label=f"12-lead        mean r = {np.nanmean(r_12):.3f}")

    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.set_xticks(x);  ax.set_xticklabels(LEAD_NAMES, fontsize=11)
    ax.set_ylabel("Pearson r", fontsize=12)
    ax.set_ylim(-0.05, 1.05);  ax.set_xlim(-0.6, 11.6)
    ax.set_title(
        "Per-lead Pearson r (val): QT prediction vs per-lead ground truth\n"
        "1-lead (Lead II input) vs 12-lead input",
        fontsize=13, pad=18,
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=2, fontsize=10, frameon=True,
              facecolor="white", edgecolor="#ccc")

    fig.tight_layout()
    out_path = "/ptbxl/qt_per_lead.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    ptbxl_vol.commit()
    print(f"Saved → {out_path}")
    return open(out_path, "rb").read()


@app.local_entrypoint()
def plot_per_lead_entrypoint():
    """Generate per-lead QT Pearson r bar chart and save locally.

    Usage:
        modal run modal_finetune.py::plot_per_lead_entrypoint
    """
    import pathlib
    print("Generating per-lead QT bar chart on Modal ...")
    png_bytes = plot_per_lead_qt.remote()
    out = pathlib.Path("qt_finetune/qt_per_lead.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(png_bytes)
    print(f"Saved → {out}  ({len(png_bytes) / 1024:.0f} KB)")


# ---------------------------------------------------------------------------
# Validation function — runs on Modal with volume data + checkpoint
# ---------------------------------------------------------------------------

@app.function(
    image=training_image,
    volumes={"/ptbxl": ptbxl_vol},
    timeout=3600,
    memory=16384,
    cpu=4,
)
def validate_qt_modal() -> None:
    """Run 12-lead and 1-lead validation against the checkpoint saved in the volume."""
    import subprocess, sys
    cmd = [
        sys.executable, "/repo/qt_finetune/validate_qt.py",
        "--checkpoint",  "/ptbxl/best_qt_model.pt",
        "--ptbxl-dir",   "/ptbxl/ptbxl_signals/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1",
        "--labels-csv",  "/repo/qt_finetune/qt_labels.csv",
        "--num-workers", "4",
    ]
    subprocess.run(cmd, check=True)


@app.local_entrypoint()
def validate_entrypoint():
    """Run QT validation on Modal (uses the checkpoint + data already in the volume).

    Usage:
        modal run modal_finetune.py::validate_entrypoint
    """
    print("Running QT validation on Modal ...")
    validate_qt_modal.remote()
    print("Done.")


# ---------------------------------------------------------------------------
# Results plotting — Pearson grid + ROC + PRC curves
# ---------------------------------------------------------------------------

@app.function(
    image=training_image,
    volumes={"/ptbxl": ptbxl_vol},
    timeout=3600,
    memory=16384,
    cpu=4,
)
def plot_qt_results() -> dict[str, bytes]:
    """
    Run inference on the test set for all 4 lead configs, then produce:
      1. qt_pearson_grid.png  — predicted vs GT QT (2×2 panel)
      2. qt_roc_curves.png    — QTc prolongation ROC (all configs overlaid)
      3. qt_prc_curves.png    — QTc prolongation PRC (all configs overlaid)

    QTc = predicted_QT / sqrt(RR_Mean_Global_ms / 1000)  [Bazett]
    Prolongation: Narrow QRS (≤100 ms) → QTc > 500 ms
                  Wide   QRS  (>100 ms) → QTc > 550 ms

    All plots are saved to the volume at /ptbxl/plots/ and returned as PNG bytes.
    """
    import sys
    sys.path.insert(0, "/repo/models")
    sys.path.insert(0, "/repo/qt_finetune")

    from pathlib import Path
    from plot_qt_results import run_all

    plots = run_all(
        checkpoint   = Path("/ptbxl/best_qt_model.pt"),
        ptbxl_dir    = Path("/ptbxl/ptbxl_signals/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"),
        features_dir = Path("/ptbxl/ptbxl_features/ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1"),
        labels_csv   = Path("/repo/qt_finetune/qt_labels.csv"),
        output_dir   = Path("/ptbxl/plots"),
        batch_size   = 32,
    )

    # Commit plots to volume so they persist
    ptbxl_vol.commit()

    # Return PNG bytes for all three plots
    result = {}
    for name, path in plots.items():
        result[name] = path.read_bytes()
    return result


@app.local_entrypoint()
def plot_results_entrypoint():
    """Generate all QT results plots (Pearson grid + ROC + PRC) and save locally.

    Usage:
        modal run modal_finetune.py::plot_results_entrypoint
    """
    import pathlib
    print("Running QT results plotting on Modal ...")
    all_plots = plot_qt_results.remote()

    out_dir = pathlib.Path("qt_finetune/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    filename_map = {
        "pearson_grid": "qt_pearson_grid.png",
        "roc_curves":   "qt_roc_curves.png",
        "prc_curves":   "qt_prc_curves.png",
    }
    for name, png_bytes in all_plots.items():
        out = out_dir / filename_map[name]
        out.write_bytes(png_bytes)
        print(f"  Saved → {out}  ({len(png_bytes) / 1024:.0f} KB)")

    print("\nDone. Run locally to view:")
    print("  open qt_finetune/plots/qt_pearson_grid.png")
    print("  open qt_finetune/plots/qt_roc_curves.png")
    print("  open qt_finetune/plots/qt_prc_curves.png")


# ---------------------------------------------------------------------------
# Training entrypoint  (default: modal run modal_finetune.py)
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def train():
    """Default entrypoint — runs QT fine-tuning.

    Attached (streams live logs, laptop must stay on):
        modal run modal_finetune.py::train

    Detached (safe to close laptop; Modal keeps the job alive):
        modal run --detach modal_finetune.py::train

    NOTE: We use .remote() (not .spawn()) so that --detach properly
    keeps the function alive and logs are tracked on Modal's side.

    After training finishes, download the checkpoint:
        modal volume get ptbxl-data /best_qt_model.pt \\
            qt_finetune/checkpoints/best_qt_model.pt
    """
    print("Launching QT fine-tuning on Modal (A10G GPU, full fine-tune, 500 Hz) ...")
    print("Logs will stream below. Run with --detach to close your laptop safely.\n")
    finetune_qt.remote(epochs=30, batch_size=32)
    print("\nTraining complete. Checkpoint saved to volume.")
    print("Download with:")
    print("  modal volume get ptbxl-data /best_qt_model.pt qt_finetune/checkpoints/best_qt_model.pt")
