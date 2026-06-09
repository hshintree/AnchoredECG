"""
Modal app for the 1-D ResNet-18 QT ablation (PTB-XL, 500 Hz).

Trains two from-scratch supervised baselines IN PARALLEL on separate H100s:
  * lead1  — ResNet-18 on Lead I only          (in_channels=1)
  * all12  — ResNet-18 on all 12 leads         (in_channels=12)

These are the control for the electrode-graph GNN: "is our lead-I GNN
meaningfully better than a vanilla CNN on the same task?"

Reuses the existing `ptbxl-data` volume (already populated by modal_finetune.py).

Launch BOTH in parallel (default entrypoint):
    modal run modal_resnet_qt.py
    modal run --detach modal_resnet_qt.py        # safe to close laptop

Launch a single config:
    modal run modal_resnet_qt.py::train_one --leads lead1
    modal run modal_resnet_qt.py::train_one --leads all12

Checkpoints land in the volume at:
    /ptbxl/best_resnet_lead1.pt
    /ptbxl/best_resnet_all12.pt
Download with:
    modal volume get ptbxl-data /best_resnet_lead1.pt qt_finetune/checkpoints/best_resnet_lead1.pt
    modal volume get ptbxl-data /best_resnet_all12.pt qt_finetune/checkpoints/best_resnet_all12.pt
"""
from __future__ import annotations

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent

# --- Container image (PyTorch + CUDA 12.1; no PyG needed for the CNN) --------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.0",
        "torchvision==0.18.0",
        "torchaudio==2.3.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install("wfdb", "pandas", "scipy", "tqdm", "numpy", "scikit-learn")
    .pip_install("wandb")
)

training_image = (
    image
    .add_local_dir(str(REPO_ROOT / "models"), "/repo/models")
    .add_local_dir(str(REPO_ROOT / "qt_finetune"), "/repo/qt_finetune")
)

ptbxl_vol = modal.Volume.from_name("ptbxl-data", create_if_missing=True)

app = modal.App("anchoredecg-qt-resnet")

SIGNALS_DIR = "/ptbxl/ptbxl_signals/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
LABELS_CSV = "/repo/qt_finetune/qt_labels.csv"


@app.function(
    image=training_image,
    gpu="H100",
    volumes={"/ptbxl": ptbxl_vol},
    secrets=[modal.Secret.from_name("wandb")],
    timeout=4 * 3600,
    memory=32768,
    cpu=8.0,
)
def train_resnet(leads: str, epochs: int = 50, batch_size: int = 128,
                 patience: int = 5) -> dict:
    """Train one ResNet-18 config inside the container.

    Saves the best checkpoint to /ptbxl/best_resnet_<leads>.pt in the volume.
    Returns a small summary dict.
    """
    import subprocess
    import sys
    import shutil

    output_dir = f"/tmp/resnet_{leads}"
    ckpt_name = f"best_resnet_{leads}.pt"

    cmd = [
        sys.executable,
        "/repo/qt_finetune/train_resnet_qt.py",
        "--leads", leads,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--patience", str(patience),
        "--num-workers", "8",
        "--ptbxl-dir", SIGNALS_DIR,
        "--labels-csv", LABELS_CSV,
        "--output-dir", output_dir,
        "--ckpt-name", ckpt_name,
        "--run-name", f"resnet_{leads}",
        "--wandb-project", "anchoredecg-qt-resnet",
    ]
    subprocess.run(cmd, check=True)

    src = f"{output_dir}/{ckpt_name}"
    dst = f"/ptbxl/{ckpt_name}"
    shutil.copy2(src, dst)
    ptbxl_vol.commit()
    print(f"\n[{leads}] checkpoint committed to volume at {dst}")

    import torch
    ckpt = torch.load(dst, map_location="cpu", weights_only=False)
    return {
        "leads": leads,
        "epoch": ckpt.get("epoch"),
        "val_mae": ckpt.get("val_mae"),
        "val_r": ckpt.get("val_r"),
        "ckpt": dst,
    }


@app.local_entrypoint()
def train_one(leads: str = "lead1", epochs: int = 50, batch_size: int = 128,
              patience: int = 5):
    """Train a single ResNet config.

    Usage:
        modal run modal_resnet_qt.py::train_one --leads lead1
        modal run modal_resnet_qt.py::train_one --leads all12
    """
    print(f"Launching ResNet-18 QT training on Modal (H100) — leads={leads} ...")
    res = train_resnet.remote(leads, epochs, batch_size, patience)
    print("\nResult:", res)


@app.local_entrypoint()
def main(epochs: int = 50, batch_size: int = 128, patience: int = 5):
    """Train BOTH configs (lead1 + all12) in parallel on separate H100s.

    Usage:
        modal run modal_resnet_qt.py
        modal run --detach modal_resnet_qt.py
    """
    print("Launching TWO ResNet-18 QT jobs in parallel on Modal (H100 each):")
    print("  1) lead1  — Lead I only")
    print("  2) all12  — all 12 leads\n")

    # spawn() returns immediately so both H100 containers run concurrently.
    h_lead1 = train_resnet.spawn("lead1", epochs, batch_size, patience)
    h_all12 = train_resnet.spawn("all12", epochs, batch_size, patience)

    print("Both jobs spawned. Waiting for completion (Ctrl-C is safe with --detach) ...\n")
    res_lead1 = h_lead1.get()
    res_all12 = h_all12.get()

    print("\n=== Done ===")
    for r in (res_lead1, res_all12):
        print(f"  {r['leads']:>6}: best val MAE={r['val_mae']:.2f} ms  "
              f"R={r['val_r']:.3f}  (epoch {r['epoch']})  -> {r['ckpt']}")
    print("\nDownload checkpoints:")
    print("  modal volume get ptbxl-data /best_resnet_lead1.pt qt_finetune/checkpoints/best_resnet_lead1.pt")
    print("  modal volume get ptbxl-data /best_resnet_all12.pt qt_finetune/checkpoints/best_resnet_all12.pt")
