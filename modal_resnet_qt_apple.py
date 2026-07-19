"""
Modal app for the Apple-Watch-oriented 1-D ResNet-18 QT runs.

Same model, same task, same target (QT_Int_Global, ms) as modal_resnet_qt.py —
the only thing that changes is the input sample rate. We are optimising for
Apple Watch reads, so the rates are chosen to be the ones the *watch* reaches
with the least processing, and PTB-XL absorbs the resampling instead:

    512 Hz  Apple Watch native. User data is fed to the model untouched.
            PTB-XL is upsampled 500 -> 512 (128/125). Window = 5120 samples.
    128 Hz  Exact /4 decimation of the watch rate — the only clean lower rate.
            PTB-XL is downsampled 500 -> 128 (32/125). Window = 1280 samples.

Both divide a 30 s Apple recording into exactly three 10 s windows.

Trains the two rates IN PARALLEL on separate H100s:
    modal run --detach modal_resnet_qt_apple.py                  # both rates
    modal run --detach modal_resnet_qt_apple.py --rates 512      # one rate
    modal run --detach modal_resnet_qt_apple.py --rates 128 --leads all12

Checkpoints land in the ptbxl-data volume at:
    /ptbxl/best_resnet_lead1_512hz.pt
    /ptbxl/best_resnet_lead1_128hz.pt
Download with:
    modal volume get ptbxl-data /best_resnet_lead1_512hz.pt qt_finetune/checkpoints/
    modal volume get ptbxl-data /best_resnet_lead1_128hz.pt qt_finetune/checkpoints/
"""
from __future__ import annotations

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent

# The two rates this ablation compares. Change here to switch what runs.
RATES = (512, 128)

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

app = modal.App("anchoredecg-qt-resnet-apple")

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
def train_resnet(leads: str, target_fs: int, epochs: int = 50,
                 batch_size: int = 128, patience: int = 5) -> dict:
    """Train one (leads, target_fs) config inside the container.

    Saves the best checkpoint to /ptbxl/best_resnet_<leads>_<fs>hz.pt.
    """
    import subprocess
    import sys
    import shutil

    tag = f"{leads}_{target_fs}hz"
    output_dir = f"/tmp/resnet_{tag}"
    ckpt_name = f"best_resnet_{tag}.pt"

    cmd = [
        sys.executable,
        "/repo/qt_finetune/train_resnet_qt.py",
        "--leads", leads,
        "--target-fs", str(target_fs),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--patience", str(patience),
        "--num-workers", "8",
        "--ptbxl-dir", SIGNALS_DIR,
        "--labels-csv", LABELS_CSV,
        "--output-dir", output_dir,
        "--ckpt-name", ckpt_name,
        "--run-name", f"resnet_{tag}",
        "--wandb-project", "anchoredecg-qt-resnet-apple",
    ]
    subprocess.run(cmd, check=True)

    src = f"{output_dir}/{ckpt_name}"
    dst = f"/ptbxl/{ckpt_name}"
    shutil.copy2(src, dst)
    ptbxl_vol.commit()
    print(f"\n[{tag}] checkpoint committed to volume at {dst}")

    import torch
    ckpt = torch.load(dst, map_location="cpu", weights_only=False)
    return {
        "leads": leads,
        "target_fs": target_fs,
        "input_len": ckpt.get("input_len"),
        "epoch": ckpt.get("epoch"),
        "val_mae": ckpt.get("val_mae"),
        "val_r": ckpt.get("val_r"),
        "ckpt": dst,
    }


# NOTE: exactly ONE local entrypoint on purpose. Modal only auto-selects an
# entrypoint when there is a single one, so a bare `modal run <file>` works.
@app.local_entrypoint()
def main(rates: str = ",".join(map(str, RATES)), leads: str = "lead1",
         epochs: int = 50, batch_size: int = 128, patience: int = 5):
    """Train one rate per H100, in parallel.

    Usage:
        modal run --detach modal_resnet_qt_apple.py                 # both rates
        modal run --detach modal_resnet_qt_apple.py --rates 512     # just 512
        modal run --detach modal_resnet_qt_apple.py --rates 128 --leads all12
    """
    fs_list = [int(r) for r in str(rates).replace(" ", "").split(",") if r]

    print(f"Launching {len(fs_list)} ResNet-18 QT job(s) in parallel "
          f"on Modal (H100 each), leads={leads}:")
    for fs in fs_list:
        note = ("Apple Watch native — user data untouched" if fs == 512 else
                "exact /4 of the watch rate" if fs == 128 else "PTB-XL native")
        print(f"  * {fs:>4} Hz — {note}")
    print()

    # Batch size is held equal across rates so the comparison is clean, even
    # though 128 Hz could afford a larger one.
    handles = [(fs, train_resnet.spawn(leads, fs, epochs, batch_size, patience))
               for fs in fs_list]

    print("All jobs spawned. Waiting (Ctrl-C is safe with --detach) ...\n")
    results = [(fs, h.get()) for fs, h in handles]

    print("\n=== Done ===")
    for fs, r in results:
        print(f"  {fs:>4} Hz (L={r['input_len']}): best val MAE={r['val_mae']:.2f} ms  "
              f"R={r['val_r']:.3f}  (epoch {r['epoch']})")
    print("\nDownload checkpoints:")
    for fs, r in results:
        print(f"  modal volume get ptbxl-data /{Path(r['ckpt']).name} "
              f"qt_finetune/checkpoints/")
