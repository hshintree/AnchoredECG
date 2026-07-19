"""
Modal app for the tail-robust QT ResNet — the retrain that replaces the
MSE checkpoints.

Why: the MSE baselines were accurate in the normal range (~11 ms MAE below
450 ms) but collapsed exactly where it matters — mean error -37 ms in the
450-500 band, -59 ms above 500, pred~ref slope 0.71, and 0/5 sensitivity at the
QTc>500 torsades threshold. The discrimination was there (ranking AUC 0.85);
the point estimate was simply shrunk toward the training mean.

Two fixes, applied together:
  * pinball (quantile) loss  — median point estimate instead of a conditional
    mean, so the objective stops rewarding hedging; the outer quantiles also
    give a prediction interval, and an upper quantile gives a far more
    sensitive prolongation flag than the median ever could.
  * inverse-frequency tail weighting — QT>450 is 7.7% of PTB-XL folds 1-8 and
    QT>500 is 0.94%. Nothing is discarded: the full 17,418-record training set
    is retained and only re-weighted.

This is a 2x2 over rate and tail-upweighting method, one H100 per cell:

    rate       512  Apple Watch native; user data needs no resampling at all
               128  exact /4 of the watch rate; 4x shorter sequences.
                    128 won the MSE ablation on both PTB-XL (13.01 vs 13.31 ms)
                    and LUDB (18.29 vs 19.11 ms), but 512 is kept trained so
                    the option to wire it in stays open.

    variant    tailloss     per-sample loss weights; every record still seen
                            once per epoch (gentler, less prone to memorising
                            the ~163 extreme records)
               tailsampler  WeightedRandomSampler; tail records drawn more often

Run:
    modal run --detach modal_resnet_qt_robust.py                      # all four
    modal run --detach modal_resnet_qt_robust.py --rates 128          # one rate
    modal run --detach modal_resnet_qt_robust.py --variants tailloss  # one method
    modal run --detach modal_resnet_qt_robust.py --rates 512 --variants tailloss

Checkpoints land in the ptbxl-data volume as:
    /ptbxl/best_resnet_lead1_<fs>hz_pinball_<variant>.pt
so the existing MSE checkpoints are never overwritten.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent

RATES = (512, 128)                    # both kept, so 512 stays an option
QUANTILES = "0.05,0.25,0.5,0.75,0.95"

# variant -> --tail-weighting value
VARIANTS = {"tailloss": "loss", "tailsampler": "sampler"}

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

app = modal.App("anchoredecg-qt-resnet-robust")

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
def train_resnet(variant: str, target_fs: int, leads: str = "lead1",
                 epochs: int = 50, batch_size: int = 128, patience: int = 10,
                 tail_alpha: float = 0.5, tail_clip: float = 10.0) -> dict:
    import subprocess
    import sys
    import shutil

    tail_mode = VARIANTS[variant]
    tag = f"{leads}_{target_fs}hz_pinball_{variant}"
    output_dir = f"/tmp/resnet_{tag}"
    ckpt_name = f"best_resnet_{tag}.pt"

    cmd = [
        sys.executable,
        "/repo/qt_finetune/train_resnet_qt.py",
        "--leads", leads,
        "--target-fs", str(target_fs),
        "--loss", "pinball",
        "--quantiles", QUANTILES,
        "--tail-weighting", tail_mode,
        "--tail-alpha", str(tail_alpha),
        "--tail-clip", str(tail_clip),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--patience", str(patience),
        "--num-workers", "8",
        "--ptbxl-dir", SIGNALS_DIR,
        "--labels-csv", LABELS_CSV,
        "--output-dir", output_dir,
        "--ckpt-name", ckpt_name,
        "--run-name", f"resnet_{tag}",
        "--wandb-project", "anchoredecg-qt-resnet-robust",
    ]
    subprocess.run(cmd, check=True)

    src, dst = f"{output_dir}/{ckpt_name}", f"/ptbxl/{ckpt_name}"
    shutil.copy2(src, dst)
    ptbxl_vol.commit()
    print(f"\n[{tag}] checkpoint committed to volume at {dst}")

    import torch
    ck = torch.load(dst, map_location="cpu", weights_only=False)
    return {
        "variant": variant,
        "tail_weighting": tail_mode,
        "target_fs": target_fs,
        "input_len": ck.get("input_len"),
        "epoch": ck.get("epoch"),
        "val_mae": ck.get("val_mae"),
        "val_r": ck.get("val_r"),
        "val_coverage": ck.get("val_coverage"),
        "ckpt": dst,
    }


# Exactly ONE local entrypoint so a bare `modal run <file>` works.
@app.local_entrypoint()
def main(rates: str = ",".join(map(str, RATES)),
         variants: str = ",".join(VARIANTS), leads: str = "lead1",
         epochs: int = 50, batch_size: int = 128, patience: int = 10,
         tail_alpha: float = 0.5, tail_clip: float = 10.0):
    """Train the rate x tail-method grid, one H100 per cell, in parallel."""
    fs_list = [int(r) for r in str(rates).replace(" ", "").split(",") if r]
    names = [v for v in str(variants).replace(" ", "").split(",") if v]
    bad = [v for v in names if v not in VARIANTS]
    if bad:
        raise SystemExit(f"unknown variant(s) {bad}; choose from {list(VARIANTS)}")

    cells = list(product(fs_list, names))
    print(f"Tail-robust QT retrain — pinball {QUANTILES}, leads={leads}")
    print(f"{len(cells)} job(s), one H100 each:")
    for fs, v in cells:
        note = "Apple Watch native" if fs == 512 else "exact /4 of watch rate"
        print(f"  * {fs:>4} Hz ({note:<22}) x {v:<12} "
              f"--tail-weighting {VARIANTS[v]}")
    print()

    # Batch size is held equal across cells so the comparison stays clean.
    handles = [((fs, v), train_resnet.spawn(v, fs, leads, epochs, batch_size,
                                            patience, tail_alpha, tail_clip))
               for fs, v in cells]
    print("Spawned. Waiting (Ctrl-C is safe with --detach) ...\n")
    results = [(k, h.get()) for k, h in handles]

    print("\n=== Done ===")
    print(f"  {'rate':>6} {'variant':<13} {'val MAE':>9} {'R':>7} {'cov':>6} {'epoch':>6}")
    for (fs, v), r in results:
        cov = r.get("val_coverage")
        print(f"  {fs:>4}Hz {v:<13} {r['val_mae']:>8.2f}ms {r['val_r']:>7.3f} "
              f"{(f'{cov*100:.0f}%' if cov else '-'):>6} {r['epoch']:>6}")

    print("\nDownload, then revalidate against human annotations:")
    for (fs, v), r in results:
        print(f"  modal volume get ptbxl-data /{Path(r['ckpt']).name} "
              f"qt_finetune/checkpoints/")
    print()
    for (fs, v), r in results:
        print(f"  python qt_finetune/validate_human_qt.py "
              f"--ckpt qt_finetune/checkpoints/{Path(r['ckpt']).name} "
              f"--local-dir <ludb>")
