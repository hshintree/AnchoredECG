"""
Modal app for JEPA end-to-end QT fine-tuning on a GPU cloud container.

IMPORTANT — two-step workflow:

  Step 1 (once, or after any code change):
      modal deploy modal_jepa.py

  Step 2 (trigger training — terminal can be closed immediately after):
      modal run modal_jepa.py::train_linear_lqt_sweep   # λ_QT sweep
      modal run modal_jepa.py::train_linear              # single backbone
      modal run modal_jepa.py::train_all                 # all 5 backbones

Why deploy first?  `modal run` creates an ephemeral app run; .spawn() calls
within it target the ephemeral copies of functions and are cancelled when the
run ends.  By calling modal.Function.from_name() in local_entrypoints we
explicitly target the deployed (persistent) function, so spawned jobs live in
the deployed app and are never cancelled when the local run exits.

Monitor:   modal app list  →  modal app logs <app-id>
Download:  modal run modal_jepa.py::download_checkpoints
Eval:      modal run modal_jepa.py::eval_all
"""

from __future__ import annotations
from pathlib import Path
import modal

REPO_ROOT = Path(__file__).resolve().parent
APP_NAME  = "anchoredecg-jepa"

# ---------------------------------------------------------------------------
# Image + volume
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.0", "torchvision==0.18.0", "torchaudio==2.3.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install("torch_geometric==2.5.3")
    .pip_install(
        "torch_scatter", "torch_sparse", "torch_cluster", "torch_spline_conv",
        find_links="https://data.pyg.org/whl/torch-2.3.0+cu121.html",
    )
    .pip_install("wfdb", "pandas", "scipy", "tqdm", "numpy", "scikit-learn")
    .pip_install("matplotlib", "seaborn")
    .pip_install("wandb")
)

ptbxl_vol = modal.Volume.from_name("ptbxl-data", create_if_missing=True)

training_image = (
    image
    .add_local_dir(str(REPO_ROOT / "models"),      "/repo/models")
    .add_local_dir(str(REPO_ROOT / "qt_finetune"), "/repo/qt_finetune")
    .add_local_dir(str(REPO_ROOT / "checkpoints"), "/repo/checkpoints")
)

app = modal.App(APP_NAME)

# ---------------------------------------------------------------------------
# Helper: get the deployed train function so .spawn() targets the persistent
# app rather than the ephemeral modal run context.
# ---------------------------------------------------------------------------

def _get_deployed_train_fn():
    """Return a reference to the deployed train_jepa_fn."""
    try:
        return modal.Function.from_name(APP_NAME, "train_jepa_fn")
    except modal.exception.NotFoundError:
        print(f"\nERROR: '{APP_NAME}' is not deployed yet.")
        print("Run this first:  modal deploy modal_jepa.py")
        print("Then retry your command.")
        raise


# ---------------------------------------------------------------------------
# Core GPU training function (deployed via `modal deploy modal_jepa.py`)
# ---------------------------------------------------------------------------

@app.function(
    image=training_image,
    gpu="A100",
    volumes={"/ptbxl": ptbxl_vol},
    secrets=[modal.Secret.from_name("wandb")],
    timeout=8 * 3600,
    memory=32768,
    cpu=8,
    retries=5,   # auto-restart after preemption; resume ckpt in volume picks up from last eval
)
def train_jepa_fn(
    predictor_name: str,
    warmup_epochs:   int   = 20,
    finetune_epochs: int   = 80,
    eval_every:      int   = 10,
    lambda_jepa:     float = 1.0,
    lambda_qt:       float = 1.0,
    ema_tau:         float = 0.999,
    lr_encoder:      float = 5e-5,
    lr_predictor:    float = 1e-3,
    batch_size:      int   = 32,
    force_retrain:   bool  = False,
    run_name:        str   = "",
) -> None:
    """Train one JEPA predictor variant. Checkpoint committed to volume on completion."""
    import subprocess, sys, os, time, shutil
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    jepa_vol_dir = "/ptbxl/jepa"
    os.makedirs(jepa_vol_dir, exist_ok=True)

    ckpt_stem   = run_name if run_name else predictor_name
    ckpt_dst    = f"{jepa_vol_dir}/jepa_{ckpt_stem}.pt"
    resume_dst  = f"{jepa_vol_dir}/jepa_{ckpt_stem}_resume.pt"

    # Training is truly complete only when the final checkpoint exists AND the
    # resume file has been deleted (train_jepa.py removes it on clean exit).
    # If both exist, training is still in progress — the checkpoint was saved
    # mid-run when val MAE first improved but hasn't finished yet.
    training_complete = os.path.exists(ckpt_dst) and not os.path.exists(resume_dst)
    if training_complete and not force_retrain:
        print(f"[train_jepa_fn] Training complete: {ckpt_dst} — skipping.")
        return
    if os.path.exists(ckpt_dst) and os.path.exists(resume_dst) and not force_retrain:
        print(f"[train_jepa_fn] Mid-run checkpoint found — resuming training for {ckpt_stem}.")

    # ── Local SSD cache for WFDB files ──────────────────────────────────────
    # Modal Volumes use a network filesystem.  On a cold container, reading
    # 17 k WFDB files one-at-a-time incurs ~50-70 ms of network latency per
    # file → 15-20 minutes for the very first epoch.  Copying everything to
    # the container's local NVMe (/tmp) with 16 parallel threads reduces that
    # to ~2-3 minutes, and every subsequent epoch then reads at SSD speed.
    #
    # Cost: paid once per container lifetime (2-3 min).
    # Benefit: all epochs after the copy run at full 24 s/epoch, and any
    #          preemption + retry only repeats the 2-3 min copy, not 18 min.
    VOL_SIGNALS = Path(
        "/ptbxl/ptbxl_signals/"
        "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
    )
    LOCAL_CACHE  = Path("/tmp/ptbxl_cache")
    LOCAL_SIGNALS = LOCAL_CACHE / VOL_SIGNALS.name

    # Try to build the local SSD cache; fall back to the volume path if anything
    # goes wrong so a caching error never kills training.
    ptbxl_dir = str(VOL_SIGNALS)   # default: network volume (always works)
    if LOCAL_SIGNALS.exists():
        print(f"[train_jepa_fn] Warm SSD cache found → {LOCAL_SIGNALS}")
        ptbxl_dir = str(LOCAL_SIGNALS)
    else:
        try:
            print(f"[train_jepa_fn] Caching {VOL_SIGNALS.name} to local SSD …")
            t_copy = time.time()
            LOCAL_CACHE.mkdir(parents=True, exist_ok=True)

            all_files = [f for f in VOL_SIGNALS.rglob("*") if f.is_file()]
            print(f"[train_jepa_fn]   {len(all_files):,} files to copy with 16 threads …")

            def _copy_one(src_file: Path) -> None:
                dst = LOCAL_SIGNALS / src_file.relative_to(VOL_SIGNALS)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst)

            with ThreadPoolExecutor(max_workers=16) as pool:
                list(pool.map(_copy_one, all_files))

            elapsed = time.time() - t_copy
            size_mb = sum(
                f.stat().st_size for f in LOCAL_SIGNALS.rglob("*") if f.is_file()
            ) / 1e6
            print(
                f"[train_jepa_fn]   SSD cache ready: {size_mb:.0f} MB "
                f"in {elapsed:.1f} s ({elapsed/60:.1f} min)"
            )
            ptbxl_dir = str(LOCAL_SIGNALS)
        except Exception as e:
            print(f"[train_jepa_fn] WARNING: SSD cache failed ({e}), falling back to volume path.")
            ptbxl_dir = str(VOL_SIGNALS)

    cmd = [
        sys.executable, "/repo/qt_finetune/train_jepa.py",
        "--predictor",       predictor_name,
        "--checkpoint",      "/ptbxl/best_qt_model.pt",
        "--ptbxl-dir",       ptbxl_dir,   # ← local NVMe if cache succeeded, else volume
        "--labels-csv",      "/repo/qt_finetune/qt_labels.csv",
        "--jepa-dir",        jepa_vol_dir,
        "--warmup-epochs",   str(warmup_epochs),
        "--finetune-epochs", str(finetune_epochs),
        "--eval-every",      str(eval_every),
        "--lambda-jepa",     str(lambda_jepa),
        "--lambda-qt",       str(lambda_qt),
        "--ema-tau",         str(ema_tau),
        "--lr-encoder",      str(lr_encoder),
        "--lr-predictor",    str(lr_predictor),
        "--batch-size",      str(batch_size),
    ]
    if force_retrain:
        cmd.append("--force-retrain")
    if run_name:
        cmd += ["--run-name", run_name]

    print(f"[train_jepa_fn] JEPA-{predictor_name.upper()}  run={ckpt_stem}")
    print(f"  λ_QT={lambda_qt}  λ_JEPA={lambda_jepa}  "
          f"warmup={warmup_epochs}  finetune={finetune_epochs}  bs={batch_size}")
    subprocess.run(cmd, check=True)

    ptbxl_vol.commit()
    print(f"\n[train_jepa_fn] Checkpoint committed → {ckpt_dst}")


# ---------------------------------------------------------------------------
# Local entrypoints — use _get_deployed_train_fn().spawn() so jobs run in
# the persistent deployed app, not the ephemeral modal run context.
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def respawn_all():
    """Respawn all three λ_QT runs from their last saved epoch.

    Safe to run even if some are still running — each job checks its own
    resume checkpoint.  Terminal can be closed immediately after.

    modal run modal_jepa.py::respawn_all
    """
    deployed = _get_deployed_train_fn()
    sweep = [(0.5, "linear_lqt0.5"), (1.0, "linear_lqt1.0"), (2.0, "linear_lqt2.0")]
    for lqt, run_name in sweep:
        deployed.spawn(
            predictor_name="linear",
            lambda_qt=lqt,
            lambda_jepa=1.0,
            run_name=run_name,
        )
        print(f"  Spawned {run_name}")
    print("\nAll 3 runs spawned in deployed app. Safe to close terminal.")
    print("Monitor:  modal container list")
    print("Status:   modal volume ls ptbxl-data /jepa/")


@app.local_entrypoint()
def train_linear_lqt_sweep():
    """Sweep λ_QT ∈ {0.5, 1.0, 2.0} for the Linear predictor (3 parallel A10G jobs).

    Deploy first, then run — close terminal immediately after:
        modal deploy modal_jepa.py
        modal run modal_jepa.py::train_linear_lqt_sweep

    Saves:  jepa_linear_lqt0.5.pt / jepa_linear_lqt1.0.pt / jepa_linear_lqt2.0.pt

    After training:
        modal run modal_jepa.py::download_checkpoints
        python qt_finetune/train_jepa.py --predictor linear --run-name linear_lqt0.5 --eval-only
        python qt_finetune/train_jepa.py --predictor linear --run-name linear_lqt1.0 --eval-only
        python qt_finetune/train_jepa.py --predictor linear --run-name linear_lqt2.0 --eval-only
    """
    deployed = _get_deployed_train_fn()
    sweep = [(0.5, "linear_lqt0.5"), (1.0, "linear_lqt1.0"), (2.0, "linear_lqt2.0")]
    for lqt, run_name in sweep:
        deployed.spawn(
            predictor_name="linear",
            lambda_qt=lqt,
            lambda_jepa=1.0,
            run_name=run_name,
        )
        print(f"  Spawned {run_name}  (λ_QT={lqt})")
    print(f"\n{len(sweep)} jobs running in parallel on Modal. Safe to close terminal.")
    print("Monitor:  modal app list  →  modal app logs <app-id>")
    print("Download: modal run modal_jepa.py::download_checkpoints")


@app.local_entrypoint()
def respawn_lqt05():
    """Respawn linear_lqt0.5 ONLY — safe while lqt1.0 and lqt2.0 are running.

    Detects the existing _resume.pt and picks up from where it crashed.

    modal run modal_jepa.py::respawn_lqt05
    """
    _get_deployed_train_fn().spawn(
        predictor_name="linear",
        lambda_qt=0.5,
        lambda_jepa=1.0,
        run_name="linear_lqt0.5",
    )
    print("linear_lqt0.5 respawned — will resume from last saved epoch.")
    print("lqt1.0 and lqt2.0 are unaffected.")


@app.local_entrypoint()
def respawn_lqt10():
    """Respawn linear_lqt1.0 ONLY — safe while lqt0.5 and lqt2.0 are running.

    Detects the existing _resume.pt and picks up from where it crashed.

    modal run modal_jepa.py::respawn_lqt10
    """
    _get_deployed_train_fn().spawn(
        predictor_name="linear",
        lambda_qt=1.0,
        lambda_jepa=1.0,
        run_name="linear_lqt1.0",
    )
    print("linear_lqt1.0 respawned — will resume from last saved epoch.")
    print("lqt0.5 and lqt2.0 are unaffected.")


@app.local_entrypoint()
def respawn_lqt20():
    """Respawn linear_lqt2.0 ONLY — safe while lqt0.5 and lqt1.0 are running.

    Detects the existing _resume.pt and picks up from where it crashed.

    modal run modal_jepa.py::respawn_lqt20
    """
    _get_deployed_train_fn().spawn(
        predictor_name="linear",
        lambda_qt=2.0,
        lambda_jepa=1.0,
        run_name="linear_lqt2.0",
    )
    print("linear_lqt2.0 respawned — will resume from last saved epoch.")
    print("lqt0.5 and lqt1.0 are unaffected.")


@app.local_entrypoint()
def train_linear():
    """Train JEPA-Linear (default λ_QT=1, λ_JEPA=1).

    modal deploy modal_jepa.py   # if not already deployed
    modal run modal_jepa.py::train_linear
    """
    _get_deployed_train_fn().spawn(predictor_name="linear")
    print("JEPA-Linear spawned in deployed app. Safe to close terminal.")
    print("Download: modal volume get ptbxl-data /jepa/jepa_linear.pt "
          "qt_finetune/checkpoints/jepa/jepa_linear.pt")


@app.local_entrypoint()
def train_mlp():
    """Train JEPA-MLP.  modal run modal_jepa.py::train_mlp"""
    _get_deployed_train_fn().spawn(predictor_name="mlp")
    print("JEPA-MLP spawned in deployed app. Safe to close terminal.")


@app.local_entrypoint()
def train_attention():
    """Train JEPA-Attention.  modal run modal_jepa.py::train_attention"""
    _get_deployed_train_fn().spawn(predictor_name="attention")
    print("JEPA-Attention spawned in deployed app. Safe to close terminal.")


@app.local_entrypoint()
def train_lora():
    """Train JEPA-LoRA.  modal run modal_jepa.py::train_lora"""
    _get_deployed_train_fn().spawn(predictor_name="lora")
    print("JEPA-LoRA spawned in deployed app. Safe to close terminal.")


@app.local_entrypoint()
def train_grelu():
    """Train JEPA-GReLU.  modal run modal_jepa.py::train_grelu"""
    _get_deployed_train_fn().spawn(predictor_name="grelu")
    print("JEPA-GReLU spawned in deployed app. Safe to close terminal.")


@app.local_entrypoint()
def train_all():
    """Train all 5 original JEPA backbones in parallel.  modal run modal_jepa.py::train_all"""
    deployed  = _get_deployed_train_fn()
    predictors = ["linear", "mlp", "attention", "lora", "grelu"]
    for pname in predictors:
        deployed.spawn(predictor_name=pname)
        print(f"  Spawned JEPA-{pname.upper()}")
    print(f"\n{len(predictors)} jobs running in parallel. Safe to close terminal.")
    print("Download: modal run modal_jepa.py::download_checkpoints")


@app.local_entrypoint()
def respawn_cgrelu_original():
    """Resume the cgrelu (m=1024) run from its Phase-2 resume checkpoint.

    The run was stopped at finetune epoch 35/120 with a resume checkpoint on
    the volume.  train_jepa_fn detects phase=finetune in the resume file and
    skips warmup entirely, picking up from epoch 36 → 120.

    modal deploy modal_jepa.py
    modal run modal_jepa.py::respawn_cgrelu_original
    """
    _get_deployed_train_fn().spawn(
        predictor_name="cgrelu",
        warmup_epochs=50,
        finetune_epochs=120,
    )
    print("JEPA-CvxGReLU (m=1024) respawned — will resume from finetune epoch 36.")
    print("Download: modal volume get ptbxl-data /jepa/jepa_cgrelu.pt "
          "qt_finetune/checkpoints/jepa/jepa_cgrelu.pt")


@app.local_entrypoint()
def train_rwna():
    """Train JEPA-RWNA (k=16, nonlinear rank-16 predictor).

    Tests whether a nonlinear rank-16 bottleneck breaks the rank-2 collapse
    observed in JEPA-Linear (where linear rank-2 gradients kept z_c degenerate).

    modal deploy modal_jepa.py
    modal run modal_jepa.py::train_rwna
    """
    _get_deployed_train_fn().spawn(
        predictor_name="rwna",
        warmup_epochs=20,
        finetune_epochs=80,
    )
    print("JEPA-RWNA (k=16) spawned. Safe to close terminal.")
    print("Download: modal volume get ptbxl-data /jepa/jepa_rwna.pt "
          "qt_finetune/checkpoints/jepa/jepa_rwna.pt")


@app.local_entrypoint()
def train_lora16():
    """Train JEPA-LoRA16 (rank=16, linear ablation partner to RWNA).

    Same rank-16 bottleneck as RWNA but linear — isolates whether the
    nonlinearity inside the k-dim space is what matters, vs the rank alone.

    modal deploy modal_jepa.py
    modal run modal_jepa.py::train_lora16
    """
    _get_deployed_train_fn().spawn(
        predictor_name="lora16",
        warmup_epochs=20,
        finetune_epochs=80,
    )
    print("JEPA-LoRA16 (rank=16) spawned. Safe to close terminal.")
    print("Download: modal volume get ptbxl-data /jepa/jepa_lora16.pt "
          "qt_finetune/checkpoints/jepa/jepa_lora16.pt")


@app.local_entrypoint()
def train_cgrelu():
    """Train JEPA-CvxGReLU-512 and JEPA-CvxGReLU-256 in parallel.

    Both use the same convex-GReLU training constraints (vs plain JEPA-GReLU):
      1. proj.weight clamped >= 0 after every optimizer step — guarantees the
         network is a convex function of z_c (non-negative output weights over
         ReLU-gated features).
      2. L1 penalty (1e-4) on linear1.weight added to the loss — sparsity prior
         that mimics R-FISTA's proximal soft-threshold in the frozen adapter.

    Neuron-count ablation:
      cgrelu512  →  m=512  (~197K trainable params)
      cgrelu256  →  m=256  (~98K  trainable params)

    Epoch schedule: warmup=50 + finetune=120 = 170 total.
    More than the 100 used for RWNA/LoRA16; the convex projection + L1 constraint
    slow early convergence and benefit from a longer search.

    modal deploy modal_jepa.py
    modal run modal_jepa.py::train_cgrelu
    """
    deployed = _get_deployed_train_fn()
    for pname in ["cgrelu512", "cgrelu256"]:
        deployed.spawn(
            predictor_name=pname,
            warmup_epochs=50,
            finetune_epochs=120,
        )
        print(f"  Spawned JEPA-{pname.upper()}")
    print("\n2 CvxGReLU jobs running in parallel on A100. Safe to close terminal.")
    print("Download when done:")
    print("  modal volume get ptbxl-data /jepa/jepa_cgrelu512.pt "
          "qt_finetune/checkpoints/jepa/jepa_cgrelu512.pt")
    print("  modal volume get ptbxl-data /jepa/jepa_cgrelu256.pt "
          "qt_finetune/checkpoints/jepa/jepa_cgrelu256.pt")


@app.local_entrypoint()
def train_rank16_ablation():
    """Train JEPA-RWNA and JEPA-LoRA16 in parallel (the rank-16 ablation pair).

    Together these answer: at the same bottleneck rank (16), does nonlinearity help?
    Both use warmup=20, finetune=80 (reduced from 50+150 based on convergence analysis).

    modal deploy modal_jepa.py
    modal run modal_jepa.py::train_rank16_ablation
    """
    deployed = _get_deployed_train_fn()
    for pname in ["rwna", "lora16"]:
        deployed.spawn(
            predictor_name=pname,
            warmup_epochs=20,
            finetune_epochs=80,
        )
        print(f"  Spawned JEPA-{pname.upper()}")
    print("\n2 jobs running in parallel. Safe to close terminal.")
    print("Monitor: modal app list  →  modal container list")
    print("Download: modal run modal_jepa.py::download_checkpoints")


# ---------------------------------------------------------------------------
# Read-only helpers (.remote() fine here — fast, user waits for output)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={"/ptbxl": ptbxl_vol},
    timeout=60,
    memory=512,
    cpu=1,
)
def list_jepa_checkpoints() -> list:
    import os
    jepa_dir = "/ptbxl/jepa"
    if not os.path.isdir(jepa_dir):
        return []
    return sorted(f for f in os.listdir(jepa_dir) if f.endswith(".pt"))


@app.local_entrypoint()
def download_checkpoints():
    """Print modal volume get commands for all JEPA checkpoints.

    modal run modal_jepa.py::download_checkpoints
    """
    import pathlib
    found = list_jepa_checkpoints.remote()
    if not found:
        print("No JEPA checkpoints in volume yet.")
        print("Check training status:  modal app list")
        return
    local_dir = pathlib.Path("qt_finetune/checkpoints/jepa")
    print(f"\nFound {len(found)} checkpoint(s):\n")
    for fname in found:
        print(f"  modal volume get ptbxl-data /jepa/{fname} {local_dir / fname}")
    print("\nAfter downloading:\n  python qt_finetune/analyze_qtc.py --split val")


@app.function(
    image=training_image,
    volumes={"/ptbxl": ptbxl_vol},
    timeout=3600,
    memory=16384,
    cpu=4,
)
def eval_jepa_fn(predictor_name: str, run_name: str = "", split: str = "val") -> None:
    import subprocess, sys, os
    stem = run_name if run_name else predictor_name
    if not os.path.exists(f"/ptbxl/jepa/jepa_{stem}.pt"):
        print(f"No checkpoint for {stem}.")
        return
    cmd = [
        sys.executable, "/repo/qt_finetune/train_jepa.py",
        "--predictor",  predictor_name,
        "--checkpoint", "/ptbxl/best_qt_model.pt",
        "--ptbxl-dir",  "/ptbxl/ptbxl_signals/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1",
        "--labels-csv", "/repo/qt_finetune/qt_labels.csv",
        "--jepa-dir",   "/ptbxl/jepa",
        "--split",      split,
        "--eval-only",
    ]
    if run_name:
        cmd += ["--run-name", run_name]
    subprocess.run(cmd, check=True)


@app.local_entrypoint()
def eval_all():
    """Eval all JEPA checkpoints on val set.  Keep terminal open for output.

    modal run modal_jepa.py::eval_all
    """
    found = list_jepa_checkpoints.remote()
    if not found:
        print("No checkpoints found.")
        return
    print(f"\nEvaluating {len(found)} checkpoint(s)...\n")
    for fname in found:
        stem      = fname.replace("jepa_", "").replace(".pt", "")
        predictor = stem.split("_")[0]
        run_name  = stem if stem != predictor else ""
        eval_jepa_fn.remote(predictor_name=predictor, run_name=run_name)
