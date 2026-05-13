"""
Bundle train/val embeddings + regression head for Hakeem's repo.

Run AFTER extract_embeddings.py has produced embeddings.npz.

Usage (from project root):
    python ee364b_stress_ecg/hakeem_grelu_adapter/bundle.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_EE364B_DIR = _SCRIPT_DIR.parent
_PROJECT_ROOT = _EE364B_DIR.parent

EMBEDDINGS_PATH = _EE364B_DIR / "adapter" / "results" / "embeddings.npz"
CKPT_PATH = _EE364B_DIR / "checkpoints" / "best_st80_model.pt"
DATA_DIR = _SCRIPT_DIR / "data"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # -- load full embeddings --
    assert EMBEDDINGS_PATH.exists(), (
        f"Run extract_embeddings.py first. Expected: {EMBEDDINGS_PATH}")
    d = np.load(str(EMBEDDINGS_PATH), allow_pickle=True)
    splits = d["splits"]

    for split_name in ["train", "val"]:
        m = splits == split_name
        out = DATA_DIR / f"{split_name}.npz"
        np.savez(
            out,
            z_rest=d["z_rest"][m],
            z_stress=d["z_stress"][m],
            phi_stress=d["phi_stress"][m],
            bl_st80=d["bl_st80"][m],
            ischemia=d["ischemia"][m],
        )
        n = m.sum()
        y = d["ischemia"][m].sum()
        print(f"  {split_name}: {n} patients ({y} ischemic) -> {out.name}")

    # -- extract regression head --
    ckpt = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
    head_sd = {
        k.replace("classifier.", ""): v
        for k, v in ckpt["model_state_dict"].items()
        if k.startswith("classifier.")
    }
    head_path = DATA_DIR / "regression_head.pt"
    torch.save(head_sd, head_path)
    print(f"  Regression head -> {head_path.name}")

    print(f"\nDone. Upload {DATA_DIR}/ to the repo.")
    print("NOTE: test split is intentionally excluded.")


if __name__ == "__main__":
    main()
