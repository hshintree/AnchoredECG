"""
Evaluate adapters on the QT prediction task.

Setup:
  - z_full = encoder(12-lead ECG)  -- the target embedding
  - phi = encoder(reduced leads)   -- what the adapter observes
  - Adapter: z_hat = f(phi) such that z_hat ~ z_full
  - Decode: QT_pred = regression_head(z_hat)

Metrics:
  - Latent: MSE(z_hat, z_full), cosine similarity
  - QT: MAE(QT_pred, QT_gt), Pearson R, stratified by prolonged QT

Includes baseline comparisons:
  - Oracle: pass z_full through head (upper bound)
  - Direct reduced: pass phi directly through head (lower bound)
  - Linear adapter (Ridge): W @ phi ~ z_full

Usage:
    set KMP_DUPLICATE_LIB_OK=TRUE
    python qt_finetune/qt_adapter_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge

_SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = _SCRIPT_DIR / "results"


def load_qt_head(head_path):
    head = nn.Sequential(
        nn.Linear(199, 64), nn.Mish(inplace=True), nn.Linear(64, 1),
    )
    head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=False))
    head.eval()
    return head


def decode_qt(head, z):
    z_t = torch.tensor(z, dtype=torch.float32)
    pad = torch.zeros(z_t.shape[0], 7)
    fused = torch.cat([z_t, pad], dim=1)
    with torch.no_grad():
        return head(fused).numpy().flatten()


def main():
    emb = np.load(str(RESULTS_DIR / "embeddings.npz"), allow_pickle=True)
    head = load_qt_head(RESULTS_DIR / "qt_regression_head.pt")

    folds = emb["strat_fold"]
    train_mask = folds <= 8
    val_mask = folds == 9

    qt_gt_tr = emb["qt_gt"][train_mask]
    qt_gt_va = emb["qt_gt"][val_mask]

    z_full_tr = emb["z_full_12"][train_mask]
    z_full_va = emb["z_full_12"][val_mask]

    configs_to_test = ["lead_II", "lead_V5", "limb_6", "precord_6"]

    print(f"Train: {train_mask.sum()} records, Val: {val_mask.sum()} records")
    print(f"QT range (val): [{qt_gt_va.min():.0f}, {qt_gt_va.max():.0f}] ms")
    print(f"Prolonged QT (val, >450ms): {(qt_gt_va > 450).sum()} / {len(qt_gt_va)}")

    # Oracle
    qt_oracle = decode_qt(head, z_full_va)
    oracle_mae = np.abs(qt_oracle - qt_gt_va).mean()
    oracle_r, _ = pearsonr(qt_oracle, qt_gt_va)

    hdr = f"{'Method':<30s} {'QT MAE (ms)':>12s} {'Pearson R':>10s} {'Latent MSE':>11s} {'Cos Sim':>8s}"
    print(f"\n{hdr}")
    print("-" * len(hdr))
    print(f"{'Oracle (12-lead)':<30s} {oracle_mae:12.2f} {oracle_r:10.3f} {'--':>11s} {'--':>8s}")

    for config in configs_to_test:
        phi_tr = emb[f"z_{config}"][train_mask]
        phi_va = emb[f"z_{config}"][val_mask]

        # Direct: pass reduced embedding through head
        qt_direct = decode_qt(head, phi_va)
        direct_mae = np.abs(qt_direct - qt_gt_va).mean()
        direct_r, _ = pearsonr(qt_direct, qt_gt_va)

        # Linear adapter (Ridge): W @ phi ~ z_full
        ridge = Ridge(alpha=1.0)
        ridge.fit(phi_tr, z_full_tr)
        z_hat_va = ridge.predict(phi_va)

        lat_mse = np.mean(np.sum((z_hat_va - z_full_va) ** 2, axis=1))
        norms = np.linalg.norm(z_hat_va, axis=1) * np.linalg.norm(z_full_va, axis=1) + 1e-8
        cos = np.mean(np.sum(z_hat_va * z_full_va, axis=1) / norms)

        qt_adapted = decode_qt(head, z_hat_va)
        adapted_mae = np.abs(qt_adapted - qt_gt_va).mean()
        adapted_r, _ = pearsonr(qt_adapted, qt_gt_va)

        print(f"{'Direct ' + config:<30s} {direct_mae:12.2f} {direct_r:10.3f} {'--':>11s} {'--':>8s}")
        print(f"{'Ridge ' + config:<30s} {adapted_mae:12.2f} {adapted_r:10.3f} {lat_mse:11.2f} {cos:8.3f}")

    print("\nDone. Replace Ridge with your convex GReLU adapter and compare.")


if __name__ == "__main__":
    main()
