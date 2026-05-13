"""
Base adapter interface and evaluation utilities.

All adapters (Linear, MLP, GReLU) must implement AdapterBase so they
can be evaluated identically in the comparison pipeline.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc


TERRITORIES = {
    "Inferior": [1, 2, 5],         # II, III, aVF
    "Anterior": [8, 9],            # V3, V4
    "Lateral":  [0, 4, 10, 11],    # I, aVL, V5, V6
}


class AdapterBase:
    """Interface that all adapters must implement."""

    def fit(self, phi_train, z_rest_train, z_stress_train):
        """Train the adapter.

        Args:
            phi_train:    (N, 192) ICM stress embeddings
            z_rest_train: (N, 192) resting 12-lead embeddings
            z_stress_train: (N, 192) stress 12-lead embeddings (target)

        The adapter should learn to predict the residual:
            r = z_stress - z_rest
        from the input:
            u = [phi_stress; z_rest]  (384-dim)
        """
        raise NotImplementedError

    def predict(self, phi, z_rest):
        """Predict recovered stress embedding.

        Args:
            phi:    (N, 192) ICM stress embeddings
            z_rest: (N, 192) resting 12-lead embeddings

        Returns:
            z_hat: (N, 192) = z_rest + f([phi; z_rest])
        """
        raise NotImplementedError

    def save(self, path):
        """Save trained adapter to a single file."""
        raise NotImplementedError

    @classmethod
    def load(cls, path):
        """Load a trained adapter from file."""
        raise NotImplementedError


# -- evaluation utilities --------------------------------------------------

def load_regression_head(head_path):
    """Load the frozen ST80 regression head."""
    head = nn.Sequential(
        nn.Linear(199, 64), nn.Mish(inplace=True), nn.Linear(64, 12),
    )
    head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=False))
    head.eval()
    return head


def decode_st80(head, z_hat):
    """z_hat (N, 192) -> predicted ST80 (N, 12) via regression head."""
    z = torch.tensor(z_hat, dtype=torch.float32)
    pad = torch.zeros(z.shape[0], 7)
    fused = torch.cat([z, pad], dim=1)
    with torch.no_grad():
        return head(fused).numpy()


def ischemia_patient_score(pred_st80, bl_st80):
    """Continuous patient-level ischemia score.

    Score = max over territories of 2nd-largest |predicted_delta|.
    This is the continuous relaxation of '>=2 leads >=1mm in a territory'.
    """
    pred_delta = np.abs(pred_st80 - bl_st80)
    scores = np.full(pred_delta.shape[0], -np.inf)
    for idx in TERRITORIES.values():
        td = np.sort(pred_delta[:, idx], axis=1)[:, ::-1]
        k = min(1, td.shape[1] - 1)
        scores = np.maximum(scores, td[:, k])
    return scores


def evaluate_adapter(adapter, data, head):
    """Compute all metrics for an adapter on a data split.

    Args:
        adapter: AdapterBase instance (already trained)
        data: dict with z_rest, z_stress, phi_stress, bl_st80, ischemia
        head: ST80 regression head (nn.Module)

    Returns:
        dict of metrics
    """
    z_hat = adapter.predict(data["phi_stress"], data["z_rest"])
    z_true = data["z_stress"]
    y = data["ischemia"]

    # latent metrics
    mse = np.mean(np.sum((z_hat - z_true) ** 2, axis=1))
    norms = np.linalg.norm(z_hat, axis=1) * np.linalg.norm(z_true, axis=1) + 1e-8
    cos = np.mean(np.sum(z_hat * z_true, axis=1) / norms)

    # ischemia metrics
    pred_st80 = decode_st80(head, z_hat)
    scores = ischemia_patient_score(pred_st80, data["bl_st80"])
    auroc = roc_auc_score(y, scores) if len(np.unique(y)) >= 2 else float("nan")

    fpr, tpr, _ = roc_curve(y, scores)
    pr, rc, _ = precision_recall_curve(y, scores)
    auprc = auc(rc, pr)

    return dict(
        latent_MSE=mse, cosine_sim=cos,
        AUROC=auroc, AUPRC=auprc,
    )
