"""
Latent-space adapters for the QT fine-tuning experiment.

Maps the single-lead (Lead I) graph_embed → 12-lead graph_embed using
the same residual formulation as the ST-segment adapters in src/:

    r_i = z_12lead_i - z_lead_i             (residual to predict)
    z_12lead_hat = z_lead_i + adapter(z_lead_i)

Two adapter families, matching the ST experiments:
  Linear  : Ridge regression         (l1_ratio = 0 in the ST ElasticNet sweep)
  Kernel  : KernelRidge + RBF kernel + StandardScaler  (same as kernel_adapter.py)

Key differences from the ST adapters:
  - Input is z_lead_i alone (192-dim), not concat(phi_stress, z_rest) (384-dim)
  - Output residual r is 192-dim (same as input)
  - No stress/rest split — train on the standard training fold (strat_fold 1-8)

Usage (imported by analyze_qtc.py):
    from qt_adapter import extract_latents, predict_from_latents
    from qt_adapter import LinearResidualAdapter, KernelResidualAdapter
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader as PyGDataLoader


# ---------------------------------------------------------------------------
# Latent extraction  (model.graph_embed, 192-dim)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_latents(
    model: nn.Module,
    ds,
    device: torch.device,
    batch_size: int = 64,
    desc: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """
    Run the model on *ds* and return raw graph embeddings.

    Returns
    -------
    latents : (N, 192) float32 — graph_embed BEFORE classifier
    pred_qt : (N,)     float32 — QT predictions from the unmodified model
    tgt_qt  : (N,)     float32 — ground-truth QT_Int_Global
    ecg_ids : [N] int
    """
    model.eval()
    loader = PyGDataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    latents, preds, tgts = [], [], []
    for bd, qt in loader:
        bd  = bd.to(device)
        B   = bd.num_graphs
        out = model(bd, tabular=torch.zeros(B, 0, device=device))
        latents.append(out["graph_embed"].cpu().numpy())
        preds.append(out["logits"].flatten().cpu().numpy())
        tgts.append(qt.flatten().numpy())
    if desc:
        print(f"    {desc}: {sum(len(p) for p in preds)} samples")
    return (
        np.concatenate(latents).astype(np.float32),
        np.concatenate(preds),
        np.concatenate(tgts),
        [i["ecg_id"] for i in ds.index],
    )


# ---------------------------------------------------------------------------
# Classifier-only inference from pre-computed latents
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_from_latents(
    model: nn.Module,
    latents: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """
    Pass (N, 192) latents through model.classifier (with empty tabular).
    Returns (N,) QT predictions in ms.
    """
    model.eval()
    preds = []
    for i in range(0, len(latents), batch_size):
        z = torch.tensor(latents[i : i + batch_size], dtype=torch.float32, device=device)
        # model.classifier expects fused = cat([graph_embed, zeros(0)]) = graph_embed
        logits = model.classifier(z)
        preds.append(logits.flatten().cpu().numpy())
    return np.concatenate(preds)


# ---------------------------------------------------------------------------
# Residual adapter base
# ---------------------------------------------------------------------------

class _ResidualAdapterBase:
    """
    Common interface: fit(z_src, z_tgt), transform(z_src).

    Follows the ST residual convention:
        r = z_tgt - z_src            (fit target)
        z_adapted = z_src + r_hat    (apply at inference)
    """

    def fit(self, z_src: np.ndarray, z_tgt: np.ndarray) -> "_ResidualAdapterBase":
        r = (z_tgt - z_src).astype(np.float32)
        self._fit_residual(z_src.astype(np.float32), r)
        r_hat = self._predict_residual(z_src.astype(np.float32))
        train_mse = float(((r - r_hat) ** 2).mean())
        print(f"    {self.__class__.__name__}: train residual MSE = {train_mse:.5f}")
        return self

    def transform(self, z_src: np.ndarray) -> np.ndarray:
        z_src = z_src.astype(np.float32)
        r_hat = self._predict_residual(z_src)
        return (z_src + r_hat).astype(np.float32)

    def _fit_residual(self, z_src: np.ndarray, r: np.ndarray) -> None:
        raise NotImplementedError

    def _predict_residual(self, z_src: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Linear Ridge adapter  (matches ElasticNet l1_ratio=0 from ST sweep)
# ---------------------------------------------------------------------------

class LinearResidualAdapter(_ResidualAdapterBase):
    """
    Ridge regression: r_hat = W @ z_src + b
    Multi-output (sklearn Ridge handles 192 outputs natively).
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._model: Ridge | None = None

    def _fit_residual(self, z_src: np.ndarray, r: np.ndarray) -> None:
        self._model = Ridge(alpha=self.alpha, fit_intercept=True)
        self._model.fit(z_src, r)

    def _predict_residual(self, z_src: np.ndarray) -> np.ndarray:
        return self._model.predict(z_src).astype(np.float32)


# ---------------------------------------------------------------------------
# Kernel Ridge adapter  (matches kernel_adapter.py from ST experiments)
# ---------------------------------------------------------------------------

class KernelResidualAdapter(_ResidualAdapterBase):
    """
    KernelRidge (RBF) + StandardScaler, matching src/kernel_adapter.py.

    Fits on a random subsample (n_subsample) of training points to keep the
    N×N kernel matrix tractable.  Gamma defaults to the "scale" heuristic:
        gamma = 1 / (d * X.var())
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float | str = "scale",
        n_subsample: int = 3000,
        random_state: int = 42,
    ):
        self.alpha        = alpha
        self.gamma_arg    = gamma
        self.n_subsample  = n_subsample
        self.random_state = random_state
        self._model: KernelRidge | None = None
        self._scaler: StandardScaler | None = None
        self._z_sub: np.ndarray | None = None   # kept for reporting only

    def _fit_residual(self, z_src: np.ndarray, r: np.ndarray) -> None:
        rng = np.random.RandomState(self.random_state)
        n   = min(len(z_src), self.n_subsample)
        idx = rng.choice(len(z_src), n, replace=False)
        Zs, Rs = z_src[idx], r[idx]

        self._scaler = StandardScaler()
        Zs_s = self._scaler.fit_transform(Zs)

        if self.gamma_arg == "scale":
            gamma = float(1.0 / (Zs_s.shape[1] * Zs_s.var()))
        else:
            gamma = float(self.gamma_arg)
        self._gamma = gamma

        self._model = KernelRidge(alpha=self.alpha, kernel="rbf", gamma=gamma)
        self._model.fit(Zs_s, Rs)
        self._z_sub = Zs_s   # store for prediction (dual solution anchored here)

        print(f"    KernelResidualAdapter: n_sub={n}  gamma={gamma:.3e}  alpha={self.alpha}")

    def _predict_residual(self, z_src: np.ndarray) -> np.ndarray:
        z_s = self._scaler.transform(z_src)
        return self._model.predict(z_s).astype(np.float32)
