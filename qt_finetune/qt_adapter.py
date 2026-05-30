"""
Latent-space adapters for the QT fine-tuning experiment.

Maps the single-lead (Lead I) graph_embed → 12-lead graph_embed using
the same residual formulation as the ST-segment adapters in src/:

    r_i = z_12lead_i - z_lead_i             (residual to predict)
    z_12lead_hat = z_lead_i + adapter(z_lead_i)

Six adapter families:
  Linear    : Ridge regression         (l1_ratio = 0 in the ST ElasticNet sweep)
  Kernel    : KernelRidge + RBF kernel + StandardScaler  (same as kernel_adapter.py)
  GReLU     : Convex fixed-gate GReLU network + R-FISTA  (same as grelu_adapter.py)
  MLP       : 3-layer MLP (GELU + Dropout) trained with AdamW + CosineAnnealingLR
  Attention : Reshape (192→12×16) → TransformerEncoder (pre-norm) → project back
  LoRA      : Low-rank adapter B(A(z))·(α/r); A~Kaiming, B=0 init

Key differences from the ST adapters:
  - Input is z_lead_i alone (192-dim), not concat(phi_stress, z_rest) (384-dim)
  - Output residual r is 192-dim (same as input)
  - No stress/rest split — train on the standard training fold (strat_fold 1-8)

Latent caching (avoids re-running the encoder on every invocation):
  save_latents / load_latents   — thin np.savez / np.load wrappers
  get_latents                   — load from .npz cache if present, else extract + save
                                  Embeds checkpoint mtime; warns if cache is stale.

Checkpoint helpers (save / load):
  save_linear / load_linear     — joblib pickle of LinearResidualAdapter
  save_krr    / load_krr        — joblib pickle of KernelResidualAdapter
  save_grelu  / load_grelu      — torch.save / torch.load of GReLUResidualAdapter
  save_mlp    / load_mlp        — torch.save / torch.load of MLPResidualAdapter
  save_attention / load_attention — torch.save / torch.load of AttentionResidualAdapter
  save_lora   / load_lora       — torch.save / torch.load of LoRAResidualAdapter

Usage (imported by analyze_qtc.py and train_adapters.py):
    from qt_adapter import get_latents, predict_from_latents
    from qt_adapter import (LinearResidualAdapter, KernelResidualAdapter,
                            GReLUResidualAdapter, MLPResidualAdapter,
                            AttentionResidualAdapter, LoRAResidualAdapter)
    from qt_adapter import (save_linear, load_linear, save_krr, load_krr,
                            save_grelu, load_grelu, save_mlp, load_mlp,
                            save_attention, load_attention, save_lora, load_lora)
"""
from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
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
# Latent cache helpers — save / load / get
# ---------------------------------------------------------------------------

def save_latents(
    path: str | Path,
    latents: np.ndarray,
    pred_qt: np.ndarray,
    tgt_qt: np.ndarray,
    ecg_ids: list[int],
    ckpt_mtime: float | None = None,
) -> None:
    """Save extracted latents to a compressed .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        latents=latents.astype(np.float32),
        pred_qt=pred_qt.astype(np.float32),
        tgt_qt=tgt_qt.astype(np.float32),
        ecg_ids=np.array(ecg_ids, dtype=np.int64),
        ckpt_mtime=np.array(ckpt_mtime if ckpt_mtime is not None else 0.0),
    )


def load_latents(
    path: str | Path,
    ckpt_mtime: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """
    Load cached latents from *path*.

    If *ckpt_mtime* is supplied, checks that the cache was built with the same
    checkpoint and prints a warning (but still loads) if there is a mismatch.
    """
    path = Path(path)
    d = np.load(path)
    stored_mtime = float(d["ckpt_mtime"])
    if ckpt_mtime is not None and stored_mtime != 0.0:
        if abs(stored_mtime - ckpt_mtime) > 2.0:   # >2 s tolerance
            print(
                f"  ⚠  Latent cache {path.name} was built from a different "
                f"checkpoint (stored mtime {stored_mtime:.0f} ≠ current "
                f"{ckpt_mtime:.0f}). Pass --force-reextract to rebuild."
            )
    return (
        d["latents"].astype(np.float32),
        d["pred_qt"].astype(np.float32),
        d["tgt_qt"].astype(np.float32),
        d["ecg_ids"].tolist(),
    )


def get_latents(
    model: nn.Module,
    ds,
    device: torch.device,
    batch_size: int,
    cache_path: str | Path,
    ckpt_mtime: float | None = None,
    force: bool = False,
    desc: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """
    Return (latents, pred_qt, tgt_qt, ecg_ids) — from cache if available.

    On a cache miss (or *force=True*), runs the encoder forward pass through
    *ds* and saves the result to *cache_path* (.npz).  On a cache hit the
    model is not touched at all, so you can safely call this before even
    moving the model to GPU.

    Parameters
    ----------
    cache_path : path to the .npz file (created on first call)
    ckpt_mtime : os.path.getmtime(checkpoint_path) — stored in the cache so
                 stale caches are detected automatically.
    force      : if True, re-extract even if the cache exists.
    desc       : short label printed alongside progress messages.
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        print(f"  [{desc}] loading cached latents from {cache_path.name} …")
        result = load_latents(cache_path, ckpt_mtime)
        n = len(result[0])
        print(f"    → {n:,} samples")
        return result

    # Cache miss: extract and save
    print(f"  [{desc}] extracting latents …")
    latents, pred_qt, tgt_qt, ecg_ids = extract_latents(
        model, ds, device, batch_size, desc=desc
    )
    save_latents(cache_path, latents, pred_qt, tgt_qt, ecg_ids, ckpt_mtime)
    print(f"    → cached to {cache_path}")
    return latents, pred_qt, tgt_qt, ecg_ids


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


# ---------------------------------------------------------------------------
# GReLU Residual Adapter  (R-FISTA, matches src/grelu_adapter.py)
# ---------------------------------------------------------------------------

class _GReLUNet(nn.Module):
    """
    Internal two-layer fixed-gate convex GReLU network.

    Architecture (mirrors src/grelu_adapter.py exactly, d_input=192 for QT):
        linear1: nn.Linear(d_input, m, bias=False)   # trainable features
        linear2: nn.Linear(d_input, m, bias=False)   # FROZEN random gate
        proj   : nn.Linear(m, d_output, bias=True)   # trainable projection

    Forward:
        gate     = (linear2(u) >= 0).float()
        features = linear1(u) * gate
        return     proj(features)
    """

    def __init__(self, d_input: int, d_output: int, m: int, seed: int = 42):
        super().__init__()
        self.d_input  = d_input
        self.d_output = d_output
        self.m        = m
        self.seed     = seed

        gate_gen = torch.Generator().manual_seed(seed)

        self.linear1 = nn.Linear(d_input, m, bias=False)
        self.linear2 = nn.Linear(d_input, m, bias=False)
        self.proj    = nn.Linear(m, d_output, bias=True)

        # Kaiming-uniform style gate init, then freeze
        bound = (1.0 / d_input) ** 0.5
        with torch.no_grad():
            self.linear2.weight.uniform_(-bound, bound, generator=gate_gen)
        self.linear2.weight.requires_grad_(False)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        gate     = (self.linear2(u) >= 0).to(u.dtype)
        features = self.linear1(u) * gate
        return self.proj(features)


def _rfista_step(
    net: _GReLUNet,
    u_tr: torch.Tensor,
    r_tr: torch.Tensor,
    state: dict,
    lambda_l1: float,
    lambda_l2: float,
) -> None:
    """
    One R-FISTA iteration on the GReLU net.  Mutates *state* in-place.

    Ported verbatim from src/grelu_adapter.py, except the model argument
    is a _GReLUNet (d_input=192) rather than the ST 384-dim adapter.

    state keys: W1, Wp, bp  (current x)
                yW1, yWp, ybp  (momentum y)
                t  (FISTA momentum scalar)
                eta  (current step size, adapted by backtracking)
    """
    yW1, yWp, ybp = state["yW1"], state["yWp"], state["ybp"]

    # 1) Smooth loss + gradient at the momentum point y
    net.linear1.weight.data.copy_(yW1)
    net.proj.weight.data.copy_(yWp)
    net.proj.bias.data.copy_(ybp)
    net.zero_grad(set_to_none=True)
    pred  = net(u_tr)
    data  = 0.5 * ((pred - r_tr) ** 2).mean()
    l2    = 0.5 * lambda_l2 * (
        net.linear1.weight.pow(2).sum() + net.proj.weight.pow(2).sum()
    )
    f_y_t = data + l2
    f_y_t.backward()
    f_y = f_y_t.item()
    gW1 = net.linear1.weight.grad.detach().clone()
    gWp = net.proj.weight.grad.detach().clone()
    gbp = net.proj.bias.grad.detach().clone()

    # 2) Backtracking proximal gradient step.
    #    Prox of lambda_l1 * ||.||_1  is element-wise soft threshold on W1.
    eta = state["eta"]
    for _ in range(50):
        W1n = torch.sign(yW1 - eta * gW1) * torch.clamp(
            (yW1 - eta * gW1).abs() - eta * lambda_l1, min=0.0
        )
        Wpn = yWp - eta * gWp
        bpn = ybp - eta * gbp

        with torch.no_grad():
            net.linear1.weight.data.copy_(W1n)
            net.proj.weight.data.copy_(Wpn)
            net.proj.bias.data.copy_(bpn)
            pred_n = net(u_tr)
            f_n = (
                0.5 * ((pred_n - r_tr) ** 2).mean()
                + 0.5 * lambda_l2 * (W1n.pow(2).sum() + Wpn.pow(2).sum())
            ).item()

        # Quadratic upper bound check
        d_W1   = W1n - yW1;  d_Wp = Wpn - yWp;  d_bp = bpn - ybp
        inner  = ((gW1 * d_W1).sum() + (gWp * d_Wp).sum()
                  + (gbp * d_bp).sum())
        sqdist = (d_W1.pow(2).sum() + d_Wp.pow(2).sum()
                  + d_bp.pow(2).sum())
        ub = f_y + inner.item() + sqdist.item() / (2.0 * eta)
        if f_n <= ub + 1e-9 or eta < 1e-12:
            break
        eta *= 0.5
    # Grow step slightly to recover from over-shrink
    state["eta"] = eta * 1.1

    # 3) FISTA momentum + O'Donoghue-Candes gradient-adaptive restart
    W1, Wp, bp   = state["W1"], state["Wp"], state["bp"]
    step_W1 = W1n - W1;  step_Wp = Wpn - Wp;  step_bp = bpn - bp
    g_dot = (
        (gW1 * step_W1).sum().item()
        + (gWp * step_Wp).sum().item()
        + (gbp * step_bp).sum().item()
    )
    t_k   = state["t"]
    t_new = 0.5 * (1.0 + (1.0 + 4.0 * t_k * t_k) ** 0.5)
    if g_dot > 0:          # uphill step → restart momentum
        t_new = 1.0
        yW1n, yWpn, ybpn = W1n.clone(), Wpn.clone(), bpn.clone()
    else:
        ratio = (t_k - 1.0) / t_new
        yW1n  = W1n + ratio * step_W1
        yWpn  = Wpn + ratio * step_Wp
        ybpn  = bpn + ratio * step_bp

    state["W1"],  state["Wp"],  state["bp"]  = W1n,  Wpn,  bpn
    state["yW1"], state["yWp"], state["ybp"] = yW1n, yWpn, ybpn
    state["t"]   = t_new

    # Leave the net holding the new post-prox parameters
    net.linear1.weight.data.copy_(W1n)
    net.proj.weight.data.copy_(Wpn)
    net.proj.bias.data.copy_(bpn)


# Default config for the QT GReLU adapter (d_input=192, not 384 as in ST)
GRELU_DEFAULT_CONFIG: dict = {
    "d_input":    192,    # z_lead_i only (Lead I latent)
    "d_output":   192,    # 12-lead latent residual
    "m":          1024,   # GReLU neurons
    "lr":         1e-3,   # initial step size for R-FISTA
    "epochs":     1000,    # R-FISTA is cheap per-iteration; 500 is a reasonable default
    "eval_every": 50,     # print / val-checkpoint every N iterations
    "seed":       42,
    "lambda_l2":  1e-4,
    "lambda_l1":  0.0,
}


class GReLUResidualAdapter:
    """
    Convex fixed-gate GReLU residual adapter trained with R-FISTA.

    Numpy-facing interface matching LinearResidualAdapter / KernelResidualAdapter:

        adapter = GReLUResidualAdapter(...)
        adapter.fit(z_src_tr, z_tgt_tr)             # or with val split below
        z_adapted = adapter.transform(z_src_val)    # numpy in, numpy out

    For training with validation-split checkpoint tracking:

        adapter.fit(z_src_tr, z_tgt_tr,
                    z_src_val=zi_val, z_tgt_val=z12_val)

    Architecture (see _GReLUNet):
        gate     = (linear2(u) >= 0).float()    # FROZEN random gate
        features = linear1(u) * gate             # GReLU
        residual = proj(features)                # projection
        output   = z_src + residual              # via transform()

    For QT: d_input = 192 (Lead I latent only, not 384-dim concat as in ST).
    """

    def __init__(
        self,
        m: int            = 1024,
        seed: int         = 42,
        lr: float         = 1e-3,
        epochs: int       = 100,
        eval_every: int   = 10,
        lambda_l2: float  = 1e-4,
        lambda_l1: float  = 0.0,
        device: str | None = None,
    ):
        self.m          = m
        self.seed       = seed
        self.lr         = lr
        self.epochs     = epochs
        self.eval_every = eval_every
        self.lambda_l2  = lambda_l2
        self.lambda_l1  = lambda_l1
        self._device_str = device   # None → auto-detect at fit time
        self._net: _GReLUNet | None = None

    # ------------------------------------------------------------------
    def _pick_device(self) -> torch.device:
        if self._device_str is not None:
            return torch.device(self._device_str)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # ------------------------------------------------------------------
    def fit(
        self,
        z_src: np.ndarray,
        z_tgt: np.ndarray,
        z_src_val: np.ndarray | None = None,
        z_tgt_val: np.ndarray | None = None,
        verbose: bool = True,
    ) -> "GReLUResidualAdapter":
        """
        Train R-FISTA on (z_src, z_tgt) pairs.

        If z_src_val / z_tgt_val are supplied, the best state (lowest val
        latent MSE) is restored at the end of training.
        """
        z_src = z_src.astype(np.float32)
        z_tgt = z_tgt.astype(np.float32)
        r_tr  = z_tgt - z_src          # residuals (N, 192)

        d_in  = z_src.shape[1]
        d_out = r_tr.shape[1]
        device = self._pick_device()
        torch.manual_seed(self.seed)

        u_tr  = torch.tensor(z_src, device=device)
        r_tr_t = torch.tensor(r_tr,  device=device)

        has_val = (z_src_val is not None) and (z_tgt_val is not None)
        if has_val:
            z_src_val = z_src_val.astype(np.float32)
            z_tgt_val = z_tgt_val.astype(np.float32)
            u_val     = torch.tensor(z_src_val, device=device)
            r_val_t   = torch.tensor(z_tgt_val - z_src_val, device=device)
            z_tgt_val_t = torch.tensor(z_tgt_val, device=device)

        net = _GReLUNet(d_input=d_in, d_output=d_out,
                        m=self.m, seed=self.seed).to(device)

        state: dict = {
            "W1":  net.linear1.weight.detach().clone(),
            "Wp":  net.proj.weight.detach().clone(),
            "bp":  net.proj.bias.detach().clone(),
            "yW1": net.linear1.weight.detach().clone(),
            "yWp": net.proj.weight.detach().clone(),
            "ybp": net.proj.bias.detach().clone(),
            "t":   1.0,
            "eta": float(self.lr),
        }

        best_val_mse  = float("inf")
        best_sd: dict | None = None

        for epoch in range(1, self.epochs + 1):
            net.train()
            _rfista_step(net, u_tr, r_tr_t, state,
                         self.lambda_l1, self.lambda_l2)

            do_eval = (epoch % self.eval_every == 0
                       or epoch == self.epochs)

            with torch.no_grad():
                net.eval()
                if do_eval or verbose:
                    tr_mse = ((net(u_tr) - r_tr_t) ** 2).mean().item()

                if has_val and do_eval:
                    r_hat_v  = net(u_val)
                    # latent recovery MSE (sum over feature dim)
                    z_hat_v  = u_val + r_hat_v
                    val_lat  = ((z_hat_v - z_tgt_val_t) ** 2).sum(dim=1).mean().item()
                    val_res  = ((r_hat_v - r_val_t) ** 2).mean().item()
                    if val_lat < best_val_mse:
                        best_val_mse = val_lat
                        best_sd = {k: v.clone() if torch.is_tensor(v) else v
                                   for k, v in net.state_dict().items()}

            if verbose and do_eval:
                if has_val:
                    print(f"    GReLU iter {epoch:4d}  "
                          f"eta={state['eta']:.2e}  "
                          f"train_mse={tr_mse:.6f}  "
                          f"val_res={val_res:.6f}  "
                          f"val_lat_mse={val_lat:.6f}")
                else:
                    print(f"    GReLU iter {epoch:4d}  "
                          f"eta={state['eta']:.2e}  "
                          f"train_mse={tr_mse:.6f}")
            elif verbose and epoch % max(1, self.eval_every // 2) == 0:
                with torch.no_grad():
                    net.eval()
                    tr_mse = ((net(u_tr) - r_tr_t) ** 2).mean().item()
                print(f"    GReLU iter {epoch:4d}  "
                      f"eta={state['eta']:.2e}  "
                      f"train_mse={tr_mse:.6f}")

        # Restore best val checkpoint if tracked
        if has_val and best_sd is not None:
            net.load_state_dict(best_sd)
            if verbose:
                print(f"    GReLU: restored best (val_lat_mse={best_val_mse:.6f})")

        self._net = net.cpu().eval()
        self._d_input  = d_in
        self._d_output = d_out

        # Final train residual MSE (same printout style as Linear/KRR)
        with torch.no_grad():
            r_hat_tr = self._net(torch.tensor(z_src))
            train_res_mse = float(
                ((r_hat_tr - torch.tensor(r_tr)) ** 2).mean().item()
            )
        print(f"    GReLUResidualAdapter: train residual MSE = {train_res_mse:.5f}")
        return self

    # ------------------------------------------------------------------
    def transform(self, z_src: np.ndarray) -> np.ndarray:
        """Apply the residual adapter: z_adapted = z_src + adapter(z_src)."""
        assert self._net is not None, "Call fit() (or load_grelu()) before transform()"
        z_src = z_src.astype(np.float32)
        with torch.no_grad():
            u     = torch.tensor(z_src)
            r_hat = self._net(u).numpy()
        return (z_src + r_hat).astype(np.float32)


# ---------------------------------------------------------------------------
# Checkpoint helpers — save / load for all three adapter types
# ---------------------------------------------------------------------------

def save_grelu(
    adapter: GReLUResidualAdapter,
    path: str | Path,
    config: dict | None = None,
) -> None:
    """Save a fitted GReLUResidualAdapter to *path* via torch.save."""
    assert adapter._net is not None, "Adapter must be fitted before saving"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": adapter._net.state_dict(),
        "d_input":    adapter._net.d_input,
        "d_output":   adapter._net.d_output,
        "m":          adapter._net.m,
        "seed":       adapter._net.seed,
        "config": {
            "m":          adapter.m,
            "seed":       adapter.seed,
            "lr":         adapter.lr,
            "epochs":     adapter.epochs,
            "eval_every": adapter.eval_every,
            "lambda_l2":  adapter.lambda_l2,
            "lambda_l1":  adapter.lambda_l1,
            **(config or {}),
        },
    }, path)
    print(f"    GReLU adapter saved → {path}")


def load_grelu(path: str | Path) -> GReLUResidualAdapter:
    """Load a GReLUResidualAdapter from *path*.  Returns adapter ready for transform()."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg  = ckpt.get("config") or {}
    adapter = GReLUResidualAdapter(
        m          = ckpt["m"],
        seed       = ckpt["seed"],
        lr         = float(cfg.get("lr",         GRELU_DEFAULT_CONFIG["lr"])),
        epochs     = int(  cfg.get("epochs",     GRELU_DEFAULT_CONFIG["epochs"])),
        eval_every = int(  cfg.get("eval_every", GRELU_DEFAULT_CONFIG["eval_every"])),
        lambda_l2  = float(cfg.get("lambda_l2",  GRELU_DEFAULT_CONFIG["lambda_l2"])),
        lambda_l1  = float(cfg.get("lambda_l1",  GRELU_DEFAULT_CONFIG["lambda_l1"])),
    )
    net = _GReLUNet(
        d_input=ckpt["d_input"], d_output=ckpt["d_output"],
        m=ckpt["m"], seed=ckpt["seed"],
    )
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    adapter._net      = net
    adapter._d_input  = ckpt["d_input"]
    adapter._d_output = ckpt["d_output"]
    print(f"    GReLU adapter loaded ← {path}")
    return adapter


def save_linear(adapter: LinearResidualAdapter, path: str | Path) -> None:
    """Save a fitted LinearResidualAdapter to *path* via joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(adapter, path)
    print(f"    Linear adapter saved → {path}")


def load_linear(path: str | Path) -> LinearResidualAdapter:
    """Load a LinearResidualAdapter from *path*."""
    adapter = joblib.load(path)
    print(f"    Linear adapter loaded ← {path}")
    return adapter


def save_krr(adapter: KernelResidualAdapter, path: str | Path) -> None:
    """Save a fitted KernelResidualAdapter to *path* via joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(adapter, path)
    print(f"    KRR adapter saved → {path}")


def load_krr(path: str | Path) -> KernelResidualAdapter:
    """Load a KernelResidualAdapter from *path*."""
    adapter = joblib.load(path)
    print(f"    KRR adapter loaded ← {path}")
    return adapter


# ---------------------------------------------------------------------------
# Shared mini-batch training loop for neural adapters (MLP, Attention, LoRA)
# ---------------------------------------------------------------------------

def _train_nn_adapter(
    net: nn.Module,
    z_src_tr: torch.Tensor,
    r_tr: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    device: torch.device,
    eval_every: int,
    z_src_val: torch.Tensor | None = None,
    r_val: torch.Tensor | None = None,
    verbose: bool = True,
    name: str = "NN",
) -> nn.Module:
    """
    Mini-batch AdamW + CosineAnnealingLR training loop for MLP/Attention/LoRA.

    If *z_src_val* / *r_val* are supplied the best state (lowest val residual
    MSE) is restored at the end of training.

    Returns *net* in eval() mode on CPU.
    """
    net = net.to(device)
    optimizer = AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 1e-3)

    ds = TensorDataset(z_src_tr, r_tr)
    loader = TorchDataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    has_val = (z_src_val is not None) and (r_val is not None)
    best_val_mse = float("inf")
    best_sd: dict | None = None

    for epoch in range(1, epochs + 1):
        net.train()
        for zb, rb in loader:
            zb = zb.to(device)
            rb = rb.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(net(zb), rb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        do_eval = (epoch % eval_every == 0 or epoch == epochs)
        if do_eval:
            net.eval()
            with torch.no_grad():
                tr_mse = F.mse_loss(
                    net(z_src_tr.to(device)), r_tr.to(device)
                ).item()
                if has_val:
                    vr_hat  = net(z_src_val.to(device))
                    val_mse = F.mse_loss(vr_hat, r_val.to(device)).item()
                    if val_mse < best_val_mse:
                        best_val_mse = val_mse
                        best_sd = {k: v.clone() for k, v in net.state_dict().items()}
            if verbose:
                if has_val:
                    print(
                        f"    {name} epoch {epoch:4d}  "
                        f"train_mse={tr_mse:.6f}  val_mse={val_mse:.6f}"
                    )
                else:
                    print(
                        f"    {name} epoch {epoch:4d}  train_mse={tr_mse:.6f}"
                    )

    if has_val and best_sd is not None:
        net.load_state_dict(best_sd)
        if verbose:
            print(f"    {name}: restored best checkpoint (val_mse={best_val_mse:.6f})")

    return net.cpu().eval()


# ---------------------------------------------------------------------------
# MLP Residual Adapter
# ---------------------------------------------------------------------------

class _MLPNet(nn.Module):
    """
    Three-layer MLP with GELU activations and Dropout.

    Architecture:
        Linear(d_input → hidden) → GELU → Dropout(p)
        Linear(hidden  → hidden) → GELU → Dropout(p)
        Linear(hidden  → d_output)
    """

    def __init__(
        self,
        d_input: int,
        d_output: int,
        hidden: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_output),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


MLP_DEFAULT_CONFIG: dict = {
    "d_input":      192,
    "d_output":     192,
    "hidden":       512,
    "dropout":      0.1,
    "lr":           1e-3,
    "weight_decay": 1e-4,
    "epochs":       200,
    "eval_every":   20,
    "batch_size":   256,
    "seed":         42,
}


class MLPResidualAdapter:
    """
    Three-layer MLP residual adapter trained with AdamW + CosineAnnealingLR.

    Numpy-facing interface matching the other adapter classes:

        adapter = MLPResidualAdapter(...)
        adapter.fit(z_src_tr, z_tgt_tr)
        z_adapted = adapter.transform(z_src_val)
    """

    def __init__(
        self,
        hidden: int        = 512,
        dropout: float     = 0.1,
        lr: float          = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int        = 200,
        eval_every: int    = 20,
        batch_size: int    = 256,
        seed: int          = 42,
        device: str | None = None,
    ):
        self.hidden       = hidden
        self.dropout      = dropout
        self.lr           = lr
        self.weight_decay = weight_decay
        self.epochs       = epochs
        self.eval_every   = eval_every
        self.batch_size   = batch_size
        self.seed         = seed
        self._device_str  = device
        self._net: _MLPNet | None = None

    def _pick_device(self) -> torch.device:
        if self._device_str is not None:
            return torch.device(self._device_str)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def fit(
        self,
        z_src: np.ndarray,
        z_tgt: np.ndarray,
        z_src_val: np.ndarray | None = None,
        z_tgt_val: np.ndarray | None = None,
        verbose: bool = True,
    ) -> "MLPResidualAdapter":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        z_src = z_src.astype(np.float32)
        z_tgt = z_tgt.astype(np.float32)
        r_tr  = z_tgt - z_src
        d_in, d_out = z_src.shape[1], r_tr.shape[1]
        device = self._pick_device()

        net = _MLPNet(d_in, d_out, hidden=self.hidden, dropout=self.dropout)

        zs_val_t = rs_val_t = None
        if z_src_val is not None and z_tgt_val is not None:
            zs_val_t = torch.tensor(z_src_val.astype(np.float32))
            rs_val_t = torch.tensor((z_tgt_val - z_src_val).astype(np.float32))

        self._net = _train_nn_adapter(
            net,
            torch.tensor(z_src), torch.tensor(r_tr),
            epochs=self.epochs, lr=self.lr, weight_decay=self.weight_decay,
            batch_size=self.batch_size, device=device, eval_every=self.eval_every,
            z_src_val=zs_val_t, r_val=rs_val_t,
            verbose=verbose, name="MLP",
        )
        self._d_input  = d_in
        self._d_output = d_out

        with torch.no_grad():
            r_hat = self._net(torch.tensor(z_src))
            mse   = F.mse_loss(r_hat, torch.tensor(r_tr)).item()
        print(f"    MLPResidualAdapter: train residual MSE = {mse:.5f}")
        return self

    def transform(self, z_src: np.ndarray) -> np.ndarray:
        assert self._net is not None, "Call fit() (or load_mlp()) before transform()"
        z_src = z_src.astype(np.float32)
        with torch.no_grad():
            r_hat = self._net(torch.tensor(z_src)).numpy()
        return (z_src + r_hat).astype(np.float32)


# ---------------------------------------------------------------------------
# Attention Residual Adapter
# ---------------------------------------------------------------------------

class _AttentionNet(nn.Module):
    """
    Token-based self-attention residual adapter.

    Reshapes the d_input-dim latent into n_tokens × token_dim, applies
    n_layers × TransformerEncoderLayer (pre-norm, batch_first), flattens,
    then projects back to d_input via a final linear layer.

    Architecture:
        Reshape (B, d_input) → (B, n_tokens, token_dim)
        + learned positional embedding (1, n_tokens, token_dim)
        → n_layers × TransformerEncoderLayer(d_model=token_dim, nhead=nhead,
              dim_feedforward=dim_feedforward, dropout=dropout,
              batch_first=True, norm_first=True)
        → Flatten → Linear(d_input, d_input)

    Default for d_input=192: 12 tokens × 16-dim, nhead=4.
    """

    def __init__(
        self,
        d_input: int       = 192,
        n_tokens: int      = 12,
        token_dim: int     = 16,
        nhead: int         = 4,
        n_layers: int      = 2,
        dim_feedforward: int = 64,
        dropout: float     = 0.1,
    ):
        super().__init__()
        assert d_input == n_tokens * token_dim, (
            f"d_input ({d_input}) must equal n_tokens ({n_tokens}) × "
            f"token_dim ({token_dim})"
        )
        self.n_tokens  = n_tokens
        self.token_dim = token_dim

        self.pos_emb = nn.Parameter(torch.zeros(1, n_tokens, token_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj    = nn.Linear(d_input, d_input)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B      = x.shape[0]
        tokens = x.view(B, self.n_tokens, self.token_dim) + self.pos_emb
        tokens = self.encoder(tokens)
        return self.proj(tokens.reshape(B, -1))


ATTENTION_DEFAULT_CONFIG: dict = {
    "d_input":        192,
    "n_tokens":       12,
    "token_dim":      16,
    "nhead":          4,
    "n_layers":       2,
    "dim_feedforward": 64,
    "dropout":        0.1,
    "lr":             1e-3,
    "weight_decay":   1e-4,
    "epochs":         200,
    "eval_every":     20,
    "batch_size":     256,
    "seed":           42,
}


class AttentionResidualAdapter:
    """
    Self-attention residual adapter (reshape → TransformerEncoder → project).

    Numpy-facing interface matching the other adapter classes:

        adapter = AttentionResidualAdapter(...)
        adapter.fit(z_src_tr, z_tgt_tr)
        z_adapted = adapter.transform(z_src_val)
    """

    def __init__(
        self,
        n_tokens: int        = 12,
        token_dim: int       = 16,
        nhead: int           = 4,
        n_layers: int        = 2,
        dim_feedforward: int = 64,
        dropout: float       = 0.1,
        lr: float            = 1e-3,
        weight_decay: float  = 1e-4,
        epochs: int          = 200,
        eval_every: int      = 20,
        batch_size: int      = 256,
        seed: int            = 42,
        device: str | None   = None,
    ):
        self.n_tokens        = n_tokens
        self.token_dim       = token_dim
        self.nhead           = nhead
        self.n_layers        = n_layers
        self.dim_feedforward = dim_feedforward
        self.dropout         = dropout
        self.lr              = lr
        self.weight_decay    = weight_decay
        self.epochs          = epochs
        self.eval_every      = eval_every
        self.batch_size      = batch_size
        self.seed            = seed
        self._device_str     = device
        self._net: _AttentionNet | None = None

    def _pick_device(self) -> torch.device:
        if self._device_str is not None:
            return torch.device(self._device_str)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def fit(
        self,
        z_src: np.ndarray,
        z_tgt: np.ndarray,
        z_src_val: np.ndarray | None = None,
        z_tgt_val: np.ndarray | None = None,
        verbose: bool = True,
    ) -> "AttentionResidualAdapter":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        z_src = z_src.astype(np.float32)
        z_tgt = z_tgt.astype(np.float32)
        r_tr  = z_tgt - z_src
        d_in, d_out = z_src.shape[1], r_tr.shape[1]
        device = self._pick_device()

        net = _AttentionNet(
            d_input=d_in,
            n_tokens=self.n_tokens, token_dim=self.token_dim,
            nhead=self.nhead, n_layers=self.n_layers,
            dim_feedforward=self.dim_feedforward, dropout=self.dropout,
        )

        zs_val_t = rs_val_t = None
        if z_src_val is not None and z_tgt_val is not None:
            zs_val_t = torch.tensor(z_src_val.astype(np.float32))
            rs_val_t = torch.tensor((z_tgt_val - z_src_val).astype(np.float32))

        self._net = _train_nn_adapter(
            net,
            torch.tensor(z_src), torch.tensor(r_tr),
            epochs=self.epochs, lr=self.lr, weight_decay=self.weight_decay,
            batch_size=self.batch_size, device=device, eval_every=self.eval_every,
            z_src_val=zs_val_t, r_val=rs_val_t,
            verbose=verbose, name="Attention",
        )
        self._d_input  = d_in
        self._d_output = d_out

        with torch.no_grad():
            r_hat = self._net(torch.tensor(z_src))
            mse   = F.mse_loss(r_hat, torch.tensor(r_tr)).item()
        print(f"    AttentionResidualAdapter: train residual MSE = {mse:.5f}")
        return self

    def transform(self, z_src: np.ndarray) -> np.ndarray:
        assert self._net is not None, "Call fit() (or load_attention()) before transform()"
        z_src = z_src.astype(np.float32)
        with torch.no_grad():
            r_hat = self._net(torch.tensor(z_src)).numpy()
        return (z_src + r_hat).astype(np.float32)


# ---------------------------------------------------------------------------
# LoRA Residual Adapter
# ---------------------------------------------------------------------------

class _LoRANet(nn.Module):
    """
    LoRA-style low-rank residual adapter.

    Architecture:
        A : Linear(d_input, rank, bias=False)   — Kaiming-uniform init
        B : Linear(rank, d_output, bias=True)   — zeros init (so r_hat=0 at start)
        r_hat = B(A(z)) * (alpha / rank)

    The zero-init of B means the residual is exactly zero at the start of
    training, so the model begins as an identity adapter.
    """

    def __init__(
        self,
        d_input: int,
        d_output: int,
        rank: int   = 32,
        alpha: float = 32.0,
    ):
        super().__init__()
        self.rank  = rank
        self.scale = alpha / rank
        self.A = nn.Linear(d_input, rank, bias=False)
        self.B = nn.Linear(rank, d_output, bias=True)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        nn.init.zeros_(self.B.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.B(self.A(x)) * self.scale


# ---------------------------------------------------------------------------
# _RWNANet  (Rank-k Whitened Nonlinear Adapter)
# ---------------------------------------------------------------------------

class _RWNANet(nn.Module):
    """
    Rank-k Whitened Nonlinear Adapter (RWNA).

    Motivated by the SVD analysis of the JEPA-Linear weight matrix W, which
    shows only k=2 dominant singular values and PC1+PC2 = 99.4% of the
    combined embedding-space variance.  Rather than applying a full 192×192
    linear map (37k parameters), RWNA:

      1. Projects z_c to a k-dimensional signal subspace  (proj_in, 192→k)
      2. Applies a small MLP in that k-dimensional space   (k→hidden→k)
      3. Projects the correction back to 192-dim           (proj_out, k→192)
      4. Adds a learned bias term b ∈ ℝ^192               (matches the
         dominant ||b|| >> ||W||_F pattern in JEPA-Linear)

    Total parameters with k=2, hidden=32: ~700 vs 37,056 for linear.

    Identity initialisation:
      proj_in/proj_out — Kaiming uniform (small initial correction)
      MLP final layer  — zeros (so the residual starts at b only)
      b                — zeros (pure identity at init)

    Forward:
        t      = proj_in(z)              (B, k)
        delta  = mlp(t)                  (B, k)
        return proj_out(delta) + b       (B, d_output)
    """

    def __init__(
        self,
        d_input:  int,
        d_output: int,
        k:        int = 2,
        hidden:   int = 32,
    ):
        super().__init__()
        self.k = k
        self.proj_in  = nn.Linear(d_input, k, bias=False)
        self.mlp      = nn.Sequential(
            nn.Linear(k, hidden),
            nn.GELU(),
            nn.Linear(hidden, k),
        )
        self.proj_out = nn.Linear(k, d_output, bias=False)
        self.bias     = nn.Parameter(torch.zeros(d_output))
        # Zero-init final MLP layer so residual correction starts at 0
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t     = self.proj_in(x)          # (B, k)
        delta = self.mlp(t)              # (B, k)
        return self.proj_out(delta) + self.bias   # (B, d_output)


RWNA_DEFAULT_CONFIG: dict = {
    "d_input":      192,
    "d_output":     192,
    "k":            16,    # rank of projection subspace; k=2 for frozen z_c (near-degenerate),
                           # k=16 for JEPA E2E (encoder learns richer representations)
    "hidden":       64,    # MLP hidden size in the k-dim projected space
    "lr":           1e-3,
    "weight_decay": 1e-4,
    "epochs":       200,
    "eval_every":   20,
    "batch_size":   256,
    "seed":         42,
}

# Frozen-adapter variant: small k because frozen z_c is near-degenerate (2 dominant SVD modes)
RWNA_FROZEN_CONFIG: dict = {**RWNA_DEFAULT_CONFIG, "k": 2, "hidden": 32}


LORA_DEFAULT_CONFIG: dict = {
    "d_input":      192,
    "d_output":     192,
    "rank":         32,
    "lora_alpha":   32.0,
    "lr":           1e-3,
    "weight_decay": 1e-4,
    "epochs":       200,
    "eval_every":   20,
    "batch_size":   256,
    "seed":         42,
}


class LoRAResidualAdapter:
    """
    Low-rank (LoRA-style) residual adapter trained with AdamW + CosineAnnealingLR.

    Numpy-facing interface matching the other adapter classes:

        adapter = LoRAResidualAdapter(...)
        adapter.fit(z_src_tr, z_tgt_tr)
        z_adapted = adapter.transform(z_src_val)
    """

    def __init__(
        self,
        rank: int           = 32,
        lora_alpha: float   = 32.0,
        lr: float           = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int         = 200,
        eval_every: int     = 20,
        batch_size: int     = 256,
        seed: int           = 42,
        device: str | None  = None,
    ):
        self.rank         = rank
        self.lora_alpha   = lora_alpha
        self.lr           = lr
        self.weight_decay = weight_decay
        self.epochs       = epochs
        self.eval_every   = eval_every
        self.batch_size   = batch_size
        self.seed         = seed
        self._device_str  = device
        self._net: _LoRANet | None = None

    def _pick_device(self) -> torch.device:
        if self._device_str is not None:
            return torch.device(self._device_str)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def fit(
        self,
        z_src: np.ndarray,
        z_tgt: np.ndarray,
        z_src_val: np.ndarray | None = None,
        z_tgt_val: np.ndarray | None = None,
        verbose: bool = True,
    ) -> "LoRAResidualAdapter":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        z_src = z_src.astype(np.float32)
        z_tgt = z_tgt.astype(np.float32)
        r_tr  = z_tgt - z_src
        d_in, d_out = z_src.shape[1], r_tr.shape[1]
        device = self._pick_device()

        net = _LoRANet(d_in, d_out, rank=self.rank, alpha=self.lora_alpha)

        zs_val_t = rs_val_t = None
        if z_src_val is not None and z_tgt_val is not None:
            zs_val_t = torch.tensor(z_src_val.astype(np.float32))
            rs_val_t = torch.tensor((z_tgt_val - z_src_val).astype(np.float32))

        self._net = _train_nn_adapter(
            net,
            torch.tensor(z_src), torch.tensor(r_tr),
            epochs=self.epochs, lr=self.lr, weight_decay=self.weight_decay,
            batch_size=self.batch_size, device=device, eval_every=self.eval_every,
            z_src_val=zs_val_t, r_val=rs_val_t,
            verbose=verbose, name="LoRA",
        )
        self._d_input  = d_in
        self._d_output = d_out

        with torch.no_grad():
            r_hat = self._net(torch.tensor(z_src))
            mse   = F.mse_loss(r_hat, torch.tensor(r_tr)).item()
        print(f"    LoRAResidualAdapter: train residual MSE = {mse:.5f}")
        return self

    def transform(self, z_src: np.ndarray) -> np.ndarray:
        assert self._net is not None, "Call fit() (or load_lora()) before transform()"
        z_src = z_src.astype(np.float32)
        with torch.no_grad():
            r_hat = self._net(torch.tensor(z_src)).numpy()
        return (z_src + r_hat).astype(np.float32)


# ---------------------------------------------------------------------------
# Checkpoint helpers — MLP, Attention, LoRA
# ---------------------------------------------------------------------------

def save_mlp(
    adapter: MLPResidualAdapter,
    path: str | Path,
    config: dict | None = None,
) -> None:
    """Save a fitted MLPResidualAdapter to *path* via torch.save."""
    assert adapter._net is not None, "Adapter must be fitted before saving"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": adapter._net.state_dict(),
            "d_input":    adapter._d_input,
            "d_output":   adapter._d_output,
            "config": {
                "hidden":       adapter.hidden,
                "dropout":      adapter.dropout,
                "lr":           adapter.lr,
                "weight_decay": adapter.weight_decay,
                "epochs":       adapter.epochs,
                "eval_every":   adapter.eval_every,
                "batch_size":   adapter.batch_size,
                "seed":         adapter.seed,
                **(config or {}),
            },
        },
        path,
    )
    print(f"    MLP adapter saved → {path}")


def load_mlp(path: str | Path) -> MLPResidualAdapter:
    """Load a MLPResidualAdapter from *path*.  Returns adapter ready for transform()."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg  = ckpt.get("config") or {}
    adapter = MLPResidualAdapter(
        hidden       = int(  cfg.get("hidden",       MLP_DEFAULT_CONFIG["hidden"])),
        dropout      = float(cfg.get("dropout",      MLP_DEFAULT_CONFIG["dropout"])),
        lr           = float(cfg.get("lr",           MLP_DEFAULT_CONFIG["lr"])),
        weight_decay = float(cfg.get("weight_decay", MLP_DEFAULT_CONFIG["weight_decay"])),
        epochs       = int(  cfg.get("epochs",       MLP_DEFAULT_CONFIG["epochs"])),
        eval_every   = int(  cfg.get("eval_every",   MLP_DEFAULT_CONFIG["eval_every"])),
        batch_size   = int(  cfg.get("batch_size",   MLP_DEFAULT_CONFIG["batch_size"])),
        seed         = int(  cfg.get("seed",         MLP_DEFAULT_CONFIG["seed"])),
    )
    net = _MLPNet(
        d_input=ckpt["d_input"], d_output=ckpt["d_output"],
        hidden=adapter.hidden, dropout=adapter.dropout,
    )
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    adapter._net      = net
    adapter._d_input  = ckpt["d_input"]
    adapter._d_output = ckpt["d_output"]
    print(f"    MLP adapter loaded ← {path}")
    return adapter


def save_attention(
    adapter: AttentionResidualAdapter,
    path: str | Path,
    config: dict | None = None,
) -> None:
    """Save a fitted AttentionResidualAdapter to *path* via torch.save."""
    assert adapter._net is not None, "Adapter must be fitted before saving"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict":     adapter._net.state_dict(),
            "d_input":        adapter._d_input,
            "d_output":       adapter._d_output,
            "n_tokens":       adapter.n_tokens,
            "token_dim":      adapter.token_dim,
            "nhead":          adapter.nhead,
            "n_layers":       adapter.n_layers,
            "dim_feedforward": adapter.dim_feedforward,
            "dropout":        adapter.dropout,
            "config": {
                "lr":           adapter.lr,
                "weight_decay": adapter.weight_decay,
                "epochs":       adapter.epochs,
                "eval_every":   adapter.eval_every,
                "batch_size":   adapter.batch_size,
                "seed":         adapter.seed,
                **(config or {}),
            },
        },
        path,
    )
    print(f"    Attention adapter saved → {path}")


def load_attention(path: str | Path) -> AttentionResidualAdapter:
    """Load an AttentionResidualAdapter from *path*.  Returns adapter ready for transform()."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg  = ckpt.get("config") or {}
    adapter = AttentionResidualAdapter(
        n_tokens       = int(  ckpt.get("n_tokens",        ATTENTION_DEFAULT_CONFIG["n_tokens"])),
        token_dim      = int(  ckpt.get("token_dim",       ATTENTION_DEFAULT_CONFIG["token_dim"])),
        nhead          = int(  ckpt.get("nhead",           ATTENTION_DEFAULT_CONFIG["nhead"])),
        n_layers       = int(  ckpt.get("n_layers",        ATTENTION_DEFAULT_CONFIG["n_layers"])),
        dim_feedforward = int( ckpt.get("dim_feedforward", ATTENTION_DEFAULT_CONFIG["dim_feedforward"])),
        dropout        = float(ckpt.get("dropout",         ATTENTION_DEFAULT_CONFIG["dropout"])),
        lr             = float(cfg.get("lr",               ATTENTION_DEFAULT_CONFIG["lr"])),
        weight_decay   = float(cfg.get("weight_decay",     ATTENTION_DEFAULT_CONFIG["weight_decay"])),
        epochs         = int(  cfg.get("epochs",           ATTENTION_DEFAULT_CONFIG["epochs"])),
        eval_every     = int(  cfg.get("eval_every",       ATTENTION_DEFAULT_CONFIG["eval_every"])),
        batch_size     = int(  cfg.get("batch_size",       ATTENTION_DEFAULT_CONFIG["batch_size"])),
        seed           = int(  cfg.get("seed",             ATTENTION_DEFAULT_CONFIG["seed"])),
    )
    net = _AttentionNet(
        d_input=ckpt["d_input"],
        n_tokens=adapter.n_tokens,
        token_dim=adapter.token_dim,
        nhead=adapter.nhead,
        n_layers=adapter.n_layers,
        dim_feedforward=adapter.dim_feedforward,
        dropout=adapter.dropout,
    )
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    adapter._net      = net
    adapter._d_input  = ckpt["d_input"]
    adapter._d_output = ckpt["d_output"]
    print(f"    Attention adapter loaded ← {path}")
    return adapter


def save_lora(
    adapter: LoRAResidualAdapter,
    path: str | Path,
    config: dict | None = None,
) -> None:
    """Save a fitted LoRAResidualAdapter to *path* via torch.save."""
    assert adapter._net is not None, "Adapter must be fitted before saving"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": adapter._net.state_dict(),
            "d_input":    adapter._d_input,
            "d_output":   adapter._d_output,
            "rank":       adapter.rank,
            "lora_alpha": adapter.lora_alpha,
            "config": {
                "lr":           adapter.lr,
                "weight_decay": adapter.weight_decay,
                "epochs":       adapter.epochs,
                "eval_every":   adapter.eval_every,
                "batch_size":   adapter.batch_size,
                "seed":         adapter.seed,
                **(config or {}),
            },
        },
        path,
    )
    print(f"    LoRA adapter saved → {path}")


def load_lora(path: str | Path) -> LoRAResidualAdapter:
    """Load a LoRAResidualAdapter from *path*.  Returns adapter ready for transform()."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg  = ckpt.get("config") or {}
    adapter = LoRAResidualAdapter(
        rank         = int(  ckpt.get("rank",       LORA_DEFAULT_CONFIG["rank"])),
        lora_alpha   = float(ckpt.get("lora_alpha", LORA_DEFAULT_CONFIG["lora_alpha"])),
        lr           = float(cfg.get("lr",           LORA_DEFAULT_CONFIG["lr"])),
        weight_decay = float(cfg.get("weight_decay", LORA_DEFAULT_CONFIG["weight_decay"])),
        epochs       = int(  cfg.get("epochs",       LORA_DEFAULT_CONFIG["epochs"])),
        eval_every   = int(  cfg.get("eval_every",   LORA_DEFAULT_CONFIG["eval_every"])),
        batch_size   = int(  cfg.get("batch_size",   LORA_DEFAULT_CONFIG["batch_size"])),
        seed         = int(  cfg.get("seed",         LORA_DEFAULT_CONFIG["seed"])),
    )
    net = _LoRANet(
        d_input=ckpt["d_input"], d_output=ckpt["d_output"],
        rank=adapter.rank, alpha=adapter.lora_alpha,
    )
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    adapter._net      = net
    adapter._d_input  = ckpt["d_input"]
    adapter._d_output = ckpt["d_output"]
    print(f"    LoRA adapter loaded ← {path}")
    return adapter


# ---------------------------------------------------------------------------
# RWNAResidualAdapter  — frozen-encoder, latent-space adapter
# ---------------------------------------------------------------------------

class RWNAResidualAdapter:
    """
    Rank-k Whitened Nonlinear Adapter trained with AdamW + CosineAnnealingLR
    on cached latents.  Numpy-facing interface matching all other adapters.

    Args:
        k       : rank of the projection subspace (default 2, motivated by SVD)
        hidden  : hidden size of the 2-layer MLP in the projected space
    """

    def __init__(
        self,
        k:            int   = 2,
        hidden:       int   = 32,
        lr:           float = 1e-3,
        weight_decay: float = 1e-4,
        epochs:       int   = 200,
        eval_every:   int   = 20,
        batch_size:   int   = 256,
        seed:         int   = 42,
        device:       str | None = None,
    ):
        self.k            = k
        self.hidden       = hidden
        self.lr           = lr
        self.weight_decay = weight_decay
        self.epochs       = epochs
        self.eval_every   = eval_every
        self.batch_size   = batch_size
        self.seed         = seed
        self._device_str  = device
        self._net: _RWNANet | None = None
        self._d_input:  int = 192
        self._d_output: int = 192

    # -- internals -----------------------------------------------------------

    def _get_device(self) -> torch.device:
        if self._device_str:
            return torch.device(self._device_str)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # -- public API ----------------------------------------------------------

    def fit(
        self,
        z_src: np.ndarray,   # (N, d_input)  — Lead-I latents
        z_tgt: np.ndarray,   # (N, d_output) — 12-lead latents
    ) -> "RWNAResidualAdapter":
        torch.manual_seed(self.seed)
        device = self._get_device()
        self._d_input  = z_src.shape[1]
        self._d_output = z_tgt.shape[1]

        net = _RWNANet(self._d_input, self._d_output, k=self.k, hidden=self.hidden)
        net = net.to(device)
        self._net = net

        optimizer = AdamW(net.parameters(),
                          lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)

        # Residuals: predict z_tgt - z_src
        r_tr = torch.from_numpy(z_tgt - z_src).float()
        z_tr = torch.from_numpy(z_src).float()
        ds   = TensorDataset(z_tr, r_tr)
        dl   = TorchDataLoader(ds, batch_size=self.batch_size, shuffle=True)

        net.train()
        for ep in range(1, self.epochs + 1):
            total_loss = 0.0
            for zb, rb in dl:
                zb, rb = zb.to(device), rb.to(device)
                pred = net(zb)
                loss = F.mse_loss(pred, rb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * zb.size(0)
            scheduler.step()
            if ep % self.eval_every == 0 or ep == self.epochs:
                print(f"    RWNA ep {ep:4d}/{self.epochs}  "
                      f"MSE={total_loss/len(ds):.4f}")

        net.eval()
        return self

    def transform(self, z_src: np.ndarray) -> np.ndarray:
        assert self._net is not None, "Call fit() first"
        device = self._get_device()
        self._net.to(device).eval()
        with torch.no_grad():
            z = torch.from_numpy(z_src).float().to(device)
            r = self._net(z)
            return (z + r).cpu().numpy()


def save_rwna(
    adapter: RWNAResidualAdapter,
    path: str | Path,
    config: dict | None = None,
) -> None:
    """Save a fitted RWNAResidualAdapter to *path* via torch.save."""
    assert adapter._net is not None, "Adapter must be fitted before saving"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": adapter._net.state_dict(),
            "d_input":    adapter._d_input,
            "d_output":   adapter._d_output,
            "k":          adapter.k,
            "hidden":     adapter.hidden,
            "config": {
                "lr":           adapter.lr,
                "weight_decay": adapter.weight_decay,
                "epochs":       adapter.epochs,
                "eval_every":   adapter.eval_every,
                "batch_size":   adapter.batch_size,
                "seed":         adapter.seed,
                **(config or {}),
            },
        },
        path,
    )
    print(f"    RWNA adapter saved → {path}")


def load_rwna(path: str | Path) -> RWNAResidualAdapter:
    """Load a RWNAResidualAdapter from *path*.  Returns adapter ready for transform()."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg  = ckpt.get("config") or {}
    adapter = RWNAResidualAdapter(
        k            = int(  ckpt.get("k",      RWNA_DEFAULT_CONFIG["k"])),
        hidden       = int(  ckpt.get("hidden", RWNA_DEFAULT_CONFIG["hidden"])),
        lr           = float(cfg.get("lr",           RWNA_DEFAULT_CONFIG["lr"])),
        weight_decay = float(cfg.get("weight_decay", RWNA_DEFAULT_CONFIG["weight_decay"])),
        epochs       = int(  cfg.get("epochs",       RWNA_DEFAULT_CONFIG["epochs"])),
        eval_every   = int(  cfg.get("eval_every",   RWNA_DEFAULT_CONFIG["eval_every"])),
        batch_size   = int(  cfg.get("batch_size",   RWNA_DEFAULT_CONFIG["batch_size"])),
        seed         = int(  cfg.get("seed",         RWNA_DEFAULT_CONFIG["seed"])),
    )
    net = _RWNANet(
        d_input=ckpt["d_input"], d_output=ckpt["d_output"],
        k=adapter.k, hidden=adapter.hidden,
    )
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    adapter._net      = net
    adapter._d_input  = ckpt["d_input"]
    adapter._d_output = ckpt["d_output"]
    print(f"    RWNA adapter loaded ← {path}")
    return adapter
