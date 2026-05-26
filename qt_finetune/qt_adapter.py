"""
Latent-space adapters for the QT fine-tuning experiment.

Maps the single-lead (Lead I) graph_embed → 12-lead graph_embed using
the same residual formulation as the ST-segment adapters in src/:

    r_i = z_12lead_i - z_lead_i             (residual to predict)
    z_12lead_hat = z_lead_i + adapter(z_lead_i)

Three adapter families, matching the ST experiments:
  Linear  : Ridge regression         (l1_ratio = 0 in the ST ElasticNet sweep)
  Kernel  : KernelRidge + RBF kernel + StandardScaler  (same as kernel_adapter.py)
  GReLU   : Convex fixed-gate GReLU network + R-FISTA  (same as grelu_adapter.py)

Key differences from the ST adapters:
  - Input is z_lead_i alone (192-dim), not concat(phi_stress, z_rest) (384-dim)
  - Output residual r is 192-dim (same as input)
  - No stress/rest split — train on the standard training fold (strat_fold 1-8)

Latent caching (avoids re-running the encoder on every invocation):
  save_latents / load_latents   — thin np.savez / np.load wrappers
  get_latents                   — load from .npz cache if present, else extract + save
                                  Embeds checkpoint mtime; warns if cache is stale.

Checkpoint helpers (save / load):
  save_linear / load_linear   — joblib pickle of LinearResidualAdapter
  save_krr    / load_krr      — joblib pickle of KernelResidualAdapter
  save_grelu  / load_grelu    — torch.save / torch.load of GReLUResidualAdapter

Usage (imported by analyze_qtc.py and train_adapters.py):
    from qt_adapter import get_latents, predict_from_latents
    from qt_adapter import LinearResidualAdapter, KernelResidualAdapter, GReLUResidualAdapter
    from qt_adapter import save_linear, load_linear, save_krr, load_krr, save_grelu, load_grelu
"""
from __future__ import annotations

from pathlib import Path

import joblib
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
