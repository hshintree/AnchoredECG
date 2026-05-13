"""
Convex GReLU Residual Adapter for prior-anchored stress ECG latent recovery.

Adapts the fixed-gate convex distillation idea (Varshney & Pilanci, 2024) from
the SVHN MyConvexLayer4 prototype to a fully-connected 1D embedding adapter.

Problem:
    u = [phi_stress; z_rest]    in R^384   (input)
    r = z_stress - z_rest       in R^192   (target residual)
    z_hat_stress = z_rest + adapter(u)

The adapter is a two-layer fixed-gate convex GReLU network:
    gate     = (linear2(u) >= 0)        # FROZEN after random init
    features = linear1(u) * gate        # GReLU
    output   = proj(features)

Because the gate weights linear2 are sampled and frozen, the optimization in
(linear1, proj) is convex (it's an MSE objective in the remaining linear
parameters with the per-sample gate pattern fixed).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapter_interface import AdapterBase, decode_st80, ischemia_patient_score, load_regression_head


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "d_input": 384,          # 192 (phi_stress) + 192 (z_rest)
    "d_output": 192,         # residual dimension
    "m": 1024,               # number of GReLU neurons
    "lr": 1e-3,
    "epochs": 100,
    "batch_size": 256,
    "eval_every": 10,
    "seed": 42,
    "lambda_l2": 1e-4,       # weight decay on trainable params
    "device": None,          # None -> auto-detect cuda/mps/cpu
}


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GReLUResidualAdapter(nn.Module):
    """Two-layer fixed-gate convex GReLU adapter.

    Mirrors the conv1 (trainable) / conv2 (frozen gate) / conv3 (projection)
    pattern of MyConvexLayer4 from the SVHN notebook, but on 1D embeddings:

        linear1: nn.Linear(d_input, m, bias=False)        # trainable features
        linear2: nn.Linear(d_input, m, bias=False)        # FROZEN gate
        proj   : nn.Linear(m, d_output, bias=True)        # trainable projection

    Forward:
        gate     = (linear2(u) >= 0).float()
        features = linear1(u) * gate
        return     proj(features)
    """

    def __init__(self, d_input: int = 384, d_output: int = 192, m: int = 1024,
                 seed: int = 42):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output
        self.m = m
        self.seed = seed

        # Reproducible gate init independent of any later global seeding
        gate_gen = torch.Generator().manual_seed(seed)

        self.linear1 = nn.Linear(d_input, m, bias=False)
        self.linear2 = nn.Linear(d_input, m, bias=False)
        self.proj = nn.Linear(m, d_output, bias=True)

        # Reseed linear2 deterministically from the gate generator, then freeze.
        # We use Kaiming-uniform style scaling matching nn.Linear's default.
        bound = (1.0 / d_input) ** 0.5
        with torch.no_grad():
            self.linear2.weight.uniform_(-bound, bound, generator=gate_gen)
        self.linear2.weight.requires_grad_(False)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # Hard binary gate, no gradient flows through it
        gate = (self.linear2(u) >= 0).to(u.dtype)
        features = self.linear1(u) * gate
        return self.proj(features)

    # convenience: predict the full recovered latent
    @torch.no_grad()
    def predict_latent(self, phi: torch.Tensor, z_rest: torch.Tensor) -> torch.Tensor:
        u = torch.cat([phi, z_rest], dim=1)
        return z_rest + self.forward(u)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_split(data_dir: Path | str, name: str) -> dict:
    """Load a precomputed embedding split. Splits are patient-level upstream."""
    d = np.load(Path(data_dir) / f"{name}.npz")
    return {k: d[k] for k in d.files}


def build_tensors(split: dict, device: torch.device | None = None
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (U, R, z_rest, z_stress) tensors from a split dict."""
    phi = torch.from_numpy(split["phi_stress"]).float()
    z_rest = torch.from_numpy(split["z_rest"]).float()
    z_stress = torch.from_numpy(split["z_stress"]).float()
    u = torch.cat([phi, z_rest], dim=1)
    r = z_stress - z_rest
    if device is not None:
        u, r, z_rest, z_stress = (t.to(device) for t in (u, r, z_rest, z_stress))
    return u, r, z_rest, z_stress


def make_loader(u: torch.Tensor, r: torch.Tensor, batch_size: int, shuffle: bool,
                seed: int = 0) -> DataLoader:
    ds = TensorDataset(u, r)
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainHistory:
    train_mse: list[float] = field(default_factory=list)
    val_residual_mse: list[float] = field(default_factory=list)
    val_latent_mse: list[float] = field(default_factory=list)
    val_auroc: list[float] = field(default_factory=list)
    best_val_latent_mse: float = float("inf")
    best_epoch: int = -1


def train_grelu_adapter(
    train_split: dict,
    val_split: dict | None = None,
    head: nn.Module | None = None,
    config: dict | None = None,
    ckpt_path: str | Path | None = None,
    verbose: bool = True,
) -> tuple[GReLUResidualAdapter, TrainHistory]:
    """Train the GReLU residual adapter on MSE between adapter(u) and r.

    Args:
        train_split: dict with z_rest, z_stress, phi_stress (numpy float32).
        val_split:   same shape, used for periodic eval. Optional.
        head:        frozen ST80 regression head for ischemia AUROC. Optional.
        config:      see DEFAULT_CONFIG.
        ckpt_path:   if given, the best (lowest val_latent_mse) model state_dict
                     is saved to this path during training.

    Returns:
        (adapter, history)
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    device = pick_device(cfg["device"])
    torch.manual_seed(cfg["seed"])

    # Build tensors
    u_tr, r_tr, _, _ = build_tensors(train_split, device=device)
    if val_split is not None:
        u_va, r_va, zr_va, zs_va = build_tensors(val_split, device=device)
        y_va = val_split.get("ischemia")
        bl_va = val_split.get("bl_st80")

    # Model and optimizer
    adapter = GReLUResidualAdapter(
        d_input=cfg["d_input"], d_output=cfg["d_output"],
        m=cfg["m"], seed=cfg["seed"],
    ).to(device)
    optim = torch.optim.Adam(
        adapter.trainable_parameters(), lr=cfg["lr"], weight_decay=cfg["lambda_l2"],
    )
    loss_fn = nn.MSELoss()

    loader = make_loader(u_tr, r_tr, cfg["batch_size"], shuffle=True, seed=cfg["seed"])
    history = TrainHistory()

    for epoch in range(1, cfg["epochs"] + 1):
        adapter.train()
        running, n = 0.0, 0
        for ub, rb in loader:
            optim.zero_grad()
            pred = adapter(ub)
            loss = loss_fn(pred, rb)
            loss.backward()
            optim.step()
            running += loss.item() * ub.size(0)
            n += ub.size(0)
        train_mse = running / max(n, 1)
        history.train_mse.append(train_mse)

        do_eval = val_split is not None and (
            epoch % cfg["eval_every"] == 0 or epoch == cfg["epochs"]
        )
        if do_eval:
            metrics = evaluate_residual_adapter(adapter, u_va, r_va, zr_va, zs_va,
                                                head=head, y=y_va, bl_st80=bl_va)
            history.val_residual_mse.append(metrics["residual_mse"])
            history.val_latent_mse.append(metrics["latent_mse"])
            history.val_auroc.append(metrics.get("auroc", float("nan")))

            if metrics["latent_mse"] < history.best_val_latent_mse:
                history.best_val_latent_mse = metrics["latent_mse"]
                history.best_epoch = epoch
                if ckpt_path is not None:
                    save_adapter(adapter, ckpt_path, cfg)

            if verbose:
                print(
                    f"  epoch {epoch:4d}  train_mse={train_mse:.6f}  "
                    f"val_resid_mse={metrics['residual_mse']:.6f}  "
                    f"val_latent_mse={metrics['latent_mse']:.6f}  "
                    f"val_auroc={metrics.get('auroc', float('nan')):.4f}"
                )
        elif verbose and epoch % max(1, cfg["eval_every"] // 2) == 0:
            print(f"  epoch {epoch:4d}  train_mse={train_mse:.6f}")

    return adapter, history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_residual_adapter(
    adapter: GReLUResidualAdapter,
    u: torch.Tensor,
    r: torch.Tensor,
    z_rest: torch.Tensor,
    z_stress: torch.Tensor,
    head: nn.Module | None = None,
    y: np.ndarray | None = None,
    bl_st80: np.ndarray | None = None,
) -> dict:
    """Compute residual MSE, latent recovery MSE, and (optionally) AUROC."""
    adapter.eval()
    r_hat = adapter(u)
    z_hat = z_rest + r_hat

    # Per-sample squared error summed over feature dim, averaged over batch.
    residual_mse = ((r_hat - r) ** 2).sum(dim=1).mean().item()
    latent_mse = ((z_hat - z_stress) ** 2).sum(dim=1).mean().item()

    out = {"residual_mse": residual_mse, "latent_mse": latent_mse}

    if head is not None and y is not None and bl_st80 is not None:
        try:
            from sklearn.metrics import roc_auc_score
            pred_st80 = decode_st80(head, z_hat.cpu().numpy())
            scores = ischemia_patient_score(pred_st80, bl_st80)
            if len(np.unique(y)) >= 2:
                out["auroc"] = float(roc_auc_score(y, scores))
            else:
                out["auroc"] = float("nan")
        except ImportError:
            out["auroc"] = float("nan")

    return out


def seed_variance(
    train_split: dict,
    val_split: dict,
    head: nn.Module | None = None,
    seeds: tuple[int, ...] = (42, 43, 44),
    config: dict | None = None,
    verbose: bool = True,
) -> dict:
    """Run training `len(seeds)` times and report mean/std of val metrics."""
    results = {"latent_mse": [], "residual_mse": [], "auroc": []}
    for s in seeds:
        cfg = {**(config or {}), "seed": int(s)}
        if verbose:
            print(f"\n=== seed {s} ===")
        adapter, hist = train_grelu_adapter(
            train_split, val_split, head=head, config=cfg, verbose=verbose,
        )
        # Final epoch metrics from history
        results["latent_mse"].append(hist.val_latent_mse[-1])
        results["residual_mse"].append(hist.val_residual_mse[-1])
        if hist.val_auroc:
            results["auroc"].append(hist.val_auroc[-1])

    summary = {}
    for k, v in results.items():
        if v:
            arr = np.array(v, dtype=float)
            summary[f"{k}_mean"] = float(np.nanmean(arr))
            summary[f"{k}_std"] = float(np.nanstd(arr, ddof=0))
    summary["seeds"] = list(seeds)
    return summary


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_adapter(adapter: GReLUResidualAdapter, path: str | Path,
                 config: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": adapter.state_dict(),
        "d_input": adapter.d_input,
        "d_output": adapter.d_output,
        "m": adapter.m,
        "seed": adapter.seed,
        "config": config,
    }, path)


def load_adapter(path: str | Path, device: torch.device | None = None
                 ) -> GReLUResidualAdapter:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    adapter = GReLUResidualAdapter(
        d_input=ckpt["d_input"], d_output=ckpt["d_output"],
        m=ckpt["m"], seed=ckpt["seed"],
    )
    adapter.load_state_dict(ckpt["state_dict"])
    if device is not None:
        adapter = adapter.to(device)
    adapter.eval()
    return adapter


# ---------------------------------------------------------------------------
# AdapterBase wrapper (for compatibility with src/evaluate.py)
# ---------------------------------------------------------------------------

class GReLUAdapter(AdapterBase):
    """Numpy-facing wrapper around GReLUResidualAdapter."""

    def __init__(self, d_in: int = 384, d_out: int = 192, n_gates: int = 1024,
                 lam: float = 1e-4, seed: int = 42, config: dict | None = None):
        # Accept both the original (d_in/d_out/n_gates/lam) signature used by
        # the existing evaluate.py and the new config-dict style.
        self.config = {
            **DEFAULT_CONFIG,
            "d_input": d_in,
            "d_output": d_out,
            "m": n_gates,
            "lambda_l2": lam,
            "seed": seed,
        }
        if config:
            self.config.update(config)
        self.adapter: GReLUResidualAdapter | None = None
        self.device = pick_device(self.config["device"])

    def fit(self, phi_train, z_rest_train, z_stress_train):
        train_split = {
            "phi_stress": np.asarray(phi_train, dtype=np.float32),
            "z_rest":     np.asarray(z_rest_train, dtype=np.float32),
            "z_stress":   np.asarray(z_stress_train, dtype=np.float32),
        }
        self.adapter, _ = train_grelu_adapter(
            train_split, val_split=None, head=None, config=self.config, verbose=True,
        )
        return self

    @torch.no_grad()
    def predict(self, phi, z_rest):
        assert self.adapter is not None, "Call fit() or load() before predict()"
        phi_t = torch.from_numpy(np.asarray(phi, dtype=np.float32)).to(self.device)
        zr_t = torch.from_numpy(np.asarray(z_rest, dtype=np.float32)).to(self.device)
        z_hat = self.adapter.predict_latent(phi_t, zr_t)
        return z_hat.cpu().numpy()

    def save(self, path):
        assert self.adapter is not None, "Nothing to save -- adapter is untrained"
        save_adapter(self.adapter, path, config=self.config)

    @classmethod
    def load(cls, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config") or {}
        instance = cls(
            d_in=ckpt["d_input"], d_out=ckpt["d_output"],
            n_gates=ckpt["m"], seed=ckpt["seed"],
            config=cfg,
        )
        instance.adapter = load_adapter(path, device=instance.device)
        return instance
