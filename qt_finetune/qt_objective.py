"""
Tail-robust training objective for QT regression.

The 500 Hz / 128 Hz MSE baselines were accurate in the normal range (~11 ms MAE
for QT < 450) and collapsed on the prolonged cases that actually matter:
-37 ms mean error in the 450-500 band, -59 ms above 500, a pred~ref slope of
0.71, and 0/5 sensitivity at the QTc>500 torsades threshold. Two causes, and
this module addresses both.

1.  **MSE rewards hedging.** The conditional mean minimises squared error, so a
    model facing an uncertain, right-skewed target shrinks toward the centre of
    the training distribution. Pinball (quantile) loss instead makes the network
    predict conditional *quantiles*: tau=0.5 is the median (an L1 objective,
    already far less shrink-prone than L2), and the outer quantiles model the
    spread rather than averaging it away.

    Quantiles are built monotonically by construction -- a base level plus
    softplus increments -- so q05 <= q25 <= q50 <= q75 <= q95 always holds and
    no post-hoc sorting is needed.

    Deployment payoff: use q50 as the point estimate, and flag prolongation on
    an upper quantile. The MSE model already ranked long-QT cases well
    (AUC 0.85) but placed its point estimate too low to ever cross 500 ms; an
    upper quantile crosses far more readily, converting that latent
    discrimination into actual sensitivity.

2.  **The tail is rare.** In PTB-XL folds 1-8, QT>450 ms is 7.7% of records and
    QT>500 ms is 0.94%. ``TailWeighter`` computes inverse-frequency sample
    weights over QT bins, usable either as per-sample loss weights (every record
    still seen once per epoch -- gentler, less prone to memorising the ~163
    extreme records) or to drive a WeightedRandomSampler.

    The full dataset is retained either way: nothing is discarded or truncated,
    only re-weighted.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


# --------------------------------------------------------------------------
# monotonic quantile head
# --------------------------------------------------------------------------
def monotonic_quantiles(raw: torch.Tensor) -> torch.Tensor:
    """(B, K) unconstrained -> (B, K) non-decreasing across the quantile axis.

    q_0 = raw_0;  q_k = q_{k-1} + softplus(raw_k).
    """
    base = raw[:, :1]
    if raw.shape[1] == 1:
        return base
    steps = F.softplus(raw[:, 1:])
    return torch.cat([base, base + torch.cumsum(steps, dim=1)], dim=1)


def _inv_softplus(y: float) -> float:
    return float(np.log(np.expm1(max(y, 1e-4))))


def init_quantile_head(linear: nn.Linear, quantiles, center: float = 396.0,
                       spread: float = 45.0) -> None:
    """Bias-initialise the head so the quantiles start near the QT distribution.

    Without this the head starts around 0 ms against a ~400 ms target and burns
    epochs just travelling to the right neighbourhood.
    """
    q = np.asarray(quantiles, dtype=float)
    start = center + spread * (2.0 * q - 1.0)          # linear stand-in for ppf
    with torch.no_grad():
        linear.bias[0] = float(start[0])
        for k in range(1, len(q)):
            linear.bias[k] = _inv_softplus(start[k] - start[k - 1])
        linear.weight.mul_(0.01)                       # start near the prior


# --------------------------------------------------------------------------
# tail weighting
# --------------------------------------------------------------------------
class TailWeighter:
    """Inverse-frequency sample weights over QT bins.

    weight(qt) proportional to (1 / P(bin(qt))) ** alpha, normalised to mean 1
    over the training set and clipped to [1/clip, clip].

    alpha=0   uniform (no reweighting)
    alpha=0.5 sqrt-inverse frequency (default; a moderate, stable correction)
    alpha=1   full inverse frequency (aggressive, can destabilise training)
    """

    def __init__(self, qts, bin_width: float = 25.0, alpha: float = 0.5,
                 clip: float = 10.0, qt_min: float = 250.0, qt_max: float = 650.0):
        qts = np.asarray(qts, dtype=float)
        self.alpha, self.clip, self.bin_width = alpha, clip, bin_width
        self.edges = np.arange(qt_min, qt_max + bin_width, bin_width)
        idx = np.clip(np.digitize(qts, self.edges) - 1, 0, len(self.edges) - 2)
        counts = np.bincount(idx, minlength=len(self.edges) - 1).astype(float)
        p = np.maximum(counts, 1.0) / max(counts.sum(), 1.0)

        if alpha <= 0:
            w = np.ones_like(p)
        else:
            w = (1.0 / p) ** alpha
            w /= w[idx].mean()                          # E[w] = 1 on the train set
            w = np.clip(w, 1.0 / clip, clip)
            w /= w[idx].mean()                          # renormalise after clipping
        self.weights = w
        self._counts = counts
        self.sample_weights = w[idx]                    # per-record, for a sampler

    def torch_table(self, device):
        return (torch.as_tensor(self.edges, dtype=torch.float32, device=device),
                torch.as_tensor(self.weights, dtype=torch.float32, device=device))

    def describe(self) -> str:
        lines = [f"TailWeighter(alpha={self.alpha}, clip={self.clip}, "
                 f"bin={self.bin_width:g} ms)  — full dataset retained, reweighted only"]
        tot = max(self._counts.sum(), 1.0)
        for i in range(len(self.weights)):
            if self._counts[i] == 0:
                continue
            lo, hi = self.edges[i], self.edges[i + 1]
            if hi <= 350 or lo >= 450 or i % 2 == 0:     # keep the print short
                lines.append(f"    QT {lo:5.0f}-{hi:<5.0f} n={int(self._counts[i]):>6} "
                             f"({self._counts[i]/tot*100:5.2f}%)  weight {self.weights[i]:.2f}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# objective
# --------------------------------------------------------------------------
class Objective:
    """Dispatch for the two training modes, so the loop stays mode-agnostic."""

    def __init__(self, mode: str = "pinball", quantiles=DEFAULT_QUANTILES,
                 weighter: TailWeighter | None = None, device=None):
        assert mode in ("mse", "pinball")
        self.mode = mode
        self.quantiles = tuple(quantiles) if mode == "pinball" else (0.5,)
        self.weighter = weighter
        self._edges = self._wtab = None
        if weighter is not None and device is not None:
            self._edges, self._wtab = weighter.torch_table(device)
        self.median_idx = int(np.argmin(np.abs(np.asarray(self.quantiles) - 0.5)))
        self._tau = None
        self.device = device

    @property
    def num_outputs(self) -> int:
        return len(self.quantiles) if self.mode == "pinball" else 1

    def to(self, device):
        self.device = device
        if self.weighter is not None:
            self._edges, self._wtab = self.weighter.torch_table(device)
        self._tau = torch.as_tensor(self.quantiles, dtype=torch.float32,
                                    device=device).view(1, -1)
        return self

    def _weights(self, target):
        if self._wtab is None:
            return None
        idx = torch.clamp(torch.bucketize(target.flatten(), self._edges) - 1,
                          0, self._wtab.numel() - 1)
        return self._wtab[idx].view(-1, 1)

    def transform(self, logits):
        """Raw head output -> usable predictions."""
        return monotonic_quantiles(logits) if self.mode == "pinball" else logits

    def loss(self, logits, target):
        pred = self.transform(logits)
        w = self._weights(target)
        if self.mode == "mse":
            l = (pred - target) ** 2
        else:
            e = target - pred                       # (B, K) broadcast over quantiles
            l = torch.maximum(self._tau * e, (self._tau - 1.0) * e)
        if w is not None:
            l = l * w
        return l.mean()

    def point(self, logits):
        """Point estimate (median quantile) as (B, 1)."""
        pred = self.transform(logits)
        return pred[:, self.median_idx:self.median_idx + 1]

    def interval(self, logits):
        """(lo, hi) from the outermost quantiles, or (None, None) in mse mode."""
        if self.mode != "pinball" or len(self.quantiles) < 2:
            return None, None
        pred = self.transform(logits)
        return pred[:, :1], pred[:, -1:]

    def coverage(self, logits, target):
        """Fraction of targets inside the outer quantile band (calibration check)."""
        lo, hi = self.interval(logits)
        if lo is None:
            return float("nan")
        return ((target >= lo) & (target <= hi)).float().mean().item()
