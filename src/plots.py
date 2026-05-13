"""
Interpretability + diagnostic plots for the convex GReLU residual adapter.

Mirrors the kind of figures emphasized in ConvexECG (Ansari et al., 2024):
the convex formulation lets us reason directly about individual hidden
neurons (gate-activation patterns, output-weight magnitudes, contribution
sparsity), not just black-box accuracy.

Figures produced (all written to --out-dir, default: figures/):

  1.  neuron_importance.png        Sorted per-neuron output-weight L2 norm
                                   (`||proj[:, j]||_2`). Shows how many of
                                   the m gates carry meaningful signal.
  2.  gate_activation_rates.png    For each gate, fraction of training
                                   samples where the gate fires
                                   (`P(linear2(u) >= 0)` empirical).
  3.  pruning_sweep.png            Reads the CSV produced by polish.py and
                                   plots val MSE / AUROC vs kept-k.
  4.  ischemia_roc.png             ROC curve for the ischemia endpoint
                                   AUROC reported on the val split, with
                                   95% bootstrap CI.
  5.  per_dim_recovery.png         Per-latent-dim cosine similarity between
                                   z_hat and z_stress, sorted descending.
  6.  recovery_by_label.png        Latent recovery error (per patient)
                                   split by ischemia status.
  7.  baseline_vs_adapter.png      Latent MSE: rest-only vs adapter, paired
                                   per-patient scatter.

Usage:
    python src/plots.py --ckpt checkpoints/grelu_best.pt
    python src/plots.py --ckpt checkpoints/grelu_polished.pt --sweep-csv checkpoints/polish_sweep.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapter_interface import decode_st80, ischemia_patient_score, load_regression_head
from grelu_adapter import (
    build_tensors,
    load_adapter,
    load_split,
    pick_device,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bootstrap_auroc(scores: np.ndarray, labels: np.ndarray, n_boot: int = 1000,
                    seed: int = 0) -> tuple[float, float, float]:
    """Return (point, lo95, hi95) for AUROC via patient-level bootstrap."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    n = len(labels)
    point = float(roc_auc_score(labels, scores))
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(labels[idx])) < 2:
            continue
        samples.append(roc_auc_score(labels[idx], scores[idx]))
    samples = np.array(samples)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return point, float(lo), float(hi)


def per_dim_cosine(z_hat: np.ndarray, z_true: np.ndarray) -> np.ndarray:
    """Cosine similarity per latent dimension, computed across samples.

    For each column j, returns
        <z_hat[:, j], z_true[:, j]> / (||z_hat[:, j]|| ||z_true[:, j]||)
    """
    a = z_hat - z_hat.mean(axis=0, keepdims=True)
    b = z_true - z_true.mean(axis=0, keepdims=True)
    num = (a * b).sum(axis=0)
    den = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0) + 1e-12
    return num / den


# ---------------------------------------------------------------------------
# Individual figures
# ---------------------------------------------------------------------------

def plot_neuron_importance(adapter, out_path: Path):
    import matplotlib.pyplot as plt
    s = adapter.proj.weight.detach().cpu().norm(dim=0).numpy()
    s_sorted = np.sort(s)[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(s_sorted, lw=1.5)
    ax[0].set(xlabel="Neuron rank (sorted by importance)",
              ylabel="||proj column||_2",
              title=f"Per-neuron output magnitude (m={adapter.m})")
    ax[0].set_yscale("log")
    ax[0].grid(True, alpha=0.3)

    cum = np.cumsum(s_sorted ** 2) / np.sum(s_sorted ** 2)
    ax[1].plot(cum, lw=1.5)
    ax[1].axhline(0.9, color="r", ls="--", lw=1, label="90% energy")
    ax[1].axhline(0.99, color="orange", ls="--", lw=1, label="99% energy")
    ax[1].set(xlabel="Top-k neurons kept",
              ylabel="Cumulative squared importance",
              title="Cumulative energy of neurons (sorted)")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    eff90 = int(np.searchsorted(cum, 0.9) + 1)
    eff99 = int(np.searchsorted(cum, 0.99) + 1)
    print(f"  effective neurons:  90% energy = {eff90}   99% energy = {eff99}")


def plot_gate_activation_rates(adapter, u_tr: torch.Tensor, out_path: Path):
    import matplotlib.pyplot as plt
    with torch.no_grad():
        rates = (adapter.linear2(u_tr) >= 0).float().mean(dim=0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rates, bins=40, color="steelblue", edgecolor="white")
    ax.axvline(0.5, color="red", ls="--", lw=1, label="0.5 (random gate)")
    ax.set(xlabel="P(gate fires) on training set",
           ylabel="Number of neurons",
           title="Per-neuron gate activation rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  gate firing rate:  mean={rates.mean():.3f}  std={rates.std():.3f}  "
          f"min={rates.min():.3f}  max={rates.max():.3f}")


def plot_pruning_sweep(csv_path: Path, out_path: Path):
    import matplotlib.pyplot as plt
    if not csv_path.exists():
        print(f"  (skipping pruning sweep: {csv_path} not found -- run polish.py first)")
        return
    rows = [line.strip().split(",") for line in csv_path.read_text().splitlines() if line.strip()]
    header = rows[0]
    data = rows[1:]
    keep, res, lat, au = [], [], [], []
    extras = []
    for r in data:
        try:
            k = int(r[0])
        except ValueError:
            extras.append((r[0], float(r[1]), float(r[2]), float(r[3])))
            continue
        keep.append(k); res.append(float(r[1])); lat.append(float(r[2])); au.append(float(r[3]))
    if not keep:
        print(f"  (no integer keep rows in {csv_path})")
        return
    order = np.argsort(keep)
    keep = np.array(keep)[order]
    lat = np.array(lat)[order]
    au = np.array(au)[order]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(keep, lat, "o-", color="C0", lw=2, label="val latent MSE")
    ax1.set(xlabel="Top-k neurons kept (refit-then-evaluate)",
            ylabel="Val latent MSE")
    ax1.set_xscale("log", base=2)
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(keep, au, "s--", color="C3", lw=2, label="val AUROC")
    ax2.set_ylabel("Val AUROC")
    for name, _, lat_e, au_e in extras:
        ax1.axhline(lat_e, color="gray", ls=":", lw=1)
        ax1.text(keep.min(), lat_e, f" {name}", color="gray", va="bottom", fontsize=8)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax1.set_title("Pruning sweep: accuracy vs effective width")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ischemia_roc(scores: np.ndarray, labels: np.ndarray, out_path: Path):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    auc_pt, lo, hi = bootstrap_auroc(scores, labels)
    fpr, tpr, _ = roc_curve(labels, scores)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2,
            label=f"GReLU adapter (AUROC = {auc_pt:.3f}, 95% CI [{lo:.3f}, {hi:.3f}])")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="Ischemia detection through frozen ST80 head")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  AUROC = {auc_pt:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")


def plot_per_dim_recovery(z_hat: np.ndarray, z_true: np.ndarray, out_path: Path):
    import matplotlib.pyplot as plt
    cos = per_dim_cosine(z_hat, z_true)
    order = np.argsort(cos)[::-1]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(np.arange(len(cos)), cos[order], color="steelblue", width=1.0)
    ax.set(xlabel="Latent dimension (sorted by recovery quality)",
           ylabel="Cosine similarity (across samples)",
           title=f"Per-dimension latent recovery  (mean={cos.mean():.3f})")
    ax.axhline(0, color="k", lw=0.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  per-dim cosine:  mean={cos.mean():.3f}  median={np.median(cos):.3f}  "
          f"frac>0.5={np.mean(cos > 0.5):.2%}")


def plot_recovery_by_label(z_hat: np.ndarray, z_true: np.ndarray,
                           labels: np.ndarray, out_path: Path):
    import matplotlib.pyplot as plt
    err = np.linalg.norm(z_hat - z_true, axis=1)
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, np.quantile(err, 0.99), 40)
    ax.hist(err[labels == 0], bins=bins, alpha=0.6, label="non-ischemic", color="C0")
    ax.hist(err[labels == 1], bins=bins, alpha=0.6, label="ischemic", color="C3")
    ax.set(xlabel="||z_hat - z_stress||_2  (per patient)",
           ylabel="Count",
           title="Latent recovery error by ischemia status")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  recovery error:  non-ischemic mean={err[labels==0].mean():.3f}   "
          f"ischemic mean={err[labels==1].mean():.3f}")


LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def plot_st80_per_lead_pearson(z_hat: np.ndarray, z_true: np.ndarray,
                                bl_st80: np.ndarray, head, out_path: Path):
    """Per-lead Pearson correlation between predicted and true ST80 on val.

    Two panels:
        (a) absolute ST80:   corr( head(z_hat)[:, k] ,  head(z_true)[:, k] )
        (b) delta ST80:      corr( head(z_hat)[:, k] - bl_st80[:, k] ,
                                   head(z_true)[:, k] - bl_st80[:, k] )

    Delta is what clinicians actually use (stress - rest), so it's the
    more meaningful number. Bars are coloured by ischemia territory.
    """
    import matplotlib.pyplot as plt
    from adapter_interface import TERRITORIES

    pred_st80 = decode_st80(head, z_hat)
    true_st80 = decode_st80(head, z_true)
    pred_delta = pred_st80 - bl_st80
    true_delta = true_st80 - bl_st80

    n_leads = pred_st80.shape[1]
    assert n_leads == len(LEAD_NAMES), (
        f"Expected 12 leads, got {n_leads}"
    )

    def per_lead_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        out = np.zeros(a.shape[1])
        for k in range(a.shape[1]):
            ak = a[:, k] - a[:, k].mean()
            bk = b[:, k] - b[:, k].mean()
            denom = (np.linalg.norm(ak) * np.linalg.norm(bk)) + 1e-12
            out[k] = float((ak * bk).sum() / denom)
        return out

    rho_abs = per_lead_pearson(pred_st80, true_st80)
    rho_delta = per_lead_pearson(pred_delta, true_delta)

    # Map each lead index -> territory color (gray if not in any territory)
    territory_color = {
        "Inferior": "#C44E52",
        "Anterior": "#4C72B0",
        "Lateral":  "#55A868",
    }
    lead_color = ["#888888"] * n_leads
    for name, idxs in TERRITORIES.items():
        for i in idxs:
            lead_color[i] = territory_color[name]

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)

    for ax, rho, title in [
        (axs[0], rho_abs,   "Pearson r:  predicted ST80 vs true ST80"),
        (axs[1], rho_delta, "Pearson r:  predicted dST80 vs true dST80"),
    ]:
        bars = ax.bar(range(n_leads), rho, color=lead_color, edgecolor="black", linewidth=0.4)
        ax.axhline(rho.mean(), color="black", lw=1, ls="--",
                   label=f"mean r = {rho.mean():.3f}")
        ax.set_xticks(range(n_leads))
        ax.set_xticklabels(LEAD_NAMES)
        ax.set_ylim(min(-0.05, rho.min() - 0.05), 1.05)
        ax.set_ylabel("Pearson r")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="lower right")
        # Per-bar value annotation
        for b, v in zip(bars, rho):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02 if v >= 0 else v - 0.04,
                    f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=8)

    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, label=name) for name, c in territory_color.items()]
    legend_handles.append(Patch(facecolor="#888888", label="Other"))
    fig.legend(handles=legend_handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Console summary
    print(f"  abs ST80   mean r = {rho_abs.mean():.3f}   "
          f"min = {rho_abs.min():.3f} ({LEAD_NAMES[rho_abs.argmin()]})   "
          f"max = {rho_abs.max():.3f} ({LEAD_NAMES[rho_abs.argmax()]})")
    print(f"  d  ST80   mean r = {rho_delta.mean():.3f}   "
          f"min = {rho_delta.min():.3f} ({LEAD_NAMES[rho_delta.argmin()]})   "
          f"max = {rho_delta.max():.3f} ({LEAD_NAMES[rho_delta.argmax()]})")


def plot_sweep_boxplot(jsonl_path: Path, out_path: Path, metric: str = "val_latent_mse",
                       also_auroc: bool = True, min_runs: int = 5,
                       include_configs: list[str] | None = None,
                       include_methods: list[str] | None = None):
    """Side-by-side box plots of sweep results, grouped by (method, config).

    Reads JSONL records written by `nonconvex_MLP.py` and `grelu_sweep.py`,
    groups by `method` -> `config`, and shows distribution across the
    hyperparameter grid.

    The JSONL is append-only, so old / legacy / partial sweeps can leave
    stale records around. To suppress those, configs with fewer than
    `min_runs` records are dropped. Pass explicit `include_configs` /
    `include_methods` to override.
    """
    import matplotlib.pyplot as plt

    if not jsonl_path.exists():
        print(f"  (skipping sweep box plot: {jsonl_path} not found -- "
              f"run nonconvex_MLP.py and grelu_sweep.py first)")
        return

    records = []
    for line in jsonl_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not records:
        print(f"  (no records in {jsonl_path})")
        return

    grelu_order = ["m=128", "m=256", "m=512", "m=1024", "m=2048", "m=4096"]
    mlp_order = ["tiny", "small", "medium", "wide", "deep"]

    grouped: dict[tuple[str, str], list[dict]] = {}
    for r in records:
        grouped.setdefault((r["method"], r["config"]), []).append(r)

    # Filter out groups that don't meet the min_runs threshold or the
    # explicit include lists.
    dropped: list[tuple[tuple[str, str], int, str]] = []
    filtered: dict[tuple[str, str], list[dict]] = {}
    for key, runs in grouped.items():
        method, config = key
        if include_methods is not None and method not in include_methods:
            dropped.append((key, len(runs), "method not in --include-methods"))
            continue
        if include_configs is not None and config not in include_configs:
            dropped.append((key, len(runs), "config not in --include-configs"))
            continue
        if len(runs) < min_runs:
            dropped.append((key, len(runs), f"<{min_runs} runs (likely legacy)"))
            continue
        filtered[key] = runs
    grouped = filtered

    # Stable ordering: GReLU configs in width order, then MLP configs by size
    keys: list[tuple[str, str]] = []
    for c in grelu_order:
        if ("GReLU", c) in grouped:
            keys.append(("GReLU", c))
    for c in sorted(k[1] for k in grouped if k[0] == "GReLU" and k[1] not in grelu_order):
        keys.append(("GReLU", c))
    for c in mlp_order:
        if ("MLP", c) in grouped:
            keys.append(("MLP", c))
    for c in sorted(k[1] for k in grouped if k[0] == "MLP" and k[1] not in mlp_order):
        keys.append(("MLP", c))

    for key, n, reason in dropped:
        print(f"  (dropped {key[0]:<6s} {key[1]:<10s}: {reason}, n={n})")
    if not keys:
        print(f"  (nothing left after filtering -- try lower --min-runs or "
              f"explicit --include-configs)")
        return

    labels = [f"{m}\n{c}" for m, c in keys]
    box_data = [[r[metric] for r in grouped[k]] for k in keys]
    n_per = max(len(v) for v in box_data)
    colors = ["#4C72B0" if m == "GReLU" else "#DD8452" for m, _ in keys]

    if also_auroc:
        au_data = [[r["val_auroc"] for r in grouped[k]] for k in keys]
        fig, axs = plt.subplots(1, 2, figsize=(max(11, len(keys) * 1.2), 5))
    else:
        au_data = None
        fig, axs = plt.subplots(1, 1, figsize=(max(7, len(keys) * 1.2), 5))
        axs = [axs]

    bp1 = axs[0].boxplot(box_data, tick_labels=labels, patch_artist=True, widths=0.6)
    for patch, color in zip(bp1["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    # Overlay individual seed points
    for i, runs in enumerate(box_data, start=1):
        x = np.full(len(runs), i, dtype=float) + np.random.uniform(-0.08, 0.08, size=len(runs))
        axs[0].plot(x, runs, "o", color="black", markersize=4, alpha=0.7)
    axs[0].set_title(f"Val {metric}  (n={n_per} hp-combos per config)")
    axs[0].set_ylabel(metric)
    axs[0].grid(True, axis="y", alpha=0.3)
    axs[0].tick_params(axis="x", rotation=0)

    if au_data is not None:
        bp2 = axs[1].boxplot(au_data, tick_labels=labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp2["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(0.75)
        for i, runs in enumerate(au_data, start=1):
            x = np.full(len(runs), i, dtype=float) + np.random.uniform(-0.08, 0.08, size=len(runs))
            axs[1].plot(x, runs, "o", color="black", markersize=4, alpha=0.7)
        axs[1].set_title(f"Val AUROC  (n={n_per} hp-combos per config)")
        axs[1].set_ylabel("AUROC")
        axs[1].grid(True, axis="y", alpha=0.3)
        axs[1].tick_params(axis="x", rotation=0)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#4C72B0", alpha=0.75, label="Convex GReLU"),
        Patch(facecolor="#DD8452", alpha=0.75, label="Non-convex MLP"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Print a quick summary
    print(f"  configs: {len(keys)}   total runs: {sum(len(v) for v in box_data)}")
    for k, runs in zip(keys, box_data):
        arr = np.array(runs)
        print(f"    {k[0]:<6s} {k[1]:<10s}  {metric}: {arr.mean():.4f} "
              f"+/- {arr.std():.4f}   n={len(arr)}")


def plot_baseline_vs_adapter(z_rest: np.ndarray, z_hat: np.ndarray,
                             z_stress: np.ndarray, out_path: Path):
    import matplotlib.pyplot as plt
    rest_err = np.linalg.norm(z_rest - z_stress, axis=1)
    adapter_err = np.linalg.norm(z_hat - z_stress, axis=1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    mx = max(rest_err.max(), adapter_err.max()) * 1.05
    ax[0].scatter(rest_err, adapter_err, s=8, alpha=0.5)
    ax[0].plot([0, mx], [0, mx], "--", color="gray", lw=1)
    ax[0].set(xlabel="Rest-only error  ||z_rest - z_stress||",
              ylabel="Adapter error  ||z_hat - z_stress||",
              title="Per-patient recovery error (below diagonal = adapter wins)")
    ax[0].grid(True, alpha=0.3)
    ax[0].set_xlim(0, mx); ax[0].set_ylim(0, mx)

    improvement = rest_err - adapter_err
    ax[1].hist(improvement, bins=40, color="C2", edgecolor="white")
    ax[1].axvline(0, color="r", ls="--", lw=1)
    ax[1].set(xlabel="Improvement: rest_err - adapter_err",
              ylabel="Count",
              title=f"Per-patient improvement  (mean={improvement.mean():+.3f})")
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    win = (improvement > 0).mean()
    print(f"  adapter beats rest-only on {win:.1%} of val patients")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(CKPT_DIR / "grelu_best.pt"))
    ap.add_argument("--sweep-csv", type=str, default=str(CKPT_DIR / "polish_sweep.csv"))
    ap.add_argument("--sweep-jsonl", type=str, default=str(CKPT_DIR / "sweep_results.jsonl"),
                    help="JSONL written by nonconvex_MLP.py and grelu_sweep.py")
    ap.add_argument("--min-runs", type=int, default=5,
                    help="Drop sweep configs with fewer than this many records "
                         "(filters out legacy seed-variance runs).")
    ap.add_argument("--include-configs", type=str, nargs="+", default=None,
                    help="Override --min-runs and show exactly these config names "
                         "(e.g. m=256 m=1024 m=4096 tiny medium wide).")
    ap.add_argument("--include-methods", type=str, nargs="+", default=None,
                    choices=["GReLU", "MLP"],
                    help="Limit box plot to these methods.")
    ap.add_argument("--out-dir", type=str, default=str(FIG_DIR))
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--max-train-samples", type=int, default=4000,
                    help="Cap on # train samples used for gate-rate histogram")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)

    print(f"Loading data ...   device: {device}")
    train_split = load_split(DATA_DIR, "train")
    val_split = load_split(DATA_DIR, "val")
    head = load_regression_head(DATA_DIR / "regression_head.pt")

    print(f"Loading adapter from {args.ckpt} ...")
    adapter = load_adapter(args.ckpt, device=device)
    adapter.eval()

    u_va, _, zr_va, zs_va = build_tensors(val_split, device=device)
    with torch.no_grad():
        z_hat_va = (zr_va + adapter(u_va)).cpu().numpy()
    z_true_va = val_split["z_stress"]
    z_rest_va = val_split["z_rest"]

    # Use a bounded sample of train for gate stats
    n = min(args.max_train_samples, len(train_split["z_rest"]))
    sub = {k: v[:n] for k, v in train_split.items()}
    u_tr, *_ = build_tensors(sub, device=device)

    # ---- figures ------------------------------------------------------------
    print("\n[1/7] neuron importance ...")
    plot_neuron_importance(adapter, out_dir / "neuron_importance.png")

    print("\n[2/7] gate activation rates ...")
    plot_gate_activation_rates(adapter, u_tr, out_dir / "gate_activation_rates.png")

    print("\n[3/7] pruning sweep ...")
    plot_pruning_sweep(Path(args.sweep_csv), out_dir / "pruning_sweep.png")

    print("\n[4/7] ischemia ROC ...")
    pred_st80 = decode_st80(head, z_hat_va)
    scores = ischemia_patient_score(pred_st80, val_split["bl_st80"])
    plot_ischemia_roc(scores, val_split["ischemia"], out_dir / "ischemia_roc.png")

    print("\n[5/7] per-dimension recovery ...")
    plot_per_dim_recovery(z_hat_va, z_true_va, out_dir / "per_dim_recovery.png")

    print("\n[6/7] recovery by label ...")
    plot_recovery_by_label(z_hat_va, z_true_va, val_split["ischemia"].astype(int),
                           out_dir / "recovery_by_label.png")

    print("\n[7/9] baseline vs adapter ...")
    plot_baseline_vs_adapter(z_rest_va, z_hat_va, z_true_va,
                             out_dir / "baseline_vs_adapter.png")

    print("\n[8/9] per-lead ST80 Pearson ...")
    plot_st80_per_lead_pearson(z_hat_va, z_true_va, val_split["bl_st80"], head,
                               out_dir / "st80_per_lead_pearson.png")

    print("\n[9/9] sweep box plot (GReLU vs MLP) ...")
    plot_sweep_boxplot(
        Path(args.sweep_jsonl), out_dir / "sweep_boxplot.png",
        min_runs=args.min_runs,
        include_configs=args.include_configs,
        include_methods=args.include_methods,
    )

    print(f"\nAll figures written to {out_dir}/")


if __name__ == "__main__":
    main()
