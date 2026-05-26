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
  8.  sweep / model-comparison figures (JSONL checkpoints, ROC extensions, ...)
  ... st80_per_lead_pearson.png, sweep_boxplot*.png, model_comparison.png ...

  Additional outputs when adapter checkpoints exist (see CLI paths):
  •  ischemia_roc_methods.png       Multi-method ischemia ROC overlay
  •  st80_per_lead_pearson_methods.png (+ .csv)   Per-lead dST80 Pearson — all checkpoints

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


def precision_at_sensitivity(scores: np.ndarray, labels: np.ndarray,
                              target_sens: float = 0.80) -> float:
    """Precision (PPV) at the operating point that achieves >= `target_sens` recall.

    Walks the ROC, picks the threshold where sensitivity (TPR) first reaches the
    target, returns precision at that threshold. NaN if not reachable.
    """
    from sklearn.metrics import precision_recall_curve
    prec, rec, _ = precision_recall_curve(labels, scores)
    # precision_recall_curve returns prec/rec sorted by ascending threshold,
    # which corresponds to recall going from 1 -> 0 (last point is recall=0).
    # Find points where recall >= target_sens, pick the max precision among them.
    mask = rec >= target_sens
    if not mask.any():
        return float("nan")
    return float(prec[mask].max())


def per_lead_delta_pearson(z_hat: np.ndarray, z_true: np.ndarray,
                           bl_st80: np.ndarray, head) -> np.ndarray:
    """Pearson r per lead between predicted dST80 and true dST80 (stress-rest)."""
    pred_d = decode_st80(head, z_hat) - bl_st80
    true_d = decode_st80(head, z_true) - bl_st80
    out = np.zeros(pred_d.shape[1])
    for k in range(pred_d.shape[1]):
        ak = pred_d[:, k] - pred_d[:, k].mean()
        bk = true_d[:, k] - true_d[:, k].mean()
        den = (np.linalg.norm(ak) * np.linalg.norm(bk)) + 1e-12
        out[k] = float((ak * bk).sum() / den)
    return out


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


def plot_multi_method_ischemia_roc_overlay(
    scores_by_model: dict[str, np.ndarray],
    labels: np.ndarray,
    out_path: Path,
    *,
    title: str = "Ischemia ROC — all available checkpoints (val)",
) -> None:
    """One figure, one ROC curve per model (frozen ST80 head + patient scores)."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
               "#937860", "#DA8BC3", "#8C8C8C", "#E17C05", "#76B7B2"]

    labels = np.asarray(labels)
    fig, ax = plt.subplots(figsize=(7.0, 6.2))

    print(f"  overlaying ROC for {len(scores_by_model)} models ...")
    for i, (name, scores) in enumerate(scores_by_model.items()):
        color = palette[i % len(palette)]
        if len(np.unique(labels)) < 2:
            auc = float("nan")
            fpr = np.array([0.0, 1.0])
            tpr = np.array([0.0, 1.0])
        else:
            auc = float(roc_auc_score(labels, scores))
            fpr, tpr, _ = roc_curve(labels, scores)
        ax.plot(fpr, tpr, lw=2.0, color=color,
                label=f"{name.replace(chr(10), ' ')} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1.0, label="chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=title)
    ax.legend(loc="lower right", fontsize=7.8, framealpha=0.92)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_multi_method_per_lead_pearson_panel(
    preds: dict[str, np.ndarray],
    z_true: np.ndarray,
    bl_st80: np.ndarray,
    head,
    out_path: Path,
    csv_path: Path | None = None,
) -> None:
    """Grouped bar chart: every model × 12 leads, delta-ST80 Pearson r (val).

    Matches the interpretation used in plot_model_comparison; also writes an
    optional CSV (`method × lead`).
    """
    import csv
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    from adapter_interface import TERRITORIES

    names = list(preds.keys())
    mat = np.stack([per_lead_delta_pearson(preds[k], z_true, bl_st80, head) for k in names])
    n_models, n_leads = mat.shape
    assert n_leads == len(LEAD_NAMES)

    fig, ax = plt.subplots(figsize=(16, 5.2))
    x = np.arange(n_leads)
    width = 0.8 / max(n_models, 1)
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
               "#937860", "#DA8BC3", "#8C8C8C", "#E17C05", "#76B7B2"]

    for i, name in enumerate(names):
        xpos = x + (i - (n_models - 1) / 2) * width
        rho = mat[i]
        label = f"{name.replace(chr(10), ' ')}  (mean r = {rho.mean():.3f})"
        ax.bar(xpos, rho, width=width, color=palette[i % len(palette)],
               edgecolor="black", linewidth=0.35, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(LEAD_NAMES)
    ax.set_ylabel("Pearson r")
    ax.set_ylim(-0.15, 1.05)
    ax.axhline(0.0, color="black", lw=0.6)
    ax.set_title("Per-lead Pearson r (val): predicted dST80 vs true dST80 — all checkpoints")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=2, fontsize=7.5)

    band = {"Inferior": "#C44E52", "Anterior": "#4C72B0", "Lateral": "#55A868"}
    for terr, idxs in TERRITORIES.items():
        for j in idxs:
            ax.add_patch(Rectangle(
                (j - 0.5, ax.get_ylim()[0]), 1.0, ax.get_ylim()[1] - ax.get_ylim()[0],
                facecolor=band[terr], alpha=0.06, zorder=-1,
            ))

    territory_handles = [Patch(facecolor=c, label=n) for n, c in band.items()]
    fig.legend(handles=territory_handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02), fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")

    if csv_path is not None:
        with csv_path.open("w", newline="") as cf:
            w = csv.writer(cf)
            w.writerow(["method", *LEAD_NAMES, "mean_r"])
            for name, row in zip(names, mat):
                w.writerow([name.replace("\n", " ")] + [f"{v:.5f}" for v in row]
                           + [f"{row.mean():.5f}"])
        print(f"  wrote {csv_path}")


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


def plot_sweep_boxplot(jsonl_path: Path, out_path: Path, min_runs: int = 1,
                       include_configs: list[str] | None = None,
                       include_methods: list[str] | None = None,
                       latest_only: bool = True):
    """AUROC box plot of sweep results, grouped by (method, config).

    Reads JSONL records written by `nonconvex_MLP.py` and `grelu_sweep.py`,
    groups by `method` -> `config`, and shows the val AUROC distribution
    across the hyperparameter grid (one point per hyperparameter combo).

    Each record carries a `run_id` (timestamp). With `latest_only=True`
    (default) we keep, per method, only the records belonging to the most
    recent run_id seen in the file -- so stale older sweeps don't pollute
    the boxes. Records without a run_id are treated as a 'legacy' group
    and only kept if no record in their method has a run_id.

    The box labels show the actual trial count (n=X) so you can see at a
    glance how many hyperparameter points the box summarises. Configs with
    fewer than `min_runs` records are dropped; the default 1 means show
    everything.
    """
    import matplotlib.pyplot as plt
    metric = "val_auroc"

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
    mlp_order = ["tiny", "small", "medium", "wide", "deep", "huge"]

    # Optionally filter to the most recent sweep per method. Records produced
    # before run_id was added carry no `run_id` key; if a method has *any*
    # records with run_id, drop its run-id-less ones; otherwise treat the
    # legacy group as a single bucket and keep it.
    if latest_only:
        latest_by_method: dict[str, str] = {}
        for r in records:
            rid = r.get("run_id")
            if rid is None:
                continue
            m = r["method"]
            if rid > latest_by_method.get(m, ""):
                latest_by_method[m] = rid
        n_in, kept, n_legacy = len(records), [], 0
        for r in records:
            m = r["method"]
            if m in latest_by_method:
                if r.get("run_id") == latest_by_method[m]:
                    kept.append(r)
            else:
                kept.append(r); n_legacy += 1
        records = kept
        if latest_by_method:
            ids = ", ".join(f"{k}->{v}" for k, v in sorted(latest_by_method.items()))
            print(f"  filtered to latest run_id per method ({ids})  "
                  f"kept {len(records)} / {n_in} records")
        if n_legacy:
            print(f"  ({n_legacy} legacy records kept for methods with no run_id)")
        if not records:
            print(f"  (no records left after --latest-only filtering; "
                  f"re-run grelu_sweep.py / nonconvex_MLP.py)")
            return

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

    box_data = [[r[metric] for r in grouped[k]] for k in keys]
    counts = [len(v) for v in box_data]
    labels = [f"{m}\n{c}\nn={n}" for (m, c), n in zip(keys, counts)]
    colors = ["#4C72B0" if m == "GReLU" else "#DD8452" for m, _ in keys]

    fig, ax = plt.subplots(1, 1, figsize=(max(7, len(keys) * 1.4), 5))
    bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True, widths=0.6)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    for i, runs in enumerate(box_data, start=1):
        x = np.full(len(runs), i, dtype=float) + np.random.uniform(-0.08, 0.08, size=len(runs))
        ax.plot(x, runs, "o", color="black", markersize=4, alpha=0.7)
    total = sum(counts)
    ax.set_title(f"Val AUROC across hyperparameter trials  "
                 f"({total} total, n shown per config)")
    ax.set_ylabel("AUROC")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=0, labelsize=9)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#4C72B0", alpha=0.75, label="Convex GReLU"),
        Patch(facecolor="#DD8452", alpha=0.75, label="Non-convex MLP"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  configs: {len(keys)}   total runs: {sum(len(v) for v in box_data)}")
    for k, runs in zip(keys, box_data):
        arr = np.array(runs)
        print(f"    {k[0]:<6s} {k[1]:<10s}  val_auroc: {arr.mean():.4f} "
              f"+/- {arr.std():.4f}   n={len(arr)}")


def plot_model_comparison(preds: dict[str, np.ndarray], z_true: np.ndarray,
                          val_split: dict, head, out_path: Path,
                          target_sens: float = 0.80):
    """Compare AUROC, precision-at-target-sensitivity, AUPRC, and per-lead
    Pearson across multiple models.

    Args:
        preds: ordered dict mapping label -> z_hat (N, 192) on val.
        z_true: (N, 192) ground-truth stress latent.
        val_split: dict with bl_st80, ischemia.
        head: ST80 regression head.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, average_precision_score
    from adapter_interface import TERRITORIES

    labels = list(preds.keys())
    bl_st80 = val_split["bl_st80"]
    y = val_split["ischemia"]

    aurocs, auprcs, precs, rho_per_lead = [], [], [], []
    print(f"  Scoring {len(labels)} models on val (sens target = {target_sens:.0%}) ...")
    for name in labels:
        z_hat = preds[name]
        pred_st80 = decode_st80(head, z_hat)
        scores = ischemia_patient_score(pred_st80, bl_st80)
        if len(np.unique(y)) >= 2:
            auroc = float(roc_auc_score(y, scores))
            auprc = float(average_precision_score(y, scores))
            prec = precision_at_sensitivity(scores, y, target_sens)
        else:
            auroc = auprc = prec = float("nan")
        aurocs.append(auroc); auprcs.append(auprc); precs.append(prec)
        rho_per_lead.append(per_lead_delta_pearson(z_hat, z_true, bl_st80, head))
        print(f"    {name:<22s}  auroc={auroc:.4f}  auprc={auprc:.4f}  "
              f"prec@sens{target_sens:.0%}={prec:.4f}  "
              f"mean_lead_rho={rho_per_lead[-1].mean():.3f}")

    # Stable colors per model
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
    colors = [palette[i % len(palette)] for i in range(len(labels))]

    # Layout: 2x2 grid -- top row 3 panels (AUROC, Prec, AUPRC), bottom row per-lead Pearson
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2])
    axA = fig.add_subplot(gs[0, 0])
    axP = fig.add_subplot(gs[0, 1])
    axQ = fig.add_subplot(gs[0, 2])
    axL = fig.add_subplot(gs[1, :])

    def _bar(ax, values, title, ylim=None):
        x = np.arange(len(values))
        bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_title(title); ax.grid(True, axis="y", alpha=0.3)
        if ylim is not None:
            ax.set_ylim(*ylim)
        for b, v in zip(bars, values):
            if v != v:  # NaN
                ax.text(b.get_x() + b.get_width() / 2, 0.02, "N/A",
                        ha="center", va="bottom", fontsize=8)
                continue
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    _bar(axA, aurocs, "Ischemia AUROC (val)", ylim=(0.5, 1.0))
    _bar(axP, precs,  f"Precision @ sensitivity={target_sens:.0%} (val)", ylim=(0.0, 1.0))
    _bar(axQ, auprcs, "Ischemia AUPRC (val)", ylim=(0.0, 1.0))

    # Per-lead clustered bars
    n_models = len(labels)
    n_leads = rho_per_lead[0].shape[0]
    width = 0.8 / max(n_models, 1)
    x = np.arange(n_leads)
    for i, (name, rho) in enumerate(zip(labels, rho_per_lead)):
        xpos = x + (i - (n_models - 1) / 2) * width
        axL.bar(xpos, rho, width=width, color=colors[i], edgecolor="black",
                linewidth=0.3, label=f"{name}  (mean r = {rho.mean():.3f})")
    axL.set_xticks(x); axL.set_xticklabels(LEAD_NAMES)
    axL.set_ylabel("Pearson r")
    axL.set_title("Per-lead Pearson r:  predicted dST80 vs true dST80")
    axL.axhline(0, color="black", lw=0.6)
    axL.grid(True, axis="y", alpha=0.3)
    axL.legend(loc="lower right", fontsize=8, ncol=2)
    # Territory annotations as shaded vertical bands
    from matplotlib.patches import Rectangle
    band = {"Inferior": "#C44E52", "Anterior": "#4C72B0", "Lateral": "#55A868"}
    for terr, idxs in TERRITORIES.items():
        for j in idxs:
            axL.add_patch(Rectangle((j - 0.5, axL.get_ylim()[0]), 1.0,
                                     axL.get_ylim()[1] - axL.get_ylim()[0],
                                     facecolor=band[terr], alpha=0.06, zorder=-1))
    fig.suptitle("Model comparison on validation set", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
# PCA analysis
# ---------------------------------------------------------------------------

def plot_pca_analysis(linear_ckpt: Path, kernel_ckpt: Path,
                      val_split: dict, device, out_dir: Path) -> None:
    """PCA / SVD analysis for the Linear and Kernel Ridge adapters.

    Linear adapter  →  figures/pca_linear.png
      Panel 1: SVD spectrum of weight matrix W (192×384) — effective rank of
               the learned linear map.
      Panel 2: Val samples projected onto the top-2 PCs of z_stress (ground
               truth), coloured by ischemia status.
      Panel 3: The same projection applied to z_hat_linear — shows how well the
               linear adapter preserves the principal structure of z_stress.

    Kernel Ridge adapter  →  figures/pca_kernel.png  (skipped if no checkpoint)
      The dual coefficients α (n_train × 192) are compact enough for exact PCA.
      Panel 1: Explained variance of PCA on α — how many output directions the
               kernel solution actually uses.
      Panels 2–3: Val residuals (z_stress − z_rest) vs predicted residuals
                  (z_hat_kernel − z_rest) in the top-2 PCs of the true residual,
                  coloured by ischemia.
    """
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    z_stress = val_split["z_stress"]   # (N, 192)
    z_rest   = val_split["z_rest"]     # (N, 192)
    y        = val_split["ischemia"].astype(int)

    # ---- Linear adapter -----------------------------------------------------
    if not linear_ckpt.exists():
        print(f"  (no linear ckpt at {linear_ckpt} — skipping linear PCA)")
    else:
        lin = None
        try:
            from linear_adapter import load_linear
            lin = load_linear(linear_ckpt)
        except Exception as e:
            print(f"  (skipping linear PCA: {e})")

        if lin is not None and lin.sklearn_model is not None and \
                hasattr(lin.sklearn_model, "coef_"):
            W = np.asarray(lin.sklearn_model.coef_)   # (192, 384)
            if W.ndim == 1:
                W = W[np.newaxis, :]
            sv = np.linalg.svd(W, compute_uv=False)   # (192,)
            sv_norm = sv / (sv.sum() + 1e-12)
            cum_sv  = np.cumsum(sv_norm)
            eff90 = int(np.searchsorted(cum_sv, 0.90) + 1)
            eff99 = int(np.searchsorted(cum_sv, 0.99) + 1)

            # z_hat on val
            u_va, _, zr_va, _ = build_tensors(val_split, device=device)
            with torch.no_grad():
                z_hat_lin = (zr_va + lin.to(device)(u_va)).cpu().numpy()

            # PCA fit on z_stress, project both z_stress and z_hat
            pca = PCA(n_components=2)
            pca.fit(z_stress)
            pc_stress = pca.transform(z_stress)
            pc_hat    = pca.transform(z_hat_lin)

            fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))

            # Panel 1 — SVD spectrum
            axs[0].bar(np.arange(len(sv_norm)), sv_norm, color="steelblue", width=1.0)
            ax2 = axs[0].twinx()
            ax2.plot(np.arange(len(cum_sv)), cum_sv, "r-", lw=1.5)
            ax2.axhline(0.90, color="r", ls="--", lw=1, alpha=0.5, label="90%")
            ax2.axhline(0.99, color="orange", ls="--", lw=1, alpha=0.5, label="99%")
            ax2.set_ylim(0, 1.05)
            ax2.set_ylabel("Cumulative energy", color="r", fontsize=8)
            ax2.legend(loc="center right", fontsize=7)
            axs[0].set(xlabel="Singular value index",
                       ylabel="Normalized singular value",
                       title=f"SVD of W  (90% in {eff90} dims, 99% in {eff99} dims)")
            axs[0].grid(True, axis="y", alpha=0.3)

            # Panels 2 & 3 — PCA scatter
            ev = pca.explained_variance_ratio_
            for ax, pc, title in [
                (axs[1], pc_stress, "z_stress  (ground truth)"),
                (axs[2], pc_hat,    "z_hat  (linear adapter)"),
            ]:
                sc = ax.scatter(pc[:, 0], pc[:, 1], c=y, cmap="bwr",
                                s=8, alpha=0.6, vmin=-0.5, vmax=1.5)
                ax.set(xlabel=f"PC1 ({ev[0]:.1%})", ylabel=f"PC2 ({ev[1]:.1%})",
                       title=title)
                ax.grid(True, alpha=0.3)
            plt.colorbar(sc, ax=axs[2], label="ischemia", ticks=[0, 1])

            fig.suptitle("Linear adapter — PCA of latent space (val set, z_stress basis)",
                         fontsize=12)
            fig.tight_layout()
            out = out_dir / "pca_linear.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Linear W effective rank:  90%={eff90}   99%={eff99}")
            print(f"  wrote {out}")

    # ---- Kernel Ridge adapter -----------------------------------------------
    if not kernel_ckpt.exists():
        print(f"  (no kernel ckpt at {kernel_ckpt} — skipping kernel PCA)")
        return
    ker = None
    try:
        from kernel_adapter import load_kernel
        ker = load_kernel(kernel_ckpt)
    except Exception as e:
        print(f"  (skipping kernel PCA: {e})")

    if ker is None or ker.kernel_model is None or \
            not hasattr(ker.kernel_model, "dual_coef_"):
        print("  (kernel model not fitted — skipping kernel PCA)")
        return

    alpha = np.asarray(ker.kernel_model.dual_coef_)  # (n_train, 192)
    n_comp = min(20, alpha.shape[0], alpha.shape[1])
    pca_a  = PCA(n_components=n_comp).fit(alpha)
    ev_a   = pca_a.explained_variance_ratio_
    cum_a  = np.cumsum(ev_a)
    eff90_a = int(np.searchsorted(cum_a, 0.90) + 1)

    # z_hat_kernel on val
    u_va, _, zr_va, _ = build_tensors(val_split, device=device)
    with torch.no_grad():
        z_hat_ker = (zr_va + ker.to(device)(u_va)).cpu().numpy()

    # Residual PCA: fit on true residual, project both true and predicted
    r_true = z_stress - z_rest
    r_hat  = z_hat_ker - z_rest
    pca_r  = PCA(n_components=2).fit(r_true)
    pc_r_true = pca_r.transform(r_true)
    pc_r_hat  = pca_r.transform(r_hat)
    ev_r = pca_r.explained_variance_ratio_

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel 1 — dual coefficient PCA explained variance
    axs[0].bar(np.arange(n_comp), ev_a * 100, color="steelblue")
    ax2 = axs[0].twinx()
    ax2.plot(np.arange(n_comp), cum_a * 100, "r-", lw=1.5)
    ax2.axhline(90, color="r", ls="--", lw=1, alpha=0.5)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Cumulative variance (%)", color="r", fontsize=8)
    axs[0].set(xlabel="Component", ylabel="Explained variance (%)",
               title=f"PCA of dual coefficients α  ({alpha.shape[0]}×{alpha.shape[1]})\n"
                     f"90% energy in {eff90_a} / {n_comp} components")
    axs[0].grid(True, axis="y", alpha=0.3)

    # Panels 2 & 3 — residual PCA scatter
    for ax, pc, title in [
        (axs[1], pc_r_true, "True residual  z_stress − z_rest"),
        (axs[2], pc_r_hat,  f"Predicted residual  (kernel={ker.kernel_name})"),
    ]:
        sc = ax.scatter(pc[:, 0], pc[:, 1], c=y, cmap="bwr",
                        s=8, alpha=0.6, vmin=-0.5, vmax=1.5)
        ax.set(xlabel=f"Residual PC1 ({ev_r[0]:.1%})",
               ylabel=f"Residual PC2 ({ev_r[1]:.1%})",
               title=title)
        ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=axs[2], label="ischemia", ticks=[0, 1])

    fig.suptitle(f"Kernel Ridge — PCA of residual latent space (val set, kernel={ker.kernel_name})",
                 fontsize=12)
    fig.tight_layout()
    out = out_dir / "pca_kernel.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Kernel dual-coef PCA:  90% energy in {eff90_a} / {n_comp} components")
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(CKPT_DIR / "grelu_best.pt"))
    ap.add_argument("--sweep-csv", type=str, default=str(CKPT_DIR / "polish_sweep.csv"))
    ap.add_argument("--sweep-jsonl", type=str, default=str(CKPT_DIR / "sweep_results.jsonl"),
                    help="JSONL written by nonconvex_MLP.py and grelu_sweep.py")
    ap.add_argument("--min-runs", type=int, default=1,
                    help="Drop sweep configs with fewer than this many records. "
                         "Default 1 = show everything; raise it to hide tiny groups.")
    ap.add_argument("--all-runs", action="store_true",
                    help="By default the box plot keeps only records with the "
                         "most recent run_id per method (so old / stale sweep "
                         "runs in the same JSONL don't pollute the boxes). "
                         "Pass this flag to keep everything.")
    ap.add_argument("--include-configs", type=str, nargs="+", default=None,
                    help="Override filters and show exactly these config names "
                         "(e.g. m=256 m=1024 m=4096 tiny medium wide).")
    ap.add_argument("--include-methods", type=str, nargs="+", default=None,
                    help="Limit sweep box plots to methods named in JSONL "
                         "(e.g. GReLU MLP ElasticNet KernelRidge LoRA AttentionLinear).")
    ap.add_argument("--grelu-ckpt", type=str,
                    default=str(CKPT_DIR / "grelu_best.pt"),
                    help="Convex GReLU adapter (with prior) for model comparison.")
    ap.add_argument("--grelu-zero-prior-ckpt", type=str,
                    default=str(CKPT_DIR / "grelu_best_zero_prior.pt"),
                    help="Convex GReLU adapter trained with --zero-prior.")
    ap.add_argument("--mlp-ckpt", type=str,
                    default=str(CKPT_DIR / "mlp_best.pt"),
                    help="Best non-convex MLP from the sweep.")
    ap.add_argument("--linear-ckpt", type=str,
                    default=str(CKPT_DIR / "linear_best.pt"),
                    help="ElasticNet / linear sweep best (joblib).")
    ap.add_argument("--kernel-ckpt", type=str,
                    default=str(CKPT_DIR / "kernel_best.pt"),
                    help="Kernel ridge best (joblib).")
    ap.add_argument("--lora-ckpt", type=str,
                    default=str(CKPT_DIR / "lora_best.pt"),
                    help="LoRA best (torch).")
    ap.add_argument("--attention-ckpt", type=str,
                    default=str(CKPT_DIR / "attention_best.pt"),
                    help="Attention adapter best (torch).")
    ap.add_argument("--target-sens", type=float, default=0.80,
                    help="Sensitivity at which to report precision in the comparison plot.")
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

    # Honor the zero_prior flag baked into the checkpoint's config so the
    # plots use the same network input format as training.
    ckpt_meta = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    zero_prior = bool((ckpt_meta.get("config") or {}).get("zero_prior", False))
    if zero_prior:
        print("  [mode] zero_prior=True  ->  u = [phi_stress; zeros(192)]")

    u_va, _, zr_va, zs_va = build_tensors(val_split, device=device, zero_prior=zero_prior)
    with torch.no_grad():
        z_hat_va = (zr_va + adapter(u_va)).cpu().numpy()
    z_true_va = val_split["z_stress"]
    z_rest_va = val_split["z_rest"]

    # Use a bounded sample of train for gate stats
    n = min(args.max_train_samples, len(train_split["z_rest"]))
    sub = {k: v[:n] for k, v in train_split.items()}
    u_tr, *_ = build_tensors(sub, device=device, zero_prior=zero_prior)

    # ---- figures ------------------------------------------------------------
    print("\n[1/15] neuron importance ...")
    plot_neuron_importance(adapter, out_dir / "neuron_importance.png")

    print("\n[2/15] gate activation rates ...")
    plot_gate_activation_rates(adapter, u_tr, out_dir / "gate_activation_rates.png")

    print("\n[3/15] pruning sweep ...")
    plot_pruning_sweep(Path(args.sweep_csv), out_dir / "pruning_sweep.png")

    print("\n[4/15] ischemia ROC ...")
    pred_st80 = decode_st80(head, z_hat_va)
    scores = ischemia_patient_score(pred_st80, val_split["bl_st80"])
    plot_ischemia_roc(scores, val_split["ischemia"], out_dir / "ischemia_roc.png")

    print("\n[5/15] per-dimension recovery ...")
    plot_per_dim_recovery(z_hat_va, z_true_va, out_dir / "per_dim_recovery.png")

    print("\n[6/15] recovery by label ...")
    plot_recovery_by_label(z_hat_va, z_true_va, val_split["ischemia"].astype(int),
                           out_dir / "recovery_by_label.png")

    print("\n[7/15] baseline vs adapter ...")
    plot_baseline_vs_adapter(z_rest_va, z_hat_va, z_true_va,
                             out_dir / "baseline_vs_adapter.png")

    print("\n[8/15] per-lead ST80 Pearson ...")
    plot_st80_per_lead_pearson(z_hat_va, z_true_va, val_split["bl_st80"], head,
                               out_dir / "st80_per_lead_pearson.png")

    latest_only = not args.all_runs
    print("\n[9/15] sweep box plot (combined, GReLU vs MLP) ...")
    plot_sweep_boxplot(
        Path(args.sweep_jsonl), out_dir / "sweep_boxplot.png",
        min_runs=args.min_runs,
        include_configs=args.include_configs,
        include_methods=args.include_methods,
        latest_only=latest_only,
    )
    print("\n[10/15] sweep box plot (GReLU only) ...")
    plot_sweep_boxplot(
        Path(args.sweep_jsonl), out_dir / "sweep_boxplot_grelu.png",
        min_runs=args.min_runs, include_configs=args.include_configs,
        include_methods=["GReLU"], latest_only=latest_only,
    )
    print("\n[11/15] sweep box plot (MLP only) ...")
    plot_sweep_boxplot(
        Path(args.sweep_jsonl), out_dir / "sweep_boxplot_mlp.png",
        min_runs=args.min_runs, include_configs=args.include_configs,
        include_methods=["MLP"], latest_only=latest_only,
    )

    all_preds = _collect_model_predictions(val_split, device, args)

    print("\n[12/15] model comparison (GReLU / GReLU 0prior / MLP) ...")
    cmp_keys = [
        "Convex GReLU\n(with prior)",
        "Convex GReLU\n(zero prior)",
        "Non-convex MLP",
    ]
    cmp_preds = {k: all_preds[k] for k in cmp_keys if k in all_preds}
    if cmp_preds:
        plot_model_comparison(cmp_preds, z_true_va, val_split, head,
                              out_dir / "model_comparison.png",
                              target_sens=args.target_sens)
    else:
        print("  (no GReLU/MLP triplet checkpoints -- skipping compact comparison)")

    print("\n[13/15] multi-method ischemia ROC overlay ...")
    if len(all_preds) >= 1:
        scores_by_model = {}
        for name, z_hat in all_preds.items():
            pred_st = decode_st80(head, z_hat)
            scores_by_model[name] = ischemia_patient_score(pred_st, val_split["bl_st80"])
        plot_multi_method_ischemia_roc_overlay(
            scores_by_model, val_split["ischemia"], out_dir / "ischemia_roc_methods.png",
        )
    else:
        print("  (no checkpoints found for multi-method ROC)")

    print("\n[14/15] multi-method per-lead Pearson (dST80) ...")
    if len(all_preds) >= 1:
        plot_multi_method_per_lead_pearson_panel(
            all_preds, z_true_va, val_split["bl_st80"], head,
            out_dir / "st80_per_lead_pearson_methods.png",
            csv_path=out_dir / "st80_per_lead_pearson_methods.csv",
        )
    else:
        print("  (no checkpoints -- skipping per-lead Pearson panel)")

    print("\n[15/15] PCA analysis (linear adapter + kernel ridge) ...")
    plot_pca_analysis(
        Path(args.linear_ckpt), Path(args.kernel_ckpt),
        val_split, device, out_dir,
    )

    print(f"\nAll figures written to {out_dir}/")


def _collect_model_predictions(val_split: dict, device,
                               args: argparse.Namespace,
                               ) -> dict[str, np.ndarray]:
    """Load every adapter checkpoint named on the CLI; return z_hat (val) per model."""

    preds: dict[str, np.ndarray] = {}

    # ---- No adapter: single lead embedding passed directly to regression head ----
    preds["No adapter\n(single lead)"] = val_split["phi_stress"].astype(np.float32)

    # ---- Convex GReLU (with prior) -----------------------------------------
    gp = Path(args.grelu_ckpt)
    if gp.exists():
        try:
            meta = torch.load(gp, map_location="cpu", weights_only=False)
            zp = bool((meta.get("config") or {}).get("zero_prior", False))
            m = load_adapter(gp, device=device)
            m.eval()
            uv, _, zr, _ = build_tensors(val_split, device=device, zero_prior=zp)
            with torch.no_grad():
                preds["Convex GReLU\n(with prior)"] = (zr + m(uv)).cpu().numpy()
        except Exception as e:
            print(f"  (skipping GReLU: {e})")
    else:
        print(f"  (no GReLU ckpt at {gp})")

    # ---- Convex GReLU (zero prior) ----------------------------------------
    gz = Path(args.grelu_zero_prior_ckpt)
    if gz.exists():
        try:
            m = load_adapter(gz, device=device)
            m.eval()
            uv, _, zr, _ = build_tensors(val_split, device=device, zero_prior=True)
            with torch.no_grad():
                preds["Convex GReLU\n(zero prior)"] = (zr + m(uv)).cpu().numpy()
        except Exception as e:
            print(f"  (skipping GReLU zero-prior: {e})")
    else:
        print(f"  (no zero-prior GReLU ckpt at {gz})")

    # ---- Non-convex MLP -----------------------------------------------------
    mp = Path(args.mlp_ckpt)
    if mp.exists():
        try:
            from nonconvex_MLP import load_mlp
            m = load_mlp(mp, device=device)
            m.eval()
            uv, _, zr, _ = build_tensors(val_split, device=device, zero_prior=False)
            with torch.no_grad():
                preds["Non-convex MLP"] = (zr + m(uv)).cpu().numpy()
        except Exception as e:
            print(f"  (skipping MLP: {e})")
    else:
        print(f"  (no MLP ckpt at {mp})")

    # ---- ElasticNet (sklearn joblib) --------------------------------------
    lp = Path(args.linear_ckpt)
    if lp.exists():
        try:
            from linear_adapter import load_linear
            m = load_linear(lp).to(device)
            m.eval()
            uv, _, zr, _ = build_tensors(val_split, device=device, zero_prior=False)
            with torch.no_grad():
                preds["ElasticNet (linear)"] = (zr + m(uv)).cpu().numpy()
        except Exception as e:
            print(f"  (skipping linear/ElasticNet ckpt: {e})")
    else:
        print(f"  (no linear ckpt at {lp})")

    # ---- Kernel ridge --------------------------------------------------------
    kp = Path(args.kernel_ckpt)
    if kp.exists():
        try:
            from kernel_adapter import load_kernel
            m = load_kernel(kp).to(device)
            m.eval()
            uv, _, zr, _ = build_tensors(val_split, device=device, zero_prior=False)
            with torch.no_grad():
                preds["Kernel ridge"] = (zr + m(uv)).cpu().numpy()
        except Exception as e:
            print(f"  (skipping kernel ckpt: {e})")
    else:
        print(f"  (no kernel ckpt at {kp})")

    # ---- LoRA --------------------------------------------------------------
    lop = Path(args.lora_ckpt)
    if lop.exists():
        try:
            from lora_adapter import load_lora
            m = load_lora(lop, device=device)
            m.eval()
            uv, _, zr, _ = build_tensors(val_split, device=device, zero_prior=False)
            with torch.no_grad():
                preds["LoRA"] = (zr + m(uv)).cpu().numpy()
        except Exception as e:
            print(f"  (skipping LoRA ckpt: {e})")
    else:
        print(f"  (no LoRA ckpt at {lop})")

    # ---- Attention ----------------------------------------------------------
    attn_p = Path(args.attention_ckpt)
    if attn_p.exists():
        try:
            from attention_adapter import load_attention
            meta = torch.load(attn_p, map_location="cpu", weights_only=False)
            variant = meta.get("variant")
            if variant is None and isinstance(meta.get("config"), dict):
                variant = meta["config"].get("variant")
            if variant == "attn_additive":
                alabel = "Attention (additive)"
            else:
                alabel = "Attention (linear)"
            m = load_attention(attn_p, device=device)
            m.eval()
            uv, _, zr, _ = build_tensors(val_split, device=device, zero_prior=False)
            with torch.no_grad():
                preds[alabel] = (zr + m(uv)).cpu().numpy()
        except Exception as e:
            print(f"  (skipping attention ckpt: {e})")
    else:
        print(f"  (no attention ckpt at {attn_p})")

    print(f"  Loaded {len(preds)} model(s) with valid checkpoints for aggregation plots.")
    return preds


if __name__ == "__main__":
    main()
