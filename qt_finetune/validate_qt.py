"""
Evaluate the fine-tuned QT regression model on the PTB-XL test set (strat_fold=10).

Tests four configurations in one run:
  12-lead          — all leads (standard clinical ECG)
  6 limb leads     — I, II, III, aVR, aVL, aVF (no precordials)
  1-lead (Lead I)  — index 0, single-lead patch
  ICM (V3-V2)      — non-standard single-lead (V3 minus V2 electrode potentials)

Metrics: MAE (ms), RMSE (ms), Pearson R, within-10 ms %, within-20 ms %

No tabular features — waveform-only model.

Usage (local):
    python qt_finetune/validate_qt.py
    python qt_finetune/validate_qt.py --checkpoint path/to/best_qt_model.pt

Usage (Modal — checkpoint and data already in the volume):
    modal run modal_finetune.py::validate_entrypoint
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT  = _SCRIPT_DIR.parent
_MODELS_DIR = _REPO_ROOT / "models"
for _p in (_MODELS_DIR, _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ecg_gnn_simple import ECGGNNSimple
from qt_dataset import QTFinetuneDataset, TARGET_LEN
from torch_geometric.loader import DataLoader as PyGDataLoader

DEFAULT_CHECKPOINT  = _REPO_ROOT / "qt_finetune" / "checkpoints" / "best_qt_model.pt"
DEFAULT_LABELS      = _SCRIPT_DIR / "qt_labels.csv"
DEFAULT_SIGNALS_DIR = _REPO_ROOT / "data" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"

# PTB-XL lead order: I(0), II(1), III(2), aVR(3), aVL(4), aVF(5), V1-V6(6-11)
LIMB_LEAD_INDICES = [0, 1, 2, 3, 4, 5]   # I, II, III, aVR, aVL, aVF
# V1=6, V2=7, V3=8, V4=9, V5=10, V6=11
_V2_IDX = 7
_V3_IDX = 8


# ---------------------------------------------------------------------------
# Datasets — fixed lead subsets at test time (no dropout)
# ---------------------------------------------------------------------------

class ForcedLeadDataset(QTFinetuneDataset):
    """Evaluate on a fixed set of standard leads."""

    def __init__(self, lead_indices: list[int], **kwargs):
        kwargs.setdefault("split", "test")
        kwargs["lead_dropout_prob"] = 0.0
        super().__init__(**kwargs)
        self._fixed_leads = sorted(lead_indices)

    def __getitem__(self, idx: int):
        import wfdb

        item = self.index[idx]
        record  = wfdb.rdrecord(str(self.ptbxl_dir / item["filename_hr"]))
        signals = record.p_signal.T.astype(np.float32)   # (12, 5000) — native 500 Hz
        signals = np.nan_to_num(signals) * self.scale_factor

        graph = self.builder.build_subgraph_from_indices(
            signals, self._fixed_leads, bidirectional=True
        )
        qt = torch.tensor([item["qt_ms"]], dtype=torch.float32)
        return graph, qt


class ICMLeadDataset(QTFinetuneDataset):
    """
    Evaluate with a single ICM-style lead: signal = V3 - V2.

    ICM = ('V2', 'V3') is already in ecg_graph_v2.OTHER_LEAD_DEFINITIONS.
    The graph is built via build_observed_subgraph({'ICM': signal}).
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("split", "test")
        kwargs["lead_dropout_prob"] = 0.0
        super().__init__(**kwargs)

    def __getitem__(self, idx: int):
        import wfdb

        item = self.index[idx]
        record  = wfdb.rdrecord(str(self.ptbxl_dir / item["filename_hr"]))
        signals = record.p_signal.T.astype(np.float32)   # (12, 5000) — native 500 Hz
        signals = np.nan_to_num(signals) * self.scale_factor

        # ICM signal: V3 - V2 (index 8 - index 7) → (5000,)
        icm_signal = signals[_V3_IDX] - signals[_V2_IDX]

        graph = self.builder.build_observed_subgraph(
            {"ICM": icm_signal}, bidirectional=True
        )
        qt = torch.tensor([item["qt_ms"]], dtype=torch.float32)
        return graph, qt


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    model = ECGGNNSimple(
        input_length=TARGET_LEN,   # 5000 samples @ 500 Hz
        sh_max_degree=4,
        node_dim=128,
        edge_dim=192,
        hidden_dim=192,
        num_gnn_layers=3,
        tabular_dim=7,      # base architecture; classifier replaced below
        dropout=0.3,
    )
    # Waveform-only head: 192-dim graph embed → 64 → 1
    model.classifier = nn.Sequential(
        nn.Linear(192, 64),
        nn.Mish(inplace=True),
        nn.Linear(64, 1),
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    print(f"  Checkpoint  epoch={ckpt.get('epoch','?')}  "
          f"val_MAE={ckpt.get('val_mae', float('nan')):.2f} ms  "
          f"val_R={ckpt.get('val_r', float('nan')):.3f}")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, label: str) -> dict:
    all_pred, all_tgt = [], []
    for batch_data, qt in loader:
        batch_data = batch_data.to(device)
        qt         = qt.to(device)
        # Waveform-only: empty (B, 0) tabular so cat gives 192-dim (no zero-padding)
        B       = batch_data.num_graphs
        tabular = torch.zeros(B, 0, device=device)
        pred    = model(batch_data, tabular=tabular)["logits"].flatten()
        all_pred.append(pred.cpu())
        all_tgt.append(qt.flatten().cpu())

    pred = torch.cat(all_pred).numpy()
    tgt  = torch.cat(all_tgt).numpy()
    err  = pred - tgt

    mae  = float(np.abs(err).mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    p, t = pred - pred.mean(), tgt - tgt.mean()
    r    = float((p * t).sum() / (np.linalg.norm(p) * np.linalg.norm(t) + 1e-8))
    w10  = float((np.abs(err) <= 10).mean() * 100)
    w20  = float((np.abs(err) <= 20).mean() * 100)
    bias = float(err.mean())

    return dict(label=label, n=len(tgt),
                mae=mae, rmse=rmse, r=r, w10=w10, w20=w20, bias=bias,
                pred=pred, tgt=tgt)


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_results(results: list[dict]) -> None:
    cols = f"{'Config':<28} {'N':>5} {'MAE':>9} {'RMSE':>9} {'R':>8} {'≤10ms':>7} {'≤20ms':>7} {'Bias':>8}"
    bar  = "─" * len(cols)
    print(f"\n╔{'═' * len(cols)}╗")
    print(f"║  QT Interval Regression — Test-Set Results (waveform-only){' ' * (len(cols) - 59)}║")
    print(f"║  (ms unless noted){' ' * (len(cols) - 19)}║")
    print(f"╠{bar}╣")
    print(f"  {cols}")
    print(f"  {bar}")
    for r in results:
        print(f"  {r['label']:<28} {r['n']:>5} "
              f"{r['mae']:>8.2f}  {r['rmse']:>8.2f}  {r['r']:>7.3f}  "
              f"{r['w10']:>6.1f}%  {r['w20']:>6.1f}%  {r['bias']:>+7.2f}")
    print(f"╚{'═' * len(cols)}╝")
    print()


# ---------------------------------------------------------------------------
# Sanity baseline (no model, no waveform)
# ---------------------------------------------------------------------------

def print_population_baseline(test_items: list[dict]) -> None:
    tgts = np.array([it["qt_ms"] for it in test_items], dtype=np.float32)
    pred_mean = np.full_like(tgts, tgts.mean())
    mae_m = float(np.abs(pred_mean - tgts).mean())
    print(f"\n── Sanity baseline ──────────────────────────────────────────")
    print(f"  Population mean ({tgts.mean():.0f} ms, no model, no waveform) : MAE={mae_m:.2f} ms  R=0.000")
    print(f"  If model MAE << {mae_m:.0f} ms → encoder is genuinely useful.\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate waveform-only QT model on PTB-XL test set (4 lead configs)"
    )
    parser.add_argument("--checkpoint",  default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--ptbxl-dir",   default=str(DEFAULT_SIGNALS_DIR))
    parser.add_argument("--labels-csv",  default=str(DEFAULT_LABELS))
    parser.add_argument("--batch-size",  type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = (torch.device("cuda")  if torch.cuda.is_available() else
              torch.device("mps")   if torch.backends.mps.is_available() else
              torch.device("cpu"))
    print(f"Device     : {device}")
    print(f"Checkpoint : {args.checkpoint}")
    model = load_model(Path(args.checkpoint), device)

    # Common kwargs for all datasets
    ds_kwargs = dict(
        ptbxl_signals_dir=args.ptbxl_dir,
        labels_csv=args.labels_csv,
    )

    configs = [
        ("12-lead (all)",        ForcedLeadDataset(lead_indices=list(range(12)), **ds_kwargs)),
        ("6 limb leads",         ForcedLeadDataset(lead_indices=LIMB_LEAD_INDICES, **ds_kwargs)),
        ("1-lead (Lead I)",      ForcedLeadDataset(lead_indices=[0], **ds_kwargs)),
        ("ICM (V3−V2)",          ICMLeadDataset(**ds_kwargs)),
    ]

    results = []
    for label, ds in configs:
        print(f"\nEvaluating: {label}  ({len(ds)} test samples) ...")
        loader = PyGDataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        results.append(evaluate(model, loader, device, label))

    print_results(results)
    print_population_baseline(configs[0][1].index)   # same test split for all


if __name__ == "__main__":
    main()
