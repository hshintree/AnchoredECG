"""
Paired lead dataset for JEPA end-to-end QT training.

Returns (context_graph, target_graph, qt_label) from the **same** ECG in a
single __getitem__ call, enabling online JEPA training without offline latent
caching.

Default configuration (mirrors the adapter experiments):
  context_leads = [0]          — Lead I  (2 edges, bidirectional)
  target_leads  = range(12)    — Full 12-lead  (24 edges, bidirectional)

Both graphs share the same 13 electrode node positions (fixed 3-D coordinates)
and are built from the same raw 500 Hz waveform loaded from records500/.

Lead dropout is ALWAYS disabled in this dataset — both graphs always use the
exact fixed lead sets.  (Dropout is for QTFinetuneDataset single-branch
training; here both branches must be consistent.)

Usage:
    from jepa_dataset import PairedLeadDataset, collate_paired, JEPAInferenceModel
    ds     = PairedLeadDataset(ptbxl_signals_dir, labels_csv, split='train')
    loader = DataLoader(ds, batch_size=32, collate_fn=collate_paired, shuffle=True)
    for ctx_batch, tgt_batch, qt in loader:
        ...

The module also exports JEPAInferenceModel — a thin wrapper that makes a
(context_encoder, predictor, qt_head) triple look like a standard ECGGNNSimple
for drop-in use with existing run_inference / extract_latents utilities.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import wfdb
from torch.utils.data import Dataset

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT   = _SCRIPT_DIR.parent
_MODELS_DIR  = _REPO_ROOT / "models"
for _p in (_MODELS_DIR, _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from qt_dataset import QTFinetuneDataset  # reuse index builder, graph builder, file loading
from torch_geometric.data import Batch


# ---------------------------------------------------------------------------
# Paired dataset
# ---------------------------------------------------------------------------

class PairedLeadDataset(QTFinetuneDataset):
    """
    Dataset returning (context_graph, target_graph, qt_label) for the same ECG.

    Inherits _build_index(), builder, scale_factor, ptbxl_dir, and index from
    QTFinetuneDataset.  Lead dropout is always disabled.

    Args:
        ptbxl_signals_dir : Path to PTB-XL signals root (contains records500/).
        labels_csv        : Path to qt_labels.csv.
        split             : 'train', 'val', or 'test'.
        context_leads     : Lead indices for context branch (default: [0] = Lead I).
        target_leads      : Lead indices for target branch (default: all 12 leads).
    """

    def __init__(
        self,
        ptbxl_signals_dir: str | Path,
        labels_csv: str | Path,
        split: str = "train",
        context_leads: List[int] | None = None,
        target_leads:  List[int] | None = None,
    ):
        super().__init__(
            ptbxl_signals_dir=ptbxl_signals_dir,
            labels_csv=labels_csv,
            split=split,
            lead_dropout_prob=0.0,   # always disabled; both branches must be consistent
        )
        self._context_leads = sorted(context_leads if context_leads is not None else [0])
        self._target_leads  = sorted(target_leads  if target_leads  is not None else list(range(12)))

    def __getitem__(self, idx: int):
        item = self.index[idx]

        record  = wfdb.rdrecord(str(self.ptbxl_dir / item["filename_hr"]))
        signals = record.p_signal.T.astype(np.float32)   # (12, 5000) — native 500 Hz
        signals = np.nan_to_num(signals) * self.scale_factor

        context_graph = self.builder.build_subgraph_from_indices(
            signals, self._context_leads, bidirectional=True
        )
        target_graph = self.builder.build_subgraph_from_indices(
            signals, self._target_leads, bidirectional=True
        )
        qt = torch.tensor([item["qt_ms"]], dtype=torch.float32)
        return context_graph, target_graph, qt


# ---------------------------------------------------------------------------
# Custom collate function
# ---------------------------------------------------------------------------

def collate_paired(batch):
    """
    Collate function for PairedLeadDataset.

    Batches context graphs and target graphs separately into two PyG Batch
    objects, then stacks the QT label tensors.

    Returns:
        ctx_batch : PyG Batch of context graphs (e.g. Lead I graphs)
        tgt_batch : PyG Batch of target graphs  (e.g. 12-lead graphs)
        qt        : (B, 1) float32 tensor of QT labels
    """
    ctx_graphs, tgt_graphs, qts = zip(*batch)
    ctx_batch = Batch.from_data_list(list(ctx_graphs))
    tgt_batch = Batch.from_data_list(list(tgt_graphs))
    qt        = torch.stack(qts, dim=0)   # (B, 1)
    return ctx_batch, tgt_batch, qt


# ---------------------------------------------------------------------------
# JEPA inference wrapper
# ---------------------------------------------------------------------------

class JEPAInferenceModel(nn.Module):
    """
    Thin wrapper making (context_encoder, predictor, qt_head) look like a
    standard ECGGNNSimple for drop-in use with run_inference() and
    extract_latents() in analyze_qtc.py.

    Forward:
        z_c   = context_encoder(data, tabular)["graph_embed"]   (B, 192)
        z_hat = z_c + predictor(z_c)                            (B, 192)
        logits = qt_head(z_hat)                                  (B, 1)

    Returns {"logits": (B,1), "graph_embed": (B,192)} — same keys as ECGGNNSimple.

    The residual formulation (z_c + predictor(z_c)) matches the six frozen
    adapters, making the comparison fair.
    """

    def __init__(
        self,
        context_encoder: nn.Module,
        predictor: nn.Module,
        qt_head: nn.Module,
    ):
        super().__init__()
        self.context_encoder = context_encoder
        self.predictor       = predictor
        self.qt_head         = qt_head

    def forward(self, data, tabular=None):
        B = data.num_graphs
        if tabular is None:
            tabular = torch.zeros(B, 0, device=data.x.device)
        z_c   = self.context_encoder(data, tabular=tabular)["graph_embed"]  # (B, 192)
        z_hat = z_c + self.predictor(z_c)                                    # (B, 192)
        logits = self.qt_head(z_hat)                                          # (B, 1)
        return {"logits": logits, "graph_embed": z_hat}
