"""
PTB-XL dataset for fine-tuning GraphECG on QT interval regression.

Input  : electrode graph from native 500 Hz 12-lead ECG with lead dropout.
         NO tabular features — waveform only.
Target : QT_Int_Global (ms) — single global value from GE 12SL algorithm
         (earliest Q onset to latest T offset across all 12 leads).

PTB-XL signals are 500 Hz, 10 s, 12 leads (5000 samples).
We use the full 500 Hz resolution (records500/) — NO resampling.

Lead dropout (training only): 70 % of batches use a random 1-12 lead subset,
making the encoder robust to any lead configuration at test time.

Requires: pip install wfdb
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT   = _SCRIPT_DIR.parent
_MODELS_DIR  = _REPO_ROOT / "models"
if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))

from ecg_graph_v2 import ECGGraphBuilderV2, LEAD_NAMES_12

PTBXL_FS  = 500          # native PTB-XL sampling rate
TARGET_LEN = 5000        # 500 Hz × 10 s — no resampling

PTBXL_LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF",
                     "V1", "V2", "V3", "V4", "V5", "V6"]


class QTFinetuneDataset(Dataset):
    """
    Fine-tuning dataset: variable-lead electrode graph → QT interval (ms).

    Args:
        ptbxl_signals_dir : Path to PTB-XL *signals* dataset root.
                            Must contain records500/.
        labels_csv        : Path to qt_labels.csv.
        split             : 'train', 'val', or 'test'.
        lead_dropout_prob : Fraction of training samples that use a random
                            lead subset (ignored for val/test).
        min_leads         : Minimum leads when dropout fires.
        max_leads         : Maximum leads when dropout fires.
        scale_mv_to_mm    : Multiply signals by 10 (mV → mm).
    """

    def __init__(
        self,
        ptbxl_signals_dir: str | Path,
        labels_csv: str | Path,
        split: str = "train",
        lead_dropout_prob: float = 0.7,
        min_leads: int = 1,
        max_leads: int = 12,
        scale_mv_to_mm: bool = False,
    ):
        self.ptbxl_dir        = Path(ptbxl_signals_dir)
        self.split            = split
        self.lead_dropout_prob = lead_dropout_prob if split == "train" else 0.0
        self.min_leads        = min_leads
        self.max_leads        = max_leads
        self.scale_factor     = 10.0 if scale_mv_to_mm else 1.0

        self._builder = None
        self.index    = self._build_index(labels_csv)

    @property
    def builder(self):
        if self._builder is None:
            self._builder = ECGGraphBuilderV2()
        return self._builder

    def _build_index(self, labels_csv):
        df = pd.read_csv(labels_csv)
        if self.split == "train":
            df = df[df["strat_fold"] <= 8]
        elif self.split == "val":
            df = df[df["strat_fold"] == 9]
        elif self.split == "test":
            df = df[df["strat_fold"] == 10]

        df = df.dropna(subset=["QT_Int_Global"])

        return [
            {
                "ecg_id":      int(row["ecg_id"]),
                "filename_hr": row["filename_hr"],
                "qt_ms":       float(row["QT_Int_Global"]),
            }
            for _, row in df.iterrows()
        ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        item = self.index[idx]

        # ── Waveforms ──────────────────────────────────────────────────────
        record  = wfdb.rdrecord(str(self.ptbxl_dir / item["filename_hr"]))
        signals = record.p_signal   # (5000, 12) float64, mV

        assert [s.upper() for s in record.sig_name] == \
               [s.upper() for s in PTBXL_LEAD_ORDER], \
               f"Lead order mismatch: {record.sig_name}"

        signals = signals.T.astype(np.float32)        # (12, 5000) — native 500 Hz
        signals = np.nan_to_num(signals) * self.scale_factor

        # ── Lead dropout (training only) ───────────────────────────────────
        if self.lead_dropout_prob > 0 and np.random.random() < self.lead_dropout_prob:
            num_leads    = np.random.randint(self.min_leads, self.max_leads + 1)
            lead_indices = sorted(
                np.random.choice(12, num_leads, replace=False).tolist()
            )
        else:
            lead_indices = list(range(12))

        # ── Build electrode graph ──────────────────────────────────────────
        graph = self.builder.build_subgraph_from_indices(
            signals, lead_indices, bidirectional=True
        )

        qt = torch.tensor([item["qt_ms"]], dtype=torch.float32)
        return graph, qt
