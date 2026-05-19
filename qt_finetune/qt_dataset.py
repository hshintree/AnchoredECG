"""
PTB-XL dataset for fine-tuning GraphECG on QT interval regression.

Each sample = one 12-lead ECG recording.
Input: electrode graph from resampled (250 Hz) ECG with optional lead dropout.
Target: QT_Int_Global (ms) -- single scalar.

PTB-XL signals are 500 Hz, 10 seconds, 12 leads (5000 samples).
Resampled to 250 Hz -> 2500 samples (same as original SHD pretraining).

Lead dropout: same as ST80 pipeline -- 70% of training batches use random
1-12 lead subsets. This makes the encoder robust to partial-lead inputs.

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
from scipy.signal import resample

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_MODELS_DIR = _REPO_ROOT / "models"
if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))

from ecg_graph_v2 import ECGGraphBuilderV2, LEAD_NAMES_12

PTBXL_FS = 500
TARGET_FS = 250
PTBXL_LEN = 5000
TARGET_LEN = 2500

PTBXL_LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF",
                     "V1", "V2", "V3", "V4", "V5", "V6"]


class QTFinetuneDataset(Dataset):
    """
    Fine-tuning dataset: variable-lead electrode graph -> QT interval (ms).

    Args:
        ptbxl_dir: path to PTB-XL root (contains records500/, etc.)
        labels_csv: path to qt_labels.csv
        split: 'train', 'val', or 'test'
        lead_dropout_prob: fraction of training batches with random lead subsets
        scale_mv_to_mm: if True, multiply signals by 10 (mV -> mm). The SHD
            pretrained model was trained on EchoNext which uses mV. Set False
            to stay in mV (recommended -- the model adapts during fine-tuning).
    """

    def __init__(
        self,
        ptbxl_dir: str | Path,
        labels_csv: str | Path,
        split: str = "train",
        lead_dropout_prob: float = 0.7,
        min_leads: int = 1,
        max_leads: int = 12,
        scale_mv_to_mm: bool = False,
    ):
        self.ptbxl_dir = Path(ptbxl_dir)
        self.split = split
        self.lead_dropout_prob = lead_dropout_prob if split == "train" else 0.0
        self.min_leads = min_leads
        self.max_leads = max_leads
        self.scale_factor = 10.0 if scale_mv_to_mm else 1.0

        self._builder = None
        self.index = self._build_index(labels_csv)

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

        samples = []
        for _, row in df.iterrows():
            samples.append({
                "ecg_id": int(row["ecg_id"]),
                "filename_hr": row["filename_hr"],
                "qt_ms": float(row["QT_Int_Global"]),
                "qtc_bazett": float(row["QTc_Bazett"]),
                "hr": float(row["HR"]) if pd.notna(row["HR"]) else 0.0,
            })
        return samples

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        item = self.index[idx]

        record = wfdb.rdrecord(str(self.ptbxl_dir / item["filename_hr"]))
        signals = record.p_signal  # (5000, 12) float64, mV

        assert list(record.sig_name) == PTBXL_LEAD_ORDER, (
            f"Lead order mismatch: {record.sig_name}"
        )

        signals = signals.T.astype(np.float32)  # (12, 5000)
        signals = np.nan_to_num(signals)
        signals *= self.scale_factor

        resampled = np.stack([
            resample(signals[i], TARGET_LEN).astype(np.float32)
            for i in range(12)
        ])  # (12, 2500)

        if self.lead_dropout_prob > 0 and np.random.random() < self.lead_dropout_prob:
            num_leads = np.random.randint(self.min_leads, self.max_leads + 1)
            lead_indices = sorted(
                np.random.choice(12, num_leads, replace=False).tolist()
            )
        else:
            lead_indices = list(range(12))

        graph = self.builder.build_subgraph_from_indices(
            resampled, lead_indices, bidirectional=True
        )

        qt = torch.tensor([item["qt_ms"]], dtype=torch.float32)
        return graph, qt
