"""
PTB-XL dataset for fine-tuning GraphECG on QT interval regression.

Uses TWO PTB-XL datasets:

  1. ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/
     (pass as --ptbxl-dir)
     Contains: records500/  <-- raw 500 Hz waveforms read with wfdb.
     This is the ONLY dataset needed for waveforms.

  2. ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1/
     (pass as --ptbxl-features-dir, optional but recommended)
     Contains: features/ecgdeli_features.csv  <-- keyed by ecg_id.
     Provides 7 global ECG morphology features as tabular input:
       age, sex, HR, PR_Int_Global (ms), QRS_Dur_Global (ms),
       P_Dur_Global (ms), RR_Mean_Global (ms).
     NOTE: QT_Int_Global is intentionally EXCLUDED (it is the target).
     When this directory is not supplied the 7 features default to zero,
     exactly matching the pretrained model's original behaviour.

Each sample = one 12-lead ECG recording.
Input  : electrode graph from resampled (250 Hz) ECG + optional lead dropout.
Tabular: 7-dim vector attached as data.tabular on the returned PyG Data object.
Target : QT_Int_Global (ms) -- single scalar.

PTB-XL signals are 500 Hz, 10 seconds, 12 leads (5000 samples).
Resampled to 250 Hz -> 2500 samples (same as original SHD pretraining).

Lead dropout: same as ST80 pipeline -- 70% of training batches use random
1-12 lead subsets. This makes the encoder robust to partial-lead inputs.

Requires: pip install wfdb
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

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

# 7 clinically meaningful global features from ecgdeli_features.csv.
# Must match tabular_dim=7 of the pretrained model.
# QT_Int_Global is intentionally excluded -- it is the regression target.
ECGDELI_TABULAR_COLS = [
    "PR_Int_Global",    # PR interval (ms)  -- AV conduction
    "QRS_Dur_Global",   # QRS duration (ms) -- ventricular depolarisation
    "P_Dur_Global",     # P-wave duration (ms)
    "RR_Mean_Global",   # Mean RR interval (ms) -- inverse of HR
]
# age, sex, HR come from qt_labels.csv; the four above come from ecgdeli.
# Together they form a 7-dim vector [age, sex, HR, PR, QRS_Dur, P_Dur, RR].
TABULAR_DIM = 7

# Normalisation constants for the tabular features
# (approximate population means / stds; used to z-score at load time).
_TAB_MEAN = np.array([
    62.0,    # age
    0.5,     # sex  (0=female, 1=male)
    70.0,    # HR (bpm)
    162.0,   # PR_Int_Global (ms)
    93.0,    # QRS_Dur_Global (ms)
    111.0,   # P_Dur_Global (ms)
    877.0,   # RR_Mean_Global (ms)
], dtype=np.float32)
_TAB_STD = np.array([
    17.0, 0.5, 16.0, 22.0, 14.0, 18.0, 175.0
], dtype=np.float32)


def _load_ecgdeli(features_dir: Path) -> pd.DataFrame | None:
    """Load ecgdeli_features.csv from the comprehensive feature dataset.

    Returns a DataFrame indexed by ecg_id, or None if the file is missing.
    """
    path = features_dir / "features" / "ecgdeli_features.csv"
    if not path.exists():
        print(f"[qt_dataset] WARNING: ecgdeli_features.csv not found at {path}. "
              "Tabular features will be zeros.")
        return None
    df = pd.read_csv(path, usecols=["ecg_id"] + ECGDELI_TABULAR_COLS, low_memory=False)
    df = df.set_index("ecg_id")
    return df


class QTFinetuneDataset(Dataset):
    """
    Fine-tuning dataset: variable-lead electrode graph -> QT interval (ms).

    Args:
        ptbxl_signals_dir:   Path to the PTB-XL *signals* dataset root
                             (ptb-xl-a-large-publicly-available-...-1.0.1).
                             Must contain records500/.
        labels_csv:          Path to qt_labels.csv (already merged QT values).
        split:               'train', 'val', or 'test'.
        ptbxl_features_dir:  Optional path to the PTB-XL *feature* dataset root
                             (ptb-xl-a-comprehensive-...-1.0.1).
                             When provided, loads ecgdeli global features and
                             attaches them as data.tabular on the graph object.
        lead_dropout_prob:   Fraction of training samples with random lead subsets.
        scale_mv_to_mm:      Multiply signals by 10 (mV -> mm).  False = stay mV.
    """

    def __init__(
        self,
        ptbxl_signals_dir: str | Path,
        labels_csv: str | Path,
        split: str = "train",
        ptbxl_features_dir: Optional[str | Path] = None,
        lead_dropout_prob: float = 0.7,
        min_leads: int = 1,
        max_leads: int = 12,
        scale_mv_to_mm: bool = False,
    ):
        self.ptbxl_dir = Path(ptbxl_signals_dir)
        self.split = split
        self.lead_dropout_prob = lead_dropout_prob if split == "train" else 0.0
        self.min_leads = min_leads
        self.max_leads = max_leads
        self.scale_factor = 10.0 if scale_mv_to_mm else 1.0

        # Load ecgdeli tabular features (from the comprehensive dataset)
        self._ecgdeli: pd.DataFrame | None = None
        if ptbxl_features_dir is not None:
            self._ecgdeli = _load_ecgdeli(Path(ptbxl_features_dir))
            if self._ecgdeli is not None:
                print(f"[qt_dataset] Loaded ecgdeli tabular features "
                      f"for {len(self._ecgdeli)} records "
                      f"({len(ECGDELI_TABULAR_COLS)} columns + age/sex/HR).")

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
                "age": float(row["age"]) if pd.notna(row.get("age")) else 0.0,
                "sex": float(row["sex"]) if pd.notna(row.get("sex")) else 0.0,
                "hr": float(row["HR"]) if pd.notna(row["HR"]) else 0.0,
            })
        return samples

    def _build_tabular(self, item: dict) -> torch.Tensor:
        """Build a normalised 7-dim tabular feature vector.

        Order: [age, sex, HR, PR_Int_Global, QRS_Dur_Global, P_Dur_Global, RR_Mean_Global]
        Z-scored with precomputed population statistics.
        """
        raw = np.array([
            item["age"],
            item["sex"],
            item["hr"],
            0.0, 0.0, 0.0, 0.0,  # ecgdeli slots -- filled below
        ], dtype=np.float32)

        if self._ecgdeli is not None:
            eid = item["ecg_id"]
            if eid in self._ecgdeli.index:
                row = self._ecgdeli.loc[eid]
                for i, col in enumerate(ECGDELI_TABULAR_COLS):
                    v = row[col]
                    raw[3 + i] = float(v) if (v == v and v is not None) else 0.0
                    # NaN check: v == v is False for NaN

        # Z-score normalisation
        normed = (raw - _TAB_MEAN) / (_TAB_STD + 1e-8)
        return torch.tensor(normed, dtype=torch.float32).unsqueeze(0)  # (1,7) so PyG batches to (B,7)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        item = self.index[idx]

        # ---- Waveforms (from the large signals dataset) -------------------
        record = wfdb.rdrecord(str(self.ptbxl_dir / item["filename_hr"]))
        signals = record.p_signal  # (5000, 12) float64, mV

        assert [s.upper() for s in record.sig_name] == [s.upper() for s in PTBXL_LEAD_ORDER], (
            f"Lead order mismatch: {record.sig_name}"
        )

        signals = signals.T.astype(np.float32)  # (12, 5000)
        signals = np.nan_to_num(signals)
        signals *= self.scale_factor

        resampled = np.stack([
            resample(signals[i], TARGET_LEN).astype(np.float32)
            for i in range(12)
        ])  # (12, 2500)

        # ---- Lead dropout -------------------------------------------------
        if self.lead_dropout_prob > 0 and np.random.random() < self.lead_dropout_prob:
            num_leads = np.random.randint(self.min_leads, self.max_leads + 1)
            lead_indices = sorted(
                np.random.choice(12, num_leads, replace=False).tolist()
            )
        else:
            lead_indices = list(range(12))

        # ---- Build electrode graph ----------------------------------------
        graph = self.builder.build_subgraph_from_indices(
            resampled, lead_indices, bidirectional=True
        )

        # ---- Attach tabular features to graph object ----------------------
        # finetune_qt.py reads batch_data.tabular after batching.
        graph.tabular = self._build_tabular(item)

        qt = torch.tensor([item["qt_ms"]], dtype=torch.float32)
        return graph, qt
