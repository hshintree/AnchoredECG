"""
PTB-XL waveform dataset for the 1-D ResNet-18 QT ablation.

Unlike ``QTFinetuneDataset`` (which builds an electrode graph), this returns the
raw multi-channel waveform tensor for a fixed set of leads, so it can feed a
plain CNN.

Input  : (C, L) float32 — C leads, resampled to ``target_fs``.
Target : QT_Int_Global (ms), single scalar.

The target is in **milliseconds**, so it is independent of the input sample
rate: changing ``target_fs`` changes the input length but never the label.

``target_fs`` exists because we are optimising for Apple Watch input (512 Hz
native). See ``rates.py`` for why 512 and 128 are the supported choices. The
only transform applied is polyphase resampling — no normalisation, filtering,
or augmentation.

Splits follow the rest of the QT pipeline: strat_fold <= 8 train, == 9 val,
== 10 test.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset

from rates import PTBXL_FS, DEFAULT_RATE, fit_length, resample_signal, window_len

# PTB-XL lead order: I(0), II(1), III(2), aVR(3), aVL(4), aVF(5), V1-V6(6-11)
PTBXL_LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF",
                    "V1", "V2", "V3", "V4", "V5", "V6"]

# Derived (augmented) single-lead definitions: name -> (pos_idx, neg_idx),
# producing one channel = signals[pos_idx] - signals[neg_idx].
# "v3_v2" is the ICM-style lead V3 - V2 (matches validate_qt.ICMLeadDataset).
DERIVED_LEADS = {
    "v3_v2": (8, 7),   # V3(idx 8) - V2(idx 7)
}


class ResNetQTDataset(Dataset):
    """Fixed-lead waveform → QT (ms) regression dataset.

    Args:
        ptbxl_signals_dir : PTB-XL signals root (must contain records500/).
        labels_csv        : qt_labels.csv.
        split             : 'train' | 'val' | 'test'.
        lead_indices      : leads to keep, in PTB-XL order. [0] = lead I,
                            list(range(12)) = all 12 leads.
        target_fs         : resample PTB-XL's native 500 Hz to this rate.
                            512 = Apple Watch native, 128 = exact /4 of it,
                            500 = no resampling (legacy baseline).
    """

    def __init__(
        self,
        ptbxl_signals_dir: str | Path,
        labels_csv: str | Path,
        split: str = "train",
        lead_indices: list[int] | None = None,
        derived: str | None = None,
        target_fs: int = DEFAULT_RATE,
    ):
        self.ptbxl_dir = Path(ptbxl_signals_dir)
        self.split = split
        self.derived = derived
        self.target_fs = int(target_fs)
        self.target_len = window_len(self.target_fs)
        if derived is not None and derived not in DERIVED_LEADS:
            raise ValueError(f"Unknown derived lead {derived!r}; "
                             f"options: {list(DERIVED_LEADS)}")
        self.lead_indices = sorted(lead_indices if lead_indices is not None
                                   else range(12))
        self.index = self._build_index(labels_csv)

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
                "ecg_id": int(row["ecg_id"]),
                "filename_hr": row["filename_hr"],
                "qt_ms": float(row["QT_Int_Global"]),
            }
            for _, row in df.iterrows()
        ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        item = self.index[idx]
        record = wfdb.rdrecord(str(self.ptbxl_dir / item["filename_hr"]))

        assert [s.upper() for s in record.sig_name] == \
               [s.upper() for s in PTBXL_LEAD_ORDER], \
               f"Lead order mismatch: {record.sig_name}"

        signals = record.p_signal.T.astype(np.float32)     # (12, 5000), mV
        signals = np.nan_to_num(signals)

        if self.derived is not None:
            pos, neg = DERIVED_LEADS[self.derived]
            signals = (signals[pos] - signals[neg])[None, :]   # (1, 5000)
        else:
            signals = signals[self.lead_indices]               # (C, 5000)

        # Resample after lead selection (cheaper, and resampling is per-channel).
        if self.target_fs != PTBXL_FS:
            signals = resample_signal(signals, PTBXL_FS, self.target_fs)
        signals = fit_length(signals, self.target_len)          # (C, target_len)

        x = torch.from_numpy(np.ascontiguousarray(signals))    # (C, target_len)
        qt = torch.tensor([item["qt_ms"]], dtype=torch.float32)
        return x, qt
