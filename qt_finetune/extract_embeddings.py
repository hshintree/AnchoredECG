"""
Extract frozen encoder embeddings from QT-finetuned model for adapter experiments.

For each patient in PTB-XL:
  z_full     = E_G(12-lead ECG)              in R^192   (target)
  phi_single = E_G(single lead, e.g. II)     in R^192   (observed)
  phi_limb   = E_G(limb leads: I,II,III,aVR,aVL,aVF) in R^192
  phi_precord = E_G(precordial: V1-V6)      in R^192

Also stores QT labels and the QT regression head weights.

Saves:
  qt_finetune/results/embeddings.npz
  qt_finetune/results/qt_regression_head.pt

Usage:
    set KMP_DUPLICATE_LIB_OK=TRUE
    python qt_finetune/extract_embeddings.py --ptbxl-dir /path/to/ptbxl
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wfdb
from scipy.signal import resample
from tqdm import tqdm
from torch_geometric.data import Batch

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_MODELS_DIR = _REPO_ROOT / "models"
for p in [str(_MODELS_DIR), str(_SCRIPT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ecg_gnn_simple import ECGGNNSimple
from ecg_graph_v2 import ECGGraphBuilderV2, LEAD_NAMES_12
from qt_dataset import TARGET_LEN, PTBXL_LEAD_ORDER

DEFAULT_CKPT = _SCRIPT_DIR / "checkpoints" / "best_qt_model.pt"
DEFAULT_LABELS = _SCRIPT_DIR / "qt_labels.csv"
RESULTS_DIR = _SCRIPT_DIR / "results"

LEAD_CONFIGS = {
    "full_12":   list(range(12)),
    "lead_II":   [1],
    "lead_V5":   [10],
    "limb_6":    [0, 1, 2, 3, 4, 5],
    "precord_6": [6, 7, 8, 9, 10, 11],
}


def load_model(ckpt_path, device):
    model = ECGGNNSimple(
        input_length=TARGET_LEN, sh_max_degree=4, node_dim=128,
        edge_dim=192, hidden_dim=192, num_gnn_layers=3,
        tabular_dim=7, dropout=0.3,
    )
    model.classifier = nn.Sequential(
        nn.Linear(199, 64), nn.Mish(inplace=True), nn.Linear(64, 1),
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded epoch {ckpt['epoch']}  val_MAE={ckpt['val_mae']:.2f} ms")
    return model.to(device).eval()


def load_signals(ptbxl_dir, filename_hr):
    record = wfdb.rdrecord(str(Path(ptbxl_dir) / filename_hr))
    signals = record.p_signal.T.astype(np.float32)  # (12, 5000)
    signals = np.nan_to_num(signals)
    resampled = np.stack([
        resample(signals[i], TARGET_LEN).astype(np.float32)
        for i in range(12)
    ])  # (12, 2500)
    return resampled


@torch.no_grad()
def extract_batch(model, builder, signals_list, lead_indices, device):
    graphs = []
    for sigs in signals_list:
        graph = builder.build_subgraph_from_indices(
            sigs, lead_indices, bidirectional=True
        )
        graphs.append(graph)
    batch = Batch.from_data_list(graphs).to(device)
    out = model(batch, tabular=None)
    return out["graph_embed"].cpu().numpy(), out["logits"].cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptbxl-dir", type=str, required=True)
    parser.add_argument("--labels-csv", type=str, default=str(DEFAULT_LABELS))
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, device)
    builder = ECGGraphBuilderV2()

    # Save the QT regression head separately for adapter evaluation
    head_state = model.classifier.state_dict()
    torch.save(head_state, RESULTS_DIR / "qt_regression_head.pt")
    print(f"Saved regression head to {RESULTS_DIR / 'qt_regression_head.pt'}")

    # Load labels
    df = pd.read_csv(args.labels_csv)
    print(f"Total records: {len(df)}")

    # Collect per-config embeddings
    all_embeddings = {name: [] for name in LEAD_CONFIGS}
    all_qt_pred = {name: [] for name in LEAD_CONFIGS}
    all_qt_gt = []
    all_qtc_gt = []
    all_hr = []
    all_ecg_id = []
    all_fold = []

    BS = args.batch_size
    for i in tqdm(range(0, len(df), BS), desc="Extracting"):
        chunk = df.iloc[i:i + BS]

        signals_list = [
            load_signals(args.ptbxl_dir, row["filename_hr"])
            for _, row in chunk.iterrows()
        ]

        for config_name, lead_idx in LEAD_CONFIGS.items():
            emb, pred = extract_batch(model, builder, signals_list, lead_idx, device)
            all_embeddings[config_name].append(emb)
            all_qt_pred[config_name].append(pred)

        all_qt_gt.extend(chunk["QT_Int_Global"].tolist())
        all_qtc_gt.extend(chunk["QTc_Bazett"].tolist())
        all_hr.extend(chunk["HR"].tolist())
        all_ecg_id.extend(chunk["ecg_id"].tolist())
        all_fold.extend(chunk["strat_fold"].tolist())

    # Save
    save_dict = {
        "qt_gt": np.array(all_qt_gt, dtype=np.float32),
        "qtc_bazett_gt": np.array(all_qtc_gt, dtype=np.float32),
        "hr": np.array(all_hr, dtype=np.float32),
        "ecg_id": np.array(all_ecg_id, dtype=np.int64),
        "strat_fold": np.array(all_fold, dtype=np.int32),
    }
    for config_name in LEAD_CONFIGS:
        save_dict[f"z_{config_name}"] = np.concatenate(all_embeddings[config_name])
        save_dict[f"qt_pred_{config_name}"] = np.concatenate(all_qt_pred[config_name])

    out = RESULTS_DIR / "embeddings.npz"
    np.savez(out, **save_dict)

    # Summary
    folds = save_dict["strat_fold"]
    for split_name, mask_fn in [("train", lambda f: f <= 8),
                                 ("val", lambda f: f == 9),
                                 ("test", lambda f: f == 10)]:
        mask = mask_fn(folds)
        n = mask.sum()
        qt = save_dict["qt_gt"][mask]
        print(f"  {split_name:5s}: {n} records, QT mean={qt.mean():.1f} +/- {qt.std():.1f} ms")

    print(f"\nSaved {out}")
    print(f"Keys: {list(save_dict.keys())}")
    print(f"\nEmbedding configs: {list(LEAD_CONFIGS.keys())}")
    print("z_full_12 is the target. Others are reduced-lead inputs for the adapter.")


if __name__ == "__main__":
    main()
