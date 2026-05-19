# AnchoredECG: Convex Adapters for Reduced-Lead ECG Latent Recovery

## Overview

This repo contains two tasks built on the same **GraphECG** encoder (1.21M params, 192-dim embeddings, variable-lead graph ECG model):

1. **ST80 Ischemia Adapter** (`data/` + `src/`) -- recover full 12-lead stress ECG embeddings from a single ICM lead + resting prior, then predict ST-segment deviations for ischemia detection.
2. **QT Interval Fine-Tuning + Adapter** (`qt_finetune/`) -- fine-tune GraphECG on PTB-XL for QT interval prediction with lead dropout, then train adapters to recover full embeddings from reduced leads.

Both use the same core idea: **a convex adapter (Ridge or GReLU) in the encoder's latent space outperforms nonlinear alternatives and even end-to-end fine-tuning** because the encoder's lead-dropout pretraining creates a well-structured embedding space where the recovery problem is approximately linear.

---

## 1. ST80 Ischemia Adapter (existing)

See the [original README section](#st80-adapter-details) below.

---

## 2. QT Interval Pipeline (new)

### Prerequisites

```bash
pip install torch torch_geometric wfdb scipy numpy pandas scikit-learn tqdm
```

Download PTB-XL from PhysioNet:
```bash
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
# Or use your existing PTB-XL directory
```

### Directory structure

```
qt_finetune/
    qt_labels.csv              QT labels from GE 12SL (21,799 records, 100% coverage)
    qt_dataset.py              PyG dataset: PTB-XL + lead dropout -> QT regression
    finetune_qt.py             Fine-tuning script (SHD pretrained -> QT)
    extract_embeddings.py      Post-finetune embedding extraction (5 lead configs)
    qt_adapter_eval.py         Baseline adapter comparison (Oracle/Direct/Ridge)
    checkpoints/               (created during training)
    results/                   (created during extraction)

models/
    ecg_gnn_simple.py          GraphECG model definition
    ecg_graph_v2.py            Graph builder (electrode subgraphs)
    ecg_graph_flexible.py      Electrode positions

checkpoints/
    shd_pretrained.pt          SHD pretrained weights (starting point)
```

### Step 1: Fine-tune on QT

```bash
# Smoke test (verify data loading + forward pass)
python qt_finetune/finetune_qt.py --ptbxl-dir /path/to/ptbxl --smoke-test

# Full training (~30 epochs, differential LR: encoder 5e-5, head 1e-3)
python qt_finetune/finetune_qt.py --ptbxl-dir /path/to/ptbxl --epochs 30 --batch-size 32
```

This saves `qt_finetune/checkpoints/best_qt_model.pt` (best val MAE).

Architecture:
- Encoder: GraphECG (frozen SHD weights, fine-tuned with LR 5e-5)
- Head: `Linear(199, 64) -> Mish -> Linear(64, 1)` (new, LR 1e-3)
- Lead dropout: 70% of training batches use random 1-12 lead subsets
- Loss: MSE on QT_Int_Global (ms)
- Optimizer: AdamW, weight decay 0.01, cosine schedule

### Step 2: Extract embeddings

```bash
set KMP_DUPLICATE_LIB_OK=TRUE
python qt_finetune/extract_embeddings.py --ptbxl-dir /path/to/ptbxl
```

Extracts embeddings for 5 lead configurations:
| Config | Leads | Use case |
|--------|-------|----------|
| `full_12` | All 12 | Target (z_full) |
| `lead_II` | Lead II only | Single-lead wearable |
| `lead_V5` | Lead V5 only | Chest patch |
| `limb_6` | I,II,III,aVR,aVL,aVF | Limb leads only |
| `precord_6` | V1-V6 | Precordial leads only |

Saves to `qt_finetune/results/embeddings.npz` with keys:
- `z_full_12`, `z_lead_II`, `z_limb_6`, etc. -- (N, 192) embeddings
- `qt_gt`, `qtc_bazett_gt`, `hr` -- labels
- `strat_fold` -- for train/val/test splits (1-8/9/10)

### Step 3: Train your adapter

The adapter maps reduced-lead embeddings to full 12-lead embeddings:
```
z_hat = f(phi_reduced)  such that  z_hat ~ z_full
```

Then decode QT via the frozen regression head:
```
QT_pred = head(z_hat)
```

Run the baseline comparison:
```bash
python qt_finetune/qt_adapter_eval.py
```

This trains Ridge adapters for each config and reports QT MAE and Pearson R. **Replace Ridge with your convex GReLU adapter and compare.**

### QT Labels

`qt_labels.csv` contains 12SL algorithm measurements for all 21,799 PTB-XL records:

| Column | Description |
|--------|-------------|
| `ecg_id` | PTB-XL record ID |
| `QT_Int_Global` | Raw QT interval (ms) |
| `QTc_Bazett` | Bazett-corrected QTc (ms) |
| `QTc_Fridericia` | Fridericia-corrected QTc (ms) |
| `QTc_Framingham` | Framingham-corrected QTc (ms) |
| `HR` | Ventricular heart rate (bpm) |
| `patient_id` | PTB-XL patient ID |
| `strat_fold` | Stratified fold (1-10) |
| `age`, `sex` | Demographics |
| `filename_hr` | Path to 500 Hz waveform (relative to PTB-XL root) |

Splits: train (folds 1-8, 17,418), val (fold 9, 2,183), test (fold 10, 2,198).

### Signal details

- PTB-XL: 500 Hz, 10 seconds, 12 leads (5000 samples)
- Resampled to 250 Hz (2500 samples) to match GraphECG's original input format
- Signals loaded in mV via wfdb (no unit conversion applied)
- Lead order: I, II, III, aVR, aVL, aVF, V1-V6

---

## ST80 Adapter Details

### The Problem

Given:
- `z_rest = E_G(12-lead baseline)` -- resting embedding (192-dim)
- `phi_stress = E_G(ICM stress)` -- single-lead stress embedding (192-dim)

Predict:
- `z_hat ≈ z_stress = E_G(12-lead stress)` -- full stress embedding

Using: `z_hat = z_rest + f([phi_stress; z_rest])`

### Data (data/)

| Array | Shape | Description |
|-------|-------|-------------|
| `z_rest` | (N, 192) | Resting 12-lead embedding |
| `z_stress` | (N, 192) | Stress 12-lead embedding (TARGET) |
| `phi_stress` | (N, 192) | Stress ICM embedding (OBSERVED) |
| `bl_st80` | (N, 12) | GT baseline ST80 values (mm) |
| `ischemia` | (N,) | Binary patient-level ischemia label |

Train: ~4800. Val: ~800. Test: withheld.

### Adapter interface (src/)

```python
class GReLUAdapter(AdapterBase):
    def fit(self, phi_train, z_rest_train, z_stress_train): ...
    def predict(self, phi, z_rest): ...
    def save(self, path): ...
    def load(cls, path): ...
```

Run: `python src/evaluate.py`

### Reference

Varshney & Pilanci, "Convex distillation: Efficient compression of deep networks via convex optimization," arXiv 2024.
