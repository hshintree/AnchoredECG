# Convex GReLU Adapter for Single-Lead Stress ECG Latent Recovery

## Context

We have a **frozen graph-ECG encoder** (GraphECG, fine-tuned on stress ECG data) that maps any observed-lead ECG subgraph to a 192-dim embedding. We've already trained a **linear (Ridge) adapter** and an **MLP adapter** that recover the full 12-lead stress embedding from a single ICM stress lead + resting 12-lead prior. Your job is to implement the **convex GReLU residual adapter** from the proposal.

## The Problem

Given:
- `z_rest = E_G(12-lead baseline)` — patient's resting embedding (192-dim)
- `phi_stress = E_G(ICM stress)` — single-lead stress embedding (192-dim)

Predict:
- `z_hat_stress ≈ z_stress = E_G(12-lead stress)` — the full 12-lead stress embedding

Using the **residual formulation**:
```
z_hat_stress = z_rest + f_theta([phi_stress; z_rest])
```
where `f_theta` is a two-layer fixed-gate convex (GReLU) network, trained via group-lasso.

The target residual is `r = z_stress - z_rest`.

## What's in this repo

```
data/
    train.npz          z_rest, z_stress, phi_stress, bl_st80, ischemia (train split)
    val.npz            same (val split)
    regression_head.pt ST80 regression head weights (for ischemia eval)
src/
    adapter_interface.py   Base class + evaluation utilities
    grelu_adapter.py       YOUR CODE HERE — template with TODOs
    evaluate.py            Run to check your adapter on val
bundle.py              Script to populate data/ (run by Nico before sharing)
```

## Data shapes

| Array | Shape | Description |
|-------|-------|-------------|
| `z_rest` | (N, 192) | Resting 12-lead embedding |
| `z_stress` | (N, 192) | Stress 12-lead embedding (TARGET) |
| `phi_stress` | (N, 192) | Stress ICM embedding (OBSERVED) |
| `bl_st80` | (N, 12) | GT baseline ST80 values (mm) for ischemia eval |
| `ischemia` | (N,) | Binary patient-level ischemia label |

Train: ~4800 patients. Val: ~800 patients. Test is withheld.

## Adapter interface

Your adapter must implement the class in `src/adapter_interface.py`:

```python
class GReLUAdapter(AdapterBase):
    def fit(self, phi_train, z_rest_train, z_stress_train):
        # Train adapter: input u = [phi; z_rest] (384-dim), target r = z_stress - z_rest (192-dim)
        ...

    def predict(self, phi, z_rest):
        # Return z_hat = z_rest + f_theta([phi; z_rest])
        ...

    def save(self, path):
        # Save trained weights to a file
        ...

    @classmethod
    def load(cls, path):
        # Load trained weights from a file
        ...
```

## Evaluation

Run `python src/evaluate.py` to check your adapter against val set:
- **Latent MSE**: ||z_hat - z_stress||^2
- **Cosine similarity**: cos(z_hat, z_stress)
- **Ischemia AUROC**: using clinical thresholds on predicted ST80 deltas
  - Territory-level: >=1 mm ST deviation in Inferior (II/III/aVF), Anterior (V3/V4), Lateral (I/aVL/V5/V6)
  - Patient-level: >=2 contiguous leads in any territory >=1 mm

## How to submit

1. Implement `GReLUAdapter` in `src/grelu_adapter.py`
2. Run `python src/evaluate.py` and include the output
3. Your adapter's `save()` should produce a single file (e.g., `grelu_weights.pt`)
4. Push your branch and open a PR, or send the weights file + code

We'll plug your adapter into the full comparison pipeline (Linear, MLP, GReLU) and evaluate on the held-out test set.

## Reference

Varshney & Pilanci, "Convex distillation: Efficient compression of deep networks via convex optimization," arXiv 2024.

The key idea: sample random gates, fix their activation patterns on the training set, and train the resulting convex group-lasso problem. The adapter solves:
```
min_theta  sum_i ||N_theta(u_i) - r_i||^2  +  lambda * Omega(theta)
```
where `Omega(theta)` is the convex group-lasso regularizer.
