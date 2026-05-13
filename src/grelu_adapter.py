"""
Convex GReLU Residual Adapter -- IMPLEMENT THIS.

Reference: Varshney & Pilanci, "Convex distillation" (2024)

The adapter learns:
    r_hat = N_theta(u)
where:
    u = [phi_stress; z_rest]   in R^384   (input)
    r = z_stress - z_rest      in R^192   (target)

N_theta is a two-layer fixed-gate convex network:
    N_theta(u) = sum_k  w_k * max(D_k * V * u, 0)

where D_k are FIXED diagonal gate matrices (randomly sampled, then frozen),
V is a learnable weight matrix, and w_k are learnable output weights.

Because the gates D_k are fixed, the training problem is convex in (V, w_k)
and can be solved via group-lasso.

The prediction is:
    z_hat = z_rest + N_theta(u)
"""
from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from adapter_interface import AdapterBase


class GReLUAdapter(AdapterBase):
    """Convex GReLU residual adapter with fixed gates."""

    def __init__(self, d_in=384, d_out=192, n_gates=512, lam=1e-3, seed=42):
        """
        Args:
            d_in:    input dimension (384 = 192 phi + 192 z_rest)
            d_out:   output dimension (192 = residual)
            n_gates: number of random gate patterns to sample
            lam:     group-lasso regularization strength
            seed:    random seed for gate sampling (MUST be fixed for reproducibility)
        """
        self.d_in = d_in
        self.d_out = d_out
        self.n_gates = n_gates
        self.lam = lam
        self.seed = seed

        # TODO: initialize your model parameters here
        self.weights = None

    def fit(self, phi_train, z_rest_train, z_stress_train):
        """Train the GReLU adapter.

        Steps (following Varshney & Pilanci 2024):
        1. Form input u = [phi; z_rest]  shape (N, 384)
        2. Form target r = z_stress - z_rest  shape (N, 192)
        3. Sample n_gates random gate patterns D_k (diagonal 0/1 matrices)
           applied to a random projection of u
        4. Fix the gate patterns (compute activation masks on training data)
        5. Solve the resulting convex group-lasso problem for the weights

        The optimization is:
            min_{w,V}  sum_i ||sum_k w_k * relu(D_k * V * u_i) - r_i||^2
                       + lambda * sum_k ||w_k||_2

        Because D_k are fixed, this is convex in the remaining parameters.
        """
        u = np.hstack([phi_train, z_rest_train])    # (N, 384)
        r = z_stress_train - z_rest_train            # (N, 192)

        # TODO: implement the convex GReLU training
        # Hint: see Algorithm 1 in Varshney & Pilanci 2024
        raise NotImplementedError("Implement GReLU training here")

    def predict(self, phi, z_rest):
        """Predict z_hat = z_rest + N_theta([phi; z_rest])."""
        u = np.hstack([phi, z_rest])                 # (N, 384)

        # TODO: compute r_hat = N_theta(u) using trained weights
        r_hat = np.zeros((u.shape[0], self.d_out))   # placeholder
        raise NotImplementedError("Implement GReLU prediction here")

        return z_rest + r_hat

    def save(self, path):
        """Save trained adapter."""
        # TODO: save your weights
        # Example: torch.save({'weights': self.weights, ...}, path)
        raise NotImplementedError

    @classmethod
    def load(cls, path):
        """Load trained adapter."""
        # TODO: load weights and return a ready-to-use adapter
        raise NotImplementedError
