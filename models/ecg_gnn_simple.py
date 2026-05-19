"""
ECG GNN Simple: Classification-only model for SHD prediction.

This is a streamlined version of ECGGNNV2 that removes the reconstruction
auxiliary task. The core hypothesis being tested:

    ECG leads are differences between electrode potentials; treating them
    as independent channels hides this structure.

    We re-represent ECGs as electrode graphs: electrodes are nodes, observed
    leads are directed edges carrying waveforms.

    This yields a model interface that naturally supports missing and
    nonstandard leads via observed subgraphs, without changing architecture.

Architecture:
    1. Node encoding: Spherical harmonic features from electrode 3D positions
    2. Edge encoding: Conv1D encoder for lead waveforms -> embeddings
    3. GNN: Message passing that aggregates edge info into node representations
    4. Classification: Pool graph -> fuse with tabular -> predict SHD

Key design choices:
    - Bidirectional edges with physics: reverse edge = negated signal
    - Same architecture handles any lead subset (6-lead, 12-lead, ICM, etc.)
    - No reconstruction task = simpler training, cleaner story
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Optional, Dict
import math


class ResBlock1d(nn.Module):
    """Residual block for 1D convolutions."""

    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.activation = nn.Mish(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.activation(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.norm2(self.conv2(out))
        return self.activation(out + residual)


class SphericalHarmonicEncoding(nn.Module):
    """
    Encode 3D electrode positions using spherical harmonic basis functions.

    This captures the geometric relationships between electrodes in a
    rotation-equivariant manner.
    """

    def __init__(self, max_degree: int = 4, include_radial: bool = True):
        super().__init__()
        self.max_degree = max_degree
        self.include_radial = include_radial
        self.sh_dim = sum(2 * l + 1 for l in range(max_degree + 1))
        self.output_dim = self.sh_dim + (4 if include_radial else 0)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        r = torch.sqrt(x**2 + y**2 + z**2).clamp(min=1e-8)
        x_n, y_n, z_n = x / r, y / r, z / r

        sh_features = []

        # l=0 (constant)
        sh_features.append(torch.ones_like(x_n) * 0.5 * math.sqrt(1 / math.pi))

        # l=1 (dipole)
        if self.max_degree >= 1:
            c1 = 0.5 * math.sqrt(3 / math.pi)
            sh_features.extend([c1 * y_n, c1 * z_n, c1 * x_n])

        # l=2 (quadrupole)
        if self.max_degree >= 2:
            c2 = 0.5 * math.sqrt(15 / math.pi)
            c2_0 = 0.25 * math.sqrt(5 / math.pi)
            sh_features.extend([
                c2 * x_n * y_n,
                c2 * y_n * z_n,
                c2_0 * (3 * z_n**2 - 1),
                c2 * x_n * z_n,
                0.5 * c2 * (x_n**2 - y_n**2)
            ])

        # l=3
        if self.max_degree >= 3:
            sh_features.extend([
                y_n * (3 * x_n**2 - y_n**2),
                x_n * y_n * z_n,
                y_n * (5 * z_n**2 - 1),
                z_n * (5 * z_n**2 - 3),
                x_n * (5 * z_n**2 - 1),
                z_n * (x_n**2 - y_n**2),
                x_n * (x_n**2 - 3 * y_n**2)
            ])

        # l=4
        if self.max_degree >= 4:
            xy, xz, yz = x_n * y_n, x_n * z_n, y_n * z_n
            x2, y2, z2 = x_n**2, y_n**2, z_n**2
            sh_features.extend([
                xy * (x2 - y2),
                yz * (3 * x2 - y2),
                xy * (7 * z2 - 1),
                yz * (7 * z2 - 3),
                35 * z2**2 - 30 * z2 + 3,
                xz * (7 * z2 - 3),
                (x2 - y2) * (7 * z2 - 1),
                xz * (x2 - 3 * y2),
                x2**2 - 6 * x2 * y2 + y2**2
            ])

        sh = torch.stack(sh_features, dim=-1)

        if self.include_radial:
            radial = torch.stack([
                r, torch.sin(math.pi * r),
                torch.cos(math.pi * r), torch.exp(-r)
            ], dim=-1)
            sh = torch.cat([sh, radial], dim=-1)

        return sh


class LeadSignalEncoder(nn.Module):
    """
    Encode ECG lead signals (edge features) via Conv1D.

    Produces a fixed-size embedding for each lead that captures
    morphological features (QRS complex, T-wave, etc.).
    """

    def __init__(
        self,
        input_length: int = 2500,
        hidden_dim: int = 128,
        output_dim: int = 192,
    ):
        super().__init__()
        self.input_length = input_length

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),  # 2500 -> 1250
            nn.BatchNorm1d(64),
            nn.Mish(inplace=True),
            ResBlock1d(64),
            nn.Conv1d(64, hidden_dim, kernel_size=5, stride=2, padding=2),  # 1250 -> 625
            nn.BatchNorm1d(hidden_dim),
            nn.Mish(inplace=True),
            ResBlock1d(hidden_dim),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim),
            nn.Mish(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (E, T) lead signals where E = number of edges

        Returns:
            (E, output_dim) embeddings for each lead
        """
        x = x.unsqueeze(1)  # (E, 1, T)
        features = self.encoder(x)  # (E, output_dim, T')
        embedding = self.pool(features).squeeze(-1)  # (E, output_dim)
        return embedding


class ECGMessagePassing(MessagePassing):
    """
    Message passing layer for ECG graphs.

    Messages are computed from source node, target node, and edge (lead) features.
    This allows the model to learn how information flows through the electrode graph.
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__(aggr='add')

        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.Mish(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.Mish(inplace=True),
            nn.Linear(hidden_dim, node_dim),
        )

        self.norm = nn.LayerNorm(node_dim)

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.update_mlp(torch.cat([x, out], dim=-1))
        return self.norm(x + out)

    def message(self, x_i, x_j, edge_attr):
        return self.message_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))


class ECGGNNSimple(nn.Module):
    """
    Simple ECG Graph Neural Network for classification.

    The model processes ECG data represented as a graph where:
    - Nodes are electrodes (13 total: RA, LA, LL, WCT, V1-V6, 3 virtual midpoints)
    - Edges are leads (directed, with bidirectional physics: reverse = negated)

    This representation naturally handles:
    - Missing leads: just fewer edges in the subgraph
    - Nonstandard leads: any electrode pair can be an edge
    - Partial data: 6-lead, 3-lead, or any configuration

    Architecture:
        Input subgraph -> Node encoding -> Edge encoding -> GNN layers -> Pool -> Classify
    """

    def __init__(
        self,
        input_length: int = 2500,
        sh_max_degree: int = 4,
        node_dim: int = 128,
        edge_dim: int = 192,
        hidden_dim: int = 192,
        num_gnn_layers: int = 3,
        tabular_dim: int = 7,
        dropout: float = 0.5,  # Matching EchoNext baseline's dropout
    ):
        super().__init__()

        self.input_length = input_length
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim

        # Node encoding: 3D position -> spherical harmonics -> embedding
        self.pos_encoder = SphericalHarmonicEncoding(max_degree=sh_max_degree)
        self.node_proj = nn.Linear(self.pos_encoder.output_dim, node_dim)

        # Edge encoding: lead signal -> embedding
        self.signal_encoder = LeadSignalEncoder(
            input_length=input_length,
            hidden_dim=128,
            output_dim=edge_dim,
        )

        # GNN layers
        self.gnn_layers = nn.ModuleList([
            ECGMessagePassing(node_dim, edge_dim, node_dim * 2)
            for _ in range(num_gnn_layers)
        ])

        # Graph-level pooling and projection (to match baseline's 256-dim)
        # Baseline uses 128 avg + 128 max = 256, we use projection to 256
        self.graph_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.Mish(inplace=True),
        )

        # Dropout before fusion (matching baseline's 0.5 dropout)
        self.pre_fusion_dropout = nn.Dropout(dropout)

        # Classification head: Late fusion matching EchoNext baseline
        # Baseline: concat(CNN_256, tabular_7) -> Linear(263, num_classes)
        # Ours: concat(GNN_192, tabular_7) -> Linear(199, 1)
        #
        # NOTE: We concatenate RAW tabular features (no encoding MLP)
        # to match the baseline's approach for fair comparison.
        self.classifier = nn.Linear(hidden_dim + tabular_dim, 1)

    def forward(
        self,
        data: Data,
        tabular: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            data: PyG Data/Batch with:
                - x: (N, 3) node positions (electrodes)
                - edge_index: (2, E) edge indices
                - edge_attr: (E, T) edge signals (leads)
                - batch: (N,) batch assignment (for batched graphs)
            tabular: (B, 7) tabular features (optional)

        Returns:
            Dict with 'logits' and 'graph_embed'
        """
        batch = getattr(data, 'batch', None)

        # Encode nodes from positions
        node_features = self.node_proj(self.pos_encoder(data.x))

        # Encode edges from lead signals
        edge_embed = self.signal_encoder(data.edge_attr)

        # Message passing
        x = node_features
        for gnn in self.gnn_layers:
            x = gnn(x, data.edge_index, edge_embed)

        # Global pooling
        if batch is None:
            graph_embed = x.mean(dim=0, keepdim=True)
        else:
            graph_embed = global_mean_pool(x, batch)

        # Project to hidden dim
        graph_embed = self.graph_proj(graph_embed)

        # Dropout before fusion (matching baseline)
        graph_embed = self.pre_fusion_dropout(graph_embed)

        # Late fusion: concatenate RAW tabular features (matching EchoNext baseline)
        # No tabular encoding MLP - just raw concatenation
        if tabular is not None:
            fused = torch.cat([graph_embed, tabular], dim=-1)
        else:
            # Pad with zeros if no tabular (7 features)
            zeros = torch.zeros(graph_embed.shape[0], 7, device=graph_embed.device)
            fused = torch.cat([graph_embed, zeros], dim=-1)

        # Single linear layer to output (matching baseline's simple classifier)
        logits = self.classifier(fused)

        return {
            'logits': logits,
            'graph_embed': graph_embed,
        }


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print("Testing ECG GNN Simple")
    print("=" * 60)

    from ecg_graph_v2 import ECGGraphBuilderV2

    builder = ECGGraphBuilderV2()
    T = 2500

    # Create model (matching baseline's architecture choices)
    model = ECGGNNSimple(
        input_length=T,
        sh_max_degree=4,
        node_dim=128,
        edge_dim=192,
        hidden_dim=192,
        num_gnn_layers=3,
        tabular_dim=7,
        dropout=0.5,  # Match EchoNext baseline
    )
    print(f"Model parameters: {count_parameters(model):,}")

    # Test with 12-lead
    print("\nTesting 12-lead input...")
    dummy_ecg = np.random.randn(12, T).astype(np.float32)
    graph = builder.build_12lead_subgraph(dummy_ecg, bidirectional=True)
    tabular = torch.randn(1, 7)

    with torch.no_grad():
        outputs = model(graph, tabular)
        print(f"  Graph embed: {outputs['graph_embed'].shape}")
        print(f"  Logits: {outputs['logits'].shape}")

    # Test with 6-lead (limb leads only)
    print("\nTesting 6-lead input (limb leads)...")
    limb_indices = [0, 1, 2, 3, 4, 5]
    graph_6 = builder.build_subgraph_from_indices(dummy_ecg, limb_indices, bidirectional=True)

    with torch.no_grad():
        outputs = model(graph_6, tabular)
        print(f"  Graph embed: {outputs['graph_embed'].shape}")
        print(f"  Logits: {outputs['logits'].shape}")

    # Test batched
    print("\nTesting batched input (4 samples, mixed leads)...")
    graphs = []
    for i in range(4):
        ecg = np.random.randn(12, T).astype(np.float32)
        num_leads = np.random.randint(6, 13)
        indices = sorted(np.random.choice(12, num_leads, replace=False))
        graphs.append(builder.build_subgraph_from_indices(ecg, indices, bidirectional=True))

    batch = Batch.from_data_list(graphs)
    tabular_batch = torch.randn(4, 7)

    with torch.no_grad():
        outputs = model(batch, tabular_batch)
        print(f"  Graph embed: {outputs['graph_embed'].shape}")
        print(f"  Logits: {outputs['logits'].shape}")

    print("\n" + "=" * 60)
    print("ECG GNN Simple test passed!")
