"""
1-D ResNet-18 for ECG regression (QT interval prediction).

A standard supervised baseline for the QT ablation. Unlike the electrode-graph
GNN (``ECGGNNSimple``), this treats the ECG as a plain multi-channel 1-D signal
and runs a textbook ResNet-18 over it. There is no pretraining and no graph
structure — it is the "is our lead-I GNN meaningfully better than a vanilla
CNN?" control.

Input  : (B, C, L) float32 — C leads, L=5000 samples @ 500 Hz.
Output : dict with "logits" of shape (B, 1) — predicted QT (ms).

The forward signature mirrors ``ECGGNNSimple`` loosely (returns a dict with a
"logits" key), but it takes a raw tensor rather than a PyG batch, so the QT
training/eval loops use the plain ``ResNetQTDataset`` + ``torch`` DataLoader.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BasicBlock1d(nn.Module):
    """ResNet basic block, 1-D variant (two 3-wide convs + identity skip)."""

    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)

        self.downsample = None
        if stride != 1 or in_ch != out_ch * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch * self.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_ch * self.expansion),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ECGResNet1d(nn.Module):
    """1-D ResNet-18 regression model.

    Args:
        in_channels : number of ECG leads fed in (1 for lead I, 12 for all).
        num_outputs : regression targets (1 = QT in ms).
        layers      : blocks per stage. (2,2,2,2) = ResNet-18.
        widths      : channel widths per stage (ResNet default 64/128/256/512).
        dropout     : dropout before the final FC.
    """

    def __init__(
        self,
        in_channels: int = 12,
        num_outputs: int = 1,
        layers: tuple[int, ...] = (2, 2, 2, 2),
        widths: tuple[int, ...] = (64, 128, 256, 512),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, widths[0], kernel_size=7, stride=2,
                      padding=3, bias=False),
            nn.BatchNorm1d(widths[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self._cur_ch = widths[0]
        self.layer1 = self._make_stage(widths[0], layers[0], stride=1)
        self.layer2 = self._make_stage(widths[1], layers[1], stride=2)
        self.layer3 = self._make_stage(widths[2], layers[2], stride=2)
        self.layer4 = self._make_stage(widths[3], layers[3], stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(widths[3] * BasicBlock1d.expansion, num_outputs)

        self._init_weights()

    def _make_stage(self, out_ch: int, n_blocks: int, stride: int) -> nn.Sequential:
        blocks = [BasicBlock1d(self._cur_ch, out_ch, stride=stride)]
        self._cur_ch = out_ch * BasicBlock1d.expansion
        for _ in range(1, n_blocks):
            blocks.append(BasicBlock1d(self._cur_ch, out_ch, stride=1))
        return nn.Sequential(*blocks)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, tabular: torch.Tensor | None = None) -> dict:
        # x: (B, C, L). tabular is ignored (kept for interface symmetry).
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feat = self.pool(x).flatten(1)          # (B, 512)
        logits = self.classifier(self.dropout(feat))   # (B, 1)
        return {"logits": logits, "embed": feat}
