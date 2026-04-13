"""
vision_encoder.py

Vision encoder: frozen ResNet-50 backbone → trainable projection MLP.
Input:  pre-cached ResNet-50 features, shape (B, 2048)
Output: L2-normalized embeddings, shape (B, 256)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import VISION_INPUT_DIM, HIDDEN_DIM, EMBEDDING_DIM, DROPOUT


class ProjectionMLP(nn.Module):
    """Two-layer projection MLP with ReLU and Dropout."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)      # L2 normalize — critical


class VisionEncoder(nn.Module):
    """
    Wraps pre-cached ResNet-50 features through a trainable projection head.
    The ResNet backbone is NOT included here — it runs in precompute_features.py.
    """

    def __init__(self):
        super().__init__()
        self.projection = ProjectionMLP(
            input_dim=VISION_INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=EMBEDDING_DIM,
            dropout=DROPOUT,
        )

    def forward(self, cached_features):
        """
        Args:
            cached_features: Tensor of shape (B, 2048) from pre-cached ResNet-50
        Returns:
            Tensor of shape (B, 256), L2-normalized
        """
        return self.projection(cached_features)
