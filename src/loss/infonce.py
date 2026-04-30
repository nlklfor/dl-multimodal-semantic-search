"""
infonce.py

InfoNCE (Noise-Contrastive Estimation) loss from scratch.

Given a batch of B matched (image, text) pairs:
- Constructs a B×B cosine similarity matrix
- Treats each row as a B-class classification problem
- Ground truth: diagonal (index i matches index i)
- Loss is symmetric: computed in both image→text and text→image directions
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import INIT_TEMPERATURE


class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE loss with optional learnable temperature.

    Args:
        temperature (float or None):
            - None  → learnable parameter, initialized at INIT_TEMPERATURE
            - float → fixed temperature for ablation experiments
    """

    def __init__(self, temperature=None):
        super().__init__()
        if temperature is None:
            # Learnable log-temperature (more numerically stable than raw τ)
            self.log_temperature = nn.Parameter(
                torch.tensor(math.log(1.0 / INIT_TEMPERATURE))
            )
            self.fixed = False
        else:
            self.log_temperature = math.log(1.0 / temperature)
            self.fixed = True

    @property
    def temperature(self):
        if self.fixed:
            return math.exp(-self.log_temperature)
        return torch.exp(-self.log_temperature)

    def forward(self, image_emb, text_emb):
        """
        Args:
            image_emb: Tensor (B, D) — L2-normalized image embeddings
            text_emb:  Tensor (B, D) — L2-normalized text embeddings
        Returns:
            Scalar loss value
        """
        assert image_emb.shape == text_emb.shape, \
            f"Shape mismatch: {image_emb.shape} vs {text_emb.shape}"

        B = image_emb.shape[0]

        # Cosine similarity matrix — (B, B)
        # Since embeddings are L2-normalized, dot product = cosine similarity
        logits = (image_emb @ text_emb.T) / self.temperature

        # Ground truth: diagonal indices [0, 1, 2, ..., B-1]
        labels = torch.arange(B, device=image_emb.device)

        # Symmetric loss: both directions
        loss_i2t = F.cross_entropy(logits,   labels)    # image→text
        loss_t2i = F.cross_entropy(logits.T, labels)    # text→image

        return (loss_i2t + loss_t2i) / 2.0
