"""
text_encoder.py

Text encoder: frozen DistilBERT backbone → trainable projection MLP.
Input:  list of caption strings (raw text)
Output: L2-normalized embeddings, shape (B, 256)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertTokenizerFast
from config import TEXT_MODEL_NAME, TEXT_INPUT_DIM, HIDDEN_DIM, EMBEDDING_DIM, DROPOUT, MAX_TOKEN_LENGTH


class TextEncoder(nn.Module):
    """
    Frozen DistilBERT + trainable projection MLP.
    Uses [CLS] token as the sentence representation.
    """

    def __init__(self):
        super().__init__()
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(TEXT_MODEL_NAME)
        self.bert      = DistilBertModel.from_pretrained(TEXT_MODEL_NAME)

        # Freeze all DistilBERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(TEXT_INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, EMBEDDING_DIM),
        )

    def forward(self, captions, device):
        """
        Args:
            captions: list of B caption strings
            device:   torch device
        Returns:
            Tensor of shape (B, 256), L2-normalized
        """
        encoded = self.tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():                         # No grad through frozen BERT
            outputs = self.bert(**encoded)

        cls_token = outputs.last_hidden_state[:, 0, :]    # (B, 768) — [CLS] token
        projected = self.projection(cls_token)             # (B, 256)
        return F.normalize(projected, dim=-1)              # L2 normalize
