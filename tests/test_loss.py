"""
test_loss.py

Unit tests for the InfoNCE loss implementation.
Run with: python -m pytest tests/ -v
"""

import math
import torch
import pytest
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.loss.infonce import InfoNCELoss


class TestInfoNCELoss:

    def test_output_is_scalar(self):
        loss_fn = InfoNCELoss()
        img = torch.randn(16, 256)
        img = torch.nn.functional.normalize(img, dim=-1)
        txt = torch.randn(16, 256)
        txt = torch.nn.functional.normalize(txt, dim=-1)
        loss = loss_fn(img, txt)
        assert loss.shape == torch.Size([]), "Loss should be a scalar"

    def test_perfect_alignment_gives_low_loss(self):
        """When image and text embeddings are identical, loss should be near 0."""
        loss_fn = InfoNCELoss(temperature=0.07)
        emb = torch.nn.functional.normalize(torch.randn(32, 256), dim=-1)
        loss = loss_fn(emb, emb)
        assert loss.item() < 0.5, f"Loss with perfect alignment too high: {loss.item()}"

    def test_random_embeddings_give_high_loss(self):
        """Completely random uncorrelated embeddings should yield high loss."""
        loss_fn = InfoNCELoss(temperature=0.07)
        img = torch.nn.functional.normalize(torch.randn(64, 256), dim=-1)
        txt = torch.nn.functional.normalize(torch.randn(64, 256), dim=-1)
        loss = loss_fn(img, txt)
        # For B=64, random loss ≈ log(64) ≈ 4.15
        assert loss.item() > 2.0, f"Loss with random embeddings too low: {loss.item()}"

    def test_learnable_temperature(self):
        """Temperature should be learnable and not None."""
        loss_fn = InfoNCELoss(temperature=None)
        assert not loss_fn.fixed
        assert isinstance(loss_fn.log_temperature, torch.nn.Parameter)

    def test_fixed_temperature(self):
        loss_fn = InfoNCELoss(temperature=0.1)
        assert loss_fn.fixed
        assert abs(loss_fn.temperature - 0.1) < 1e-5

    def test_shape_mismatch_raises(self):
        loss_fn = InfoNCELoss()
        img = torch.randn(16, 256)
        txt = torch.randn(32, 256)
        with pytest.raises(AssertionError):
            loss_fn(img, txt)

    def test_loss_is_differentiable(self):
        """Gradients should flow through the loss."""
        loss_fn = InfoNCELoss()
        img = torch.nn.functional.normalize(torch.randn(16, 256, requires_grad=True), dim=-1)
        txt = torch.nn.functional.normalize(torch.randn(16, 256, requires_grad=True), dim=-1)
        loss = loss_fn(img, txt)
        loss.backward()
        assert img.grad is not None
        assert txt.grad is not None


class TestVisionEncoder:

    def test_output_shape(self):
        from src.encoders.vision_encoder import VisionEncoder
        enc   = VisionEncoder()
        feats = torch.randn(8, 2048)
        out   = enc(feats)
        assert out.shape == (8, 256), f"Expected (8, 256), got {out.shape}"

    def test_output_is_normalized(self):
        from src.encoders.vision_encoder import VisionEncoder
        enc   = VisionEncoder()
        feats = torch.randn(8, 2048)
        out   = enc(feats)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(8), atol=1e-5), "Outputs not L2-normalized"


class TestTextEncoder:

    def test_output_shape(self):
        from src.encoders.text_encoder import TextEncoder
        enc      = TextEncoder()
        captions = ["a dog running", "a cat sleeping", "a bird flying"]
        device   = torch.device("cpu")
        out      = enc(captions, device)
        assert out.shape == (3, 256), f"Expected (3, 256), got {out.shape}"

    def test_output_is_normalized(self):
        from src.encoders.text_encoder import TextEncoder
        enc      = TextEncoder()
        captions = ["hello world"] * 4
        device   = torch.device("cpu")
        out      = enc(captions, device)
        norms    = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5), "Outputs not L2-normalized"
