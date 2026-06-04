"""Tests for TradingResMLP backbone (ADR-038)."""

from __future__ import annotations

import torch

from models.drl.backbone import TradingResMLP


class TestTradingResMLP:
    def test_output_shape(self) -> None:
        model = TradingResMLP(obs_dim=42, hidden_dim=256, n_blocks=3)
        x = torch.randn(4, 42)
        out = model(x)
        assert out.shape == (4, 256)

    def test_no_nan_forward(self) -> None:
        model = TradingResMLP()
        x = torch.randn(8, 42)
        out = model(x)
        assert not torch.isnan(out).any().item()

    def test_gradients_flow(self) -> None:
        model = TradingResMLP()
        x = torch.randn(4, 42)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any().item()

    def test_skip_connections(self) -> None:
        model = TradingResMLP()
        x = torch.zeros(2, 42)
        out = model(x)
        assert not torch.allclose(out, torch.zeros_like(out))
