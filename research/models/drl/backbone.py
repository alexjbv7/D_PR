"""
Shared ResMLP backbone for DRL agents (ADR-038).

TradingResMLP maps tabular market state to a fixed embedding used by DQN/PPO/SAC heads.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


def init_weights(module: nn.Module) -> None:
    """
    Orthogonal weight initialization for RL stability (Schulman et al.).

    Parameters
    ----------
    module : nn.Module
        Module tree to initialize (typically applied via ``module.apply``).
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ResBlock(nn.Module):
    """Residual block with SwiGLU and pre-activation LayerNorm."""

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        gate, linear = self.gate(h).chunk(2, dim=-1)
        h = self.proj(torch.sigmoid(gate) * linear)
        return x + self.drop(h)


class TradingResMLP(nn.Module):
    """
    Residual MLP backbone for trading observation vectors.

    Parameters
    ----------
    obs_dim : int
        Input dimension (default 42 per ADR-037).
    hidden_dim : int
        Hidden embedding size (default 256).
    n_blocks : int
        Number of residual SwiGLU blocks.
    dropout : float
        Dropout probability inside residual blocks.

    Examples
    --------
    >>> model = TradingResMLP()
    >>> x = torch.randn(4, 42)
    >>> model(x).shape
    torch.Size([4, 256])
    """

    def __init__(
        self,
        obs_dim: int = 42,
        hidden_dim: int = 256,
        n_blocks: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(obs_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.apply(init_weights)
        # Diverse input bias: zero obs still yields non-degenerate activations (LayerNorm needs variance).
        with torch.no_grad():
            self.input_proj.bias.copy_(torch.linspace(-0.01, 0.01, hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.output_norm(h)
