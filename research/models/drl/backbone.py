"""
Shared ResMLP backbone for DRL agents (ADR-038).

TradingResMLP maps tabular market state to a fixed embedding used by DQN/PPO/SAC heads.

ResBlock is imported from models.nn_layers (shared with zoo.ResMLPClassifier per ADR-034).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.nn_layers import ResBlock  # shared with ResMLPClassifier (ADR-034)


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
