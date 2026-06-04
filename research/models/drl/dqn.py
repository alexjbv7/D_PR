"""
Deep Q-Network for discrete trading actions (ADR-038).
"""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Sequence
from typing import NamedTuple

import torch
import torch.nn as nn

from models.drl.backbone import TradingResMLP, init_weights


class Transition(NamedTuple):
    """Single experience tuple for replay buffer."""

    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool


class ReplayBuffer:
    """
    Fixed-size experience replay buffer (deque-backed).

    Parameters
    ----------
    capacity : int
        Maximum number of transitions stored.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._buffer: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buffer)

    def push(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        self._buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Sequence[Transition]:
        """
        Sample a random minibatch of transitions.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.

        Returns
        -------
        Sequence[Transition]
            List of sampled transitions.

        Raises
        ------
        ValueError
            If batch_size exceeds buffer length.
        """
        if batch_size > len(self._buffer):
            raise ValueError(
                f"batch_size {batch_size} exceeds buffer size {len(self._buffer)}"
            )
        return random.sample(self._buffer, batch_size)


class TradingDQN(nn.Module):
    """
    Deep Q-Network for trading with discrete actions {SELL, HOLD, BUY}.

    Parameters
    ----------
    obs_dim : int
        Observation dimension (default 42).
    hidden_dim : int
        Backbone hidden size.
    n_blocks : int
        Number of residual blocks in backbone.
    n_actions : int
        Number of discrete actions (default 3).

    Examples
    --------
    >>> net = TradingDQN()
    >>> q = net(torch.randn(4, 42))
    >>> q.shape
    torch.Size([4, 3])
    """

    def __init__(
        self,
        obs_dim: int = 42,
        hidden_dim: int = 256,
        n_blocks: int = 3,
        n_actions: int = 3,
    ) -> None:
        super().__init__()
        self.backbone = TradingResMLP(obs_dim, hidden_dim, n_blocks)
        self.q_head = nn.Linear(hidden_dim, n_actions)
        self.apply(init_weights)
        nn.init.orthogonal_(self.q_head.weight, gain=1.0)
        nn.init.zeros_(self.q_head.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for all actions.

        Parameters
        ----------
        state : torch.Tensor
            Batch of observations, shape ``(batch, obs_dim)``.

        Returns
        -------
        torch.Tensor
            Q-values, shape ``(batch, n_actions)``.
        """
        return self.q_head(self.backbone(state))

    def select_action(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        """
        ε-greedy action selection.

        Parameters
        ----------
        state : torch.Tensor
            Single observation ``(obs_dim,)`` or batch ``(1, obs_dim)``.
        epsilon : float
            Exploration probability in ``[0, 1]``.

        Returns
        -------
        int
            Selected action index.
        """
        if torch.rand(1).item() < epsilon:
            return int(torch.randint(3, (1,)).item())

        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            q_values = self.forward(state)
            return int(q_values.argmax(dim=-1).item())
