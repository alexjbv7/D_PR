"""Tests for TradingDQN and ReplayBuffer (ADR-038)."""

from __future__ import annotations

import collections

import torch

from models.drl.dqn import ReplayBuffer, TradingDQN, Transition


class TestTradingDQN:
    def test_q_values_shape(self) -> None:
        net = TradingDQN(obs_dim=42, hidden_dim=256, n_blocks=3)
        x = torch.randn(4, 42)
        q = net(x)
        assert q.shape == (4, 3)

    def test_select_action_greedy(self) -> None:
        torch.manual_seed(0)
        net = TradingDQN()
        state = torch.randn(42)
        action_a = net.select_action(state, epsilon=0.0)
        action_b = net.select_action(state, epsilon=0.0)
        assert action_a == action_b
        assert 0 <= action_a <= 2

    def test_select_action_random(self) -> None:
        net = TradingDQN()
        state = torch.randn(42)
        counts = collections.Counter(
            net.select_action(state, epsilon=1.0) for _ in range(1000)
        )
        assert len(counts) == 3
        for c in counts.values():
            assert 200 < c < 500  # uniform-ish

    def test_no_nan_q_values(self) -> None:
        net = TradingDQN()
        x = torch.randn(16, 42)
        q = net(x)
        assert not torch.isnan(q).any().item()


class TestReplayBuffer:
    def test_replay_buffer_sample(self) -> None:
        buf = ReplayBuffer(capacity=200)
        for i in range(100):
            s = torch.randn(42)
            ns = torch.randn(42)
            buf.push(s, i % 3, float(i), ns, i % 10 == 0)

        batch = buf.sample(32)
        assert len(batch) == 32
        assert all(isinstance(t, Transition) for t in batch)
        assert all(t.state.shape == (42,) for t in batch)
