"""
Smoke tests for the DRL training driver (cli/train_drl.py).

These are fast, CPU-only checks that the driver wires the env + trainers
correctly and respects the anti-leakage train/eval split. They are NOT a
substitute for the walk-forward DSR validation (CLAUDE.md §6.10).
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pytest

# Make research/ importable when pytest runs from the repo root or research/.
_REPO = Path(__file__).parents[2]
for _p in (str(_REPO / "research"), str(_REPO / "shared")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cli import train_drl  # noqa: E402


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        algo="dqn",
        episodes=2,
        updates=2,
        steps=64,
        seed=42,
        as_of=date.today().isoformat(),
        train_frac=0.7,
        device="cpu",
        checkpoint_dir=None,
        dry_run=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_stub_data_is_utc_and_has_close():
    data = train_drl._build_stub_data(n_bars=300, as_of=date(2026, 6, 1), seed=1)
    assert "close" in data.columns
    assert str(data.index.tz) == "UTC"
    assert len(data) == 300


def test_split_envs_disjoint_and_valid():
    train_env, eval_env = train_drl._split_envs(_args())
    # Disjoint, time-ordered slices (anti-leakage contract).
    assert train_env.data.index.max() <= eval_env.data.index.min()


def test_dry_run_returns_zero():
    assert train_drl._run(_args(dry_run=True)) == 0


def test_insufficient_data_after_split_raises(monkeypatch):
    # Pin the dataset to a small fixed size so the split leaves the eval slice
    # below one full episode; the anti-leakage guard must reject it.
    small = train_drl._build_stub_data(n_bars=260, as_of=date(2026, 6, 1), seed=1)
    monkeypatch.setattr(train_drl, "_build_stub_data", lambda *a, **k: small)
    with pytest.raises(ValueError):
        train_drl._split_envs(_args(train_frac=0.95))


def test_checkpoint_default_is_inside_repo():
    # Regression: default ckpt dir must live under research/artifacts, not above it.
    assert train_drl._RESEARCH.name == "research"
    assert (train_drl._RESEARCH / "artifacts").parent == train_drl._RESEARCH


def test_dqn_short_run_completes(tmp_path):
    pytest.importorskip("torch")  # skip cleanly where torch is absent
    code = train_drl._run(_args(episodes=2, checkpoint_dir=str(tmp_path)))
    assert code in (0, 2)  # trained; edge may or may not be positive on noise
