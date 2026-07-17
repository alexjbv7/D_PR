"""
Y-001 regression: PPO/SAC must not auto-promote without a DSR gate.

Structural test (no training): the CLI source must not set ``edge = True``
for ppo/sac, and must return exit code 2 on those paths after train.
"""
from __future__ import annotations

from pathlib import Path

_CLI = Path(__file__).resolve().parents[1] / "cli" / "train_drl.py"


def test_y001_ppo_sac_source_never_auto_promotes() -> None:
    src = _CLI.read_text(encoding="utf-8")
    assert "edge = True" not in src, (
        "Y-001 regression: train_drl must not set edge=True for ungated algos"
    )
    # Both algos document NO PROMOTE / exit 2
    assert "algo=ppo" in src and "Exit 2" in src
    assert "algo=sac" in src
    assert "no DSR walk-forward gate yet" in src
