"""
Tests for research/models/drift/ece.py
=======================================

4 cases:
  1. Near-perfect calibration → ECE close to 0
  2. Total miscalibration     → ECE close to 1
  3. n_bins parameter         → ECE always in [0, 1]
  4. compute_brier            → exact values for perfect / worst-case
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_REPO = Path(__file__).parents[3]
for _p in [str(_REPO / "research"), str(_REPO / "shared")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.drift.ece import compute_ece, compute_brier


# ===========================================================================
# compute_ece
# ===========================================================================

class TestComputeEce:
    def test_near_perfect_calibration(self) -> None:
        """
        Model predicts class 0 with 0.90 confidence; 90 % of true labels are 0.
        All 1000 samples fall in a single bin with conf ≈ 0.90 and acc ≈ 0.90
        → ECE ≈ 0.
        """
        n   = 1000
        rng = np.random.default_rng(0)
        probas = np.full((n, 3), [0.90, 0.05, 0.05])
        labels = rng.choice([0, 1], size=n, p=[0.90, 0.10])
        ece = compute_ece(probas, labels)
        assert ece < 0.05, f"Near-perfect calibration → ECE < 0.05, got {ece:.4f}"

    def test_total_miscalibration(self) -> None:
        """
        Model predicts class 0 with 100 % confidence; every true label is 1.
        All samples are in the [1.0, 1.0] bin with acc = 0, conf = 1 → ECE = 1.
        """
        n      = 200
        probas = np.zeros((n, 3))
        probas[:, 0] = 1.0          # always predicts class 0, certainty = 1
        labels = np.ones(n, dtype=int)  # true = class 1 always
        ece = compute_ece(probas, labels)
        assert ece > 0.90, f"Total miscalibration → ECE > 0.90, got {ece:.4f}"

    def test_n_bins_returns_bounded_value(self) -> None:
        """ECE ∈ [0, 1] for any n_bins in {5, 10, 20}."""
        rng    = np.random.default_rng(3)
        n      = 300
        probas = rng.dirichlet([5.0, 1.0, 1.0], size=n)
        labels = rng.integers(0, 3, size=n)
        for bins in [5, 10, 20]:
            ece = compute_ece(probas, labels, n_bins=bins)
            assert 0.0 <= ece <= 1.0, f"ECE={ece:.4f} outside [0,1] with n_bins={bins}"

    def test_empty_probas_returns_zero(self) -> None:
        """Empty input → ECE = 0 (no samples, no error)."""
        probas = np.empty((0, 3))
        labels = np.empty(0, dtype=int)
        assert compute_ece(probas, labels) == 0.0


# ===========================================================================
# compute_brier
# ===========================================================================

class TestComputeBrier:
    def test_perfect_prediction_zero_brier(self) -> None:
        """Predicting the true class with probability 1 → Brier = 0."""
        n         = 100
        n_classes = 3
        probas    = np.zeros((n, n_classes))
        labels    = np.zeros(n, dtype=int)
        probas[:, 0] = 1.0          # always predicts class 0, which is always correct
        brier = compute_brier(probas, labels)
        assert brier == pytest.approx(0.0, abs=1e-10)

    def test_worst_case_brier(self) -> None:
        """
        Predicting the wrong class with probability 1 → Brier = 2.
        One-hot truth = [1,0,0], prediction = [0,1,0].
        Σ_c (p_c − y_c)² = (0-1)² + (1-0)² + (0-0)² = 2.
        """
        n         = 100
        n_classes = 3
        probas    = np.zeros((n, n_classes))
        probas[:, 1] = 1.0          # predicts class 1 with certainty
        labels = np.zeros(n, dtype=int)  # true = class 0
        brier = compute_brier(probas, labels)
        assert brier == pytest.approx(2.0, abs=1e-10)

    def test_empty_input_returns_zero(self) -> None:
        probas = np.empty((0, 3))
        labels = np.empty(0, dtype=int)
        assert compute_brier(probas, labels) == 0.0
