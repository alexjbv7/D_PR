"""
Tests for research/models/drift/psi.py
=======================================

6 cases:
  1. Identical distributions  → PSI ≈ 0
  2. Highly shifted distribution → PSI in "severe" tier
  3. Pre-computed bucket_edges produce same result (anti-leakage path)
  4. severity_label boundary values
  5. compute_psi_categorical — identical and shifted regimes
  6. Epsilon protection — no NaN / inf when a bucket is empty in either dist
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

from models.drift.psi import compute_psi, severity_label, compute_psi_categorical


# ===========================================================================
# compute_psi
# ===========================================================================

class TestComputePsi:
    def test_identical_distributions_near_zero(self) -> None:
        """PSI of a distribution against itself must be ≈ 0."""
        rng = np.random.default_rng(0)
        x = rng.normal(0.0, 1.0, 2000)
        psi, contributions = compute_psi(x, x.copy())
        assert psi < 0.01, f"Expected PSI ≈ 0 for identical distributions, got {psi:.6f}"
        assert len(contributions) == 10  # default n_buckets

    def test_shifted_distribution_severe(self) -> None:
        """Mean shift of 3σ must produce severe PSI (≥ 0.25)."""
        rng = np.random.default_rng(1)
        train  = rng.normal(0.0, 1.0, 2000)
        recent = rng.normal(3.0, 1.0, 2000)
        psi, contributions = compute_psi(train, recent)
        assert psi >= 0.25, f"Expected severe drift, got PSI={psi:.4f}"
        assert severity_label(psi) == "severe"
        assert len(contributions) == 10

    def test_precomputed_edges_match_internal(self) -> None:
        """PSI with externally supplied edges ≡ PSI with internally derived edges."""
        rng = np.random.default_rng(2)
        train  = rng.normal(0.0, 1.0, 2000)
        recent = rng.normal(0.5, 1.0, 500)

        # Internal path: edges derived from train inside compute_psi
        psi_auto, _ = compute_psi(train, recent, n_buckets=10)

        # Explicit path (mirrors what BucketEdgesRepository saves/loads)
        edges = np.percentile(train, np.linspace(0.0, 100.0, 11))
        edges[0]  = -np.inf
        edges[-1] =  np.inf
        psi_explicit, _ = compute_psi(train, recent, bucket_edges=edges)

        assert abs(psi_auto - psi_explicit) < 1e-10, (
            f"Internal vs explicit edges diverge: {psi_auto:.8f} vs {psi_explicit:.8f}"
        )

    def test_contributions_length_matches_n_buckets(self) -> None:
        """contributions list length == n_buckets."""
        rng = np.random.default_rng(3)
        x = rng.normal(0, 1, 500)
        for k in [5, 10, 20]:
            _, contribs = compute_psi(x, x, n_buckets=k)
            assert len(contribs) == k


# ===========================================================================
# severity_label
# ===========================================================================

class TestSeverityLabel:
    def test_boundaries(self) -> None:
        assert severity_label(0.00) == "stable"
        assert severity_label(0.09) == "stable"
        assert severity_label(0.10) == "moderate"
        assert severity_label(0.24) == "moderate"
        assert severity_label(0.25) == "severe"
        assert severity_label(1.00) == "severe"


# ===========================================================================
# compute_psi_categorical
# ===========================================================================

class TestComputePsiCategorical:
    def test_identical_categorical_near_zero(self) -> None:
        labels = ["bull", "bear", "range", "bull", "bear"] * 100
        psi = compute_psi_categorical(labels, labels)
        assert psi < 0.01, f"Expected PSI ≈ 0, got {psi:.6f}"

    def test_shifted_categorical_positive(self) -> None:
        """Regime flipping from mostly bull to mostly bear → PSI > 0."""
        train  = ["bull"] * 80 + ["bear"] * 20
        recent = ["bear"] * 80 + ["bull"] * 20
        psi = compute_psi_categorical(train, recent)
        assert psi > 0.10, f"Expected moderate/severe drift, got {psi:.4f}"


# ===========================================================================
# Epsilon protection (no NaN / inf)
# ===========================================================================

class TestEpsilonProtection:
    def test_empty_bucket_no_nan_inf(self) -> None:
        """
        When recent data covers only part of the training distribution,
        empty buckets must not produce NaN, inf, or ZeroDivisionError.
        """
        train  = np.concatenate([
            np.full(500,  1.0),   # cluster 1
            np.full(500, 10.0),   # cluster 2
        ])
        recent = np.full(1000, 1.0)  # only cluster 1 present

        psi, contribs = compute_psi(train, recent, n_buckets=10)
        assert np.isfinite(psi), f"PSI not finite: {psi}"
        assert all(np.isfinite(c) for c in contribs), "Some contribution is NaN / inf"
