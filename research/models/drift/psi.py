"""
PSI — Population Stability Index.

Formula
-------
PSI = Σ_i (p_recent_i − p_train_i) × ln(p_recent_i / p_train_i)

Severity thresholds
-------------------
PSI < 0.10  → stable    (monitor only)
PSI < 0.25  → moderate  (P2 alert)
PSI ≥ 0.25  → severe    (P1 alert + retrain trigger)

Anti-leakage invariant (ADR-030)
----------------------------------
``bucket_edges`` MUST be derived from training data and persisted via
``BucketEdgesRepository``.  Re-computing edges from recent data is a
tautology: each bucket contains the same fraction of both distributions
by construction, so PSI ≈ 0 regardless of real drift.

References
----------
Yurdakul (2018) "Statistical Properties of Population Stability Index"
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["compute_psi", "severity_label", "compute_psi_categorical"]

# Small constant to guard against log(0) and 0/0 in empty buckets.
_EPS: float = 1e-6


def compute_psi(
    train_values: np.ndarray,
    recent_values: np.ndarray,
    n_buckets: int = 10,
    bucket_edges: np.ndarray | None = None,
) -> tuple[float, list[float]]:
    """Compute Population Stability Index.

    Parameters
    ----------
    train_values : array-like, shape (N,)
        Values seen during model training.
    recent_values : array-like, shape (M,)
        Values from the monitoring window (e.g. last 7 days).
    n_buckets : int
        Number of quantile buckets.  Ignored when ``bucket_edges`` is given.
    bucket_edges : np.ndarray of shape (n_buckets+1,), optional
        Pre-computed edges from training data.  **Must be provided in
        production to satisfy the anti-leakage invariant** (ADR-030).
        When None, edges are derived from ``train_values`` internally
        (acceptable only in exploratory / research context).

    Returns
    -------
    psi : float
        Total PSI.  Use :func:`severity_label` to map to a tier.
    contributions : list[float]
        Per-bucket contribution to PSI (length == n_buckets).
    """
    train_arr  = np.asarray(train_values,  dtype=float)
    recent_arr = np.asarray(recent_values, dtype=float)

    if bucket_edges is None:
        edges = np.percentile(train_arr, np.linspace(0.0, 100.0, n_buckets + 1))
        edges[0]  = -np.inf
        edges[-1] =  np.inf
    else:
        edges = np.asarray(bucket_edges, dtype=float)
        n_buckets = len(edges) - 1

    p_train,  _ = np.histogram(train_arr,  bins=edges)
    p_recent, _ = np.histogram(recent_arr, bins=edges)

    # Normalise with epsilon guard — prevents log(0) / zero-division
    # when a bucket is empty in either distribution.
    denom_train  = float(p_train.sum())  + _EPS * n_buckets
    denom_recent = float(p_recent.sum()) + _EPS * n_buckets
    p_t = (p_train.astype(float)  + _EPS) / denom_train
    p_r = (p_recent.astype(float) + _EPS) / denom_recent

    contributions = (p_r - p_t) * np.log(p_r / p_t)
    return float(contributions.sum()), contributions.tolist()


def severity_label(psi: float) -> str:
    """Map a PSI value to a human-readable severity tier.

    Parameters
    ----------
    psi : float

    Returns
    -------
    "stable"   if psi < 0.10
    "moderate" if 0.10 <= psi < 0.25
    "severe"   if psi >= 0.25
    """
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "moderate"
    return "severe"


def compute_psi_categorical(
    train_values: Sequence[str],
    recent_values: Sequence[str],
) -> float:
    """PSI for categorical features (e.g. regime labels, day-of-week bins).

    Parameters
    ----------
    train_values  : sequence of category strings
    recent_values : sequence of category strings

    Returns
    -------
    psi : float — same thresholds as numeric PSI.
    """
    categories = sorted(set(train_values) | set(recent_values))
    n_cat    = len(categories)
    n_train  = len(train_values)
    n_recent = len(recent_values)

    psi = 0.0
    for cat in categories:
        p_t = (sum(v == cat for v in train_values)  + _EPS) / (n_train  + _EPS * n_cat)
        p_r = (sum(v == cat for v in recent_values) + _EPS) / (n_recent + _EPS * n_cat)
        psi += (p_r - p_t) * float(np.log(p_r / p_t))

    return float(psi)
