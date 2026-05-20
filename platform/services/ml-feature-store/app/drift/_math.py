"""
Inline PSI & ECE formulas for the platform service.

These are lean re-implementations of the canonical versions in
``research/models/drift/``.  They exist here so the ml-feature-store
service remains self-contained (research is not in requirements.txt).

Do NOT add extra logic here — keep in sync with the research module.
"""
from __future__ import annotations

import numpy as np

_EPS: float = 1e-6


def compute_psi(
    train_values: list[float] | np.ndarray,
    recent_values: list[float] | np.ndarray,
    n_buckets: int = 10,
    bucket_edges: list[float] | np.ndarray | None = None,
) -> tuple[float, list[float]]:
    """PSI = Σ (p_recent - p_train) × ln(p_recent / p_train)."""
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

    p_t = (p_train.astype(float)  + _EPS) / (float(p_train.sum())  + _EPS * n_buckets)
    p_r = (p_recent.astype(float) + _EPS) / (float(p_recent.sum()) + _EPS * n_buckets)

    contributions = (p_r - p_t) * np.log(p_r / p_t)
    return float(contributions.sum()), contributions.tolist()


def severity_label(psi: float) -> str:
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "moderate"
    return "severe"


def compute_ece(
    probas: list[list[float]] | np.ndarray,
    labels: list[int] | np.ndarray,
    n_bins: int = 10,
) -> float:
    """ECE = Σ_m (|B_m|/N) × |acc(B_m) − conf(B_m)|."""
    p = np.asarray(probas, dtype=float)
    y = np.asarray(labels, dtype=int)
    n = len(y)
    if n == 0:
        return 0.0

    max_proba = p.max(axis=1)
    predicted = p.argmax(axis=1)
    correct   = (predicted == y).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece   = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (max_proba >= lo) & (max_proba <= hi if hi == 1.0 else max_proba < hi)
        if not mask.any():
            continue
        ece += (int(mask.sum()) / n) * abs(float(correct[mask].mean()) - float(max_proba[mask].mean()))
    return float(ece)


def compute_brier(
    probas: list[list[float]] | np.ndarray,
    labels: list[int] | np.ndarray,
) -> float:
    """Multi-class Brier score."""
    p = np.asarray(probas, dtype=float)
    y = np.asarray(labels, dtype=int)
    n = len(y)
    if n == 0:
        return 0.0
    one_hot = np.zeros_like(p)
    one_hot[np.arange(n), y] = 1.0
    return float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))
