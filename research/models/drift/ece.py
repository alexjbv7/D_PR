"""
ECE — Expected Calibration Error.

Formula
-------
ECE = Σ_m (|B_m| / N) × |acc(B_m) − conf(B_m)|

where:
  B_m    = samples whose max-class confidence falls in bin m
  acc    = fraction correctly classified in B_m
  conf   = mean max-class confidence in B_m
  N      = total sample count

Threshold: ECE < 0.05 = well-calibrated (CLAUDE.md §1.2).

Also provides multi-class Brier score as secondary metric.

References
----------
Naeini et al. (2015) "Obtaining Well Calibrated Probabilities Using
Bayesian Binning into Quantiles"
"""
from __future__ import annotations

import numpy as np

__all__ = ["compute_ece", "compute_brier"]


def compute_ece(
    probas: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error.

    Parameters
    ----------
    probas : np.ndarray, shape (N, C)
        Predicted class probabilities.  Each row should sum to ≈ 1.
    labels : np.ndarray, shape (N,)
        True integer class labels in [0, C-1].
    n_bins : int
        Number of equal-width confidence bins over [0, 1].

    Returns
    -------
    ece : float  (0 = perfect calibration, 1 = maximum miscalibration)
    """
    probas = np.asarray(probas, dtype=float)
    labels = np.asarray(labels, dtype=int)

    n = len(labels)
    if n == 0:
        return 0.0

    # For multi-class calibration: confidence = max probability,
    # correctness = 1 iff argmax == true label.
    max_proba = probas.max(axis=1)
    predicted = probas.argmax(axis=1)
    correct   = (predicted == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        # Include the upper bound in the last bin [hi-eps, 1.0].
        if hi == 1.0:
            mask = (max_proba >= lo) & (max_proba <= hi)
        else:
            mask = (max_proba >= lo) & (max_proba < hi)
        if not mask.any():
            continue
        acc  = float(correct[mask].mean())
        conf = float(max_proba[mask].mean())
        ece += (int(mask.sum()) / n) * abs(acc - conf)

    return float(ece)


def compute_brier(
    probas: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Multi-class Brier score.

    BS = (1/N) × Σ_n Σ_c (p_{n,c} − y_{n,c})²

    where y_{n,c} is the one-hot indicator of the true class.

    Parameters
    ----------
    probas : np.ndarray, shape (N, C)
    labels : np.ndarray, shape (N,)

    Returns
    -------
    brier : float  (0 = perfect, 2 = worst-case for binary tasks)
    """
    probas = np.asarray(probas, dtype=float)
    labels = np.asarray(labels, dtype=int)

    n = len(labels)
    if n == 0:
        return 0.0

    one_hot = np.zeros_like(probas)
    one_hot[np.arange(n), labels] = 1.0

    return float(np.mean(np.sum((probas - one_hot) ** 2, axis=1)))
