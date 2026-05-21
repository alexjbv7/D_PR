"""
KS test — two-sample Kolmogorov-Smirnov test for distribution drift.

Uses ``scipy.stats.ks_2samp`` (available via ``scipy>=1.11`` in the
research environment, see ``pyproject.toml``).

Interpretation
--------------
statistic  : float in [0, 1] — maximum absolute CDF difference
p_value    : float — probability of observing this or more extreme
             difference under the null hypothesis (same distribution)

Thresholds
----------
p_value > 0.05  → stable    (no evidence of drift)
p_value ≤ 0.05  → moderate  (statistically significant drift)
p_value ≤ 0.01  → severe    (strong evidence of drift)

Relation to PSI
---------------
KS test complements PSI: KS is sensitive to differences in the *shape*
of the distribution (mean shift, spread change), whereas PSI quantifies
the *magnitude* of bucket-wise divergence.  Use both for a complete
picture.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp  # type: ignore[import-untyped]

__all__ = ["compute_ks", "ks_severity"]


def compute_ks(
    train_values: np.ndarray,
    recent_values: np.ndarray,
) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test.

    Parameters
    ----------
    train_values  : array-like, shape (N,)
    recent_values : array-like, shape (M,)

    Returns
    -------
    statistic : float  (0 = identical CDFs, 1 = maximally different)
    p_value   : float  (< 0.05 suggests significant drift)
    """
    train_arr  = np.asarray(train_values,  dtype=float)
    recent_arr = np.asarray(recent_values, dtype=float)
    result = ks_2samp(train_arr, recent_arr)
    return float(result.statistic), float(result.pvalue)


def ks_severity(statistic: float, p_value: float) -> str:
    """Map KS test result to a severity tier.

    Returns
    -------
    "stable"   p_value > 0.05
    "moderate" 0.01 < p_value ≤ 0.05
    "severe"   p_value ≤ 0.01
    """
    if p_value > 0.05:
        return "stable"
    if p_value > 0.01:
        return "moderate"
    return "severe"
