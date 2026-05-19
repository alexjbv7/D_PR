"""Equity-specific feature generators for the daily horizon."""
from __future__ import annotations

from .factor_exposures import compute_ff5_exposures, load_ff5_factors
from .earnings_calendar import compute_earnings_features
from .sector_neutralizer import demean_by_sector

__all__ = [
    "compute_ff5_exposures",
    "load_ff5_factors",
    "compute_earnings_features",
    "demean_by_sector",
]
