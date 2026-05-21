"""
NaN / data-quality validator for feature matrices.
====================================================

Validates that a feature DataFrame produced by the feature pipeline does not
contain NaN contamination beyond acceptable warm-up bounds.  Every feature has
an inherent warm-up period (e.g., RSI-14 needs 14 bars, a 60-bar rolling
window needs 60 bars, etc.).  After the warm-up, NaNs signal a data problem.

Classes
-------
NanReport
    Immutable result of a validation run.
NanValidator
    Validates a feature DataFrame and returns a NanReport.

Example
-------
::

    validator = NanValidator(warmup_bars=200, alert_threshold=0.05)
    report = validator.validate(feature_matrix)
    if not report.passed:
        for col, pct in report.alerts.items():
            print(f"ALERT: {col} has {pct:.1%} NaN after warmup")

References
----------
* Architecture doc §1.1 (Capa A, Riesgo 3: latencia de datos)
* CLAUDE.md §7.4 (Feature Store validation)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NanReport
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NanReport:
    """
    Immutable result of a NanValidator run.

    Attributes
    ----------
    total_bars : int
        Total rows in the validated DataFrame.
    total_features : int
        Number of columns checked.
    warmup_bars : int
        Number of warm-up bars excluded from the NaN threshold check.
    alert_threshold : float
        Fraction above which a column is flagged as an alert (default 0.05 = 5 %).
    nan_counts : dict[str, int]
        Absolute NaN count per column (over the entire frame).
    nan_pct : dict[str, float]
        NaN fraction per column **after** the warm-up window.
    alerts : dict[str, float]
        Columns where ``nan_pct > alert_threshold``.
    passed : bool
        ``True`` if no column triggered an alert.
    empty_columns : list[str]
        Columns with 100 % NaN — almost certainly a pipeline bug.
    """
    total_bars:      int
    total_features:  int
    warmup_bars:     int
    alert_threshold: float
    nan_counts:      dict[str, int]  = field(default_factory=dict)
    nan_pct:         dict[str, float] = field(default_factory=dict)
    alerts:          dict[str, float] = field(default_factory=dict)
    empty_columns:   list[str]        = field(default_factory=list)
    passed:          bool             = True

    def summary(self) -> str:
        lines = [
            f"NanReport: {self.total_bars} bars × {self.total_features} features",
            f"  warmup_bars={self.warmup_bars}  alert_threshold={self.alert_threshold:.1%}",
            f"  status: {'✅ PASSED' if self.passed else '❌ ALERTS'}",
        ]
        if self.alerts:
            lines.append("  Alerts (post-warmup NaN %):")
            for col, pct in sorted(self.alerts.items(), key=lambda x: -x[1]):
                lines.append(f"    {col}: {pct:.1%}")
        if self.empty_columns:
            lines.append(f"  Empty columns: {self.empty_columns}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# NanValidator
# ---------------------------------------------------------------------------

class NanValidator:
    """
    Validate NaN levels in a feature matrix.

    Parameters
    ----------
    warmup_bars : int
        Number of leading bars to skip when computing post-warmup NaN %.
        Set to the longest rolling window used in feature engineering.
        Default 200 (covers `vol_60`, `frac_diff`, `vol_regime`).
    alert_threshold : float
        Maximum allowed NaN fraction after warm-up (default 0.05 = 5 %).
    raise_on_alert : bool
        If ``True``, raise ``ValueError`` when alerts are found.  Default
        ``False`` (only log warnings).
    """

    def __init__(
        self,
        warmup_bars: int   = 200,
        alert_threshold: float = 0.05,
        raise_on_alert: bool   = False,
    ):
        self.warmup_bars     = warmup_bars
        self.alert_threshold = alert_threshold
        self.raise_on_alert  = raise_on_alert

    def validate(self, df: pd.DataFrame) -> NanReport:
        """
        Validate *df* and return a :class:`NanReport`.

        Parameters
        ----------
        df : pd.DataFrame
            Feature matrix with numeric columns.

        Returns
        -------
        NanReport
        """
        if df.empty:
            return NanReport(
                total_bars=0, total_features=0,
                warmup_bars=self.warmup_bars,
                alert_threshold=self.alert_threshold,
                passed=True,
            )

        n_bars = len(df)
        n_feat = len(df.columns)

        # Full NaN counts (absolute, entire frame)
        nan_counts: dict[str, int]   = {}
        nan_pct:    dict[str, float] = {}
        alerts:     dict[str, float] = {}
        empty:      list[str]        = []

        # Slice after warm-up for the threshold check
        post_warmup = df.iloc[self.warmup_bars:] if n_bars > self.warmup_bars else df

        for col in df.columns:
            total_nan = int(df[col].isna().sum())
            nan_counts[col] = total_nan

            if len(post_warmup) == 0:
                pct = float(total_nan) / n_bars if n_bars > 0 else 0.0
            else:
                pct = float(post_warmup[col].isna().sum()) / len(post_warmup)

            nan_pct[col] = round(pct, 4)

            if pct >= 1.0:
                empty.append(col)
                alerts[col] = pct
            elif pct > self.alert_threshold:
                alerts[col] = pct

        passed = len(alerts) == 0

        # Logging
        if not passed:
            for col, pct in alerts.items():
                logger.warning(
                    "nan_validator.alert col=%s post_warmup_nan=%.1f%%", col, pct * 100
                )
        else:
            logger.debug(
                "nan_validator.passed bars=%d features=%d warmup=%d",
                n_bars, n_feat, self.warmup_bars,
            )

        if self.raise_on_alert and not passed:
            raise ValueError(
                f"Feature matrix has {len(alerts)} columns with NaN > "
                f"{self.alert_threshold:.1%} after warmup.  "
                f"Columns: {list(alerts)}"
            )

        report = NanReport(
            total_bars      = n_bars,
            total_features  = n_feat,
            warmup_bars     = self.warmup_bars,
            alert_threshold = self.alert_threshold,
            nan_counts      = nan_counts,
            nan_pct         = nan_pct,
            alerts          = alerts,
            empty_columns   = empty,
            passed          = passed,
        )

        logger.info(report.summary())
        return report

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return *df* with leading warm-up rows dropped and remaining NaNs
        forward-filled (last-observation-carried-forward).

        This is the standard preparation for WalkForwardRunner input.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        df = df.iloc[self.warmup_bars:].copy()
        df = df.ffill().bfill()
        return df
