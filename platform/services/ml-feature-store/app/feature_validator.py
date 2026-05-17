"""
FeatureValidator — Validación de feature vectors antes de ML inference.

Detecta:
  - NaN / None en features críticas
  - Valores fuera de rango (range checks por feature)
  - Feature drift (PSI comparando distribución reciente vs histórica)
  - Staleness (features no actualizadas en > TTL)

Si la validación falla, emite un warning estructurado pero NO bloquea
la inferencia (degraded mode) — el modelo puede operar con features
parciales si el caller lo indica.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Range constraints per feature
# ---------------------------------------------------------------------------

FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "rsi_14":          (0.0,   100.0),
    "rsi_7":           (0.0,   100.0),
    "bb_pct":          (-0.5,  1.5),
    "ob_imbalance":    (-1.0,  1.0),
    "funding_z":       (-10.0, 10.0),
    "volume_ratio":    (0.0,   50.0),
    "spread_bps":      (0.0,   500.0),
    "recession_prob":  (0.0,   1.0),
    "yield_inv":       (0.0,   1.0),
    "whale_sentiment": (-1.0,  1.0),
    "p_win_ml":        (0.0,   1.0),
    "p_win_bayesian":  (0.0,   1.0),
    "regime_id":       (0.0,   10.0),
}

# Features that MUST be present for a signal to be emitted
CRITICAL_FEATURES = {"rsi_14", "ob_imbalance", "p_win_ml", "regime_id"}


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    valid:      bool = True
    warnings:   list[str] = field(default_factory=list)
    errors:     list[str] = field(default_factory=list)
    nan_count:  int = 0
    oor_count:  int = 0   # out-of-range

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.valid = False

    def to_dict(self) -> dict:
        return {
            "valid":    self.valid,
            "warnings": self.warnings,
            "errors":   self.errors,
            "nan_count": self.nan_count,
            "oor_count": self.oor_count,
        }


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class FeatureValidator:
    """
    Validates feature dicts and vectors before model inference.

    Usage
    -----
    validator = FeatureValidator()
    result = validator.validate(features, updated_at=datetime.now(utc))
    if not result.valid:
        # skip inference or use degraded mode
    """

    def __init__(
        self,
        critical_features: set[str] | None = None,
        stale_threshold_s: int = 120,
        psi_threshold:     float = 0.25,
    ):
        self._critical   = critical_features or CRITICAL_FEATURES
        self._stale_th   = stale_threshold_s
        self._psi_th     = psi_threshold
        self._histograms: dict[str, np.ndarray] = {}  # for PSI

    def validate(
        self,
        features:   dict[str, Any],
        updated_at: Optional[datetime] = None,
        symbol:     str = "UNKNOWN",
    ) -> ValidationResult:
        result = ValidationResult()

        # 1. Staleness
        if updated_at:
            age_s = (datetime.now(timezone.utc) - updated_at).total_seconds()
            if age_s > self._stale_th:
                result.add_warning(f"Features stale: {age_s:.0f}s old (threshold={self._stale_th}s)")

        # 2. Critical features present
        for feat in self._critical:
            if feat not in features or features[feat] is None:
                result.add_error(f"Critical feature missing or None: {feat}")
                result.nan_count += 1

        # 3. NaN / None sweep
        for name, val in features.items():
            if val is None or (isinstance(val, float) and np.isnan(val)):
                result.add_warning(f"NaN in feature: {name}")
                result.nan_count += 1

        # 4. Range checks
        for name, val in features.items():
            if val is None:
                continue
            if name in FEATURE_RANGES:
                lo, hi = FEATURE_RANGES[name]
                if not (lo <= float(val) <= hi):
                    result.add_warning(
                        f"Out-of-range: {name}={val:.4f} (expected [{lo}, {hi}])"
                    )
                    result.oor_count += 1

        # 5. Probability sum invariant for regime probs
        regime_keys = [k for k in features if k.startswith("regime_prob_")]
        if regime_keys:
            total = sum(float(features[k] or 0) for k in regime_keys)
            if not (0.95 <= total <= 1.05):
                result.add_warning(f"regime_probs sum={total:.3f} (expected ~1.0)")

        if result.warnings or result.errors:
            logger.debug("feature_validator.result",
                         symbol=symbol,
                         valid=result.valid,
                         warnings=len(result.warnings),
                         errors=len(result.errors))

        return result

    def validate_vector(
        self,
        vector:        list[float | None],
        feature_names: list[str],
        **kwargs,
    ) -> ValidationResult:
        features = dict(zip(feature_names, vector))
        return self.validate(features, **kwargs)

    def update_histogram(self, feature_name: str, values: np.ndarray) -> None:
        """Update reference histogram for PSI drift detection."""
        counts, _ = np.histogram(values, bins=10, range=FEATURE_RANGES.get(feature_name, (None, None)))
        self._histograms[feature_name] = counts.astype(float) / counts.sum()

    def compute_psi(self, feature_name: str, current_values: np.ndarray) -> float:
        """
        Population Stability Index. PSI > 0.25 → significant drift.
        """
        if feature_name not in self._histograms:
            return 0.0
        ref = self._histograms[feature_name]
        cur_counts, _ = np.histogram(
            current_values, bins=len(ref),
            range=FEATURE_RANGES.get(feature_name, (None, None))
        )
        cur = cur_counts.astype(float) / max(cur_counts.sum(), 1)
        eps = 1e-8
        psi = np.sum((cur - ref) * np.log((cur + eps) / (ref + eps)))
        return float(psi)

    def check_drift(
        self,
        features_batch: dict[str, list[float]],
    ) -> dict[str, float]:
        """Compute PSI for all features and flag > threshold."""
        results = {}
        for feat, values in features_batch.items():
            arr = np.array([v for v in values if v is not None], dtype=float)
            if len(arr) < 10:
                continue
            psi = self.compute_psi(feat, arr)
            results[feat] = psi
            if psi > self._psi_th:
                logger.warning("feature_drift.detected",
                               feature=feat, psi=round(psi, 4))
        return results
