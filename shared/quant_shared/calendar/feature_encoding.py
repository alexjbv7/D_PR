"""Numeric encoding of :class:`SessionPhase` for ML / feature-engine pipelines."""
from __future__ import annotations

from datetime import datetime

from .session_phase import SessionPhase, get_session_phase

# Ordinal map — stable for models; document changes in feature_set semver.
_PHASE_VALUE: dict[SessionPhase, float] = {
    SessionPhase.CRYPTO_24_7: 0.0,
    SessionPhase.CLOSED_EQUITY: 1.0,
    SessionPhase.PRE_MARKET: 2.0,
    SessionPhase.RTH: 3.0,
    SessionPhase.POST_MARKET: 4.0,
}


def session_phase_value(symbol: str, ts_utc: datetime) -> float:
    """
    Scalar feature for ``session_phase`` at ``ts_utc``.

    Intended for feature-engine / offline jobs — not part of the canonical
    19-feature vector in ``quant_shared.features.definitions``.
    """
    return _PHASE_VALUE[get_session_phase(symbol, ts_utc)]
