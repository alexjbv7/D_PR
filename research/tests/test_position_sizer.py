"""
Property tests for StackedPositionSizer (DIAGNOSTICO §4) — CPU, no magic values.

Verifican propiedades del stack (monotonías, caps, atenuación, presupuesto
CVaR, coherencia de signo), no números concretos.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import beta

_REPO = Path(__file__).parents[2]
for _p in [str(_REPO / "research"), str(_REPO / "shared")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from portfolio.sizing import (
    SizerConfig,
    StackedPositionSizer,
    _expected_shortfall_unit,
)
from quant_shared.contracts import PositionSizer, SizeDecision


@pytest.fixture
def sizer() -> StackedPositionSizer:
    return StackedPositionSizer()


def _w(decision: SizeDecision) -> float:
    return abs(decision.target_weight)


# ---------------------------------------------------------------------------
# Contrato
# ---------------------------------------------------------------------------


def test_implements_contract(sizer: StackedPositionSizer) -> None:
    assert isinstance(sizer, PositionSizer)
    decision = sizer.size(0.5, 0.20, "normal", 0.60)
    assert isinstance(decision, SizeDecision)
    assert decision.method
    assert decision.reason  # siempre explica qué término mandó


# ---------------------------------------------------------------------------
# Vol-targeting: inverso en vol
# ---------------------------------------------------------------------------


def test_inverse_in_vol(sizer: StackedPositionSizer) -> None:
    low_vol = sizer.size(1.0, 0.10, "normal", 0.60)
    high_vol = sizer.size(1.0, 0.40, "normal", 0.60)
    assert _w(high_vol) < _w(low_vol), "mayor vol_forecast debe reducir el tamaño"


# ---------------------------------------------------------------------------
# Monótono en edge (hasta el cap)
# ---------------------------------------------------------------------------


def test_monotone_in_edge(sizer: StackedPositionSizer) -> None:
    cfg = sizer.config
    vol = cfg.vol_target  # vol_scale = 1: aísla el término Kelly
    weights = [_w(sizer.size(1.0, vol, "normal", p)) for p in (0.52, 0.60, 0.70, 0.80)]
    assert all(b >= a for a, b in zip(weights, weights[1:])), (
        f"el tamaño debe ser no-decreciente en el edge: {weights}"
    )
    assert weights[-1] > weights[0], "más edge debe dar estrictamente más tamaño"


def test_no_edge_means_zero(sizer: StackedPositionSizer) -> None:
    decision = sizer.size(1.0, 0.20, "normal", 0.30)  # p < breakeven
    assert decision.target_weight == 0.0
    assert "kelly" in decision.reason


# ---------------------------------------------------------------------------
# Atenuación por régimen
# ---------------------------------------------------------------------------


def test_regime_attenuation(sizer: StackedPositionSizer) -> None:
    normal = sizer.size(1.0, 0.20, "normal", 0.65)
    high_vol = sizer.size(1.0, 0.20, "high_vol", 0.65)
    defensive = sizer.size(1.0, 0.20, "defensive", 0.65)

    assert _w(high_vol) < _w(normal), "régimen alta-vol debe atenuar el tamaño"
    assert defensive.target_weight == 0.0, "régimen defensivo → no operar"
    assert "regime" in defensive.reason


# ---------------------------------------------------------------------------
# Caps: posición y Kelly
# ---------------------------------------------------------------------------


def test_respects_position_cap(sizer: StackedPositionSizer) -> None:
    # Edge extremo + vol mínima → el stack pediría mucho más que el cap
    decision = sizer.size(1.0, 0.01, "normal", 0.99)
    assert _w(decision) <= sizer.config.max_position_cap + 1e-12


def test_respects_kelly_cap(sizer: StackedPositionSizer) -> None:
    # vol_scale = 1 y régimen normal → |raw| == término Kelly ≤ kelly_cap
    cfg = sizer.config
    decision = sizer.size(1.0, cfg.vol_target, "normal", 0.99)
    assert _w(decision) <= cfg.kelly_cap + 1e-12
    assert cfg.kelly_cap <= 0.25  # ADR-003: nunca por encima de quarter-Kelly


# ---------------------------------------------------------------------------
# Restricción CVaR-lite
# ---------------------------------------------------------------------------


def test_cvar_budget_is_respected() -> None:
    cfg = SizerConfig(cvar_budget=0.01)  # presupuesto chico
    sizer = StackedPositionSizer(cfg)
    vol = 0.50  # vol alta

    decision = sizer.size(1.0, vol, "normal", 0.90)
    es = _w(decision) * _expected_shortfall_unit(vol, cfg.cvar_confidence)
    assert es <= cfg.cvar_budget + 1e-12, (
        f"ES de la posición ({es:.4f}) excede el presupuesto ({cfg.cvar_budget})"
    )
    assert "cvar" in decision.reason


# ---------------------------------------------------------------------------
# Coherencia de signo: el sizer nunca invierte la dirección
# ---------------------------------------------------------------------------


def test_sign_follows_weight(sizer: StackedPositionSizer) -> None:
    long_d = sizer.size(0.7, 0.20, "normal", 0.65)
    short_d = sizer.size(-0.7, 0.20, "normal", 0.65)
    flat_d = sizer.size(0.0, 0.20, "normal", 0.65)

    assert long_d.target_weight > 0.0
    assert short_d.target_weight < 0.0
    assert short_d.target_weight == pytest.approx(-long_d.target_weight)
    assert flat_d.target_weight == 0.0


# ---------------------------------------------------------------------------
# Kelly bayesiano: incertidumbre alta encoge el tamaño
# ---------------------------------------------------------------------------


def test_uncertainty_shrinks_size(sizer: StackedPositionSizer) -> None:
    certain = sizer.size(1.0, 0.20, "normal", (0.65, 0.0))
    uncertain = sizer.size(1.0, 0.20, "normal", (0.65, 0.10))
    assert _w(uncertain) < _w(certain), (
        "a igual media, más incertidumbre del posterior debe dar menos tamaño"
    )


def test_accepts_scipy_beta_posterior(sizer: StackedPositionSizer) -> None:
    """El posterior puede ser una distribución (p.ej. Beta desde bayesian_sizer)."""
    sharp = beta(650, 350)   # media 0.65, std pequeña (mucha evidencia)
    diffuse = beta(6.5, 3.5)  # misma media, std grande (poca evidencia)
    assert _w(sizer.size(1.0, 0.20, "normal", sharp)) > _w(
        sizer.size(1.0, 0.20, "normal", diffuse)
    )


def test_unsupported_posterior_raises(sizer: StackedPositionSizer) -> None:
    with pytest.raises(TypeError, match="edge_posterior"):
        sizer.size(1.0, 0.20, "normal", object())


# ---------------------------------------------------------------------------
# Separación de capas: dimensiona el sizer, no el agente
# ---------------------------------------------------------------------------


def test_sizing_lives_in_sizer_not_in_agent(sizer: StackedPositionSizer) -> None:
    """La señal del agente trae kelly/size en 0; el peso sale del sizer."""
    from data.drl_dataset import _MARKET_FEATURES, _REGIME_FEATURES
    from models.zoo import XGBoostClassifier
    from quant_shared.contracts import MarketContext
    from alpha.agents.xgb_agent import XgbAlphaAgent

    names = [*_MARKET_FEATURES, *_REGIME_FEATURES]
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((150, len(names))), columns=names)
    y = pd.Series(rng.choice([-1, 0, 1], size=150))
    model = XGBoostClassifier(n_estimators=5, max_depth=2, n_jobs=1)
    model.fit(X, y, all_classes=[-1, 0, 1])
    agent = XgbAlphaAgent(model)

    signal = agent.predict(
        MarketContext(symbol="SPY", features={n: 0.1 for n in names})
    )
    assert signal.kelly_fraction == 0.0 and signal.size_usd == 0.0

    decision = sizer.size(1.0, 0.20, "normal", 0.65)
    assert decision.target_weight != 0.0, (
        "con edge real, el peso lo produce el sizer — no el agente"
    )
