"""
E5 — Screening de pares cointegrados sobre un universo (ADR-043 · Rama B).

El veredicto del gate (E1) falsó el direccional → el pivote market-neutral pasa a
P0. El par por defecto SPY/QQQ ya falló (no cointegra). E5 generaliza: dado un
UNIVERSO de instrumentos (cripto en 4h, anchor XRP/USD), prueba todos los pares
candidatos en walk-forward y rankea por **LB95 del Sharpe** frente a ZERO, con
**deflación honesta** por nº de pares evaluados (buscar entre muchos infla el sesgo
de selección — ADR-043 §9).

Reutiliza (§20.2), NO reimplementa:
- ``alpha.statarb.pairs.PairStatArb`` / ``walk_forward_pair_returns`` — cointegración
  Engle-Granger, β por OLS, half-life, z-score, doble-fee, anti-leakage por fold.
- ``models.drl.dsr_gate.evaluate_zero_gate`` — DSR deflactado vs ZERO.
- ``models.drl.e1_baseline_comparison.block_bootstrap_sharpe_ci`` — IC del Sharpe
  para series deterministas (mismo método que las reglas de E1).

El núcleo de orquestación (generación de pares, alineación, sizing del splitter,
ranking, veredicto) es numpy/pandas puro y testeable; ``screen_pair`` invoca
``statsmodels`` (vía ``PairStatArb.fit``) solo al correr sobre datos reales.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from models.drl.e1_baseline_comparison import _ann_sharpe, block_bootstrap_sharpe_ci
from models.validation import WalkForwardSplitter

logger = logging.getLogger(__name__)

_PPY_DAILY = 252
_MIN_EMBARGO = 60   # ADR-043 §4 / ADR-040 §4.3


# =====================================================================
# Generación de pares y alineación (puro)
# =====================================================================

def candidate_pairs(
    symbols: Sequence[str], anchor: Optional[str] = None
) -> list[tuple[str, str]]:
    """
    Pares candidatos del universo. Con ``anchor`` (p.ej. 'XRP/USD') devuelve solo
    los pares que incluyen el anchor; si no, todas las combinaciones sin repetir.
    """
    syms = list(dict.fromkeys(symbols))            # dedup preservando orden
    if anchor is not None:
        if anchor not in syms:
            raise ValueError(f"anchor '{anchor}' no está en el universo {syms}")
        return [(anchor, s) for s in syms if s != anchor]
    return [(a, b) for a, b in combinations(syms, 2)]


def align_pair_prices(
    prices_by_symbol: Mapping[str, pd.DataFrame], y_sym: str, x_sym: str
) -> pd.DataFrame:
    """
    Frame de 2 patas alineado por la INTERSECCIÓN de timestamps, columnas
    ``y``/``x`` (cierres). Un par se evalúa solo sobre las barras comunes.
    """
    for s in (y_sym, x_sym):
        if s not in prices_by_symbol:
            raise ValueError(f"falta '{s}' en prices_by_symbol")
    y = prices_by_symbol[y_sym]["close"].astype(float)
    x = prices_by_symbol[x_sym]["close"].astype(float)
    df = pd.concat({"y": y, "x": x}, axis=1).dropna().sort_index()
    return df


def make_pair_splitter(
    n: int, n_folds: int = 5, *, embargo: int = _MIN_EMBARGO,
    min_train_frac: float = 0.3, min_train_floor: int = 250,
) -> WalkForwardSplitter:
    """
    ``WalkForwardSplitter`` expandible dimensionado sobre ``n`` barras alineadas
    (misma regla que ``dsr_gate.make_wf_splitter``: train inicial = max(floor,
    30% n); el resto menos embargo se reparte en ``n_folds`` ventanas de test).
    """
    if n_folds < 1:
        raise ValueError(f"n_folds debe ser >= 1, got {n_folds}")
    if embargo < _MIN_EMBARGO:
        raise ValueError(f"embargo {embargo} < {_MIN_EMBARGO} (ADR-043 §4)")
    min_train = max(min_train_floor, int(min_train_frac * n))
    test_size = (n - min_train - embargo) // n_folds
    if test_size < 10:
        raise ValueError(
            f"insuficientes barras ({n}) para {n_folds} folds "
            f"(min_train={min_train}, embargo={embargo})"
        )
    train_size = n - embargo - n_folds * test_size
    return WalkForwardSplitter(
        train_size=train_size, test_size=test_size, expanding=True, embargo=embargo
    )


# =====================================================================
# Resultado del screening + ranking + veredicto (puro)
# =====================================================================

@dataclass
class PairScreenResult:
    y_sym: str
    x_sym: str
    sharpe: float
    lb95: float
    p95: float
    n_oos: int
    frac_traded: float                 # fracción de barras OOS con posición (no flat)
    oos_returns: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))

    @property
    def label(self) -> str:
        return f"{self.y_sym}/{self.x_sym}"


def rank_pairs(results: Sequence[PairScreenResult]) -> list[PairScreenResult]:
    """Ordena por LB95 descendente (mejor primero) — por cota inferior, no media."""
    return sorted(results, key=lambda r: (-r.lb95, -r.sharpe))


@dataclass(frozen=True)
class ScreenVerdict:
    best: Optional[PairScreenResult]
    best_lb95: float
    branch: str                        # "VIABLE" | "MARGINAL" | "NONE"
    n_pairs: int
    dsr_deflated: float
    gate_passed: bool
    reason: str


def evaluate_screen(
    results: Sequence[PairScreenResult],
    *,
    materiality: float = 0.0,
    periods_per_year: int = _PPY_DAILY,
    dsr_threshold: float = 0.4,
) -> ScreenVerdict:
    """
    Veredicto del screening sobre el MEJOR par por LB95, con deflación honesta.

    Benchmark = ZERO (market-neutral): ``S_Δ = Sharpe``. ``n_trials`` = nº de pares
    evaluados (deflación de selección, ADR-043 §9). El DSR deflactado del mejor par
    sale de ``evaluate_zero_gate``.
    """
    from models.drl.dsr_gate import evaluate_zero_gate   # lazy: deps del gate

    if not results:
        raise ValueError("results vacío: nada que evaluar")
    ranked = rank_pairs(results)
    best = ranked[0]
    n_pairs = len(results)

    gate = evaluate_zero_gate(
        best.oos_returns, n_trials=n_pairs,
        dsr_threshold=dsr_threshold, periods_per_year=periods_per_year,
    )

    if best.lb95 > materiality and gate.passed:
        branch = "VIABLE"
        reason = (
            f"VIABLE: mejor par {best.label} con LB95={best.lb95:.3f} > "
            f"{materiality:.2f} y DSR deflactado {gate.dsr_agent:.3f} > {dsr_threshold} "
            f"(deflactado por {n_pairs} pares) → candidato a paper market-neutral."
        )
    elif best.lb95 > 0.0:
        branch = "MARGINAL"
        reason = (
            f"MARGINAL: mejor par {best.label} LB95={best.lb95:.3f} > 0 pero no pasa "
            f"el gate ZERO deflactado (DSR={gate.dsr_agent:.3f}). Más datos / mejores "
            f"pares antes de capital."
        )
    else:
        branch = "NONE"
        reason = (
            f"SIN PAR VIABLE: mejor LB95={best.lb95:.3f} ≤ 0 sobre {n_pairs} pares. "
            f"Ampliar universo / timeframe; ningún par del set actual tiene edge."
        )

    return ScreenVerdict(
        best=best, best_lb95=float(best.lb95), branch=branch, n_pairs=n_pairs,
        dsr_deflated=float(gate.dsr_agent), gate_passed=bool(gate.passed and branch == "VIABLE"),
        reason=reason,
    )


# =====================================================================
# Screening (usa PairStatArb → statsmodels en datos reales)
# =====================================================================

def screen_pair(
    prices_by_symbol: Mapping[str, pd.DataFrame],
    y_sym: str,
    x_sym: str,
    *,
    n_folds: int = 5,
    config=None,
    periods_per_year: int = _PPY_DAILY,
    n_boot: int = 2000,
) -> PairScreenResult:
    """
    Evalúa un par en walk-forward y devuelve su métrica de screening.

    Folds que no cointegran quedan flat (retorno 0) — honesto. ``frac_traded`` mide
    cuánto del OOS estuvo en posición (proxy de cuántos folds cointegraron).
    """
    from alpha.statarb.pairs import (   # lazy: statsmodels solo al correr real
        PairStatArb,
        PairStatArbConfig,
        walk_forward_pair_returns,
    )

    frame = align_pair_prices(prices_by_symbol, y_sym, x_sym)
    splitter = make_pair_splitter(len(frame), n_folds)
    strat = PairStatArb("y", "x", config or PairStatArbConfig())
    r = walk_forward_pair_returns(frame, splitter, strat)
    mean, lb, p95, _ = block_bootstrap_sharpe_ci(r, n_boot=n_boot, ppy=periods_per_year)
    frac_traded = float(np.mean(r != 0.0)) if len(r) else 0.0
    logger.info(
        "screen %s/%s: sharpe=%.3f LB95=%.3f traded=%.0f%% n=%d",
        y_sym, x_sym, _ann_sharpe(r, periods_per_year), lb, 100 * frac_traded, len(r),
    )
    return PairScreenResult(
        y_sym=y_sym, x_sym=x_sym,
        sharpe=_ann_sharpe(r, periods_per_year), lb95=lb, p95=p95,
        n_oos=int(len(r)), frac_traded=frac_traded, oos_returns=r,
    )


def screen_universe(
    prices_by_symbol: Mapping[str, pd.DataFrame],
    symbols: Sequence[str],
    *,
    anchor: Optional[str] = None,
    n_folds: int = 5,
    config=None,
    periods_per_year: int = _PPY_DAILY,
) -> list[PairScreenResult]:
    """Corre ``screen_pair`` sobre todos los pares candidatos del universo."""
    results: list[PairScreenResult] = []
    for y_sym, x_sym in candidate_pairs(symbols, anchor=anchor):
        try:
            results.append(
                screen_pair(prices_by_symbol, y_sym, x_sym,
                            n_folds=n_folds, config=config,
                            periods_per_year=periods_per_year)
            )
        except Exception as exc:   # noqa: BLE001 — un par malo no tumba el screening
            logger.warning("par %s/%s saltado: %s", y_sym, x_sym, exc)
    return results
