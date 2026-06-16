"""
E1 — Comparación troncal DQN vs XGBoost vs reglas (arbitraje D · gate pre-registrado).

El experimento de mayor ROI del arbitraje: poner el DQN, un XGBoost calibrado y
reglas simples (momentum / mean-reversion) en el MISMO walk-forward, con N≥20 seeds,
y reportar el **Sharpe con intervalo de confianza** frente a un benchmark neutral.
La regla de decisión está pre-registrada en ``ACTA_GATE_DECISION_FASE0.md``:

    LB95(S_Δ*) > 0.20  -> RAMA A   (direccional vivo)
    0 < LB95 <= 0.20   -> RAMA A'  (vivo pero marginal)
    LB95(S_Δ*) <= 0    -> RAMA B   (direccional falsado -> pivote stat-arb = P0)

donde ``S_Δ = Sharpe(modelo) − Sharpe(benchmark neutral)`` y ``LB95`` es la cota
inferior unilateral al 95% (percentil 5 sobre seeds para modelos estocásticos;
moving-block bootstrap para modelos deterministas).

Diseño (reutiliza, NO reimplementa — §20.2):
- DQN OOS:  ``dsr_gate.walk_forward_oos_returns`` (torch, perezoso).
- XGBoost:  ``dsr_gate.xgb_oos_returns``.
- Reglas:   posiciones trailing (sin look-ahead) + ``dsr_gate.positions_to_returns``.
- DSR/PSR:  ``walk_forward_runner.deflated_sharpe_ratio`` (deflación honesta por n_trials).

El núcleo estadístico (reglas, IC, veredicto) es **torch-free** y testeable en
aislamiento; los runners de DQN/XGBoost importan sus dependencias pesadas de forma
perezosa dentro de la función.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PPY_DAILY = 252


# =====================================================================
# Sharpe helper (numpy-only; espeja dsr_gate._annualized_sharpe para el
# tooling de IC. La métrica OFICIAL del gate sigue saliendo de dsr_gate.)
# =====================================================================

def _ann_sharpe(returns: np.ndarray, ppy: int = _PPY_DAILY) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return float("nan")
    sigma = float(r.std(ddof=1))
    if sigma < 1e-12:
        return 0.0
    return float(r.mean() / sigma * np.sqrt(ppy))


# =====================================================================
# Reglas deterministas (baseline mínimo viable — solo datos trailing)
# =====================================================================

def momentum_positions(
    closes: np.ndarray, lookback: int = 20, fee_bps: float = 5.0
) -> np.ndarray:
    """
    Momentum: ``pos_t = sign(close_t/close_{t-lookback} - 1)`` con banda muerta
    de costo (movimientos sub-fee → flat). Solo usa datos pasados (sin look-ahead);
    las primeras ``lookback`` barras quedan flat.
    """
    px = np.asarray(closes, dtype=float)
    n = len(px)
    pos = np.zeros(n, dtype=float)
    if n <= lookback:
        return pos
    trail = px[lookback:] / px[:-lookback] - 1.0
    dead = fee_bps / 1e4
    sig = np.where(np.abs(trail) <= dead, 0.0, np.sign(trail))
    pos[lookback:] = sig
    return pos


def meanrev_positions(
    closes: np.ndarray, lookback: int = 20, z_entry: float = 1.0
) -> np.ndarray:
    """
    Mean-reversion contrarian: z-score trailing del precio; ``pos_t = -sign(z_t)``
    cuando ``|z_t| > z_entry``, si no flat. Solo datos pasados (rolling trailing).
    """
    px = pd.Series(np.asarray(closes, dtype=float))
    mean = px.rolling(lookback).mean()
    std = px.rolling(lookback).std(ddof=1)
    z = (px - mean) / std.replace(0.0, np.nan)
    pos = np.where(z.abs() > z_entry, -np.sign(z), 0.0)
    return np.nan_to_num(pos, nan=0.0)


_RULES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "momentum": lambda c: momentum_positions(c),
    "mean_rev": lambda c: meanrev_positions(c),
}


# =====================================================================
# Resultados + intervalos de confianza
# =====================================================================

@dataclass(frozen=True)
class ModelSharpeResult:
    """Distribución del Sharpe OOS de un modelo y su cota inferior al 95%."""

    name: str
    sharpe_by_seed: np.ndarray          # un Sharpe por seed (o por bootstrap)
    mean: float
    lb95: float                         # percentil 5 unilateral (cota inferior 95%)
    p95: float
    n_seeds: int
    n_oos_bars: int
    ci_method: str                      # "seeds" | "block_bootstrap"
    dsr: float = float("nan")           # DSR deflactado del run mediano (informativo)


def sharpe_ci_from_seeds(sharpes: Sequence[float]) -> tuple[float, float, float]:
    """(media, LB95=percentil 5, P95) de una muestra de Sharpes por seed."""
    s = np.asarray([x for x in sharpes if not np.isnan(x)], dtype=float)
    if len(s) == 0:
        return float("nan"), float("nan"), float("nan")
    return float(s.mean()), float(np.percentile(s, 5)), float(np.percentile(s, 95))


def block_bootstrap_sharpe_ci(
    returns: np.ndarray,
    *,
    n_boot: int = 2000,
    block: int = 5,
    ppy: int = _PPY_DAILY,
    seed: int = 0,
) -> tuple[float, float, float, np.ndarray]:
    """
    IC del Sharpe por moving-block bootstrap (respeta autocorrelación) para
    modelos DETERMINISTAS (reglas / XGBoost de config única).

    Returns
    -------
    (media, LB95, P95, dist) — ``dist`` es la muestra bootstrap de Sharpes.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < max(2 * block, 4):
        s = _ann_sharpe(r, ppy)
        return s, s, s, np.asarray([s])
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts_max = n - block
    out = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        starts = rng.integers(0, starts_max + 1, size=n_blocks)
        sample = np.concatenate([r[s : s + block] for s in starts])[:n]
        out[b] = _ann_sharpe(sample, ppy)
    out = out[~np.isnan(out)]
    return (
        float(out.mean()),
        float(np.percentile(out, 5)),
        float(np.percentile(out, 95)),
        out,
    )


# =====================================================================
# Veredicto del gate pre-registrado (acta Fase 0)
# =====================================================================

@dataclass(frozen=True)
class E1Verdict:
    best_model: str
    best_lb95: float
    branch: str                         # "A" | "A_prime" | "B"
    directional_falsified: bool
    materiality: float
    neutral_sharpe: float
    reason: str
    per_model: Mapping[str, ModelSharpeResult] = field(default_factory=dict)


def evaluate_e1_decision(
    results: Mapping[str, ModelSharpeResult],
    *,
    materiality: float = 0.20,
    neutral_sharpe: float = 0.0,
) -> E1Verdict:
    """
    Aplica la regla pre-registrada del acta sobre el MEJOR modelo por LB95.

    El "mejor" es el de mayor cota inferior (penaliza varianza por seed, no premia
    la media optimista). ``S_Δ = Sharpe − neutral_sharpe``; con benchmark neutral
    risk-free≈0, ``S_Δ ≈ Sharpe``.
    """
    if not results:
        raise ValueError("results vacío: nada que evaluar")

    # LB95 del Sharpe diferencial vs benchmark neutral.
    diff_lb = {name: r.lb95 - neutral_sharpe for name, r in results.items()}
    best = max(diff_lb, key=diff_lb.__getitem__)
    best_lb = diff_lb[best]

    if best_lb > materiality:
        branch, falsified = "A", False
        reason = (
            f"RAMA A: mejor modelo '{best}' con LB95(S_Δ)={best_lb:.3f} > "
            f"materialidad {materiality:.2f} → direccional vivo; endurecer hacia "
            f"robustez (E6) → paper."
        )
    elif best_lb > 0.0:
        branch, falsified = "A_prime", False
        reason = (
            f"RAMA A': mejor modelo '{best}' con LB95(S_Δ)={best_lb:.3f} ∈ (0, "
            f"{materiality:.2f}] → vivo pero marginal; robustez reforzada antes de "
            f"cualquier capital."
        )
    else:
        branch, falsified = "B", True
        reason = (
            f"RAMA B: mejor modelo '{best}' con LB95(S_Δ)={best_lb:.3f} ≤ 0 → "
            f"direccional FALSADO; el pivote a stat-arb pasa a P0 (E5/E6)."
        )

    return E1Verdict(
        best_model=best,
        best_lb95=float(best_lb),
        branch=branch,
        directional_falsified=falsified,
        materiality=materiality,
        neutral_sharpe=neutral_sharpe,
        reason=reason,
        per_model=dict(results),
    )


# =====================================================================
# Runners por modelo (DQN/XGBoost: dependencias pesadas perezosas)
# =====================================================================

def rule_oos_returns(
    raw_ohlcv: pd.DataFrame,
    splitter,
    rule: str,
    *,
    fee_bps: Optional[float] = None,
) -> np.ndarray:
    """Retornos OOS concatenados de una regla determinista sobre los MISMOS folds."""
    from models.drl.dsr_gate import (  # lazy: evita acoplar el núcleo testeable
        EnvironmentConfig,
        _validated_folds,
        clean_close_series,
        positions_to_returns,
    )

    if rule not in _RULES:
        raise ValueError(f"regla desconocida '{rule}'; opciones: {list(_RULES)}")
    fee = EnvironmentConfig().fee_bps if fee_bps is None else fee_bps
    closes_all = clean_close_series(raw_ohlcv).to_numpy()
    positions_all = _RULES[rule](closes_all)
    out: list[np.ndarray] = []
    for _, test_idx in _validated_folds(raw_ohlcv, splitter):
        n_t = len(test_idx) - 1
        idx = test_idx[:n_t]
        out.append(positions_to_returns(positions_all[idx], closes_all[idx], fee))
    return np.concatenate(out)


def run_rules_model(
    raw_ohlcv: pd.DataFrame, splitter, rule: str, *, fee_bps: Optional[float] = None
) -> ModelSharpeResult:
    """Modelo determinista → IC por block bootstrap sobre los retornos OOS."""
    r = rule_oos_returns(raw_ohlcv, splitter, rule, fee_bps=fee_bps)
    mean, lb, p95, dist = block_bootstrap_sharpe_ci(r)
    return ModelSharpeResult(
        name=rule, sharpe_by_seed=dist, mean=mean, lb95=lb, p95=p95,
        n_seeds=1, n_oos_bars=int(len(r)), ci_method="block_bootstrap",
    )


def run_xgb_model(
    raw_ohlcv: pd.DataFrame, splitter, seeds: Sequence[int],
    *, fee_bps: Optional[float] = None, xgb_params: Optional[dict] = None,
) -> ModelSharpeResult:
    """XGBoost calibrado por fold; IC por seeds (random_state varía el ajuste)."""
    from models.drl.dsr_gate import xgb_oos_returns

    sharpes, last_r = [], np.empty(0)
    for s in seeds:
        last_r = xgb_oos_returns(
            raw_ohlcv, splitter, fee_bps=fee_bps, seed=s, xgb_params=xgb_params
        )
        sharpes.append(_ann_sharpe(last_r))
    mean, lb, p95 = sharpe_ci_from_seeds(sharpes)
    return ModelSharpeResult(
        name="xgboost", sharpe_by_seed=np.asarray(sharpes), mean=mean, lb95=lb,
        p95=p95, n_seeds=len(seeds), n_oos_bars=int(len(last_r)), ci_method="seeds",
    )


def run_dqn_model(
    raw_ohlcv: pd.DataFrame, splitter, env_cfg, seeds: Sequence[int],
    *, episodes: int = 100, n_jobs: int = 1,
) -> ModelSharpeResult:
    """
    DQN sobre N seeds (el estimador de alta varianza que motiva el IC, RC-3).

    Cada seed reentrena la política; el IC sale de la dispersión entre seeds.
    Para la rama "DQN calibrado" de E1, el serving usa
    ``alpha.agents.dqn_calibration.fit_dqn_fold_calibrator`` (E3 paso 2) — la
    calibración NO altera el Sharpe de retornos del gate (el gate usa posiciones),
    pero sí habilita el sizing aguas abajo.
    """
    from models.drl.dsr_gate import AgentSpec, walk_forward_oos_returns

    sharpes, last_r = [], np.empty(0)
    for s in seeds:
        spec = AgentSpec(algo="dqn", episodes=episodes, seed=s)
        last_r = walk_forward_oos_returns(
            spec, raw_ohlcv, splitter, env_cfg, seed=s, n_jobs=n_jobs
        )
        sharpes.append(_ann_sharpe(last_r))
        logger.info("DQN seed=%d: sharpe=%.3f oos_bars=%d", s, sharpes[-1], len(last_r))
    mean, lb, p95 = sharpe_ci_from_seeds(sharpes)
    return ModelSharpeResult(
        name="dqn", sharpe_by_seed=np.asarray(sharpes), mean=mean, lb95=lb,
        p95=p95, n_seeds=len(seeds), n_oos_bars=int(len(last_r)), ci_method="seeds",
    )


def run_e1(
    raw_ohlcv: pd.DataFrame,
    *,
    n_folds: int = 5,
    n_seeds: int = 20,
    episodes: int = 100,
    materiality: float = 0.20,
    n_jobs: int = 1,
    env_cfg=None,
) -> E1Verdict:
    """
    Orquesta E1 completo y devuelve el veredicto del gate pre-registrado.

    Requiere torch (DQN) + xgboost (baseline). Pensado para correr en el venv del
    repo; el núcleo (reglas/IC/veredicto) se testea por separado sin esas deps.
    """
    from models.drl.dsr_gate import EnvironmentConfig, make_wf_splitter

    cfg = env_cfg or EnvironmentConfig()
    splitter = make_wf_splitter(raw_ohlcv, n_folds, env_cfg=cfg)
    seeds = list(range(n_seeds))

    results: dict[str, ModelSharpeResult] = {}
    results["dqn"] = run_dqn_model(
        raw_ohlcv, splitter, cfg, seeds, episodes=episodes, n_jobs=n_jobs
    )
    results["xgboost"] = run_xgb_model(raw_ohlcv, splitter, seeds)
    for rule in _RULES:
        results[rule] = run_rules_model(raw_ohlcv, splitter, rule)

    verdict = evaluate_e1_decision(results, materiality=materiality)
    logger.info("E1 veredicto: %s", verdict.reason)
    return verdict
