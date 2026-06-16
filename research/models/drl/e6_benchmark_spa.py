"""
E6 — Sensibilidad al benchmark + control de data-snooping (arbitraje D · Fase 6).

El arbitraje dejó abierta una pregunta: ¿el "no edge" del direccional es real o un
**artefacto del régimen de prueba** (benchmark buy-and-hold durísimo en mercado
alcista)? E6 lo ataca por dos vías:

1. ``benchmark_sensitivity`` — Sharpe diferencial del modelo frente a varios
   benchmarks (risk-free/cero, buy-and-hold, neutral). Si el veredicto cambia mucho
   entre el benchmark NEUTRAL (el del acta) y el buy-and-hold alcista, el "no edge"
   era dependiente del régimen, no estructural.

2. **Control de data-snooping** sobre el conjunto de configuraciones probadas
   (buscar entre muchas y quedarse con la mejor infla falsos positivos):
   - ``reality_check_pvalue`` — Reality Check de White (2000): ¿la MEJOR config bate
     al benchmark más de lo esperable por azar al haber buscado entre muchas?
   - ``studentized_reality_check_pvalue`` — variante studentizada (estilo Hansen
     SPA 2005): divide por la SE bootstrap de cada config → más potencia. *No incluye
     el recentrado "consistente" de Hansen que descarta configs muy malas; es un RC
     studentizado, no el SPA_c completo (declarado honestamente).*

Ambos usan **stationary bootstrap** (Politis & Romano 1994) para respetar la
autocorrelación de los retornos. Núcleo numpy-only (testeable sin torch); la
construcción de las series de benchmark vive en el gate (``dsr_gate``).

Referencias: White (2000), *A Reality Check for Data Snooping*; Hansen (2005),
*A Test for Superior Predictive Ability*; Politis & Romano (1994).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping

import numpy as np

logger = logging.getLogger(__name__)

_PPY_DAILY = 252


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
# Sensibilidad al benchmark
# =====================================================================

@dataclass(frozen=True)
class BenchmarkSensitivity:
    benchmark: str
    sharpe_model: float
    sharpe_benchmark: float
    sharpe_diff: float          # modelo − benchmark
    model_beats: bool


def benchmark_sensitivity(
    model_returns: np.ndarray,
    benchmarks: Mapping[str, np.ndarray],
    *,
    ppy: int = _PPY_DAILY,
) -> dict[str, BenchmarkSensitivity]:
    """
    Sharpe diferencial del modelo frente a cada benchmark (mismas barras OOS).

    ``benchmarks`` mapea nombre → serie de retornos OOS (misma longitud que
    ``model_returns``). Para el benchmark "cero"/risk-free pasar un array de ceros.
    """
    m = np.asarray(model_returns, dtype=float)
    s_model = _ann_sharpe(m, ppy)
    out: dict[str, BenchmarkSensitivity] = {}
    for name, bench in benchmarks.items():
        b = np.asarray(bench, dtype=float)
        if b.shape != m.shape:
            raise ValueError(
                f"benchmark '{name}' {b.shape} no alinea con el modelo {m.shape}"
            )
        s_bench = _ann_sharpe(b, ppy)
        diff = s_model - s_bench
        out[name] = BenchmarkSensitivity(
            benchmark=name, sharpe_model=s_model, sharpe_benchmark=s_bench,
            sharpe_diff=float(diff), model_beats=bool(diff > 0),
        )
    return out


# =====================================================================
# Stationary bootstrap (Politis & Romano 1994)
# =====================================================================

def stationary_bootstrap_indices(
    n: int, block_avg: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Índices de un remuestreo stationary bootstrap de longitud ``n``.

    Bloques de longitud geométrica con media ``block_avg`` (prob. de reinicio
    ``1/block_avg``). Envuelve circularmente. Respeta la dependencia temporal —
    apropiado para series de retornos (no IID).
    """
    if n <= 0:
        return np.empty(0, dtype=int)
    p = 1.0 / max(block_avg, 1.0)
    idx = np.empty(n, dtype=int)
    idx[0] = int(rng.integers(0, n))
    for t in range(1, n):
        if rng.random() < p:
            idx[t] = int(rng.integers(0, n))
        else:
            idx[t] = (idx[t - 1] + 1) % n
    return idx


# =====================================================================
# Reality Check de White + variante studentizada (estilo SPA)
# =====================================================================

def _as_perf_matrix(perf: np.ndarray) -> np.ndarray:
    p = np.asarray(perf, dtype=float)
    if p.ndim == 1:
        p = p[:, None]
    if p.ndim != 2 or p.shape[0] < 2:
        raise ValueError("perf debe ser (n_obs, n_configs) con n_obs >= 2")
    return p


def reality_check_pvalue(
    perf: np.ndarray, *, n_boot: int = 1000, block_avg: float = 5.0, seed: int = 0
) -> float:
    """
    p-valor del Reality Check de White (2000).

    ``perf`` es ``(n_obs, n_configs)``: columna k = sobre-rendimiento por período de
    la config k respecto al benchmark (p.ej. ``ret_k − ret_benchmark``). H0: ninguna
    config bate al benchmark (``max_k E[d_k] <= 0``). p pequeño → la mejor config bate
    al benchmark más de lo esperable por azar tras la búsqueda.
    """
    d = _as_perf_matrix(perf)
    n = d.shape[0]
    dbar = d.mean(axis=0)
    V = np.sqrt(n) * dbar.max()
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_boot):
        idx = stationary_bootstrap_indices(n, block_avg, rng)
        dbar_star = d[idx].mean(axis=0)
        Vb = np.sqrt(n) * (dbar_star - dbar).max()
        if Vb > V:
            count += 1
    return count / n_boot


def studentized_reality_check_pvalue(
    perf: np.ndarray, *, n_boot: int = 1000, block_avg: float = 5.0, seed: int = 0
) -> dict:
    """
    Reality Check **studentizado** (estilo Hansen SPA 2005): divide el estadístico de
    cada config por su SE bootstrap → más potencia que el RC plano.

    *Nota honesta:* implementa la studentización, NO el recentrado "consistente" de
    Hansen que descarta configs muy malas (SPA_c). Es un RC studentizado, suficiente
    para controlar el data-snooping con mejor potencia; el SPA_c completo es una
    extensión conocida.

    Returns
    -------
    dict con ``p_value``, ``t_stat`` (estadístico observado) y ``n_configs``.
    """
    d = _as_perf_matrix(perf)
    n, L = d.shape
    dbar = d.mean(axis=0)
    rng = np.random.default_rng(seed)

    # Bootstrap de los promedios recentrados → SE por config + distribución del máx.
    boot_dev = np.empty((n_boot, L), dtype=float)
    for b in range(n_boot):
        idx = stationary_bootstrap_indices(n, block_avg, rng)
        boot_dev[b] = d[idx].mean(axis=0) - dbar          # recentrado (media 0 bajo H0)
    omega = boot_dev.std(axis=0, ddof=1)
    omega = np.where(omega < 1e-12, 1e-12, omega)

    t_obs = float(np.max(dbar / omega))
    t_star = (boot_dev / omega).max(axis=1)               # máx studentizado por boot
    p = float(np.mean(t_star > t_obs))
    return {"p_value": p, "t_stat": t_obs, "n_configs": int(L)}


# =====================================================================
# Reporte E6
# =====================================================================

@dataclass(frozen=True)
class E6Result:
    rc_pvalue: float
    spa_pvalue: float
    spa_tstat: float
    n_configs: int
    n_obs: int
    sensitivity: Mapping[str, BenchmarkSensitivity]
    reason: str


def e6_report(
    model_returns: np.ndarray,
    configs_perf: np.ndarray,
    benchmarks: Mapping[str, np.ndarray],
    *,
    ppy: int = _PPY_DAILY,
    n_boot: int = 1000,
    block_avg: float = 5.0,
    seed: int = 0,
    alpha: float = 0.05,
) -> E6Result:
    """
    E6 consolidado: sensibilidad al benchmark + RC/SPA sobre las configs probadas.

    ``configs_perf`` = ``(n_obs, n_configs)`` de sobre-rendimiento vs el benchmark
    NEUTRAL. ``benchmarks`` para la sensibilidad (incluir 'zero', 'buy_and_hold', …).
    """
    sens = benchmark_sensitivity(model_returns, benchmarks, ppy=ppy)
    rc_p = reality_check_pvalue(
        configs_perf, n_boot=n_boot, block_avg=block_avg, seed=seed
    )
    spa = studentized_reality_check_pvalue(
        configs_perf, n_boot=n_boot, block_avg=block_avg, seed=seed
    )
    d = _as_perf_matrix(configs_perf)

    beats_neutral = sens.get("zero")
    beats_bh = sens.get("buy_and_hold")
    snoop_ok = spa["p_value"] < alpha
    regime_flag = (
        beats_neutral is not None and beats_bh is not None
        and beats_neutral.model_beats and not beats_bh.model_beats
    )
    reason = (
        f"SPA p={spa['p_value']:.3f} (RC p={rc_p:.3f}) sobre {spa['n_configs']} configs: "
        + ("evidencia de habilidad superior tras corregir data-snooping. "
           if snoop_ok else
           "SIN evidencia de habilidad superior tras corregir data-snooping. ")
        + ("El modelo bate al benchmark NEUTRAL pero no al buy-and-hold alcista → el "
           "'no edge' vs BH es dependiente del régimen de prueba (revisar con E1)."
           if regime_flag else
           "Veredicto consistente entre benchmarks neutral y buy-and-hold.")
    )
    return E6Result(
        rc_pvalue=float(rc_p), spa_pvalue=float(spa["p_value"]),
        spa_tstat=float(spa["t_stat"]), n_configs=int(spa["n_configs"]),
        n_obs=int(d.shape[0]), sensitivity=sens, reason=reason,
    )
