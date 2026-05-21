"""
Advanced Metrics
================
Métricas estadísticamente rigurosas para evaluación de estrategias de trading.

REFERENCIAS:
- Mertens, E. (2002). "Comments on Variance of the IID Estimator in Lo (2002)".
- Bailey, D. & López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier".
  Journal of Risk, 15(2).
- Bailey, D. & López de Prado, M. (2014). "The Deflated Sharpe Ratio".
  Journal of Portfolio Management, 40(5).
- Harvey, C. R., Liu, Y. (2015). "Backtesting". JPM.

NOTACIÓN IMPORTANTE:
- "kurtosis raw" (γ₄) = E[(X-μ)⁴]/σ⁴. Para gaussiana = 3.
- "kurtosis exceso" (κ) = γ₄ - 3. Para gaussiana = 0. (pandas devuelve EXCESO).
- En este módulo SIEMPRE trabajamos con kurtosis RAW (γ₄) para que las fórmulas
  PSR/DSR coincidan literal con Bailey & LdP.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

# Constante Euler-Mascheroni (γ ≈ 0.5772156649) — usada en DSR
EULER_MASCHERONI = 0.5772156649015329


# =====================================================================
# MOMENTOS SUPERIORES
# =====================================================================

def returns_skewness(returns: Sequence[float]) -> float:
    """
    Skewness muestral (sesgo). Negativo = cola izquierda gorda (peligroso).
    Una estrategia con skew < -1 sostenido es típicamente short-vol disfrazado.
    """
    s = pd.Series(returns).dropna()
    if len(s) < 3:
        return 0.0
    return float(s.skew())


def returns_kurtosis(returns: Sequence[float], excess: bool = False) -> float:
    """
    Kurtosis muestral.

    Parameters
    ----------
    excess : bool
        Si False (default), retorna kurtosis RAW (γ₄). Gaussiana = 3.
        Si True, retorna kurtosis EXCESO (κ = γ₄ − 3). Gaussiana = 0.

    Notas
    -----
    pandas.Series.kurtosis() devuelve EXCESO. Aquí convertimos para que la
    interfaz con PSR/DSR sea inequívoca.
    """
    s = pd.Series(returns).dropna()
    if len(s) < 4:
        return 0.0 if excess else 3.0
    excess_kurt = float(s.kurtosis())  # pandas: exceso (Fisher)
    return excess_kurt if excess else excess_kurt + 3.0


# =====================================================================
# SHARPE FAMILY: SE, PSR, DSR
# =====================================================================

def sharpe_ratio_se_mertens(
    sr: float,
    n: int,
    skew: float,
    kurt_raw: float,
) -> float:
    """
    Standard error de Sharpe corregido por momentos superiores (Mertens 2002).

    σ̂(Ŝ R) = sqrt( [1 − γ₃·SR + (γ₄ − 1)/4 · SR²] / (n − 1) )

    donde γ₃ es skew, γ₄ es kurtosis RAW, SR es el Sharpe muestral por-período
    (NO anualizado).

    Parameters
    ----------
    sr : float
        Sharpe ratio muestral, EN UNIDADES DE PERÍODO (no anualizado).
        Si tienes SR_annual y T períodos por año, convierte: sr = SR_annual / sqrt(T).
    n : int
        Número de observaciones de retornos.
    skew : float
        Skewness muestral (γ₃).
    kurt_raw : float
        Kurtosis muestral RAW (γ₄). Gaussiana = 3.

    Returns
    -------
    float : SE del estimador del Sharpe (per-period).
    """
    if n < 2:
        return float("inf")
    var_sr = (1.0 - skew * sr + (kurt_raw - 1.0) / 4.0 * sr ** 2) / (n - 1)
    var_sr = max(var_sr, 0.0)  # guard numérico
    return float(np.sqrt(var_sr))


def probabilistic_sharpe_ratio(
    sr: float,
    n: int,
    skew: float,
    kurt_raw: float,
    sr_benchmark: float = 0.0,
) -> float:
    """
    Probabilistic Sharpe Ratio: P(SR_real > sr_benchmark | datos observados).

    PSR(SR*) = Φ( (SR̂ − SR*) / σ̂(SR̂) )

    Interpretación: probabilidad de que el verdadero Sharpe (poblacional) supere
    el umbral SR*, dada la muestra observada y los momentos superiores. PSR > 0.95
    es una "evidencia" estándar de Sharpe positivo.

    Todos los valores PER-PERÍODO. No anualizar.
    """
    se = sharpe_ratio_se_mertens(sr, n, skew, kurt_raw)
    if not np.isfinite(se) or se <= 0:
        return 0.0
    z = (sr - sr_benchmark) / se
    return float(norm.cdf(z))


def deflated_sharpe_ratio(
    sr: float,
    n: int,
    skew: float,
    kurt_raw: float,
    n_trials: int,
    sr_trials_std: Optional[float] = None,
) -> float:
    """
    Deflated Sharpe Ratio (Bailey & LdP, 2014).

    DSR = PSR(SR*₀) donde SR*₀ es el Sharpe esperado del MEJOR de n_trials
    estrategias bajo la hipótesis nula (todas con SR_real = 0):

        SR*₀ = sqrt(V[{SR_n}]) · [(1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(Ne))]

    γ = constante Euler-Mascheroni ≈ 0.5772.

    Si DSR > 0.95 → tu Sharpe sobrevive a la corrección por backtesting múltiple.
    Si DSR < 0.50 → tu "alpha" es probablemente data-snooping.

    Parameters
    ----------
    sr : float
        Sharpe muestral PER-PERÍODO observado.
    n : int
        Tamaño muestral de retornos.
    skew, kurt_raw : float
        Momentos superiores muestrales.
    n_trials : int
        Número de configuraciones/estrategias probadas durante la búsqueda.
        SÉ HONESTO. Si optimizaste 5 hiperparámetros con 20 valores cada uno,
        n_trials ≈ 20⁵ aunque solo reportes el mejor.
    sr_trials_std : float | None
        Desviación estándar de los Sharpe ratios de los n_trials configuraciones
        probadas. Si None, se usa una heurística conservadora (|sr|/2).

    Returns
    -------
    float ∈ [0, 1] : probabilidad de que el Sharpe verdadero supere el umbral
    inflado por selección bias.
    """
    if n_trials <= 1:
        # Sin múltiples pruebas, DSR colapsa a PSR(0)
        return probabilistic_sharpe_ratio(sr, n, skew, kurt_raw, sr_benchmark=0.0)

    if sr_trials_std is None:
        # Heurística conservadora: si no conoces la dispersión, asume la mitad del
        # Sharpe observado. Esto sobre-estima moderadamente el threshold.
        sr_trials_std = max(abs(sr) / 2.0, 1e-6)

    # Expected max under null
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    sr_zero = sr_trials_std * (
        (1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2
    )

    return probabilistic_sharpe_ratio(sr, n, skew, kurt_raw, sr_benchmark=sr_zero)


# =====================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =====================================================================

def bootstrap_sharpe_ci(
    returns: Sequence[float],
    periods_per_year: float = 252.0,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> tuple[float, float]:
    """
    Intervalo de confianza bootstrap (percentil) para Sharpe ANUALIZADO.

    Útil cuando los retornos son no-iid o no-gaussianos: el SE de Mertens asume
    iid; el bootstrap es no-paramétrico y tolerante a colas pesadas.

    Recomendado: usar bootstrap por bloques para series con autocorrelación
    (no implementado aquí; ver arch.bootstrap si lo necesitas).
    """
    rng = np.random.default_rng(seed)
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 10:
        return (float("nan"), float("nan"))

    sharpes = np.empty(n_boot)
    for i in range(n_boot):
        sample = r[rng.integers(0, n, size=n)]
        mu, sd = sample.mean(), sample.std(ddof=1)
        sharpes[i] = np.sqrt(periods_per_year) * mu / sd if sd > 0 else 0.0

    return (
        float(np.quantile(sharpes, alpha / 2)),
        float(np.quantile(sharpes, 1 - alpha / 2)),
    )


def bootstrap_metric_ci(
    returns: Sequence[float],
    metric_fn: Callable[[np.ndarray], float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> tuple[float, float]:
    """
    Bootstrap percentil para cualquier métrica.

    Example
    -------
    >>> ci = bootstrap_metric_ci(returns, lambda x: x.mean(), n_boot=2000)
    """
    rng = np.random.default_rng(seed)
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 10:
        return (float("nan"), float("nan"))

    vals = np.empty(n_boot)
    for i in range(n_boot):
        vals[i] = metric_fn(r[rng.integers(0, n, size=n)])
    return (
        float(np.quantile(vals, alpha / 2)),
        float(np.quantile(vals, 1 - alpha / 2)),
    )


# =====================================================================
# TAIL RISK
# =====================================================================

def value_at_risk(returns: Sequence[float], alpha: float = 0.05) -> float:
    """
    VaR histórico al nivel alpha. Retorna un valor NEGATIVO.

    VaR(0.05) = "en el peor 5% de períodos pierdo al menos |VaR|".
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return 0.0
    return float(np.quantile(r, alpha))


def conditional_var(returns: Sequence[float], alpha: float = 0.05) -> float:
    """
    CVaR / Expected Shortfall: media de los retornos PEORES que el VaR.

    Más informativo que VaR para colas: dice "cuánto pierdo en promedio cuando
    el evento extremo ocurre", no solo "el percentil".
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return 0.0
    var = np.quantile(r, alpha)
    tail = r[r <= var]
    return float(tail.mean()) if len(tail) > 0 else float(var)


def tail_ratio(returns: Sequence[float], alpha: float = 0.05) -> float:
    """
    Tail ratio = mean(top α%) / |mean(bottom α%)|.

    > 1 → cola derecha más gorda que izquierda (favorable).
    < 1 → cola izquierda más gorda (peligroso).
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 20:
        return 1.0
    upper_q = np.quantile(r, 1 - alpha)
    lower_q = np.quantile(r, alpha)
    right = r[r >= upper_q].mean()
    left = abs(r[r <= lower_q].mean())
    if left <= 1e-12:
        return float("inf")
    return float(right / left)


# =====================================================================
# CAPACITY / EFFICIENCY
# =====================================================================

def turnover(positions: Sequence[float]) -> float:
    """
    Turnover anualizado aproximado: suma de cambios absolutos en posición
    dividido por el número de períodos. Útil para estimar friccional cost.

    Returns
    -------
    float : turnover por período (multiplicar por períodos/año para anualizar).
    """
    p = np.asarray(positions, dtype=float)
    if len(p) < 2:
        return 0.0
    return float(np.abs(np.diff(p)).sum() / len(p))


def cost_drag(returns_gross: Sequence[float], returns_net: Sequence[float]) -> float:
    """
    Diferencia entre Sharpe gross y Sharpe net (anualizado, asumiendo 252 períodos).
    Mide cuánto te come la fricción.
    """
    rg = np.asarray(returns_gross, dtype=float)
    rn = np.asarray(returns_net, dtype=float)

    def _ann_sharpe(r):
        r = r[~np.isnan(r)]
        if len(r) < 2 or r.std(ddof=1) == 0:
            return 0.0
        return np.sqrt(252.0) * r.mean() / r.std(ddof=1)

    return float(_ann_sharpe(rg) - _ann_sharpe(rn))


# =====================================================================
# IS vs OOS DEGRADATION
# =====================================================================

def is_oos_degradation(metric_is: float, metric_oos: float) -> dict:
    """
    Cuantifica overfitting: cuánto se degrada una métrica de in-sample a out-of-sample.

    Returns dict con:
    - degradation_abs: IS - OOS
    - degradation_pct: (IS - OOS) / IS  (cuánto se evapora en %)
    - haircut: 1 - OOS/IS (interpretable: "mi backtest necesita un descuento de X%")

    Heurística empírica (Harvey-Liu 2015): un haircut del 50% es típico en
    estrategias publicadas. > 80% sugiere overfitting fuerte.
    """
    if metric_is is None or metric_oos is None or np.isnan(metric_is) or np.isnan(metric_oos):
        return {"degradation_abs": np.nan, "degradation_pct": np.nan, "haircut": np.nan}
    deg_abs = metric_is - metric_oos
    if abs(metric_is) < 1e-12:
        return {"degradation_abs": deg_abs, "degradation_pct": np.nan, "haircut": np.nan}
    deg_pct = deg_abs / metric_is
    haircut = 1.0 - metric_oos / metric_is
    return {
        "degradation_abs": float(deg_abs),
        "degradation_pct": float(deg_pct),
        "haircut": float(haircut),
    }


# =====================================================================
# ONE-SHOT: COMPUTAR TODO
# =====================================================================

def compute_advanced_metrics(
    returns: Sequence[float],
    periods_per_year: float = 252.0,
    n_trials: int = 1,
    sr_trials_std: Optional[float] = None,
    bootstrap_n: int = 2000,
    bootstrap_seed: Optional[int] = 42,
) -> dict:
    """
    Calcula el bundle completo de métricas avanzadas en una sola llamada.

    Devuelve un dict listo para serializar (JSON-friendly, sin pandas/numpy types).
    """
    r = pd.Series(returns).dropna().astype(float)
    n = len(r)
    if n < 2:
        return {"n_returns": int(n), "error": "insufficient data"}

    mu = float(r.mean())
    sigma = float(r.std(ddof=1))

    sr_period = mu / sigma if sigma > 0 else 0.0
    sr_annual = sr_period * np.sqrt(periods_per_year)

    skew = returns_skewness(r)
    kurt_raw = returns_kurtosis(r, excess=False)

    psr = probabilistic_sharpe_ratio(
        sr_period, n, skew, kurt_raw, sr_benchmark=0.0
    )
    dsr = deflated_sharpe_ratio(
        sr_period, n, skew, kurt_raw, n_trials=n_trials, sr_trials_std=sr_trials_std
    )
    se_mertens = sharpe_ratio_se_mertens(sr_period, n, skew, kurt_raw)

    boot_lo, boot_hi = bootstrap_sharpe_ci(
        r.values, periods_per_year=periods_per_year, n_boot=bootstrap_n,
        seed=bootstrap_seed,
    )

    var_05 = value_at_risk(r.values, alpha=0.05)
    cvar_05 = conditional_var(r.values, alpha=0.05)
    t_ratio = tail_ratio(r.values, alpha=0.05)

    return {
        "n_returns": int(n),
        "mean_period": mu,
        "vol_period": sigma,
        "vol_annual": float(sigma * np.sqrt(periods_per_year)),
        "sharpe_period": float(sr_period),
        "sharpe_annual": float(sr_annual),
        "sharpe_se_mertens_period": float(se_mertens),
        "sharpe_bootstrap_ci_annual": [float(boot_lo), float(boot_hi)],
        "skew": float(skew),
        "kurtosis_raw": float(kurt_raw),
        "kurtosis_excess": float(kurt_raw - 3.0),
        "psr": float(psr),
        "dsr": float(dsr),
        "n_trials_used_for_dsr": int(n_trials),
        "var_5pct_period": float(var_05),
        "cvar_5pct_period": float(cvar_05),
        "tail_ratio": float(t_ratio),
    }
