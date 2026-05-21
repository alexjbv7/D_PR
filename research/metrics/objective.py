"""
Objective Function
==================
Formalización de "qué es éxito" para una estrategia de trading.

DISEÑO (3 niveles):
  1. PRIMARY: una métrica robusta que se OPTIMIZA. Default = DSR.
  2. CONSTRAINTS: gates duros que se cumplen o no. Si fallan → estrategia rechazada.
  3. DIAGNOSTIC: métricas informativas (win rate, profit factor) — se reportan
     pero NUNCA se optimizan directamente.

Esta separación combate la ley de Goodhart: optimizar UNA métrica robusta es
seguro; optimizar varias o métricas degenerables (como win rate solo) lleva a
sistemas que parecen buenos in-sample y se evaporan out-of-sample.

REFERENCIA:
- Harvey, C., Liu, Y. (2015). "Backtesting". JPM. (selection bias)
- Bailey & López de Prado (2014). DSR.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd

from .advanced import compute_advanced_metrics, returns_skewness


# ============================================================
# CONSTRAINTS
# ============================================================

@dataclass
class HardConstraints:
    """
    Gates duros que la estrategia DEBE cumplir para ser válida.

    Si cualquiera falla, la estrategia se rechaza independientemente de su
    métrica primaria. Útil para filtrar patologías como short-vol disfrazado o
    estrategias inviables operacionalmente.
    """
    max_drawdown: float = 0.20            # |MaxDD| ≤ 20% del equity
    min_skew: float = -0.5                # skew ≥ -0.5 (no cola izquierda monstruosa)
    min_n_trades: int = 30                # ≥ 30 trades (sample size mínimo)
    min_psr: float = 0.95                 # PSR ≥ 0.95 (5% de error tipo I)
    min_dsr: float = 0.95                 # DSR ≥ 0.95 (con corrección por trials)
    min_sharpe_annual: float = 0.0        # Sharpe anualizado positivo
    max_tail_ratio_inverse: float = 2.0   # cola izquierda no > 2× cola derecha


@dataclass
class ObjectiveSpec:
    """
    Especificación completa de la función objetivo.

    Parameters
    ----------
    primary : {'dsr', 'psr', 'sharpe', 'sortino', 'calmar'}
        Métrica primaria a optimizar.
    constraints : HardConstraints
        Filtros duros que deben pasarse.
    n_trials : int
        Número de configuraciones probadas durante búsqueda. Crítico para DSR.
        SÉ HONESTO: si optimizaste 5 hiperparámetros con 10 valores cada uno,
        n_trials ≈ 10⁵ aunque solo guardes el mejor.
    sr_trials_std : float | None
        Desviación estándar de los Sharpe entre trials. Si None, heurística.
    periods_per_year : float
        Periodicidad del muestreo de retornos (252 = diario, 8760 = horario, etc.)
    """
    primary: str = "dsr"
    constraints: HardConstraints = field(default_factory=HardConstraints)
    n_trials: int = 1
    sr_trials_std: Optional[float] = None
    periods_per_year: float = 252.0


# ============================================================
# RESULT
# ============================================================

@dataclass
class ObjectiveResult:
    """Veredicto de evaluación de una estrategia."""
    primary_metric: str
    primary_value: float
    primary_threshold: float           # mínimo para considerarse "éxito"
    passed_primary: bool
    passed_constraints: bool
    constraint_violations: list[str]
    advanced_metrics: dict
    diagnostic_metrics: dict
    verdict: str                        # 'ACCEPT', 'REJECT_CONSTRAINTS', 'REJECT_PRIMARY'

    def to_dict(self) -> dict:
        d = asdict(self)
        # asdict no convierte bien dicts numpy
        return {k: (v if not isinstance(v, dict)
                    else {kk: (float(vv) if isinstance(vv, (np.floating,)) else vv)
                          for kk, vv in v.items()})
                for k, v in d.items()}

    def summary(self) -> str:
        lines = [
            "═══════════════ EVALUACIÓN DE ESTRATEGIA ═══════════════",
            f"  Métrica primaria:   {self.primary_metric.upper()}",
            f"  Valor observado:    {self.primary_value:.4f}",
            f"  Umbral requerido:   {self.primary_threshold:.4f}",
            f"  ¿Pasa primaria?     {'✓ SÍ' if self.passed_primary else '✗ NO'}",
            f"  ¿Pasa constraints?  {'✓ SÍ' if self.passed_constraints else '✗ NO'}",
        ]
        if self.constraint_violations:
            lines.append("  Violaciones:")
            for v in self.constraint_violations:
                lines.append(f"    - {v}")
        lines.append("  ────────────────────────────────────────────────────")
        lines.append(f"  VEREDICTO: {self.verdict}")
        lines.append("═══════════════════════════════════════════════════════")
        return "\n".join(lines)


# ============================================================
# EVALUACIÓN
# ============================================================

def _max_drawdown_from_returns(returns: pd.Series) -> float:
    """Calcula |MaxDD| a partir de la serie de retornos compuesta."""
    if len(returns) == 0:
        return 0.0
    equity = (1.0 + returns.fillna(0)).cumprod()
    dd = (equity - equity.cummax()) / equity.cummax()
    return float(abs(dd.min()))


def evaluate_objective(
    returns: pd.Series,
    spec: ObjectiveSpec,
    n_trades: Optional[int] = None,
    diagnostic_metrics: Optional[dict] = None,
) -> ObjectiveResult:
    """
    Evalúa una estrategia contra la especificación.

    Parameters
    ----------
    returns : pd.Series
        Serie de retornos OUT-OF-SAMPLE (idealmente concatenación de folds OOS).
    spec : ObjectiveSpec
        Función objetivo + constraints.
    n_trades : int | None
        Número de trades ejecutados. Si None, no se aplica el constraint min_n_trades.
    diagnostic_metrics : dict | None
        Métricas adicionales (win_rate, profit_factor, etc.) que se incluyen
        en el reporte pero NO afectan al veredicto.

    Returns
    -------
    ObjectiveResult
    """
    adv = compute_advanced_metrics(
        returns,
        periods_per_year=spec.periods_per_year,
        n_trials=spec.n_trials,
        sr_trials_std=spec.sr_trials_std,
    )

    # Métrica primaria
    primary_map = {
        "dsr": adv.get("dsr", 0.0),
        "psr": adv.get("psr", 0.0),
        "sharpe": adv.get("sharpe_annual", 0.0),
        # sortino y calmar requieren equity; los calculamos si están disponibles
    }

    if spec.primary == "sortino":
        excess = returns.dropna()
        downside = excess[excess < 0]
        if len(downside) > 0 and downside.std() > 0:
            sortino = (
                np.sqrt(spec.periods_per_year)
                * excess.mean() / downside.std()
            )
        else:
            sortino = 0.0
        primary_value = float(sortino)
        primary_threshold = spec.constraints.min_sharpe_annual  # mismo umbral conceptual
    elif spec.primary == "calmar":
        equity = (1 + returns.fillna(0)).cumprod()
        n_years = len(returns) / spec.periods_per_year
        cagr = equity.iloc[-1] ** (1 / max(n_years, 1e-9)) - 1 if equity.iloc[-1] > 0 else -1
        mdd = _max_drawdown_from_returns(returns)
        primary_value = float(cagr / mdd) if mdd > 0 else 0.0
        primary_threshold = spec.constraints.min_sharpe_annual
    else:
        primary_value = float(primary_map.get(spec.primary, 0.0))
        # Threshold según métrica
        if spec.primary == "dsr":
            primary_threshold = spec.constraints.min_dsr
        elif spec.primary == "psr":
            primary_threshold = spec.constraints.min_psr
        else:  # sharpe
            primary_threshold = spec.constraints.min_sharpe_annual

    passed_primary = primary_value >= primary_threshold

    # ============================================================
    # CONSTRAINTS DUROS
    # ============================================================
    violations: list[str] = []

    mdd = _max_drawdown_from_returns(returns)
    if mdd > spec.constraints.max_drawdown:
        violations.append(
            f"MaxDD={mdd:.2%} > {spec.constraints.max_drawdown:.2%}"
        )

    skew = returns_skewness(returns)
    if skew < spec.constraints.min_skew:
        violations.append(
            f"skew={skew:.3f} < {spec.constraints.min_skew} "
            "(cola izquierda peligrosa)"
        )

    if n_trades is not None and n_trades < spec.constraints.min_n_trades:
        violations.append(
            f"n_trades={n_trades} < {spec.constraints.min_n_trades} "
            "(sample insuficiente)"
        )

    psr = adv.get("psr", 0.0)
    if psr < spec.constraints.min_psr:
        violations.append(f"PSR={psr:.3f} < {spec.constraints.min_psr}")

    dsr = adv.get("dsr", 0.0)
    if dsr < spec.constraints.min_dsr:
        violations.append(
            f"DSR={dsr:.3f} < {spec.constraints.min_dsr} "
            f"(con n_trials={spec.n_trials})"
        )

    sharpe_ann = adv.get("sharpe_annual", 0.0)
    if sharpe_ann < spec.constraints.min_sharpe_annual:
        violations.append(
            f"Sharpe={sharpe_ann:.3f} < {spec.constraints.min_sharpe_annual}"
        )

    t_ratio = adv.get("tail_ratio", 1.0)
    if t_ratio > 0 and (1.0 / t_ratio) > spec.constraints.max_tail_ratio_inverse:
        violations.append(
            f"tail_ratio={t_ratio:.2f} (inverso={1/t_ratio:.2f}) "
            f"> límite {spec.constraints.max_tail_ratio_inverse}"
        )

    passed_constraints = len(violations) == 0

    # ============================================================
    # VEREDICTO
    # ============================================================
    if not passed_constraints:
        verdict = "REJECT_CONSTRAINTS"
    elif not passed_primary:
        verdict = "REJECT_PRIMARY"
    else:
        verdict = "ACCEPT"

    return ObjectiveResult(
        primary_metric=spec.primary,
        primary_value=float(primary_value),
        primary_threshold=float(primary_threshold),
        passed_primary=bool(passed_primary),
        passed_constraints=bool(passed_constraints),
        constraint_violations=violations,
        advanced_metrics=adv,
        diagnostic_metrics=diagnostic_metrics or {},
        verdict=verdict,
    )
