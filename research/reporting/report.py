"""
Reporting
=========
Reportes estandarizados que combinan métricas básicas (Sharpe, Sortino, MDD) +
métricas avanzadas (DSR, PSR, bootstrap CI) + veredicto formal de la función
objetivo.

Soporta:
  - Reportes single-period (IS o OOS individuales)
  - Comparación IS vs OOS (mide overfitting)
  - Agregación de folds de walk-forward
  - Export a JSON / CSV
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from metrics.advanced import (
    compute_advanced_metrics,
    is_oos_degradation,
)
from metrics.objective import (
    ObjectiveSpec,
    ObjectiveResult,
    evaluate_objective,
)


# =====================================================================
# DATA CLASSES
# =====================================================================

@dataclass
class StrategyReport:
    """
    Reporte completo de una estrategia evaluada sobre un período concreto.

    Combina:
    - Métricas avanzadas (DSR, PSR, bootstrap CI, tail risk, etc.)
    - Veredicto de la función objetivo
    - Métricas de diagnóstico (win rate, profit factor — solo informativos)
    """
    label: str                           # 'IS', 'OOS', 'fold_3', etc.
    period_start: Optional[str]
    period_end: Optional[str]
    n_returns: int
    n_trades: Optional[int]
    advanced_metrics: dict
    diagnostic_metrics: dict
    objective_result: Optional[dict]     # serializable

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FoldReport:
    """Reporte por fold individual de walk-forward."""
    fold_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_test_returns: int
    test_sharpe_annual: float
    test_dsr: float
    test_max_drawdown: float
    test_n_trades: int


# =====================================================================
# BUILD REPORT
# =====================================================================

def build_report(
    returns: pd.Series,
    label: str = "OOS",
    n_trades: Optional[int] = None,
    diagnostic_metrics: Optional[dict] = None,
    objective_spec: Optional[ObjectiveSpec] = None,
) -> StrategyReport:
    """
    Construye un reporte completo a partir de una serie de retornos.

    Parameters
    ----------
    returns : pd.Series con índice datetime
    label : str
        Etiqueta del período ('IS', 'OOS', 'live_2026Q1', etc.)
    n_trades : int | None
    diagnostic_metrics : dict | None
        Win rate, profit factor, expectancy, etc. SOLO INFORMATIVOS.
    objective_spec : ObjectiveSpec | None
        Si se pasa, evalúa el veredicto formal.
    """
    r = returns.dropna()

    period_start = str(r.index[0]) if len(r) > 0 else None
    period_end = str(r.index[-1]) if len(r) > 0 else None

    periods_per_year = (
        objective_spec.periods_per_year if objective_spec else 252.0
    )
    n_trials = objective_spec.n_trials if objective_spec else 1
    sr_trials_std = (
        objective_spec.sr_trials_std if objective_spec else None
    )

    adv = compute_advanced_metrics(
        r,
        periods_per_year=periods_per_year,
        n_trials=n_trials,
        sr_trials_std=sr_trials_std,
    )

    obj_dict = None
    if objective_spec is not None:
        obj_result = evaluate_objective(
            r,
            spec=objective_spec,
            n_trades=n_trades,
            diagnostic_metrics=diagnostic_metrics,
        )
        obj_dict = obj_result.to_dict()

    return StrategyReport(
        label=label,
        period_start=period_start,
        period_end=period_end,
        n_returns=int(len(r)),
        n_trades=n_trades,
        advanced_metrics=adv,
        diagnostic_metrics=diagnostic_metrics or {},
        objective_result=obj_dict,
    )


# =====================================================================
# COMPARE IS vs OOS
# =====================================================================

def compare_is_oos(
    report_is: StrategyReport,
    report_oos: StrategyReport,
    metrics_to_compare: Sequence[str] = (
        "sharpe_annual",
        "psr",
        "dsr",
        "vol_annual",
        "tail_ratio",
    ),
) -> dict:
    """
    Cuantifica overfitting comparando métricas IS vs OOS.

    Returns
    -------
    dict con degradation por métrica:
        {metric_name: {'is': ..., 'oos': ..., 'haircut': ..., 'degradation_pct': ...}}
    """
    out = {}
    for m in metrics_to_compare:
        is_val = report_is.advanced_metrics.get(m)
        oos_val = report_oos.advanced_metrics.get(m)
        if is_val is None or oos_val is None:
            continue
        deg = is_oos_degradation(is_val, oos_val)
        out[m] = {
            "is": float(is_val),
            "oos": float(oos_val),
            **deg,
        }
    return out


# =====================================================================
# AGGREGATE FOLDS
# =====================================================================

def aggregate_folds(fold_reports: Sequence[FoldReport]) -> dict:
    """
    Estadísticas agregadas a través de folds de walk-forward.

    No promediamos Sharpes (estadísticamente incorrecto). Computamos:
    - mediana / IQR de Sharpe across folds
    - fracción de folds con Sharpe positivo (consistencia)
    - DSR mediano
    - peor fold (worst-case fold) — mide tail risk de generalización
    """
    if len(fold_reports) == 0:
        return {}

    sharpes = np.array([f.test_sharpe_annual for f in fold_reports])
    dsrs = np.array([f.test_dsr for f in fold_reports])
    mdds = np.array([f.test_max_drawdown for f in fold_reports])
    n_trades = np.array([f.test_n_trades for f in fold_reports])

    return {
        "n_folds": int(len(fold_reports)),
        "sharpe_median": float(np.median(sharpes)),
        "sharpe_iqr": [float(np.quantile(sharpes, 0.25)),
                       float(np.quantile(sharpes, 0.75))],
        "sharpe_min": float(sharpes.min()),
        "sharpe_max": float(sharpes.max()),
        "frac_folds_positive_sharpe": float((sharpes > 0).mean()),
        "dsr_median": float(np.median(dsrs)),
        "dsr_min": float(dsrs.min()),
        "max_drawdown_worst_fold": float(mdds.max()),
        "n_trades_total": int(n_trades.sum()),
        "n_trades_median_per_fold": float(np.median(n_trades)),
    }


# =====================================================================
# EXPORT
# =====================================================================

def export_report(
    report: StrategyReport | dict,
    path: str | Path,
    fmt: str = "json",
) -> Path:
    """Export a report to JSON or CSV."""
    path = Path(path)
    data = report.to_dict() if isinstance(report, StrategyReport) else report

    if fmt == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
    elif fmt == "csv":
        # Aplanamos un nivel
        flat = {}
        for k, v in data.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    flat[f"{k}__{kk}"] = vv
            else:
                flat[k] = v
        pd.DataFrame([flat]).to_csv(path, index=False)
    else:
        raise ValueError(f"fmt must be 'json' or 'csv', got '{fmt}'")
    return path


# =====================================================================
# PRINT
# =====================================================================

def print_report(report: StrategyReport) -> None:
    """Pretty-print de un StrategyReport en consola."""
    adv = report.advanced_metrics
    diag = report.diagnostic_metrics or {}

    print("=" * 75)
    print(f" REPORTE: {report.label}")
    print(f" Período: {report.period_start} → {report.period_end}")
    print(f" N retornos: {report.n_returns}"
          + (f"   |   N trades: {report.n_trades}" if report.n_trades is not None else ""))
    print("-" * 75)

    print(" >> MÉTRICAS PRIMARIAS (lo que importa)")
    print(f"   Sharpe (anual):     {adv.get('sharpe_annual', float('nan')):>8.3f}")
    print(f"   Sharpe SE (Mertens):{adv.get('sharpe_se_mertens_period', float('nan')):>8.4f}  (per-period)")
    boot = adv.get("sharpe_bootstrap_ci_annual", [float("nan"), float("nan")])
    print(f"   Bootstrap 95% CI:   [{boot[0]:.3f}, {boot[1]:.3f}]  (anual)")
    print(f"   PSR (vs SR=0):      {adv.get('psr', float('nan')):>8.3f}")
    print(f"   DSR (n_trials={adv.get('n_trials_used_for_dsr', 1)}):"
          f" {adv.get('dsr', float('nan')):>10.3f}")

    print("-" * 75)
    print(" >> DISTRIBUCIÓN DE RETORNOS")
    print(f"   Vol anual:          {adv.get('vol_annual', float('nan')):>8.4f}")
    print(f"   Skew:               {adv.get('skew', float('nan')):>8.3f}")
    print(f"   Kurtosis (raw):     {adv.get('kurtosis_raw', float('nan')):>8.3f}  "
          f"(excess: {adv.get('kurtosis_excess', float('nan')):.3f})")

    print("-" * 75)
    print(" >> RIESGO DE COLA")
    print(f"   VaR 5% (período):   {adv.get('var_5pct_period', float('nan')):>8.4f}")
    print(f"   CVaR 5% (período):  {adv.get('cvar_5pct_period', float('nan')):>8.4f}")
    print(f"   Tail ratio:         {adv.get('tail_ratio', float('nan')):>8.3f}")

    if diag:
        print("-" * 75)
        print(" >> DIAGNÓSTICO (NO se optimizan, solo informan)")
        for k, v in diag.items():
            try:
                print(f"   {k:20s} {float(v):>8.4f}")
            except (TypeError, ValueError):
                print(f"   {k:20s} {v}")

    if report.objective_result:
        obj = report.objective_result
        print("-" * 75)
        print(" >> VEREDICTO DE LA FUNCIÓN OBJETIVO")
        print(f"   Primaria:           {obj['primary_metric']} = {obj['primary_value']:.4f}")
        print(f"   Umbral:             {obj['primary_threshold']:.4f}")
        print(f"   Pasa primaria:      {'SÍ' if obj['passed_primary'] else 'NO'}")
        print(f"   Pasa constraints:   {'SÍ' if obj['passed_constraints'] else 'NO'}")
        if obj.get("constraint_violations"):
            print("   Violaciones:")
            for v in obj["constraint_violations"]:
                print(f"      - {v}")
        print(f"   VEREDICTO:          {obj['verdict']}")
    print("=" * 75)
