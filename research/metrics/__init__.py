"""
Advanced metrics module.

Complementa las métricas básicas (Sharpe/Sortino/Calmar/MDD) ya presentes en
backtesting/engine.py con métricas estadísticamente rigurosas:

- Probabilistic Sharpe Ratio (Bailey & López de Prado, 2012)
- Deflated Sharpe Ratio (Bailey & López de Prado, 2014) — corrige selección bias
- Sharpe SE con corrección de momentos superiores (Mertens, 2002)
- Bootstrap confidence intervals para cualquier métrica
- Tail risk (VaR, CVaR, tail ratio)
- Capacity / efficiency (turnover, cost drag)
- Función objetivo formal con hard constraints (objective.py)
"""

from .advanced import (
    # Higher moments
    returns_skewness,
    returns_kurtosis,
    # Sharpe family
    sharpe_ratio_se_mertens,
    probabilistic_sharpe_ratio,
    deflated_sharpe_ratio,
    # Bootstrap
    bootstrap_sharpe_ci,
    bootstrap_metric_ci,
    # Tail risk
    value_at_risk,
    conditional_var,
    tail_ratio,
    # Capacity
    turnover,
    cost_drag,
    # IS vs OOS
    is_oos_degradation,
    # Compute everything in one go
    compute_advanced_metrics,
)
from .objective import (
    HardConstraints,
    ObjectiveSpec,
    ObjectiveResult,
    evaluate_objective,
)

__all__ = [
    "returns_skewness",
    "returns_kurtosis",
    "sharpe_ratio_se_mertens",
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
    "bootstrap_sharpe_ci",
    "bootstrap_metric_ci",
    "value_at_risk",
    "conditional_var",
    "tail_ratio",
    "turnover",
    "cost_drag",
    "is_oos_degradation",
    "compute_advanced_metrics",
    "HardConstraints",
    "ObjectiveSpec",
    "ObjectiveResult",
    "evaluate_objective",
]
