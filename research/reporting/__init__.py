"""
Reporting module: standardized strategy evaluation reports.

Genera reportes estandarizados que comparan IS vs OOS, agregan métricas por
fold de walk-forward, y exportan a JSON/CSV para análisis posterior.
"""
from .report import (
    StrategyReport,
    FoldReport,
    build_report,
    aggregate_folds,
    compare_is_oos,
    export_report,
    print_report,
)

__all__ = [
    "StrategyReport",
    "FoldReport",
    "build_report",
    "aggregate_folds",
    "compare_is_oos",
    "export_report",
    "print_report",
]
