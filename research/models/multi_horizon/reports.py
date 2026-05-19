"""
JSON report writer for multi-horizon training results.

Writes one report per horizon to research/reports/multi_horizon_v1/.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REPORTS_DIR = Path(__file__).parents[3] / "reports" / "multi_horizon_v1"


@dataclass
class AblativeEntry:
    """DSR outcome for one ablation configuration."""

    label: str
    dsr: float


@dataclass
class HorizonReport:
    """Full JSON-serialisable report for one horizon."""

    horizon: str
    model: str
    version: str
    as_of: str
    training_window: list[str]
    universe_size: int

    # OOS metrics
    psr: float
    dsr: float
    ece: float
    brier: float
    n_trades_oos: int
    win_rate: float
    sharpe_oos: float

    hyperparams: dict[str, Any]
    feature_importance_top10: dict[str, float]
    ablative: list[AblativeEntry]
    artifact_path: str
    seed: int

    # Promotion decision
    promoted: bool = False
    promotion_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ablative"] = {a["label"]: {"dsr": a["dsr"]} for a in d["ablative"]}
        return d


def write_horizon_report(report: HorizonReport) -> Path:
    """
    Persist a HorizonReport to research/reports/multi_horizon_v1/.

    Returns the path written.
    """
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{report.horizon}_{report.model}_v{report.version}.json"
    path = _REPORTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)
    logger.info("Report written: %s", path)
    return path


def write_ablative_summary(
    ablative_results: dict[str, dict[str, float]],
    version: str = "v1",
) -> Path:
    """
    Write cross-horizon ablative analysis summary.

    Parameters
    ----------
    ablative_results : dict
        {horizon_name: {ablation_label: dsr_value}}
    version : str
    """
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _REPORTS_DIR / f"ablative_{version}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ablative_results, f, indent=2)
    logger.info("Ablative summary written: %s", path)
    return path
