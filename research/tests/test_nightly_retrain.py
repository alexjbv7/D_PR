"""
Tests for NightlyRetrainDAG.

Coverage:
  - _check_gates: all 4 gates + happy path (no external deps)
  - _write_run_log: JSON round-trip
  - _uuid7: valid UUID format, time-ordered
  - dry_run: no training, returns exit_code=0
"""
from __future__ import annotations

import asyncio
import json
import sys
import uuid
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

_REPO = Path(__file__).parents[2]
for _p in [str(_REPO / "research"), str(_REPO / "shared")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.multi_horizon.trainer import TrainResult
from models.walk_forward_runner import WalkForwardConfig, WalkForwardResult
from quant_shared.models.registry import ModelCard, ModelRegistry
from pipelines.nightly_retrain import (
    HorizonGateResult,
    NightlyRetrainConfig,
    NightlyRetrainDAG,
    RetrainRunLog,
    _uuid7,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _dummy_wf() -> WalkForwardResult:
    return WalkForwardResult(
        config=WalkForwardConfig(),
        fold_results=[],
        oos_signals=pd.Series(dtype=float),
        oos_proba=pd.DataFrame(),
        oos_sizing=pd.DataFrame(),
        feature_importance_agg=pd.DataFrame(),
        features_to_drop=[],
        global_metrics={},
    )


def _make_result(
    horizon: str = "swing",
    dsr: float = 0.5,
    ece: float = 0.03,
    class_collapse: bool = False,
) -> TrainResult:
    return TrainResult(
        horizon_name=horizon,
        model_object=object(),
        wf_result=_dummy_wf(),
        psr=0.6, dsr=dsr, ece=ece,
        sharpe_oos=1.0, win_rate=0.55, n_trades=200,
        best_params={},
        promoted=(dsr >= 0.4 and not class_collapse),
        class_collapse=class_collapse,
    )


def _make_prod_card(dsr: float = 0.5) -> ModelCard:
    return ModelCard(
        model_id="prod-swing",
        name="multi_horizon_swing",
        version="0.1",
        model_class="xgboost",
        artifact_path="/tmp/fake.pkl",
        dsr=dsr,
        status="production",
    )


def _make_dag(tmp_path: Path) -> NightlyRetrainDAG:
    registry = ModelRegistry(path=tmp_path / "registry.json")
    config = NightlyRetrainConfig(
        as_of=date(2026, 6, 3),
        run_log_dir=tmp_path / "runs",
    )
    return NightlyRetrainDAG(config=config, registry=registry)


# ---------------------------------------------------------------------------
# _uuid7
# ---------------------------------------------------------------------------


def test_uuid7_is_valid_uuid():
    uid = _uuid7()
    parsed = uuid.UUID(uid)
    assert parsed.version == 7


def test_uuid7_time_ordered():
    ids = [_uuid7() for _ in range(10)]
    assert ids == sorted(ids), "UUID v7 values must be time-ordered (ascending)"


def test_uuid7_unique():
    ids = {_uuid7() for _ in range(100)}
    assert len(ids) == 100


# ---------------------------------------------------------------------------
# _check_gates — gate 1: DSR floor
# ---------------------------------------------------------------------------


def test_gate_dsr_below_min(tmp_path: Path):
    dag = _make_dag(tmp_path)
    result = _make_result(dsr=0.39)
    gate = dag._check_gates(result, prod_card=None)
    assert gate.promoted is False
    assert gate.skip_reason == "dsr_below_min"


def test_gate_dsr_exactly_at_min_passes(tmp_path: Path):
    dag = _make_dag(tmp_path)
    result = _make_result(dsr=0.4, ece=0.03)
    gate = dag._check_gates(result, prod_card=None)
    assert gate.promoted is True
    assert gate.skip_reason == ""


# ---------------------------------------------------------------------------
# _check_gates — gate 2: ECE ceiling
# ---------------------------------------------------------------------------


def test_gate_ece_above_max(tmp_path: Path):
    dag = _make_dag(tmp_path)
    result = _make_result(dsr=0.5, ece=0.06)
    gate = dag._check_gates(result, prod_card=None)
    assert gate.promoted is False
    assert gate.skip_reason == "ece_above_max"


def test_gate_dsr_fails_before_ece(tmp_path: Path):
    """Gate 1 (DSR) must be checked before gate 2 (ECE)."""
    dag = _make_dag(tmp_path)
    result = _make_result(dsr=0.1, ece=0.99)
    gate = dag._check_gates(result, prod_card=None)
    assert gate.skip_reason == "dsr_below_min"


# ---------------------------------------------------------------------------
# _check_gates — gate 3: class collapse
# ---------------------------------------------------------------------------


def test_gate_class_collapse(tmp_path: Path):
    dag = _make_dag(tmp_path)
    result = _make_result(dsr=0.5, ece=0.02, class_collapse=True)
    gate = dag._check_gates(result, prod_card=None)
    assert gate.promoted is False
    assert gate.skip_reason == "class_collapse"


def test_gate_ece_fails_before_collapse(tmp_path: Path):
    """Gate 2 (ECE) must be checked before gate 3 (class_collapse)."""
    dag = _make_dag(tmp_path)
    result = _make_result(dsr=0.5, ece=0.99, class_collapse=True)
    gate = dag._check_gates(result, prod_card=None)
    assert gate.skip_reason == "ece_above_max"


# ---------------------------------------------------------------------------
# _check_gates — gate 4: DSR regression vs production
# ---------------------------------------------------------------------------


def test_gate_dsr_regression(tmp_path: Path):
    dag = _make_dag(tmp_path)
    prod_card = _make_prod_card(dsr=0.60)
    # new DSR = 0.50 < 0.60 * 0.95 = 0.57
    result = _make_result(dsr=0.50, ece=0.02)
    gate = dag._check_gates(result, prod_card=prod_card)
    assert gate.promoted is False
    assert gate.skip_reason == "dsr_regression"
    assert gate.dsr_prod == pytest.approx(0.60)


def test_gate_no_prod_model_skips_regression(tmp_path: Path):
    """Gate 4 must not fire when there is no production model."""
    dag = _make_dag(tmp_path)
    result = _make_result(dsr=0.42, ece=0.03)
    gate = dag._check_gates(result, prod_card=None)
    assert gate.promoted is True
    assert gate.dsr_prod is None


def test_gate_dsr_barely_passes_regression(tmp_path: Path):
    dag = _make_dag(tmp_path)
    prod_card = _make_prod_card(dsr=0.60)
    # new DSR = 0.57 >= 0.60 * 0.95 = 0.57 (boundary — must pass)
    result = _make_result(dsr=0.570, ece=0.02)
    gate = dag._check_gates(result, prod_card=prod_card)
    assert gate.promoted is True


# ---------------------------------------------------------------------------
# _check_gates — happy path
# ---------------------------------------------------------------------------


def test_gate_all_pass_returns_promoted(tmp_path: Path):
    dag = _make_dag(tmp_path)
    prod_card = _make_prod_card(dsr=0.50)
    result = _make_result(dsr=0.60, ece=0.02)
    gate = dag._check_gates(result, prod_card=prod_card)
    assert gate.promoted is True
    assert gate.skip_reason == ""
    assert gate.dsr_new == pytest.approx(0.60)
    assert gate.ece_new == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# _write_run_log
# ---------------------------------------------------------------------------


def test_write_run_log_creates_json(tmp_path: Path):
    dag = _make_dag(tmp_path)
    log = RetrainRunLog(
        run_id="test-run-id",
        as_of="2026-06-03",
        started_at="2026-06-03T02:00:00+00:00",
        finished_at="2026-06-03T04:00:00+00:00",
        horizons=[
            HorizonGateResult(
                horizon="swing", dsr_new=0.55, dsr_prod=0.50,
                ece_new=0.03, promoted=True,
            )
        ],
        n_promoted=1,
        exit_code=0,
    )
    path = dag._write_run_log(log)
    assert path.exists()
    assert path.name == "test-run-id.json"

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["run_id"] == "test-run-id"
    assert data["n_promoted"] == 1
    assert data["exit_code"] == 0
    assert data["horizons"][0]["horizon"] == "swing"
    assert data["horizons"][0]["promoted"] is True


def test_write_run_log_creates_parent_dir(tmp_path: Path):
    """run_log_dir is created automatically if it does not exist."""
    nested = tmp_path / "a" / "b" / "runs"
    registry = ModelRegistry(path=tmp_path / "registry.json")
    config = NightlyRetrainConfig(as_of=date(2026, 6, 3), run_log_dir=nested)
    dag = NightlyRetrainDAG(config=config, registry=registry)

    log = RetrainRunLog(
        run_id="x",
        as_of="2026-06-03",
        started_at="2026-06-03T02:00:00+00:00",
        finished_at="2026-06-03T02:05:00+00:00",
        horizons=[],
        n_promoted=0,
        exit_code=2,
    )
    path = dag._write_run_log(log)
    assert path.exists()


# ---------------------------------------------------------------------------
# dry_run integration
# ---------------------------------------------------------------------------


def test_dry_run_returns_no_promotion(tmp_path: Path):
    registry = ModelRegistry(path=tmp_path / "registry.json")
    config = NightlyRetrainConfig(
        as_of=date(2026, 6, 3),
        dry_run=True,
        run_log_dir=tmp_path / "runs",
    )
    dag = NightlyRetrainDAG(config=config, registry=registry)
    log = asyncio.run(dag.run())

    assert log.n_promoted == 0
    assert log.exit_code == 0  # dry_run always exits 0
    assert all(g.skip_reason == "dry_run" for g in log.horizons)
    assert (tmp_path / "runs" / f"{log.run_id}.json").exists()


def test_dry_run_produces_one_gate_per_horizon(tmp_path: Path):
    registry = ModelRegistry(path=tmp_path / "registry.json")
    config = NightlyRetrainConfig(
        as_of=date(2026, 6, 3),
        dry_run=True,
        horizons=["swing", "daily"],
        run_log_dir=tmp_path / "runs",
    )
    dag = NightlyRetrainDAG(config=config, registry=registry)
    log = asyncio.run(dag.run())

    assert len(log.horizons) == 2
    assert {g.horizon for g in log.horizons} == {"swing", "daily"}
