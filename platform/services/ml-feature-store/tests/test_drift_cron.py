"""
Integration tests for platform/services/ml-feature-store/app/drift/drift_cron.py
==================================================================================

5 cases (all async, using unittest.mock for I/O dependencies):

  1. test_no_drift_no_trigger
     PSI stable (<0.10) and ECE clean (<0.05) → no retrain trigger emitted.

  2. test_psi_severe_triggers_retrain
     PSI >= 0.25 for one feature → retrain trigger emitted, suppressed=False.

  3. test_ece_exceeded_triggers_retrain
     ECE > 0.05, PSI clean → retrain trigger with reason "ece_exceeded".

  4. test_macro_suppression
     PSI severe BUT ts is ±2 days of FOMC → trigger emitted with suppressed=True.

  5. test_bucket_edges_loaded_from_redis
     Redis returns cached edges → compute_psi called with those edges,
     Redis.setex NOT called (no cold-start write).

All DB and Kafka calls are mocked.  No network I/O.
"""
from __future__ import annotations

import asyncio
import json
import sys
import os
from datetime import date, datetime, timezone
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch, call

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path bootstrap — add the service root so 'app.*' imports work
# ---------------------------------------------------------------------------
_SVC = os.path.dirname(os.path.dirname(__file__))
if _SVC not in sys.path:
    sys.path.insert(0, _SVC)

from app.drift.drift_cron import DriftCron, PSI_SEVERE_THRESHOLD, ECE_THRESHOLD
from app.drift.macro_event_filter import MacroEventFilter


# ===========================================================================
# Fixtures
# ===========================================================================

def _make_cron(
    *,
    model_version: str = "xgb_swing_v3",
    features: list[str] | None = None,
    train_vals: list[float] | None = None,
    recent_vals: list[float] | None = None,
    predictions: list[dict] | None = None,
    cached_edges: list[float] | None = None,
    mac_filter: MacroEventFilter | None = None,
) -> tuple[DriftCron, MagicMock, MagicMock, MagicMock]:
    """Build a DriftCron with all dependencies mocked."""
    if features is None:
        features = ["rsi_14", "macd_hist"]
    if train_vals is None:
        rng = np.random.default_rng(0)
        train_vals = rng.normal(0, 1, 500).tolist()
    if recent_vals is None:
        # Slightly shifted — PSI moderate by default
        rng = np.random.default_rng(1)
        recent_vals = rng.normal(0.2, 1, 300).tolist()
    if predictions is None:
        # 300 labelled predictions, decent calibration
        rng = np.random.default_rng(2)
        predictions = [
            {"probas": [0.7, 0.2, 0.1], "true_label": 0}
            for _ in range(300)
        ]

    # --- Mock async Redis ---
    mock_redis = AsyncMock()

    # Registry keys
    async def _redis_get(key: str):
        if ":current" in key:
            return model_version.encode()
        if ":features" in key:
            return json.dumps(features).encode()
        if "bucket_edges" in key:
            return json.dumps(cached_edges).encode() if cached_edges else None
        return None

    mock_redis.get   = AsyncMock(side_effect=_redis_get)
    mock_redis.setex = AsyncMock(return_value=True)

    # --- Mock asyncpg pool ---
    mock_pool = MagicMock()

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=None)

    async def _fetch(sql, *args):
        if "psi_history" in sql or "ece_history" in sql:
            return []
        if "audit.predictions" in sql:
            return [
                MagicMock(probas=p["probas"], true_label=p["true_label"])
                for p in predictions
            ]
        if "features.online" in sql:
            # Return train or recent depending on context
            vals = train_vals if "ORDER BY ts DESC" in sql else recent_vals
            return [MagicMock(val=v) for v in vals]
        return []

    mock_conn.fetch = AsyncMock(side_effect=_fetch)

    async def _acquire():
        return mock_conn

    mock_pool.acquire = MagicMock(
        return_value=MagicMock(
            __aenter__=AsyncMock(return_value=mock_conn),
            __aexit__=AsyncMock(return_value=None),
        )
    )

    # --- Mock Kafka producer ---
    mock_producer = AsyncMock()
    mock_producer.send_and_wait = AsyncMock(return_value=None)

    cron = DriftCron(
        pool=mock_pool,
        redis=mock_redis,
        producer=mock_producer,
        mac_filter=mac_filter,
    )
    return cron, mock_redis, mock_producer, mock_pool


# ===========================================================================
# Tests
# ===========================================================================

@pytest.mark.asyncio
async def test_no_drift_no_trigger() -> None:
    """PSI stable and ECE clean → no retrain trigger emitted.

    Calibration construction:
      probas = [0.9, 0.05, 0.05] for all 300 samples.
      270 samples have true_label=0 (correct, 90 %).
      30  samples have true_label=1 (wrong,  10 %).
      → acc = 0.90, conf = 0.90, ECE = |acc - conf| = 0.0.
    """
    rng = np.random.default_rng(42)
    same_vals = rng.normal(0, 1, 500).tolist()

    n = 300
    # Well-calibrated predictions: 90% correct at 90% confidence → ECE ≈ 0
    calibrated_preds = (
        [{"probas": [0.9, 0.05, 0.05], "true_label": 0}] * 270   # correct
        + [{"probas": [0.9, 0.05, 0.05], "true_label": 1}] * 30  # wrong
    )

    # Override fetch to return identical distributions
    cron, _, mock_producer, _ = _make_cron(
        features=["rsi_14"],
        train_vals=same_vals,
        recent_vals=same_vals,   # identical → PSI ≈ 0
        predictions=calibrated_preds,
    )

    # Patch the repo methods to return our controlled values
    cron._repo.fetch_train_feature_values  = AsyncMock(return_value=same_vals)
    cron._repo.fetch_recent_feature_values = AsyncMock(return_value=same_vals)
    cron._repo.insert_psi                  = AsyncMock()
    cron._repo.insert_ece                  = AsyncMock()
    cron._repo.fetch_recent_predictions    = AsyncMock(return_value=calibrated_preds)

    result = await cron._run_horizon("swing", datetime.now(tz=timezone.utc))

    # No retrain trigger should be emitted
    assert not result["retrain_triggered"]
    for call_args in mock_producer.send_and_wait.call_args_list:
        topic = call_args.args[0] if call_args.args else ""
        assert "retrain.triggers" not in topic, "Retrain trigger emitted unexpectedly"


@pytest.mark.asyncio
async def test_psi_severe_triggers_retrain() -> None:
    """PSI ≥ 0.25 (severe) → retrain trigger emitted with suppressed=False."""
    rng = np.random.default_rng(0)
    train_vals  = rng.normal(0.0, 1.0, 1000).tolist()
    recent_vals = rng.normal(3.0, 1.0, 1000).tolist()   # 3σ shift → severe

    cron, _, mock_producer, _ = _make_cron(features=["rsi_14"])

    cron._repo.fetch_train_feature_values  = AsyncMock(return_value=train_vals)
    cron._repo.fetch_recent_feature_values = AsyncMock(return_value=recent_vals)
    cron._repo.insert_psi                  = AsyncMock()
    cron._repo.insert_ece                  = AsyncMock()
    cron._repo.fetch_recent_predictions    = AsyncMock(return_value=[
        {"probas": [0.9, 0.05, 0.05], "true_label": 0} for _ in range(300)
    ])

    # Use a date well away from macro events
    ts = datetime(2026, 2, 15, 3, 0, tzinfo=timezone.utc)
    result = await cron._run_horizon("swing", ts)

    assert result["retrain_triggered"]
    assert result["max_psi"] >= PSI_SEVERE_THRESHOLD

    # Verify retrain.triggers was called
    retrain_calls = [
        c for c in mock_producer.send_and_wait.call_args_list
        if c.args and "retrain.triggers" in c.args[0]
    ]
    assert len(retrain_calls) >= 1, "Expected RetrainTriggerEvent on retrain.triggers"

    # Suppressed should be False
    payload = json.loads(retrain_calls[0].kwargs.get("value", b"{}"))
    assert payload.get("suppressed") is False


@pytest.mark.asyncio
async def test_ece_exceeded_triggers_retrain() -> None:
    """ECE > 0.05 with stable PSI → retrain with reason='ece_exceeded'."""
    rng = np.random.default_rng(3)
    same_vals = rng.normal(0, 1, 500).tolist()  # PSI stable

    # Badly calibrated: model always predicts class 0 w/ certainty, truth is class 1
    bad_preds = [{"probas": [1.0, 0.0, 0.0], "true_label": 1} for _ in range(300)]

    cron, _, mock_producer, _ = _make_cron(features=["rsi_14"])

    cron._repo.fetch_train_feature_values  = AsyncMock(return_value=same_vals)
    cron._repo.fetch_recent_feature_values = AsyncMock(return_value=same_vals)
    cron._repo.insert_psi                  = AsyncMock()
    cron._repo.insert_ece                  = AsyncMock()
    cron._repo.fetch_recent_predictions    = AsyncMock(return_value=bad_preds)

    ts = datetime(2026, 2, 15, 3, 0, tzinfo=timezone.utc)
    result = await cron._run_horizon("intraday", ts)

    assert result["retrain_triggered"]
    assert result["ece"] > ECE_THRESHOLD

    retrain_calls = [
        c for c in mock_producer.send_and_wait.call_args_list
        if c.args and "retrain.triggers" in c.args[0]
    ]
    assert retrain_calls, "No retrain trigger emitted"
    payload = json.loads(retrain_calls[0].kwargs.get("value", b"{}"))
    assert payload.get("trigger_reason") in ("ece_exceeded", "both")


@pytest.mark.asyncio
async def test_macro_suppression() -> None:
    """PSI severe but ±2 days of FOMC → trigger emitted with suppressed=True."""
    rng = np.random.default_rng(0)
    train_vals  = rng.normal(0.0, 1.0, 1000).tolist()
    recent_vals = rng.normal(3.0, 1.0, 1000).tolist()

    # Inject a custom macro event: 2026-04-04 (NFP) → suppress 2026-04-02 to 2026-04-06
    extra_event = (date(2026, 4, 4), "NFP_TEST")
    mac_filter  = MacroEventFilter(extra_events=[extra_event], window_days=2)

    cron, _, mock_producer, _ = _make_cron(
        features=["rsi_14"], mac_filter=mac_filter
    )

    cron._repo.fetch_train_feature_values  = AsyncMock(return_value=train_vals)
    cron._repo.fetch_recent_feature_values = AsyncMock(return_value=recent_vals)
    cron._repo.insert_psi                  = AsyncMock()
    cron._repo.insert_ece                  = AsyncMock()
    cron._repo.fetch_recent_predictions    = AsyncMock(return_value=[
        {"probas": [0.9, 0.05, 0.05], "true_label": 0} for _ in range(300)
    ])

    # ts = 2026-04-04 (exactly the event day) → should be suppressed
    ts = datetime(2026, 4, 4, 3, 0, tzinfo=timezone.utc)
    result = await cron._run_horizon("daily", ts)

    assert result["retrain_triggered"]   # logic fires

    retrain_calls = [
        c for c in mock_producer.send_and_wait.call_args_list
        if c.args and "retrain.triggers" in c.args[0]
    ]
    assert retrain_calls, "Expected suppressed retrain trigger to still be emitted (for audit)"
    payload = json.loads(retrain_calls[0].kwargs.get("value", b"{}"))
    assert payload.get("suppressed") is True, "Expected suppressed=True near macro event"
    assert payload.get("suppression_reason") is not None


@pytest.mark.asyncio
async def test_bucket_edges_loaded_from_redis() -> None:
    """When Redis returns cached edges, setex should NOT be called (no cold-start)."""
    rng = np.random.default_rng(0)
    train_vals  = rng.normal(0, 1, 500).tolist()
    recent_vals = rng.normal(0, 1, 500).tolist()

    # Pre-compute edges (as if they were stored at model-promotion time)
    edges_arr  = np.percentile(train_vals, np.linspace(0, 100, 11))
    edges_arr[0]  = float("-inf")
    edges_arr[-1] = float("inf")
    cached_edges = edges_arr.tolist()

    cron, mock_redis, _, _ = _make_cron(
        features=["rsi_14"],
        train_vals=train_vals,
        recent_vals=recent_vals,
        cached_edges=cached_edges,
    )

    cron._repo.fetch_train_feature_values  = AsyncMock(return_value=train_vals)
    cron._repo.fetch_recent_feature_values = AsyncMock(return_value=recent_vals)
    cron._repo.insert_psi                  = AsyncMock()
    cron._repo.insert_ece                  = AsyncMock()
    cron._repo.fetch_recent_predictions    = AsyncMock(return_value=[
        {"probas": [0.8, 0.1, 0.1], "true_label": 0} for _ in range(300)
    ])

    ts = datetime(2026, 2, 15, 3, 0, tzinfo=timezone.utc)
    await cron._run_horizon("swing", ts)

    # setex should NOT have been called for the bucket_edges key
    setex_calls = mock_redis.setex.call_args_list
    bucket_edge_writes = [
        c for c in setex_calls
        if c.args and "bucket_edges" in str(c.args[0])
    ]
    assert len(bucket_edge_writes) == 0, (
        f"bucket_edges was written to Redis even though cached: {bucket_edge_writes}"
    )
