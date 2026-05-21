"""
DriftCron — daily drift detection pipeline, fires at 03:00 UTC.

Pipeline per horizon (intraday / swing / daily)
-----------------------------------------------
1.  Load model_version + top-20 feature names from Redis model registry.
2.  For each feature:
    a.  Load bucket_edges from Redis (BucketEdgesRepository).
        If missing: compute from train values and persist (cold-start).
    b.  Fetch train_values from TimescaleDB (historical reference).
    c.  Fetch recent_values (last 7 days) from TimescaleDB.
    d.  Skip feature if either slice has fewer than MIN_SAMPLES points.
    e.  compute_psi(train_values, recent_values, bucket_edges=edges).
    f.  Persist PSI → drift.psi_history.
    g.  If PSI ≥ PSI_ALERT_THRESHOLD: emit DriftDetectedEvent.
3.  Fetch rolling 7-day labelled predictions → compute_ece + compute_brier.
4.  Persist ECE → drift.ece_history.
5.  If ECE > ECE_THRESHOLD: emit ECEDriftEvent.
6.  Determine RetrainTriggerEvent need:
    psi_trigger  = max_psi ≥ PSI_SEVERE_THRESHOLD
    ece_trigger  = ece > ECE_THRESHOLD
    If psi_trigger OR ece_trigger:
        reason   = "psi_severe" | "ece_exceeded" | "both"
        Check macro_event_filter.is_suppressed(now) → set suppressed flag.
        Emit RetrainTriggerEvent (always emitted, suppressed flag controls action).

Thresholds
----------
PSI_ALERT_THRESHOLD  = 0.10  (moderate → P2 alert)
PSI_SEVERE_THRESHOLD = 0.25  (severe   → P1 alert + retrain candidate)
ECE_THRESHOLD        = 0.05  (exceeds well-calibrated threshold)
MIN_SAMPLES          = 200   (minimum samples to compute meaningful PSI)
"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import structlog

from .drift_repository  import DriftRepository
from .macro_event_filter import MacroEventFilter
from .alert_emitter     import AlertEmitter
from .metrics import (
    drift_ece_value,
    drift_events_total,
    drift_psi_value,
    drift_retrain_triggers_total,
)
from ._math import compute_psi, severity_label, compute_ece, compute_brier

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PSI_ALERT_THRESHOLD  = float(os.getenv("DRIFT_PSI_ALERT",  "0.10"))
PSI_SEVERE_THRESHOLD = float(os.getenv("DRIFT_PSI_SEVERE", "0.25"))
ECE_THRESHOLD        = float(os.getenv("DRIFT_ECE_THRESH", "0.05"))
MIN_SAMPLES          = int(os.getenv("DRIFT_MIN_SAMPLES",  "200"))
ECE_WINDOW_DAYS      = int(os.getenv("DRIFT_ECE_WINDOW",   "7"))
PSI_WINDOW_DAYS      = int(os.getenv("DRIFT_PSI_WINDOW",   "7"))
TRAIN_WINDOW_DAYS    = int(os.getenv("DRIFT_TRAIN_WINDOW", "90"))
N_TOP_FEATURES       = int(os.getenv("DRIFT_TOP_FEATURES", "20"))

_HORIZONS = ["intraday", "swing", "daily"]

_CRON_HOUR_UTC   = 3   # 03:00 UTC
_CRON_MINUTE_UTC = 0


# ---------------------------------------------------------------------------
# Redis key helpers (sync — called from async context via run_in_executor)
# ---------------------------------------------------------------------------

def _redis_key_model(horizon: str) -> str:
    return f"model_registry:{horizon}:current"


def _redis_key_features(horizon: str) -> str:
    return f"model_registry:{horizon}:features"


def _redis_key_bucket_edges(model_version: str, feature_name: str) -> str:
    return f"drift:bucket_edges:{model_version}:{feature_name}"


# ---------------------------------------------------------------------------
# DriftCron
# ---------------------------------------------------------------------------

class DriftCron:
    """Orchestrates daily drift detection.

    Parameters
    ----------
    pool        : asyncpg.Pool — TimescaleDB connection pool.
    redis       : aioredis / redis.asyncio client — async Redis.
    producer    : aiokafka.AIOKafkaProducer — started Kafka producer.
    mac_filter  : MacroEventFilter — optional; created with defaults if None.
    """

    def __init__(
        self,
        pool:       Any,
        redis:      Any,
        producer:   Any,
        mac_filter: MacroEventFilter | None = None,
    ) -> None:
        self._repo       = DriftRepository(pool)
        self._redis      = redis
        self._emitter    = AlertEmitter(producer)
        self._mac_filter = mac_filter or MacroEventFilter()

    # ------------------------------------------------------------------
    # Scheduler loop — run as asyncio.create_task
    # ------------------------------------------------------------------

    async def run_loop(self) -> None:
        """Infinite loop: wait until 03:00 UTC, run pipeline, repeat."""
        logger.info("drift_cron.loop_started")
        while True:
            await self._sleep_until_next_run()
            try:
                await self.run_once()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("drift_cron.run_once.error", error=str(exc))

    async def _sleep_until_next_run(self) -> None:
        """Sleep until the next 03:00 UTC."""
        now   = datetime.now(tz=timezone.utc)
        next_ = now.replace(
            hour=_CRON_HOUR_UTC, minute=_CRON_MINUTE_UTC,
            second=0, microsecond=0,
        )
        if now >= next_:
            # Already past 03:00 today → wait for tomorrow
            from datetime import timedelta
            next_ = next_.replace(day=next_.day) + timedelta(days=1)
        delay = (next_ - now).total_seconds()
        logger.info("drift_cron.next_run", seconds_until=round(delay))
        await asyncio.sleep(delay)

    # ------------------------------------------------------------------
    # Main pipeline (also callable directly for testing / manual runs)
    # ------------------------------------------------------------------

    async def run_once(self) -> dict[str, Any]:
        """Execute the drift detection pipeline for all horizons.

        Returns a summary dict (horizon → results) for observability.
        """
        now     = datetime.now(tz=timezone.utc)
        summary: dict[str, Any] = {}

        for horizon in _HORIZONS:
            logger.info("drift_cron.horizon.start", horizon=horizon, ts=now.isoformat())
            try:
                result = await self._run_horizon(horizon, now)
                summary[horizon] = result
            except Exception as exc:
                logger.error("drift_cron.horizon.error", horizon=horizon, error=str(exc))
                summary[horizon] = {"error": str(exc)}

        logger.info("drift_cron.run_complete", horizons=list(summary.keys()))
        return summary

    async def _run_horizon(self, horizon: str, now: datetime) -> dict[str, Any]:
        model_version, feature_names = await self._load_model_registry(horizon)
        if not model_version or not feature_names:
            logger.warning("drift_cron.no_model_registry", horizon=horizon)
            return {"skipped": "no_model_registry"}

        # ----------------------------------------------------------------
        # 1. PSI per feature
        # ----------------------------------------------------------------
        psi_results: dict[str, float]  = {}
        severity_map: dict[str, str]   = {}

        for feat in feature_names[:N_TOP_FEATURES]:
            train_vals  = await self._repo.fetch_train_feature_values(
                feat, horizon, model_version, limit=5000
            )
            recent_vals = await self._repo.fetch_recent_feature_values(
                feat, window_days=PSI_WINDOW_DAYS, limit=2000
            )
            if len(train_vals) < MIN_SAMPLES or len(recent_vals) < MIN_SAMPLES:
                logger.debug(
                    "drift_cron.insufficient_samples",
                    horizon=horizon, feature=feat,
                    n_train=len(train_vals), n_recent=len(recent_vals),
                )
                continue

            # Load or compute bucket edges (anti-leakage)
            edges = await self._load_or_create_edges(
                model_version, feat, np.array(train_vals)
            )

            psi, _ = compute_psi(train_vals, recent_vals, bucket_edges=edges)
            sev    = severity_label(psi)
            psi_results[feat]  = psi
            severity_map[feat] = sev

            drift_psi_value.labels(horizon=horizon, feature=feat).set(psi)

            macro_supp = (
                sev == "severe" and self._mac_filter.is_suppressed(now)
            )

            # Persist
            await self._repo.insert_psi(
                ts=now, horizon=horizon, model_version=model_version,
                feature_name=feat, psi=psi, severity=sev,
                macro_suppressed=macro_supp,
            )

            if sev in ("moderate", "severe"):
                drift_events_total.labels(horizon=horizon, severity=sev).inc()

            # Alert if moderate or severe
            if psi >= PSI_ALERT_THRESHOLD:
                await self._emitter.emit_psi_drift(
                    horizon=horizon, model_version=model_version,
                    feature_name=feat, psi=psi, severity=sev,
                    train_window_days=TRAIN_WINDOW_DAYS,
                    recent_window_days=PSI_WINDOW_DAYS,
                )

        # ----------------------------------------------------------------
        # 2. ECE (rolling 7-day window)
        # ----------------------------------------------------------------
        rows = await self._repo.fetch_recent_predictions(
            horizon=horizon, model_version=model_version,
            window_days=ECE_WINDOW_DAYS,
        )
        ece_val  = 0.0
        brier_val = 0.0
        n_samples = len(rows)

        if n_samples >= MIN_SAMPLES:
            probas_arr = np.array([r["probas"]    for r in rows], dtype=float)
            labels_arr = np.array([r["true_label"] for r in rows], dtype=int)
            ece_val   = compute_ece(probas_arr, labels_arr)
            brier_val = compute_brier(probas_arr, labels_arr)
            drift_ece_value.labels(horizon=horizon).set(ece_val)

            await self._repo.insert_ece(
                ts=now, horizon=horizon, model_version=model_version,
                ece=ece_val, brier=brier_val, n_samples=n_samples,
            )

            if ece_val > ECE_THRESHOLD:
                await self._emitter.emit_ece_drift(
                    horizon=horizon, model_version=model_version,
                    ece=ece_val, brier=brier_val, n_samples=n_samples,
                    window_days=ECE_WINDOW_DAYS,
                )

        # ----------------------------------------------------------------
        # 3. Retrain trigger
        # ----------------------------------------------------------------
        max_psi     = max(psi_results.values(), default=0.0)
        psi_trigger = max_psi >= PSI_SEVERE_THRESHOLD
        ece_trigger = ece_val > ECE_THRESHOLD

        if psi_trigger or ece_trigger:
            if psi_trigger and ece_trigger:
                reason = "both"
            elif psi_trigger:
                reason = "psi_severe"
            else:
                reason = "ece_exceeded"

            suppressed          = self._mac_filter.is_suppressed(now)
            suppression_reason: str | None = None
            if suppressed:
                nearest = self._mac_filter.nearest_event(now)
                if nearest:
                    ev_date, ev_label, delta = nearest
                    suppression_reason = f"{ev_label} on {ev_date} (±{delta}d)"

            await self._emitter.emit_retrain_trigger(
                horizon=horizon, model_version=model_version,
                trigger_reason=reason, psi_max=max_psi, ece=ece_val,
                suppressed=suppressed, suppression_reason=suppression_reason,
            )
            drift_retrain_triggers_total.labels(
                horizon=horizon, suppressed=str(suppressed).lower(),
            ).inc()
            await self._repo.insert_retrain(
                ts=now, horizon=horizon, model_version=model_version,
                trigger_reason=reason, suppressed=suppressed,
                psi_max=max_psi, ece=ece_val,
            )

        return {
            "model_version": model_version,
            "features_checked": len(psi_results),
            "max_psi": round(max_psi, 4),
            "ece": round(ece_val, 4),
            "ece_samples": n_samples,
            "retrain_triggered": psi_trigger or ece_trigger,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _load_model_registry(
        self, horizon: str
    ) -> tuple[str, list[str]]:
        """Load model_version and feature list from Redis.

        Keys:
          model_registry:{horizon}:current  → model_version (str)
          model_registry:{horizon}:features → JSON list[str]
        """
        try:
            mv_bytes   = await self._redis.get(_redis_key_model(horizon))
            feat_bytes = await self._redis.get(_redis_key_features(horizon))
            if mv_bytes is None or feat_bytes is None:
                return "", []
            model_version  = mv_bytes.decode() if isinstance(mv_bytes, bytes) else mv_bytes
            feature_names  = json.loads(feat_bytes)
            return model_version, feature_names
        except Exception as exc:
            logger.error("drift_cron.load_registry.error", horizon=horizon, error=str(exc))
            return "", []

    async def _load_or_create_edges(
        self,
        model_version: str,
        feature_name: str,
        train_arr: np.ndarray,
        n_buckets: int = 10,
    ) -> np.ndarray:
        """Load bucket edges from Redis; compute + save on cache miss."""
        key = _redis_key_bucket_edges(model_version, feature_name)
        try:
            cached = await self._redis.get(key)
            if cached is not None:
                return np.array(json.loads(cached), dtype=float)
        except Exception:
            pass

        # Cold-start: compute from train data and cache
        edges = np.percentile(train_arr, np.linspace(0.0, 100.0, n_buckets + 1))
        edges[0]  = -np.inf
        edges[-1] =  np.inf

        ttl = 30 * 24 * 3600   # 30 days
        try:
            await self._redis.setex(key, ttl, json.dumps(edges.tolist()))
            logger.info(
                "drift_cron.edges_created",
                model_version=model_version, feature=feature_name,
            )
        except Exception as exc:
            logger.warning("drift_cron.edges_save.error", feature=feature_name, error=str(exc))

        return edges


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Drift cron dry-run")
    parser.add_argument("--once", action="store_true", help="Run one pipeline pass")
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="ISO timestamp for manual run (UTC)",
    )
    args = parser.parse_args()

    async def _main() -> None:
        if not args.once:
            parser.error("Use --once for a single dry-run pass")
        logger.info("drift_cron.cli.once", as_of=args.as_of)
        print("drift_cron CLI requires POSTGRES_DSN + REDIS_URL + Kafka in env")

    asyncio.run(_main())
