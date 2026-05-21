"""
DriftRepository — async TimescaleDB persistence for drift audit tables.

Tables (created by platform/infra/sql/migrations/008_drift_audit.sql):
  audit.predictions  — model predictions with deferred true labels
  drift.psi_history  — PSI readings per feature per horizon
  drift.ece_history  — ECE readings per horizon

Design decisions
----------------
* All writes are fire-and-forget (errors logged, not raised) so that a
  database blip never blocks the main drift-cron pipeline.
* ``fetch_*`` methods return empty lists on error for the same reason.
* Uses asyncpg.Pool; lifecycle managed by the caller (drift_cron).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class DriftRepository:
    """Async writes / reads for drift audit tables.

    Parameters
    ----------
    pool : asyncpg.Pool — connected pool, managed by caller.
    """

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    # ------------------------------------------------------------------
    # drift.psi_history
    # ------------------------------------------------------------------

    async def insert_psi(
        self,
        ts: datetime,
        horizon: str,
        model_version: str,
        feature_name: str,
        psi: float,
        severity: str,
        n_buckets: int = 10,
        macro_suppressed: bool = False,
    ) -> None:
        sql = """
            INSERT INTO drift.psi_history
                (ts, horizon, model_version, feature_name, psi, severity,
                 n_buckets, macro_suppressed)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    sql, ts, horizon, model_version,
                    feature_name, psi, severity, n_buckets, macro_suppressed,
                )
        except Exception as exc:
            logger.error(
                "drift_repo.insert_psi.error",
                horizon=horizon, feature=feature_name, error=str(exc),
            )

    # ------------------------------------------------------------------
    # drift.ece_history
    # ------------------------------------------------------------------

    async def insert_ece(
        self,
        ts: datetime,
        horizon: str,
        model_version: str,
        ece: float,
        brier: float,
        n_samples: int,
        window_days: int = 7,
    ) -> None:
        sql = """
            INSERT INTO drift.ece_history
                (ts, horizon, model_version, ece, brier, n_samples, window_days)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    sql, ts, horizon, model_version,
                    ece, brier, n_samples, window_days,
                )
        except Exception as exc:
            logger.error(
                "drift_repo.insert_ece.error",
                horizon=horizon, error=str(exc),
            )

    # ------------------------------------------------------------------
    # audit.predictions
    # ------------------------------------------------------------------

    async def insert_prediction(
        self,
        ts: datetime,
        horizon: str,
        model_version: str,
        symbol: str,
        direction: int,
        probas: list[float],
        feature_set_hash: str = "",
    ) -> None:
        sql = """
            INSERT INTO audit.predictions
                (ts, horizon, model_version, symbol, direction, probas,
                 true_label, feature_set_hash)
            VALUES ($1, $2, $3, $4, $5, $6::float[], NULL, $7)
            ON CONFLICT DO NOTHING
        """
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    sql, ts, horizon, model_version,
                    symbol, direction, probas, feature_set_hash,
                )
        except Exception as exc:
            logger.error(
                "drift_repo.insert_prediction.error",
                horizon=horizon, symbol=symbol, error=str(exc),
            )

    async def fetch_recent_predictions(
        self,
        horizon: str,
        model_version: str,
        window_days: int = 7,
    ) -> list[dict[str, Any]]:
        """Fetch labelled predictions from a closed day window.

        Window is ``[t - window_days - 1, t - 1]`` in UTC — the current UTC
        day is excluded so partial intraday data at cron time (03:00 UTC)
        does not bias PSI/ECE.

        Returns rows as ``{"probas": [float, ...], "true_label": int}``.
        Only rows with ``true_label IS NOT NULL`` are returned.
        """
        sql = """
            SELECT probas, true_label
            FROM audit.predictions
            WHERE horizon       = $1
              AND model_version = $2
              AND true_label IS NOT NULL
              AND ts >= NOW() - (($3 + 1) * INTERVAL '1 day')
              AND ts <  NOW() - INTERVAL '1 day'
            ORDER BY ts ASC
        """
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql, horizon, model_version, window_days)
            return [
                {"probas": list(r["probas"]), "true_label": int(r["true_label"])}
                for r in rows
            ]
        except Exception as exc:
            logger.error(
                "drift_repo.fetch_predictions.error",
                horizon=horizon, error=str(exc),
            )
            return []

    # ------------------------------------------------------------------
    # Feature values for PSI computation
    # ------------------------------------------------------------------

    async def fetch_train_feature_values(
        self,
        feature_name: str,
        horizon: str,
        model_version: str,
        limit: int = 5000,
    ) -> list[float]:
        """Return historical feature values (train distribution proxy)."""
        sql = """
            SELECT (features->>$1)::float AS val
            FROM features.online
            WHERE symbol IN (
                SELECT DISTINCT symbol FROM audit.predictions
                WHERE horizon = $2 AND model_version = $3
                LIMIT 10
            )
              AND features ? $1
            ORDER BY ts DESC
            LIMIT $4
        """
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql, feature_name, horizon, model_version, limit)
            return [float(r["val"]) for r in rows if r["val"] is not None]
        except Exception as exc:
            logger.error(
                "drift_repo.fetch_train_features.error",
                feature=feature_name, error=str(exc),
            )
            return []

    async def insert_retrain(
        self,
        ts: datetime,
        horizon: str,
        model_version: str,
        trigger_reason: str,
        suppressed: bool,
        psi_max: float,
        ece: float,
    ) -> None:
        sql = """
            INSERT INTO drift.retrain_history
                (ts, horizon, model_version, trigger_reason, suppressed, psi_max, ece)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    sql, ts, horizon, model_version,
                    trigger_reason, suppressed, psi_max, ece,
                )
        except Exception as exc:
            logger.error(
                "drift_repo.insert_retrain.error",
                horizon=horizon, error=str(exc),
            )

    async def fetch_recent_feature_values(
        self,
        feature_name: str,
        window_days: int = 7,
        limit: int = 2000,
    ) -> list[float]:
        """Return feature values from a closed day window (excludes today UTC)."""
        sql = """
            SELECT (features->>$1)::float AS val
            FROM features.online
            WHERE ts >= NOW() - (($2 + 1) * INTERVAL '1 day')
              AND ts <  NOW() - INTERVAL '1 day'
              AND features ? $1
            ORDER BY ts DESC
            LIMIT $3
        """
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql, feature_name, window_days, limit)
            return [float(r["val"]) for r in rows if r["val"] is not None]
        except Exception as exc:
            logger.error(
                "drift_repo.fetch_recent_features.error",
                feature=feature_name, error=str(exc),
            )
            return []
