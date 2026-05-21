"""
AlertEmitter — publishes drift events to Kafka.

Topics
------
los_ojos.drift.events     — DriftDetectedEvent + ECEDriftEvent (P1/P2 alerts)
los_ojos.retrain.triggers — RetrainTriggerEvent (compact, key = horizon)

Priority mapping
----------------
PSI severe (≥ 0.25)          → P1 alert
PSI moderate (0.10–0.25)     → P2 alert
ECE > 0.05                   → P2 alert
PSI severe AND ECE exceeded  → P1 alert + retrain trigger

The compact topic guarantees at-most-one pending trigger per horizon:
if intraday drift fires twice before the training cron reacts, only the
latest event survives compaction.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_TOPIC_DRIFT   = "los_ojos.drift.events"
_TOPIC_RETRAIN = "los_ojos.retrain.triggers"


def _utcnow_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class AlertEmitter:
    """Kafka-backed alert emitter for drift events.

    Parameters
    ----------
    producer : aiokafka.AIOKafkaProducer — already started; caller owns lifecycle.
    """

    def __init__(self, producer: Any) -> None:
        self._producer = producer

    # ------------------------------------------------------------------
    # Public emit methods
    # ------------------------------------------------------------------

    async def emit_psi_drift(
        self,
        horizon: str,
        model_version: str,
        feature_name: str,
        psi: float,
        severity: str,
        train_window_days: int,
        recent_window_days: int = 7,
    ) -> None:
        """Emit DriftDetectedEvent (PSI threshold crossed)."""
        payload = {
            "event_type":        "DriftDetectedEvent",
            "ts":                _utcnow_iso(),
            "source":            "ml-feature-store",
            "horizon":           horizon,
            "model_version":     model_version,
            "feature_name":      feature_name,
            "psi":               psi,
            "severity":          severity,
            "train_window_days": train_window_days,
            "recent_window_days": recent_window_days,
        }
        await self._send(_TOPIC_DRIFT, payload, key=horizon)
        logger.info(
            "alert_emitter.psi_drift",
            horizon=horizon, feature=feature_name,
            psi=round(psi, 4), severity=severity,
        )

    async def emit_ece_drift(
        self,
        horizon: str,
        model_version: str,
        ece: float,
        brier: float,
        n_samples: int,
        window_days: int = 7,
        threshold: float = 0.05,
    ) -> None:
        """Emit ECEDriftEvent (ECE threshold crossed)."""
        payload = {
            "event_type":    "ECEDriftEvent",
            "ts":            _utcnow_iso(),
            "source":        "ml-feature-store",
            "horizon":       horizon,
            "model_version": model_version,
            "ece":           ece,
            "brier":         brier,
            "threshold":     threshold,
            "window_days":   window_days,
            "n_samples":     n_samples,
        }
        await self._send(_TOPIC_DRIFT, payload, key=horizon)
        logger.info(
            "alert_emitter.ece_drift",
            horizon=horizon, ece=round(ece, 4), n_samples=n_samples,
        )

    async def emit_retrain_trigger(
        self,
        horizon: str,
        model_version: str,
        trigger_reason: str,
        psi_max: float,
        ece: float,
        suppressed: bool = False,
        suppression_reason: str | None = None,
    ) -> None:
        """Emit RetrainTriggerEvent to the compact retrain.triggers topic.

        Key = horizon so that only the most-recent trigger per horizon
        survives log compaction.
        """
        payload = {
            "event_type":        "RetrainTriggerEvent",
            "ts":                _utcnow_iso(),
            "source":            "ml-feature-store",
            "horizon":           horizon,
            "model_version":     model_version,
            "trigger_reason":    trigger_reason,
            "psi_max":           psi_max,
            "ece":               ece,
            "suppressed":        suppressed,
            "suppression_reason": suppression_reason,
        }
        await self._send(_TOPIC_RETRAIN, payload, key=horizon)
        if suppressed:
            logger.info(
                "alert_emitter.retrain_suppressed",
                horizon=horizon, reason=suppression_reason,
            )
        else:
            logger.warning(
                "alert_emitter.retrain_trigger",
                horizon=horizon, reason=trigger_reason,
                psi_max=round(psi_max, 4), ece=round(ece, 4),
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _send(self, topic: str, payload: dict, key: str = "") -> None:
        value_bytes = json.dumps(payload, default=str).encode()
        key_bytes   = key.encode() if key else None
        try:
            await self._producer.send_and_wait(
                topic, value=value_bytes, key=key_bytes
            )
        except Exception as exc:
            logger.error(
                "alert_emitter.send.error",
                topic=topic, key=key, error=str(exc),
            )
