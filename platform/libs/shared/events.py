"""
Kafka event schemas — backward-compatible re-export shim.
=========================================================
La fuente de verdad de todos los eventos ha sido migrada a:

    quant_shared.schemas.events  (shared/quant_shared/schemas/events.py)

Este archivo re-exporta todo desde allí para mantener compatibilidad con los
servicios que importan via `from libs.shared.events import ...`.

MIGRACIÓN PROGRESIVA: actualiza los imports de cada servicio de:
    from libs.shared.events import FundingRateEvent, KafkaTopics
a:
    from quant_shared.schemas.events import FundingRateEvent, KafkaTopics

Una vez todos los servicios hayan migrado, este archivo puede eliminarse.
"""
from quant_shared.schemas.events import (  # noqa: F401 — re-export público
    BaseEvent,
    TickEvent,
    OHLCVEvent,
    OrderBookEvent,
    FundingRateEvent,
    LiquidationEvent,
    FeatureUpdateEvent,
    RegimeUpdateEvent,
    MacroDataEvent,
    MacroRegimeEvent,
    RecessionAlertEvent,
    WhaleAlertEvent,
    SmartMoneyFlowEvent,
    OnChainSignalEvent,
    NewsSentimentEvent,
    SECFilingEvent,
    EarningsEvent,
    PredictionMarketEvent,
    RawSignalEvent,
    FinalSignalEvent,
    ExecutionIntentEvent,
    AnomalyEvent,
    KillSwitchEvent,
    KafkaTopics,
)

__all__ = [
    "BaseEvent",
    "TickEvent",
    "OHLCVEvent",
    "OrderBookEvent",
    "FundingRateEvent",
    "LiquidationEvent",
    "FeatureUpdateEvent",
    "RegimeUpdateEvent",
    "MacroDataEvent",
    "MacroRegimeEvent",
    "RecessionAlertEvent",
    "WhaleAlertEvent",
    "SmartMoneyFlowEvent",
    "OnChainSignalEvent",
    "NewsSentimentEvent",
    "SECFilingEvent",
    "EarningsEvent",
    "PredictionMarketEvent",
    "RawSignalEvent",
    "FinalSignalEvent",
    "ExecutionIntentEvent",
    "AnomalyEvent",
    "KillSwitchEvent",
    "KafkaTopics",
]
