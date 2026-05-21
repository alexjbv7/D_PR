"""
Drift detection sub-package for the ml-feature-store service.

Modules
-------
_math           — inline PSI / ECE formulas (no cross-package deps)
macro_event_filter — MacroEventFilter (±2 day suppression window)
drift_repository   — async TimescaleDB persistence for drift audit tables
alert_emitter      — Kafka producer for drift.events + retrain.triggers
drift_cron         — daily orchestration loop (runs at 03:00 UTC)
"""
