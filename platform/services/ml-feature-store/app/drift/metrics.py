"""Prometheus metrics for drift monitoring (Grafana drift_monitor.json)."""
from __future__ import annotations

from prometheus_client import Counter, Gauge

drift_psi_value = Gauge(
    "drift_psi_value",
    "Latest PSI per feature and horizon",
    ["horizon", "feature"],
)

drift_ece_value = Gauge(
    "drift_ece_value",
    "Latest rolling ECE per horizon",
    ["horizon"],
)

drift_events_total = Counter(
    "drift_events_total",
    "Drift events emitted by severity",
    ["horizon", "severity"],
)

drift_retrain_triggers_total = Counter(
    "drift_retrain_triggers_total",
    "Retrain trigger events (suppressed=true counts macro filter)",
    ["horizon", "suppressed"],
)
