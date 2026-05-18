"""Prometheus metrics for market-intelligence service.

Includes counters/gauges for universe cron and corporate actions pipeline.
Import this module early (e.g. in main.py) so metrics are registered before
the first scrape.
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# Universe cron
# ---------------------------------------------------------------------------

universe_cron_runs_total = Counter(
    "universe_cron_runs_total",
    "Total runs of the daily universe cron",
    ["result"],  # "success" | "failure"
)

universe_changes_total = Counter(
    "universe_changes_total",
    "Symbol status changes detected by universe cron",
    ["type"],  # "new_listing" | "delisting" | "metadata_update"
)

# ---------------------------------------------------------------------------
# Corporate actions pipeline
# ---------------------------------------------------------------------------

corporate_actions_fetched_total = Counter(
    "corporate_actions_fetched_total",
    "Total corporate action announcements fetched from Alpaca",
    ["ca_type"],  # "forward_split" | "reverse_split" | "stock_dividend" | ...
)

corporate_actions_applied_seconds = Histogram(
    "corporate_actions_applied_seconds",
    "Time spent applying a single corporate action to bars",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0],
)

corporate_actions_provisional_count = Gauge(
    "corporate_actions_provisional_count",
    "Number of corporate actions still marked provisional",
)
