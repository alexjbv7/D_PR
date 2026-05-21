"""Kafka topic existence checks."""
from __future__ import annotations

from confluent_kafka.admin import AdminClient


def missing_kafka_topics(
    bootstrap: str,
    expected: list[str],
    *,
    timeout_s: float = 5.0,
) -> set[str]:
    """Return topic names that are absent on the cluster."""
    admin = AdminClient({"bootstrap.servers": bootstrap})
    metadata = admin.list_topics(timeout=timeout_s)
    cluster = set(metadata.topics.keys())
    return set(expected) - cluster
