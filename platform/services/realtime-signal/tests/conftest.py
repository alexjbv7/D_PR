"""
conftest.py for realtime-signal tests.

Mocks the shared library imports (Kafka, Redis, events) so that
app/main.py can be imported in a pure-unit context without live brokers.
"""
import sys
import os
from unittest.mock import MagicMock

# Ensure service root (contains app/) is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Mock all shared-lib modules before any app.* import happens
_SHARED_MODS = [
    "libs",
    "libs.shared",
    "libs.shared.events",
    "libs.shared.kafka_client",
    "libs.shared.redis_client",
]
for _mod in _SHARED_MODS:
    sys.modules.setdefault(_mod, MagicMock())
