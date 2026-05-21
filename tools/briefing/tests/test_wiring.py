"""Tests for collector wiring."""
from __future__ import annotations

import pytest
from typing import Any

from tools.briefing.metrics_collector import StubMetricsBackend
from tools.briefing.wiring import build_collector


def test_build_collector_returns_collector() -> None:
    c = build_collector()
    assert c._backend is not None  # noqa: SLF001 — test inspects wiring


def test_build_collector_stub_when_no_db(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    fake_pg: Any = types.ModuleType("psycopg2")

    class OperationalError(Exception):
        pass

    fake_pg.OperationalError = OperationalError

    def _boom(*_a: object, **_k: object) -> None:
        raise OperationalError("connection refused")

    fake_pg.connect = _boom
    monkeypatch.setitem(sys.modules, "psycopg2", fake_pg)
    c = build_collector()
    assert isinstance(c._backend, StubMetricsBackend)  # noqa: SLF001
