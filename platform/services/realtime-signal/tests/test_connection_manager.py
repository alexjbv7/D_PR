"""
Tests for ConnectionManager — WebSocket channel routing and broadcast.

ConnectionManager is the core stateful component of the realtime-signal
service. It manages active WebSocket connections grouped by channel and
handles fanout broadcast with automatic dead-connection cleanup.

Channels: all | signals | whale | macro | system
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi import WebSocketDisconnect

from app.main import ConnectionManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ws() -> MagicMock:
    """Return a mock WebSocket with async accept() and send_text()."""
    ws = MagicMock()
    ws.accept = AsyncMock()
    ws.send_text = AsyncMock()
    return ws


# ---------------------------------------------------------------------------
# Connect
# ---------------------------------------------------------------------------

class TestConnect:
    async def test_connect_to_signals_channel(self):
        mgr = ConnectionManager()
        ws = _make_ws()
        await mgr.connect(ws, "signals")
        assert ws in mgr._connections["signals"]

    async def test_connect_to_whale_channel(self):
        mgr = ConnectionManager()
        ws = _make_ws()
        await mgr.connect(ws, "whale")
        assert ws in mgr._connections["whale"]

    async def test_connect_unknown_channel_falls_back_to_all(self):
        mgr = ConnectionManager()
        ws = _make_ws()
        await mgr.connect(ws, "nonexistent_channel")
        assert ws in mgr._connections["all"]
        assert ws not in mgr._connections["signals"]

    async def test_connect_calls_ws_accept(self):
        mgr = ConnectionManager()
        ws = _make_ws()
        await mgr.connect(ws, "all")
        ws.accept.assert_called_once()

    async def test_connect_multiple_ws_to_same_channel(self):
        mgr = ConnectionManager()
        ws1, ws2, ws3 = _make_ws(), _make_ws(), _make_ws()
        await mgr.connect(ws1, "macro")
        await mgr.connect(ws2, "macro")
        await mgr.connect(ws3, "macro")
        assert len(mgr._connections["macro"]) == 3

    async def test_connect_to_all_channels_independently(self):
        mgr = ConnectionManager()
        channels = ["all", "signals", "whale", "macro", "system"]
        ws_list = [_make_ws() for _ in channels]
        for ws, ch in zip(ws_list, channels):
            await mgr.connect(ws, ch)
        for ws, ch in zip(ws_list, channels):
            assert ws in mgr._connections[ch]


# ---------------------------------------------------------------------------
# Disconnect
# ---------------------------------------------------------------------------

class TestDisconnect:
    async def test_disconnect_removes_ws_from_channel(self):
        mgr = ConnectionManager()
        ws = _make_ws()
        await mgr.connect(ws, "signals")
        mgr.disconnect(ws, "signals")
        assert ws not in mgr._connections["signals"]

    def test_disconnect_nonexistent_ws_is_noop(self):
        """Should not raise even if WS was never connected."""
        mgr = ConnectionManager()
        ws = _make_ws()
        mgr.disconnect(ws, "signals")  # no error

    def test_disconnect_wrong_channel_leaves_original_intact(self):
        """Disconnecting from 'whale' should not affect 'signals' list."""
        mgr = ConnectionManager()
        ws = _make_ws()
        mgr._connections["signals"].append(ws)
        mgr.disconnect(ws, "whale")
        assert ws in mgr._connections["signals"]

    async def test_disconnect_removes_only_target_ws(self):
        mgr = ConnectionManager()
        ws1, ws2 = _make_ws(), _make_ws()
        await mgr.connect(ws1, "macro")
        await mgr.connect(ws2, "macro")
        mgr.disconnect(ws1, "macro")
        assert ws1 not in mgr._connections["macro"]
        assert ws2 in mgr._connections["macro"]

    async def test_disconnect_reduces_count_by_one(self):
        mgr = ConnectionManager()
        ws1, ws2 = _make_ws(), _make_ws()
        await mgr.connect(ws1, "signals")
        await mgr.connect(ws2, "signals")
        mgr.disconnect(ws1, "signals")
        assert len(mgr._connections["signals"]) == 1


# ---------------------------------------------------------------------------
# Broadcast
# ---------------------------------------------------------------------------

class TestBroadcast:
    async def test_broadcast_reaches_channel_subscriber(self):
        mgr = ConnectionManager()
        ws = _make_ws()
        await mgr.connect(ws, "signals")
        await mgr.broadcast("hello", "signals")
        ws.send_text.assert_called_once_with("hello")

    async def test_broadcast_to_signals_also_reaches_all_subscribers(self):
        """Clients on 'all' receive messages from any specific channel."""
        mgr = ConnectionManager()
        ws_all = _make_ws()
        ws_sig = _make_ws()
        await mgr.connect(ws_all, "all")
        await mgr.connect(ws_sig, "signals")
        await mgr.broadcast("event", "signals")
        ws_all.send_text.assert_called_once_with("event")
        ws_sig.send_text.assert_called_once_with("event")

    async def test_broadcast_to_all_channel_no_duplicate(self):
        """Client on 'all' receives message only once when channel='all'."""
        mgr = ConnectionManager()
        ws = _make_ws()
        await mgr.connect(ws, "all")
        await mgr.broadcast("msg", "all")
        assert ws.send_text.call_count == 1

    async def test_broadcast_to_empty_channel_no_error(self):
        mgr = ConnectionManager()
        await mgr.broadcast("msg", "whale")  # no exception, no clients

    async def test_broadcast_does_not_cross_channels(self):
        """Message to 'whale' should NOT reach 'signals'-only subscribers."""
        mgr = ConnectionManager()
        ws_whale = _make_ws()
        ws_signals = _make_ws()
        await mgr.connect(ws_whale, "whale")
        await mgr.connect(ws_signals, "signals")
        await mgr.broadcast("whale_msg", "whale")
        ws_whale.send_text.assert_called_once()
        ws_signals.send_text.assert_not_called()

    async def test_broadcast_dead_connection_removed_on_websocket_disconnect(self):
        """WS that raises WebSocketDisconnect is removed after broadcast."""
        mgr = ConnectionManager()
        ws_dead = _make_ws()
        ws_dead.send_text = AsyncMock(side_effect=WebSocketDisconnect())
        ws_alive = _make_ws()
        await mgr.connect(ws_dead, "signals")
        await mgr.connect(ws_alive, "signals")
        await mgr.broadcast("msg", "signals")
        assert ws_dead not in mgr._connections["signals"]
        assert ws_alive in mgr._connections["signals"]

    async def test_broadcast_dead_connection_removed_on_runtime_error(self):
        """WS that raises RuntimeError (closed transport) is also cleaned up."""
        mgr = ConnectionManager()
        ws = _make_ws()
        ws.send_text = AsyncMock(side_effect=RuntimeError("connection closed"))
        await mgr.connect(ws, "macro")
        await mgr.broadcast("msg", "macro")
        assert ws not in mgr._connections["macro"]

    async def test_broadcast_multiple_messages_same_channel(self):
        mgr = ConnectionManager()
        ws = _make_ws()
        await mgr.connect(ws, "whale")
        await mgr.broadcast("msg1", "whale")
        await mgr.broadcast("msg2", "whale")
        await mgr.broadcast("msg3", "whale")
        assert ws.send_text.call_count == 3

    async def test_broadcast_after_disconnect_no_error(self):
        """After a client disconnects, broadcasting should not raise."""
        mgr = ConnectionManager()
        ws = _make_ws()
        await mgr.connect(ws, "signals")
        mgr.disconnect(ws, "signals")
        await mgr.broadcast("msg", "signals")  # no exception
        ws.send_text.assert_not_called()


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_returns_all_channel_keys(self):
        mgr = ConnectionManager()
        stats = mgr.stats()
        assert set(stats.keys()) == {"all", "signals", "whale", "macro", "system"}

    def test_stats_all_zero_on_init(self):
        mgr = ConnectionManager()
        assert all(v == 0 for v in mgr.stats().values())

    async def test_stats_reflect_active_connections(self):
        mgr = ConnectionManager()
        ws1, ws2 = _make_ws(), _make_ws()
        await mgr.connect(ws1, "signals")
        await mgr.connect(ws2, "whale")
        stats = mgr.stats()
        assert stats["signals"] == 1
        assert stats["whale"] == 1
        assert stats["all"] == 0

    def test_total_connections_zero_on_init(self):
        mgr = ConnectionManager()
        assert mgr._total_connections() == 0

    def test_total_connections_sums_all_channels(self):
        mgr = ConnectionManager()
        mgr._connections["all"].append(_make_ws())
        mgr._connections["signals"].append(_make_ws())
        mgr._connections["whale"].append(_make_ws())
        mgr._connections["macro"].append(_make_ws())
        assert mgr._total_connections() == 4

    async def test_total_connections_after_disconnect(self):
        mgr = ConnectionManager()
        ws = _make_ws()
        await mgr.connect(ws, "signals")
        assert mgr._total_connections() == 1
        mgr.disconnect(ws, "signals")
        assert mgr._total_connections() == 0
