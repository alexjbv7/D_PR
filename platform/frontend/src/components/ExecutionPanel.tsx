/**
 * ExecutionPanel — live view of the execution-engine service.
 *
 * Polls the engine every `pollMs` and shows:
 *   - Service status badge (online / unreachable / kill-switched)
 *   - Counters (signals_seen / approved / rejected / submitted / errors)
 *   - Registered venues
 *   - Open internal positions (table)
 *   - Recent OrderResults (table)
 *   - Kill-switch trip / reset buttons (with confirmation)
 *
 * Backend: see platform/services/execution-engine/app/main.py
 */
import React, { useCallback, useEffect, useState } from "react";
import { ShieldAlert, Activity, RefreshCw, CheckCircle2, XCircle } from "lucide-react";

import { executionApi } from "../api/executionEngine";
import type {
  ExecutionCounters,
  ExecutionHealth,
  ExecutionOrderResult,
  ExecutionPosition,
  OrderStatusStr,
} from "../types";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fmtDecimal(s: string | null | undefined, precision = 4): string {
  if (s == null || s === "") return "—";
  const n = Number(s);
  if (!Number.isFinite(n)) return s;
  return n.toFixed(precision).replace(/\.?0+$/, "");
}

function fmtMoney(s: string | null | undefined): string {
  if (s == null) return "—";
  const n = Number(s);
  if (!Number.isFinite(n)) return s;
  return `$${n.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
}

const STATUS_COLORS: Record<OrderStatusStr, string> = {
  pending:   "text-slate-400",
  submitted: "text-blue-400",
  partial:   "text-yellow-400",
  filled:    "text-green-400",
  cancelled: "text-slate-500",
  rejected:  "text-red-400",
  expired:   "text-slate-500",
};

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function StatusPill({
  health,
  unreachable,
}: { health: ExecutionHealth | null; unreachable: boolean }) {
  if (unreachable) {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-mono bg-red-600/30 text-red-400 border border-red-500/40">
        <XCircle className="w-3 h-3" /> UNREACHABLE
      </span>
    );
  }
  if (health?.kill_switch) {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-mono bg-red-600/30 text-red-400 border border-red-500/40 animate-pulse">
        <ShieldAlert className="w-3 h-3" /> KILL SWITCH
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-mono bg-green-600/20 text-green-400 border border-green-500/40">
      <CheckCircle2 className="w-3 h-3" /> ONLINE
    </span>
  );
}

function CountersGrid({ c }: { c: ExecutionCounters }) {
  const items: Array<{ label: string; value: number; color: string }> = [
    { label: "signals",   value: c.signals_seen,  color: "text-slate-300" },
    { label: "intents",   value: c.intents_built, color: "text-slate-300" },
    { label: "approved",  value: c.approved,      color: "text-green-400" },
    { label: "rejected",  value: c.rejected,      color: "text-yellow-400" },
    { label: "submitted", value: c.submitted,     color: "text-blue-400" },
    { label: "errors",    value: c.submit_errors, color: "text-red-400" },
  ];
  return (
    <div className="grid grid-cols-6 gap-2">
      {items.map((it) => (
        <div key={it.label} className="bg-slate-800/60 rounded px-2 py-1.5">
          <div className="text-[10px] uppercase tracking-wide text-slate-500">{it.label}</div>
          <div className={`font-mono text-lg font-bold ${it.color}`}>{it.value}</div>
        </div>
      ))}
    </div>
  );
}

function PositionsTable({ positions }: { positions: ExecutionPosition[] }) {
  if (positions.length === 0) {
    return (
      <div className="text-slate-600 text-sm text-center py-4">No open positions</div>
    );
  }
  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="text-xs text-slate-500 uppercase tracking-wide">
          {["Venue", "Symbol", "Side", "Qty", "Avg Entry", "Mark", "uPnL"].map((h) => (
            <th key={h} className="text-left py-2 px-3">{h}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {positions.map((p) => {
          const pnl = p.unrealized_pnl == null ? null : Number(p.unrealized_pnl);
          const pnlColor = pnl == null
            ? "text-slate-500"
            : pnl >= 0 ? "text-green-400" : "text-red-400";
          return (
            <tr key={`${p.venue}:${p.symbol}`} className="border-t border-slate-800 hover:bg-slate-800/50">
              <td className="py-2 px-3 text-slate-400 text-xs uppercase">{p.venue}</td>
              <td className="py-2 px-3 font-mono text-white text-sm">{p.symbol}</td>
              <td className={`py-2 px-3 font-bold text-sm ${p.side === "buy" ? "text-green-400" : "text-red-400"}`}>
                {p.side === "buy" ? "▲ LONG" : "▼ SHORT"}
              </td>
              <td className="py-2 px-3 font-mono text-slate-300 text-sm">{fmtDecimal(p.qty, 6)}</td>
              <td className="py-2 px-3 font-mono text-slate-300 text-sm">{fmtMoney(p.avg_entry)}</td>
              <td className="py-2 px-3 font-mono text-slate-300 text-sm">{fmtMoney(p.current_price)}</td>
              <td className={`py-2 px-3 font-mono text-sm font-bold ${pnlColor}`}>
                {pnl == null ? "—" : (pnl >= 0 ? "+" : "") + fmtMoney(p.unrealized_pnl)}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function OrdersTable({ orders }: { orders: ExecutionOrderResult[] }) {
  if (orders.length === 0) {
    return (
      <div className="text-slate-600 text-sm text-center py-4">No orders yet</div>
    );
  }
  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="text-xs text-slate-500 uppercase tracking-wide">
          {["Time", "Venue", "Symbol", "Side", "Status", "Qty", "Filled", "Avg Price"].map((h) => (
            <th key={h} className="text-left py-2 px-3">{h}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {orders.map((o) => (
          <tr key={o.result_id} className="border-t border-slate-800 hover:bg-slate-800/50">
            <td className="py-2 px-3 text-slate-500 text-xs font-mono">
              {new Date(o.ts_updated).toLocaleTimeString()}
            </td>
            <td className="py-2 px-3 text-slate-400 text-xs uppercase">{o.venue}</td>
            <td className="py-2 px-3 font-mono text-white text-sm">{o.symbol}</td>
            <td className={`py-2 px-3 text-sm ${o.side === "buy" ? "text-green-400" : "text-red-400"}`}>
              {o.side === "buy" ? "BUY" : "SELL"}
            </td>
            <td className={`py-2 px-3 text-xs font-mono uppercase ${STATUS_COLORS[o.status]}`}>
              {o.status}
            </td>
            <td className="py-2 px-3 font-mono text-slate-300 text-sm">{fmtDecimal(o.qty, 6)}</td>
            <td className="py-2 px-3 font-mono text-slate-300 text-sm">{fmtDecimal(o.filled_qty, 6)}</td>
            <td className="py-2 px-3 font-mono text-slate-300 text-sm">{fmtMoney(o.avg_price)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ---------------------------------------------------------------------------
// Main panel
// ---------------------------------------------------------------------------

export interface ExecutionPanelProps {
  /** Polling interval in milliseconds (default 5_000). */
  pollMs?: number;
  /** Hook for parent dashboard to mirror kill-switch state in its header. */
  onKillSwitchChange?: (tripped: boolean) => void;
}

export default function ExecutionPanel({
  pollMs = 5000,
  onKillSwitchChange,
}: ExecutionPanelProps) {
  const [health, setHealth] = useState<ExecutionHealth | null>(null);
  const [positions, setPositions] = useState<ExecutionPosition[]>([]);
  const [orders, setOrders] = useState<ExecutionOrderResult[]>([]);
  const [unreachable, setUnreachable] = useState(false);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const [h, p, o] = await Promise.all([
        executionApi.health(),
        executionApi.positions(),
        executionApi.recentOrders(50),
      ]);
      setHealth(h);
      setPositions(p.positions);
      setOrders(o.orders);
      setUnreachable(false);
      onKillSwitchChange?.(h.kill_switch);
    } catch (err) {
      setUnreachable(true);
      // Keep last-known state so the UI doesn't blank out on a transient failure
      // eslint-disable-next-line no-console
      console.warn("executionApi refresh failed", err);
    }
  }, [onKillSwitchChange]);

  useEffect(() => {
    refresh();
    const id = window.setInterval(refresh, pollMs);
    return () => window.clearInterval(id);
  }, [refresh, pollMs]);

  const handleTrip = useCallback(async () => {
    if (!window.confirm(
      "Trip kill-switch?\n\nThe execution-engine will stop processing all new "
      + "signals until reset.  Open positions are NOT closed.",
    )) return;
    setBusy(true);
    try {
      const r = await executionApi.tripKillSwitch();
      onKillSwitchChange?.(r.kill_switch);
      await refresh();
    } finally {
      setBusy(false);
    }
  }, [refresh, onKillSwitchChange]);

  const handleReset = useCallback(async () => {
    setBusy(true);
    try {
      const r = await executionApi.resetKillSwitch();
      onKillSwitchChange?.(r.kill_switch);
      await refresh();
    } finally {
      setBusy(false);
    }
  }, [refresh, onKillSwitchChange]);

  return (
    <div className="bg-slate-900 rounded-xl border border-slate-800 p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wide flex items-center gap-2">
          <Activity className="w-4 h-4 text-blue-400" />
          Execution Engine
        </h2>
        <div className="flex items-center gap-2">
          <StatusPill health={health} unreachable={unreachable} />
          {(health?.venues ?? []).map((v) => (
            <span key={v} className="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 uppercase">
              {v}
            </span>
          ))}
          <button
            onClick={refresh}
            className="text-slate-400 hover:text-white transition-colors"
            title="Refresh now"
            aria-label="Refresh"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Counters */}
      {health && <CountersGrid c={health.counters} />}

      {/* Kill-switch actions */}
      <div className="flex items-center gap-2">
        {health?.kill_switch ? (
          <button
            onClick={handleReset}
            disabled={busy || unreachable}
            className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-green-600 hover:bg-green-500 text-white disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Reset kill-switch
          </button>
        ) : (
          <button
            onClick={handleTrip}
            disabled={busy || unreachable}
            className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-red-600 hover:bg-red-500 text-white disabled:opacity-50 disabled:cursor-not-allowed"
          >
            ⛔ Trip kill-switch
          </button>
        )}
        <span className="text-xs text-slate-500 font-mono ml-auto">
          polling {(pollMs / 1000).toFixed(0)}s · {orders.length} orders · {positions.length} positions
        </span>
      </div>

      {/* Positions */}
      <div>
        <div className="text-xs text-slate-500 uppercase tracking-wide mb-1">Internal positions</div>
        <div className="max-h-[200px] overflow-y-auto scrollbar-thin scrollbar-thumb-slate-700">
          <PositionsTable positions={positions} />
        </div>
      </div>

      {/* Recent orders */}
      <div>
        <div className="text-xs text-slate-500 uppercase tracking-wide mb-1">Recent orders</div>
        <div className="max-h-[280px] overflow-y-auto scrollbar-thin scrollbar-thumb-slate-700">
          <OrdersTable orders={orders} />
        </div>
      </div>
    </div>
  );
}
