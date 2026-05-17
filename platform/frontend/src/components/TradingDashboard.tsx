/**
 * TradingDashboard — Dashboard institucional principal.
 *
 * Layout:
 *   ┌──────────────────────────────────────────────────────┐
 *   │  Header: PnL summary + connection status + config    │
 *   ├─────────────┬──────────────────────┬─────────────────┤
 *   │ Chart panel │   Signal feed        │ Regime panel    │
 *   │ (price +    │   (live signals)     │ (GMM + macro)   │
 *   │  signals)   │                      │                 │
 *   ├─────────────┴──────────┬───────────┴─────────────────┤
 *   │ On-chain / Whale panel │ Order Flow + Funding panel  │
 *   ├────────────────────────┴─────────────────────────────┤
 *   │ Positions table                                       │
 *   └──────────────────────────────────────────────────────┘
 */
import React, { useCallback, useState } from "react";
import { useWebSocket } from "../hooks/useWebSocket";
import { BotConfigPanel } from "./BotConfigPanel";
import type {
  TradingSignal, WhaleAlert, RegimeState, MacroSnapshot,
  PnLSummary, Position, WSMessage, BotConfig,
} from "../types";
import { DIRECTION_META, REGIME_META } from "../types";

const ORCHESTRATOR_URL = import.meta.env.VITE_ORCHESTRATOR_URL ?? "http://localhost:8007";

const WS_URL = import.meta.env.VITE_WS_URL ?? "ws://localhost:8005/ws";
const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8005";

// ── Sub-components ────────────────────────────────────────────────────────────

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    connected:    "bg-green-500",
    connecting:   "bg-yellow-500 animate-pulse",
    disconnected: "bg-red-500",
    error:        "bg-red-600 animate-pulse",
  };
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-mono text-white ${colors[status] ?? "bg-gray-500"}`}>
      <span className="mr-1.5 h-2 w-2 rounded-full bg-white/60 inline-block" />
      {status.toUpperCase()}
    </span>
  );
}

function PnLCard({ summary }: { summary: PnLSummary }) {
  const pnlColor = summary.total_pnl_usd >= 0 ? "text-green-400" : "text-red-400";
  return (
    <div className="flex gap-6 items-center">
      <div>
        <div className="text-xs text-slate-400 uppercase tracking-wide">Total P&L</div>
        <div className={`text-xl font-bold font-mono ${pnlColor}`}>
          {summary.total_pnl_usd >= 0 ? "+" : ""}
          ${summary.total_pnl_usd.toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g, ",")}
        </div>
      </div>
      <div>
        <div className="text-xs text-slate-400 uppercase tracking-wide">Sharpe 30d</div>
        <div className="text-lg font-bold font-mono text-white">{summary.sharpe_30d.toFixed(2)}</div>
      </div>
      <div>
        <div className="text-xs text-slate-400 uppercase tracking-wide">Win Rate</div>
        <div className="text-lg font-bold font-mono text-white">{(summary.win_rate * 100).toFixed(1)}%</div>
      </div>
      <div>
        <div className="text-xs text-slate-400 uppercase tracking-wide">Max DD</div>
        <div className="text-lg font-bold font-mono text-red-400">{(summary.max_drawdown * 100).toFixed(1)}%</div>
      </div>
      <div>
        <div className="text-xs text-slate-400 uppercase tracking-wide">Trades 30d</div>
        <div className="text-lg font-bold font-mono text-white">{summary.n_trades_30d}</div>
      </div>
    </div>
  );
}

function SignalRow({ signal }: { signal: TradingSignal }) {
  const meta = DIRECTION_META[signal.direction];
  const confidence = signal.p_win > 0.65 ? "high" : signal.p_win > 0.55 ? "medium" : "low";
  const confColor = confidence === "high" ? "text-green-400" : confidence === "medium" ? "text-yellow-400" : "text-slate-400";

  return (
    <div className="flex items-center justify-between py-1.5 px-3 rounded bg-slate-800/60 hover:bg-slate-800 transition-colors text-sm">
      <span className="font-mono text-slate-300 w-24">{signal.symbol}</span>
      <span className={`font-bold w-16 ${meta.color.replace("#", "").length === 6 ? "" : ""}`}
            style={{ color: meta.color }}>
        {meta.icon} {meta.label}
      </span>
      <span className={`font-mono w-20 ${confColor}`}>
        p={signal.p_win.toFixed(3)}
      </span>
      <span className="font-mono text-slate-400 w-16">
        RR {signal.rr_ratio.toFixed(1)}
      </span>
      <span className="font-mono text-slate-500 text-xs">
        {new Date(signal.ts).toLocaleTimeString()}
      </span>
    </div>
  );
}

function WhaleRow({ alert }: { alert: WhaleAlert }) {
  const isInflow = alert.direction === "exchange_inflow";
  return (
    <div className={`flex items-center gap-3 py-1.5 px-3 rounded text-xs font-mono ${isInflow ? "bg-red-900/30" : "bg-green-900/30"}`}>
      <span className={isInflow ? "text-red-400" : "text-green-400"}>
        {isInflow ? "⬇ INFLOW" : "⬆ OUTFLOW"}
      </span>
      <span className="text-white font-bold">
        ${(alert.amount_usd / 1_000_000).toFixed(1)}M
      </span>
      <span className="text-slate-300">{alert.token}</span>
      <span className="text-slate-500">
        {alert.from_label ?? alert.blockchain} → {alert.to_label ?? "unknown"}
      </span>
      <span className="text-slate-600 ml-auto">
        {new Date(alert.ts).toLocaleTimeString()}
      </span>
    </div>
  );
}

function RegimePanel({ regime }: { regime: RegimeState | null }) {
  if (!regime) return (
    <div className="text-slate-500 text-sm text-center py-4">Loading regime...</div>
  );
  const meta = REGIME_META[regime.regime_name];
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <span className="h-3 w-3 rounded-full" style={{ backgroundColor: meta.color }} />
        <span className="font-bold text-white">{meta.label}</span>
        <span className="text-slate-400 text-xs ml-auto">
          stability {(regime.stability * 100).toFixed(0)}%
        </span>
      </div>
      {/* Regime probability bars */}
      <div className="space-y-1">
        {Object.entries(REGIME_META).map(([name, rm], i) => (
          <div key={name} className="flex items-center gap-2">
            <div className="text-xs text-slate-500 w-28">{rm.label}</div>
            <div className="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{
                  width: `${(regime.regime_probs[i] ?? 0) * 100}%`,
                  backgroundColor: rm.color,
                }}
              />
            </div>
            <div className="text-xs font-mono text-slate-400 w-10 text-right">
              {((regime.regime_probs[i] ?? 0) * 100).toFixed(0)}%
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function MacroPanel({ macro }: { macro: MacroSnapshot | null }) {
  if (!macro) return (
    <div className="text-slate-500 text-sm text-center py-4">Loading macro...</div>
  );
  const riskColor = macro.recession_probability > 0.5
    ? "text-red-400" : macro.recession_probability > 0.3
    ? "text-yellow-400" : "text-green-400";

  const regimeColors: Record<string, string> = {
    expansion: "text-green-400",
    slowdown:  "text-yellow-400",
    recession: "text-red-400",
    recovery:  "text-blue-400",
  };

  return (
    <div className="space-y-3">
      <div className="flex justify-between items-center">
        <div className="text-xs text-slate-400 uppercase">Macro Regime</div>
        <span className={`font-bold text-sm uppercase ${regimeColors[macro.regime] ?? "text-white"}`}>
          {macro.regime}
        </span>
      </div>
      <div className="flex justify-between items-center">
        <div className="text-xs text-slate-400">Recession P</div>
        <span className={`font-mono font-bold ${riskColor}`}>
          {(macro.recession_probability * 100).toFixed(1)}%
        </span>
      </div>
      <div className="flex justify-between items-center">
        <div className="text-xs text-slate-400">Rate Env</div>
        <span className="font-mono text-white text-sm uppercase">
          {macro.rate_environment}
        </span>
      </div>
      <div className="flex justify-between items-center">
        <div className="text-xs text-slate-400">Yield Curve</div>
        <span className={`font-mono text-sm ${macro.yield_curve_inverted ? "text-red-400" : "text-green-400"}`}>
          {macro.yield_curve_inverted
            ? `⚠ INVERTED (${macro.yield_inversion_days}d)`
            : "Normal"}
        </span>
      </div>
      {macro.sahm_value !== null && (
        <div className="flex justify-between items-center">
          <div className="text-xs text-slate-400">Sahm Index</div>
          <span className={`font-mono text-sm ${(macro.sahm_value ?? 0) >= 0.5 ? "text-red-400" : "text-white"}`}>
            {macro.sahm_value?.toFixed(2) ?? "—"}
          </span>
        </div>
      )}
    </div>
  );
}

function PositionRow({ pos }: { pos: Position }) {
  const pnlColor = pos.pnl_usd >= 0 ? "text-green-400" : "text-red-400";
  return (
    <tr className="border-t border-slate-800 hover:bg-slate-800/50">
      <td className="py-2 px-3 font-mono text-white text-sm">{pos.symbol}</td>
      <td className="py-2 px-3 text-slate-400 text-xs">{pos.strategy}</td>
      <td className="py-2 px-3" style={{ color: DIRECTION_META[pos.direction].color }}>
        {DIRECTION_META[pos.direction].icon} {DIRECTION_META[pos.direction].label}
      </td>
      <td className="py-2 px-3 font-mono text-slate-300 text-sm">
        ${pos.entry_price.toFixed(2)}
      </td>
      <td className="py-2 px-3 font-mono text-slate-300 text-sm">
        ${pos.current_price.toFixed(2)}
      </td>
      <td className={`py-2 px-3 font-mono text-sm font-bold ${pnlColor}`}>
        {pos.pnl_usd >= 0 ? "+" : ""}${pos.pnl_usd.toFixed(0)} ({pos.pnl_pct >= 0 ? "+" : ""}{pos.pnl_pct.toFixed(2)}%)
      </td>
      <td className="py-2 px-3 text-slate-500 text-xs">{pos.regime}</td>
    </tr>
  );
}

// ── Main Dashboard ────────────────────────────────────────────────────────────

const MOCK_PNL: PnLSummary = {
  total_pnl_usd: 12_847,
  daily_pnl_usd: 342,
  weekly_pnl_usd: 2_100,
  sharpe_30d: 1.34,
  max_drawdown: -0.082,
  win_rate: 0.587,
  n_trades_30d: 147,
};

export default function TradingDashboard() {
  const [signals,   setSignals]   = useState<TradingSignal[]>([]);
  const [whaleAlerts, setWhaleAlerts] = useState<WhaleAlert[]>([]);
  const [regime,    setRegime]    = useState<RegimeState | null>(null);
  const [macro,     setMacro]     = useState<MacroSnapshot | null>(null);
  const [positions, setPositions] = useState<Position[]>([]);
  const [selectedSymbol] = useState("BTCUSDT");
  const [killSwitchActive, setKillSwitchActive] = useState(false);
  const [showBotConfig, setShowBotConfig] = useState(false);

  const handleMessage = useCallback((msg: WSMessage) => {
    switch (msg.type) {
      case "FinalSignalEvent": {
        const s = msg.data as TradingSignal;
        setSignals((prev) => [s, ...prev].slice(0, 50));
        break;
      }
      case "WhaleAlertEvent": {
        const w = msg.data as WhaleAlert;
        setWhaleAlerts((prev) => [w, ...prev].slice(0, 30));
        break;
      }
      case "RegimeUpdateEvent": {
        const r = msg.data as RegimeState;
        if (r.symbol === selectedSymbol) setRegime(r);
        break;
      }
      case "MacroRegimeEvent":
      case "RecessionAlertEvent": {
        const m = msg.data as MacroSnapshot;
        setMacro(m);
        break;
      }
    }
  }, [selectedSymbol]);

  const { status, latency } = useWebSocket({
    url:       WS_URL,
    onMessage: handleMessage,
    debug:     import.meta.env.DEV,
  });

  return (
    <div className="min-h-screen bg-slate-950 text-white font-sans">

      {/* Header */}
      <header className="border-b border-slate-800 px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="text-lg font-bold text-white tracking-tight">
            <span className="text-blue-400">LOS</span> OJOS
          </div>
          <span className="text-slate-600 text-sm">|</span>
          <span className="text-slate-400 text-sm font-mono">{selectedSymbol}</span>
        </div>

        <PnLCard summary={MOCK_PNL} />

        <div className="flex items-center gap-3">
          {latency !== null && (
            <span className="text-xs font-mono text-slate-500">
              {latency}ms
            </span>
          )}
          {/* Kill switch indicator */}
          {killSwitchActive && (
            <span className="text-xs font-mono px-2 py-1 rounded bg-red-600/20 text-red-400 border border-red-500/40 animate-pulse">
              ⛔ HALTED
            </span>
          )}
          <StatusBadge status={status} />
          {/* Bot config toggle */}
          <button
            onClick={() => setShowBotConfig((v) => !v)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              showBotConfig
                ? "bg-blue-600 text-white"
                : "bg-slate-700 hover:bg-slate-600 text-slate-300"
            }`}
          >
            ⚙ Bot Config
          </button>
        </div>
      </header>

      {/* Bot Config Sidebar */}
      {showBotConfig && (
        <div className="fixed right-0 top-0 h-screen w-80 bg-slate-900 border-l border-slate-700 z-50 overflow-y-auto shadow-2xl animate-slide-right">
          <div className="sticky top-0 flex items-center justify-between px-4 py-3 bg-slate-900 border-b border-slate-800">
            <span className="font-semibold text-white text-sm">Bot Configuration</span>
            <button
              onClick={() => setShowBotConfig(false)}
              className="text-slate-400 hover:text-white text-lg leading-none"
            >
              ×
            </button>
          </div>
          <div className="p-4">
            <BotConfigPanel
              apiUrl={ORCHESTRATOR_URL}
              onKillSwitch={setKillSwitchActive}
            />
          </div>
        </div>
      )}

      {/* Main grid */}
      <div className={`grid grid-cols-12 gap-4 p-4 transition-all ${showBotConfig ? "mr-80" : ""}`}>

        {/* Chart placeholder */}
        <div className="col-span-7 bg-slate-900 rounded-xl border border-slate-800 p-4 min-h-[400px] flex items-center justify-center">
          <div className="text-slate-600 text-center">
            <div className="text-4xl mb-2">📈</div>
            <div className="text-sm">TradingChart — lightweight-charts</div>
            <div className="text-xs text-slate-700 mt-1">
              Mount &lt;TradingChart symbol="{selectedSymbol}" /&gt; here
            </div>
          </div>
        </div>

        {/* Signal feed */}
        <div className="col-span-5 bg-slate-900 rounded-xl border border-slate-800 p-4">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wide">
              Live Signals
            </h2>
            <span className="text-xs text-slate-500 font-mono">{signals.length} total</span>
          </div>
          <div className="space-y-1 max-h-[340px] overflow-y-auto scrollbar-thin scrollbar-thumb-slate-700">
            {signals.length === 0 ? (
              <div className="text-slate-600 text-sm text-center py-8">
                Waiting for signals...
              </div>
            ) : (
              signals.map((s) => <SignalRow key={s.event_id} signal={s} />)
            )}
          </div>
        </div>

        {/* Regime */}
        <div className="col-span-4 bg-slate-900 rounded-xl border border-slate-800 p-4">
          <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wide mb-3">
            Market Regime
          </h2>
          <RegimePanel regime={regime} />
        </div>

        {/* Macro */}
        <div className="col-span-4 bg-slate-900 rounded-xl border border-slate-800 p-4">
          <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wide mb-3">
            Macro Context
          </h2>
          <MacroPanel macro={macro} />
        </div>

        {/* Whale alerts */}
        <div className="col-span-4 bg-slate-900 rounded-xl border border-slate-800 p-4">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wide">
              Whale Alerts
            </h2>
            <span className="text-xs text-slate-500 font-mono">{whaleAlerts.length}</span>
          </div>
          <div className="space-y-1 max-h-[160px] overflow-y-auto">
            {whaleAlerts.length === 0 ? (
              <div className="text-slate-600 text-xs text-center py-4">
                Monitoring on-chain...
              </div>
            ) : (
              whaleAlerts.slice(0, 10).map((a) => (
                <WhaleRow key={a.event_id} alert={a} />
              ))
            )}
          </div>
        </div>

        {/* Positions */}
        <div className="col-span-12 bg-slate-900 rounded-xl border border-slate-800 p-4">
          <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wide mb-3">
            Open Positions
          </h2>
          {positions.length === 0 ? (
            <div className="text-slate-600 text-sm text-center py-4">No open positions</div>
          ) : (
            <table className="w-full text-sm">
              <thead>
                <tr className="text-xs text-slate-500 uppercase tracking-wide">
                  {["Symbol", "Strategy", "Side", "Entry", "Current", "P&L", "Regime"].map((h) => (
                    <th key={h} className="text-left py-2 px-3">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {positions.map((p, i) => <PositionRow key={i} pos={p} />)}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}
