/**
 * SECPanel — live view of the sec-research service.
 *
 * Shows:
 *   - Service status badge (online / unreachable)
 *   - Active regulatory risk indicator
 *   - Recent SEC filings with sentiment + key signals (table)
 *   - Institutional positions in BTC ETFs and crypto equities (table)
 *
 * Backend: platform/services/sec-research/app/main.py
 */
import React, { useCallback, useEffect, useState } from "react";
import { FileText, RefreshCw, CheckCircle2, XCircle, AlertTriangle } from "lucide-react";

import { secApi } from "../api/secResearch";
import type {
  FilingSignal,
  FilingSentimentLabel,
  InstitutionalPosition,
  RegulatorySignalsResponse,
  SECHealth,
} from "../types";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const SENTIMENT_STYLES: Record<FilingSentimentLabel, { bg: string; text: string }> = {
  positive: { bg: "bg-green-600/20",  text: "text-green-400"  },
  negative: { bg: "bg-red-600/20",    text: "text-red-400"    },
  neutral:  { bg: "bg-slate-700/40",  text: "text-slate-400"  },
  mixed:    { bg: "bg-yellow-600/20", text: "text-yellow-400" },
};

function fmtUSD(value: number): string {
  if (value >= 1e9) return `$${(value / 1e9).toFixed(1)}B`;
  if (value >= 1e6) return `$${(value / 1e6).toFixed(1)}M`;
  if (value >= 1e3) return `$${(value / 1e3).toFixed(1)}K`;
  return `$${value.toFixed(0)}`;
}

function fmtShares(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function StatusPill({ health, unreachable }: { health: SECHealth | null; unreachable: boolean }) {
  if (unreachable) {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-mono bg-red-600/30 text-red-400 border border-red-500/40">
        <XCircle className="w-3 h-3" /> UNREACHABLE
      </span>
    );
  }
  if (!health) return null;
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-mono bg-green-600/20 text-green-400 border border-green-500/40">
      <CheckCircle2 className="w-3 h-3" /> ONLINE
    </span>
  );
}

function RegulatoryBadge({ active }: { active: boolean }) {
  if (!active) return null;
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-mono bg-red-600/25 text-red-400 border border-red-500/40 animate-pulse">
      <AlertTriangle className="w-3 h-3" /> REGULATORY RISK
    </span>
  );
}

function FilingsTable({ filings }: { filings: FilingSignal[] }) {
  if (filings.length === 0) {
    return (
      <div className="text-slate-600 text-sm text-center py-4">No recent filings</div>
    );
  }
  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="text-xs text-slate-500 uppercase tracking-wide">
          {["Time", "Ticker", "Form", "Sentiment", "Score", "Signal tags"].map((h) => (
            <th key={h} className="text-left py-2 px-3">{h}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {filings.map((f) => {
          const style = SENTIMENT_STYLES[f.sentiment as FilingSentimentLabel]
            ?? SENTIMENT_STYLES.neutral;
          const topSignals = f.signals.slice(0, 3);
          return (
            <tr key={f.event_id} className="border-t border-slate-800 hover:bg-slate-800/50">
              <td className="py-2 px-3 text-slate-500 text-xs font-mono">
                {new Date(f.ts).toLocaleTimeString()}
              </td>
              <td className="py-2 px-3 font-mono text-white font-bold text-sm">
                {f.ticker}
              </td>
              <td className="py-2 px-3 text-slate-400 text-xs font-mono">
                {f.form_type}
              </td>
              <td className="py-2 px-3">
                <span className={`px-1.5 py-0.5 rounded text-xs font-mono uppercase ${style.bg} ${style.text}`}>
                  {f.sentiment}
                </span>
                {f.is_market_moving && (
                  <span className="ml-1 text-[10px] text-yellow-400 font-mono">⚡</span>
                )}
              </td>
              <td className={`py-2 px-3 font-mono text-sm ${
                f.score > 0.1 ? "text-green-400" : f.score < -0.1 ? "text-red-400" : "text-slate-400"
              }`}>
                {f.score >= 0 ? "+" : ""}{f.score.toFixed(2)}
              </td>
              <td className="py-2 px-3">
                <div className="flex flex-wrap gap-1">
                  {topSignals.map((sig) => (
                    <span key={sig} className="text-[10px] font-mono px-1 py-0.5 rounded bg-slate-800 text-slate-400">
                      {sig}
                    </span>
                  ))}
                  {f.signals.length > 3 && (
                    <span className="text-[10px] font-mono text-slate-600">
                      +{f.signals.length - 3}
                    </span>
                  )}
                </div>
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function PositionsTable({ positions }: { positions: InstitutionalPosition[] }) {
  if (positions.length === 0) {
    return (
      <div className="text-slate-600 text-sm text-center py-4">No institutional data</div>
    );
  }
  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="text-xs text-slate-500 uppercase tracking-wide">
          {["Institution", "Ticker", "Shares", "Value", "Period", "Change"].map((h) => (
            <th key={h} className="text-left py-2 px-3">{h}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {positions.map((p, i) => {
          const changeColor = p.change_pct > 0
            ? "text-green-400"
            : p.change_pct < 0
            ? "text-red-400"
            : "text-slate-400";
          return (
            <tr key={`${p.institution}-${p.ticker}-${i}`}
                className="border-t border-slate-800 hover:bg-slate-800/50">
              <td className="py-2 px-3 text-slate-300 text-sm">{p.institution}</td>
              <td className="py-2 px-3 font-mono text-white font-bold text-sm">{p.ticker}</td>
              <td className="py-2 px-3 font-mono text-slate-300 text-sm">{fmtShares(p.shares)}</td>
              <td className="py-2 px-3 font-mono text-slate-300 text-sm">{fmtUSD(p.value_usd)}</td>
              <td className="py-2 px-3 text-slate-500 text-xs font-mono">{p.period}</td>
              <td className={`py-2 px-3 font-mono text-sm font-bold ${changeColor}`}>
                {p.change_pct >= 0 ? "+" : ""}{p.change_pct.toFixed(1)}%
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

// ---------------------------------------------------------------------------
// Main panel
// ---------------------------------------------------------------------------

export interface SECPanelProps {
  /** Polling interval in milliseconds (default 60_000 — 1 minute). */
  pollMs?: number;
}

export default function SECPanel({ pollMs = 60_000 }: SECPanelProps) {
  const [health,       setHealth]       = useState<SECHealth | null>(null);
  const [filings,      setFilings]      = useState<FilingSignal[]>([]);
  const [regulatory,   setRegulatory]   = useState<RegulatorySignalsResponse | null>(null);
  const [positions,    setPositions]    = useState<InstitutionalPosition[]>([]);
  const [unreachable,  setUnreachable]  = useState(false);
  const [activeTab,    setActiveTab]    = useState<"filings" | "institutional">("filings");

  const refresh = useCallback(async () => {
    try {
      const [h, f, reg, inst] = await Promise.all([
        secApi.health(),
        secApi.recentFilings(7),
        secApi.regulatorySignals(),
        secApi.institutionalPositions(),
      ]);
      setHealth(h);
      setFilings(f);
      setRegulatory(reg);
      setPositions(inst.positions);
      setUnreachable(false);
    } catch (err) {
      setUnreachable(true);
      // eslint-disable-next-line no-console
      console.warn("secApi refresh failed", err);
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = window.setInterval(refresh, pollMs);
    return () => window.clearInterval(id);
  }, [refresh, pollMs]);

  const hasRegulatoryRisk = regulatory?.has_active_regulatory_risk ?? false;

  return (
    <div className="bg-slate-900 rounded-xl border border-slate-800 p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wide flex items-center gap-2">
          <FileText className="w-4 h-4 text-purple-400" />
          SEC Research
        </h2>
        <div className="flex items-center gap-2">
          <StatusPill health={health} unreachable={unreachable} />
          <RegulatoryBadge active={hasRegulatoryRisk} />
          <span className="text-xs text-slate-600 font-mono">
            {filings.length} filings
          </span>
          <button
            onClick={refresh}
            className="text-slate-400 hover:text-white transition-colors"
            title="Refresh now"
            aria-label="Refresh SEC data"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Tab switcher */}
      <div className="flex gap-1 p-1 bg-slate-800/50 rounded-lg w-fit">
        {(["filings", "institutional"] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
              activeTab === tab
                ? "bg-slate-700 text-white"
                : "text-slate-500 hover:text-slate-300"
            }`}
          >
            {tab === "filings" ? "Recent Filings" : "Institutional Positions"}
          </button>
        ))}
      </div>

      {/* Content */}
      {activeTab === "filings" ? (
        <div>
          <div className="max-h-[280px] overflow-y-auto scrollbar-thin scrollbar-thumb-slate-700">
            <FilingsTable filings={filings} />
          </div>
        </div>
      ) : (
        <div>
          <div className="max-h-[280px] overflow-y-auto scrollbar-thin scrollbar-thumb-slate-700">
            <PositionsTable positions={positions} />
          </div>
        </div>
      )}

      {/* Footer */}
      {regulatory && (
        <div className="text-xs text-slate-600 font-mono flex items-center gap-2">
          <span>Regulatory signals: {regulatory.count}</span>
          <span>·</span>
          <span>Polling {(pollMs / 1000).toFixed(0)}s</span>
        </div>
      )}
    </div>
  );
}
