/**
 * BotConfigPanel — Panel de configuración del bot de trading.
 *
 * Permite al usuario:
 *   - Cambiar modo paper / live
 *   - Ajustar parámetros de riesgo (max_positions, leverage, risk_per_trade, max_drawdown)
 *   - Activar/desactivar estrategias individuales
 *   - Activar kill switch
 *   - Ver estado del bot en tiempo real
 *
 * Llama a la Strategy Orchestrator API (http://localhost:8007)
 */
import React, { useState, useEffect, useCallback } from "react";
import {
  Settings,
  ShieldOff,
  Shield,
  Play,
  Square,
  AlertTriangle,
  Check,
  Loader2,
  ChevronDown,
  ChevronUp,
  Zap,
  BarChart2,
  Waves,
  TrendingUp,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface BotConfig {
  name:              string;
  description:       string;
  mode:              "paper" | "live";
  max_positions:     number;
  max_leverage:      number;
  risk_per_trade:    number;
  max_drawdown:      number;
  total_capital:     number;
  active_strategies: string[];
  exchange:          string;
  kill_switch:       boolean;
}

interface Strategy {
  name:         string;
  display_name: string;
  type:         string;
  timeframe:    string;
  is_active:    boolean;
}

interface BotConfigPanelProps {
  apiUrl?: string;
  onKillSwitch?: (active: boolean) => void;
}

// ---------------------------------------------------------------------------
// Strategy icons
// ---------------------------------------------------------------------------
const STRATEGY_ICONS: Record<string, React.FC<{ className?: string }>> = {
  momentum_ml:              ({ className }) => <Zap className={className} />,
  mean_reversion_funding:   ({ className }) => <Waves className={className} />,
  regime_adaptive:          ({ className }) => <BarChart2 className={className} />,
  whale_follow:             ({ className }) => <TrendingUp className={className} />,
};

const STRATEGY_COLORS: Record<string, string> = {
  ml:             "#3b82f6",
  mean_reversion: "#a855f7",
  trend:          "#22c55e",
};

// ---------------------------------------------------------------------------
// Utility: numeric input with constraints
// ---------------------------------------------------------------------------
function NumericInput({
  label,
  value,
  min,
  max,
  step,
  format,
  onChange,
  disabled,
}: {
  label:    string;
  value:    number;
  min:      number;
  max:      number;
  step:     number;
  format?:  (v: number) => string;
  onChange: (v: number) => void;
  disabled?: boolean;
}) {
  const fmt = format ?? ((v) => String(v));
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <span className="text-xs text-slate-400">{label}</span>
        <span className="text-xs font-mono text-white">{fmt(value)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full h-1.5 rounded-full appearance-none bg-slate-700 accent-blue-500 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
      />
      <div className="flex justify-between text-xs text-slate-600">
        <span>{fmt(min)}</span>
        <span>{fmt(max)}</span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------
export function BotConfigPanel({ apiUrl = "http://localhost:8007", onKillSwitch }: BotConfigPanelProps) {
  const [config, setConfig] = useState<BotConfig | null>(null);
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [killSwitchLoading, setKillSwitchLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [expanded, setExpanded] = useState(true);
  const [dirtyConfig, setDirtyConfig] = useState<BotConfig | null>(null);

  // Fetch current config and strategies
  const fetchAll = useCallback(async () => {
    try {
      const [cfgRes, stratRes] = await Promise.all([
        fetch(`${apiUrl}/config`),
        fetch(`${apiUrl}/strategies`),
      ]);

      if (!cfgRes.ok) throw new Error("Failed to fetch config");
      if (!stratRes.ok) throw new Error("Failed to fetch strategies");

      const cfg: BotConfig = await cfgRes.json();
      const { strategies: strats }: { strategies: Strategy[] } = await stratRes.json();

      setConfig(cfg);
      setDirtyConfig(cfg);
      setStrategies(strats);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Connection failed");
    } finally {
      setLoading(false);
    }
  }, [apiUrl]);

  useEffect(() => {
    fetchAll();
    const interval = setInterval(fetchAll, 30_000);
    return () => clearInterval(interval);
  }, [fetchAll]);

  // Save config
  const handleSave = async () => {
    if (!dirtyConfig) return;
    setSaving(true);
    try {
      const res = await fetch(`${apiUrl}/config`, {
        method:  "PUT",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(dirtyConfig),
      });
      if (!res.ok) throw new Error("Save failed");
      setConfig(dirtyConfig);
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 2000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Save error");
    } finally {
      setSaving(false);
    }
  };

  // Kill switch
  const handleKillSwitch = async (activate: boolean) => {
    setKillSwitchLoading(true);
    try {
      const res = await fetch(`${apiUrl}/kill-switch/${activate ? "on" : "off"}`, {
        method: "POST",
      });
      if (!res.ok) throw new Error("Kill switch failed");
      if (dirtyConfig) {
        const updated = { ...dirtyConfig, kill_switch: activate };
        setDirtyConfig(updated);
        setConfig(updated);
      }
      onKillSwitch?.(activate);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Kill switch error");
    } finally {
      setKillSwitchLoading(false);
    }
  };

  // Toggle strategy
  const handleToggleStrategy = async (stratName: string, active: boolean) => {
    try {
      await fetch(`${apiUrl}/strategies/toggle`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ strategy_name: stratName, active }),
      });
      setStrategies((prev) =>
        prev.map((s) => s.name === stratName ? { ...s, is_active: active } : s)
      );
      if (dirtyConfig) {
        const active_strategies = active
          ? [...dirtyConfig.active_strategies, stratName]
          : dirtyConfig.active_strategies.filter((n) => n !== stratName);
        setDirtyConfig({ ...dirtyConfig, active_strategies });
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Toggle error");
    }
  };

  const updateDirty = (partial: Partial<BotConfig>) => {
    if (dirtyConfig) setDirtyConfig({ ...dirtyConfig, ...partial });
  };

  const isDirty = JSON.stringify(config) !== JSON.stringify(dirtyConfig);

  // ---------------------------------------------------------------------------
  // Loading / Error states
  // ---------------------------------------------------------------------------
  if (loading) {
    return (
      <div className="card flex items-center justify-center gap-2 py-8">
        <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />
        <span className="text-sm text-slate-400">Loading bot config…</span>
      </div>
    );
  }

  if (error && !config) {
    return (
      <div className="card border-red-500/30 bg-red-500/5">
        <div className="flex items-center gap-2 text-red-400">
          <AlertTriangle className="w-4 h-4" />
          <span className="text-sm">Strategy Orchestrator offline: {error}</span>
        </div>
      </div>
    );
  }

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  const isKillActive = dirtyConfig?.kill_switch ?? false;

  return (
    <div className="card">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Settings className="w-4 h-4 text-blue-400" />
          <span className="font-semibold text-white text-sm">Bot Configuration</span>
          {dirtyConfig && (
            <span
              className={`text-xs px-2 py-0.5 rounded font-mono ${
                dirtyConfig.mode === "live"
                  ? "bg-red-500/10 text-red-400 border border-red-500/30"
                  : "bg-yellow-500/10 text-yellow-400 border border-yellow-500/30"
              }`}
            >
              {dirtyConfig.mode.toUpperCase()}
            </span>
          )}
        </div>
        <button
          onClick={() => setExpanded((x) => !x)}
          className="text-slate-400 hover:text-white transition-colors"
        >
          {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>
      </div>

      {expanded && dirtyConfig && (
        <>
          {/* Kill Switch */}
          <div
            className={`flex items-center justify-between p-3 rounded-lg mb-4 ${
              isKillActive
                ? "bg-red-500/10 border border-red-500/40"
                : "bg-slate-700/30 border border-slate-600/30"
            }`}
          >
            <div className="flex items-center gap-2">
              {isKillActive
                ? <ShieldOff className="w-4 h-4 text-red-400" />
                : <Shield className="w-4 h-4 text-green-400" />
              }
              <div>
                <span className={`text-sm font-medium ${isKillActive ? "text-red-400" : "text-white"}`}>
                  Kill Switch
                </span>
                <p className="text-xs text-slate-400">
                  {isKillActive ? "All signal generation halted" : "Bot is operational"}
                </p>
              </div>
            </div>
            <button
              onClick={() => handleKillSwitch(!isKillActive)}
              disabled={killSwitchLoading}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                isKillActive
                  ? "bg-green-600 hover:bg-green-500 text-white"
                  : "bg-red-600/80 hover:bg-red-500 text-white"
              } disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              {killSwitchLoading ? (
                <Loader2 className="w-3 h-3 animate-spin" />
              ) : isKillActive ? (
                <><Play className="w-3 h-3" /> Resume</>
              ) : (
                <><Square className="w-3 h-3" /> STOP ALL</>
              )}
            </button>
          </div>

          {/* Mode Toggle */}
          <div className="mb-4">
            <label className="text-xs text-slate-400 block mb-2">Trading Mode</label>
            <div className="flex rounded-lg overflow-hidden border border-slate-600/50">
              {(["paper", "live"] as const).map((mode) => (
                <button
                  key={mode}
                  onClick={() => updateDirty({ mode })}
                  className={`flex-1 py-2 text-xs font-medium capitalize transition-all ${
                    dirtyConfig.mode === mode
                      ? mode === "live"
                        ? "bg-red-600 text-white"
                        : "bg-blue-600 text-white"
                      : "bg-slate-700/50 text-slate-400 hover:text-white"
                  }`}
                >
                  {mode === "live" ? "🔴 LIVE" : "📋 Paper"}
                </button>
              ))}
            </div>
            {dirtyConfig.mode === "live" && (
              <p className="text-xs text-red-400 mt-1 flex items-center gap-1">
                <AlertTriangle className="w-3 h-3" />
                Live mode uses real capital on {dirtyConfig.exchange}
              </p>
            )}
          </div>

          {/* Risk Parameters */}
          <div className="space-y-4 mb-4">
            <h4 className="text-xs text-slate-500 uppercase tracking-wider">Risk Parameters</h4>

            <NumericInput
              label="Max Open Positions"
              value={dirtyConfig.max_positions}
              min={1} max={10} step={1}
              format={(v) => `${v} positions`}
              onChange={(v) => updateDirty({ max_positions: v })}
            />

            <NumericInput
              label="Max Leverage"
              value={dirtyConfig.max_leverage}
              min={0.5} max={5.0} step={0.5}
              format={(v) => `${v.toFixed(1)}x`}
              onChange={(v) => updateDirty({ max_leverage: v })}
            />

            <NumericInput
              label="Risk Per Trade"
              value={dirtyConfig.risk_per_trade}
              min={0.005} max={0.05} step={0.005}
              format={(v) => `${(v * 100).toFixed(1)}%`}
              onChange={(v) => updateDirty({ risk_per_trade: v })}
            />

            <NumericInput
              label="Max Drawdown (halt)"
              value={dirtyConfig.max_drawdown}
              min={0.05} max={0.30} step={0.01}
              format={(v) => `${(v * 100).toFixed(0)}%`}
              onChange={(v) => updateDirty({ max_drawdown: v })}
            />

            <NumericInput
              label="Total Capital (USD)"
              value={dirtyConfig.total_capital ?? 10000}
              min={1000} max={1_000_000} step={1000}
              format={(v) => `$${v.toLocaleString()}`}
              onChange={(v) => updateDirty({ total_capital: v })}
            />
          </div>

          {/* Strategies */}
          <div className="mb-4">
            <h4 className="text-xs text-slate-500 uppercase tracking-wider mb-3">Active Strategies</h4>
            <div className="space-y-2">
              {strategies.map((strat) => {
                const Icon = STRATEGY_ICONS[strat.name];
                const color = STRATEGY_COLORS[strat.type] ?? "#94a3b8";
                return (
                  <div
                    key={strat.name}
                    className={`flex items-center justify-between p-2.5 rounded-lg border transition-all ${
                      strat.is_active
                        ? "border-slate-600/60 bg-slate-700/30"
                        : "border-slate-700/30 bg-slate-800/20 opacity-60"
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      {Icon && (
                        <span style={{ color }}>
                          <Icon className="w-3.5 h-3.5" />
                        </span>
                      )}
                      <div>
                        <span className="text-sm text-white">{strat.display_name}</span>
                        <div className="flex items-center gap-1.5 mt-0.5">
                          <span
                            className="text-xs px-1.5 py-0 rounded font-mono"
                            style={{
                              color,
                              backgroundColor: color + "15",
                              border: `1px solid ${color}30`,
                            }}
                          >
                            {strat.type}
                          </span>
                          <span className="text-xs text-slate-500">{strat.timeframe}</span>
                        </div>
                      </div>
                    </div>

                    {/* Toggle */}
                    <button
                      onClick={() => handleToggleStrategy(strat.name, !strat.is_active)}
                      className={`relative w-10 h-5 rounded-full transition-all focus:outline-none ${
                        strat.is_active ? "bg-blue-600" : "bg-slate-600"
                      }`}
                    >
                      <span
                        className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${
                          strat.is_active ? "left-5" : "left-0.5"
                        }`}
                      />
                    </button>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Error */}
          {error && (
            <div className="flex items-center gap-2 p-2 rounded-lg bg-red-500/10 border border-red-500/20 mb-3">
              <AlertTriangle className="w-3.5 h-3.5 text-red-400 shrink-0" />
              <span className="text-xs text-red-400">{error}</span>
            </div>
          )}

          {/* Save Button */}
          <div className="flex gap-2">
            <button
              onClick={handleSave}
              disabled={!isDirty || saving}
              className="flex-1 flex items-center justify-center gap-2 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium transition-all"
            >
              {saving ? (
                <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Saving…</>
              ) : saveSuccess ? (
                <><Check className="w-3.5 h-3.5" /> Saved</>
              ) : (
                "Apply Configuration"
              )}
            </button>
            {isDirty && (
              <button
                onClick={() => setDirtyConfig(config)}
                className="px-3 py-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-slate-300 text-sm transition-all"
              >
                Reset
              </button>
            )}
          </div>
        </>
      )}
    </div>
  );
}
