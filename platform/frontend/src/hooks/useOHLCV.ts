/**
 * useOHLCV — Hook que obtiene barras OHLCV del openbb-adapter y las
 * mantiene actualizadas.
 *
 * Comportamiento:
 *   - Fetch inicial al montar o cuando cambian symbol / interval.
 *   - Re-fetch automático cada `refreshMs` (default 60 s para 1h,
 *     30 s para intervalos menores).
 *   - Append incremental: si ya hay datos, sólo solicita desde la
 *     última barra para reducir carga. Cuando hay gap > 1 bar se
 *     hace refetch completo.
 *
 * @example
 * const { bars, loading, error } = useOHLCV("BTCUSDT", "1h");
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { fetchOHLCV } from "../api/openbbAdapter";
import type { OHLCV } from "../types";

interface UseOHLCVOptions {
  /** Symbol to fetch — openbb uses base ticker, e.g. "BTC" or "BTCUSDT". */
  symbol: string;
  /** Interval string understood by openbb-adapter: "1m" | "5m" | "15m" | "1h" | "4h" | "1d". */
  interval?: string;
  /** How many days of history to seed on first load (default: 90). */
  lookbackDays?: number;
  /** Refresh interval in milliseconds (default: 60_000). */
  refreshMs?: number;
  /** If true, disable automatic refresh. */
  paused?: boolean;
}

interface UseOHLCVReturn {
  bars:    OHLCV[];
  loading: boolean;
  error:   string | null;
  refetch: () => void;
}

/** Convert a symbol like "BTCUSDT" → "BTC" for openbb-adapter queries. */
function toBaseTicker(symbol: string): string {
  // Remove common quote suffixes
  return symbol
    .replace(/USDT$/, "")
    .replace(/USD$/, "")
    .replace(/BUSD$/, "")
    .toUpperCase();
}

function daysAgoISO(days: number): string {
  const d = new Date();
  d.setDate(d.getDate() - days);
  return d.toISOString().split("T")[0]!;
}

export function useOHLCV({
  symbol,
  interval    = "1h",
  lookbackDays = 90,
  refreshMs   = 60_000,
  paused      = false,
}: UseOHLCVOptions): UseOHLCVReturn {
  const [bars,    setBars]    = useState<OHLCV[]>([]);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState<string | null>(null);

  // Avoid stale closures in the refresh callback
  const barsRef     = useRef<OHLCV[]>([]);
  const symbolRef   = useRef(symbol);
  const intervalRef = useRef(interval);

  // Sync refs
  barsRef.current     = bars;
  symbolRef.current   = symbol;
  intervalRef.current = interval;

  const doFetch = useCallback(async (fullRefetch = false) => {
    const base      = toBaseTicker(symbolRef.current);
    const existing  = barsRef.current;
    const startDate = (fullRefetch || existing.length === 0)
      ? daysAgoISO(lookbackDays)
      : daysAgoISO(2);   // incremental: only last 2 days to catch new bars

    try {
      const fresh = await fetchOHLCV(base, intervalRef.current, startDate);
      if (fresh.length === 0) {
        if (fullRefetch) setError("No data returned from openbb-adapter");
        return;
      }

      if (fullRefetch || existing.length === 0) {
        setBars(fresh);
        setError(null);
        return;
      }

      // Merge: keep existing, append/update any new bars
      const existingSet = new Set(existing.map((b) => b.time));
      const newBars = fresh.filter((b) => !existingSet.has(b.time));
      if (newBars.length > 0) {
        setBars((prev) => [...prev, ...newBars].sort((a, b) => a.time - b.time));
      }
      setError(null);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      // eslint-disable-next-line no-console
      console.warn("useOHLCV fetch error", msg);
      if (fullRefetch) setError(msg);
      // On incremental failures keep existing data — chart stays usable
    }
  }, [lookbackDays]);

  // Full refetch on symbol/interval change
  useEffect(() => {
    setLoading(true);
    setBars([]);
    setError(null);
    doFetch(true).finally(() => setLoading(false));
  }, [symbol, interval, doFetch]);

  // Periodic incremental refresh
  useEffect(() => {
    if (paused) return;
    const id = window.setInterval(() => doFetch(false), refreshMs);
    return () => window.clearInterval(id);
  }, [doFetch, refreshMs, paused]);

  const refetch = useCallback(() => {
    setLoading(true);
    doFetch(true).finally(() => setLoading(false));
  }, [doFetch]);

  return { bars, loading, error, refetch };
}
