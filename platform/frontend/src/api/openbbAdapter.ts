/**
 * openbb-adapter REST client — subset used by the frontend.
 *
 * Full API: GET /crypto/ohlcv/{symbol}  → list of OHLCV bars
 *           GET /crypto/funding/{symbol} → list of funding rate records
 *
 * Backend: platform/services/openbb-adapter/app/routers/crypto.py
 * Base URL: VITE_OPENBB_URL (defaults to http://localhost:8009)
 */
import type { OHLCV } from "../types";

const BASE_URL =
  import.meta.env.VITE_OPENBB_URL ?? "http://localhost:8009";

async function jsonFetch<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`);
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`openbb-adapter ${path} → ${res.status}: ${body}`);
  }
  return (await res.json()) as T;
}

// Shape returned by GET /crypto/ohlcv/{symbol}
interface RawOHLCVBar {
  date:   string | number;   // ISO string or unix seconds from pandas index
  open:   number;
  high:   number;
  low:    number;
  close:  number;
  volume: number;
  [k: string]: unknown;
}

/** Parse a date field from openbb-adapter into a unix timestamp (seconds). */
function toUnixSeconds(date: string | number): number {
  if (typeof date === "number") return date;
  // "2024-01-15T00:00:00" or "2024-01-15" or "2024-01-15 00:00:00"
  const d = new Date(date);
  if (!Number.isNaN(d.getTime())) return Math.floor(d.getTime() / 1000);
  // Fallback: try replacing space with T
  const d2 = new Date(date.replace(" ", "T"));
  return Math.floor(d2.getTime() / 1000);
}

/** Fetch and normalise OHLCV data for a symbol. */
export async function fetchOHLCV(
  symbol: string,
  interval = "1h",
  startDate = "2025-01-01",
): Promise<OHLCV[]> {
  const path =
    `/crypto/ohlcv/${encodeURIComponent(symbol)}` +
    `?interval=${encodeURIComponent(interval)}` +
    `&start_date=${encodeURIComponent(startDate)}`;

  const raw = await jsonFetch<RawOHLCVBar[]>(path);

  return raw
    .filter((bar) => bar.date != null && bar.close != null)
    .map((bar) => ({
      time:   toUnixSeconds(bar.date),
      open:   Number(bar.open),
      high:   Number(bar.high),
      low:    Number(bar.low),
      close:  Number(bar.close),
      volume: Number(bar.volume ?? 0),
    }))
    .sort((a, b) => a.time - b.time);   // ascending — required by lightweight-charts
}

// ---------------------------------------------------------------------------
// Funding rate (optional — used for chart overlay in future iterations)
// ---------------------------------------------------------------------------

interface RawFundingBar {
  date:         string | number;
  funding_rate: number;
  [k: string]: unknown;
}

export interface FundingBar {
  time:  number;
  rate:  number;
}

export async function fetchFundingRate(
  symbol: string,
  provider = "deribit",
): Promise<FundingBar[]> {
  const raw = await jsonFetch<RawFundingBar[]>(
    `/crypto/funding/${encodeURIComponent(symbol)}?provider=${encodeURIComponent(provider)}`,
  );
  return raw
    .filter((b) => b.date != null)
    .map((b) => ({
      time: toUnixSeconds(b.date),
      rate: Number(b.funding_rate ?? 0),
    }))
    .sort((a, b) => a.time - b.time);
}
