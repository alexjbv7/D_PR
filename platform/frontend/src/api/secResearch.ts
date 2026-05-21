/**
 * SEC Research REST client.
 *
 * Mirrors the FastAPI endpoints in platform/services/sec-research/app/main.py.
 * Base URL comes from VITE_SEC_URL (defaults to http://localhost:8008).
 */
import type {
  FilingSignal,
  InstitutionalPositionsResponse,
  RegulatorySignalsResponse,
  SECHealth,
} from "../types";

const BASE_URL =
  import.meta.env.VITE_SEC_URL ?? "http://localhost:8008";

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`sec-research ${path} → ${res.status}: ${body}`);
  }
  return (await res.json()) as T;
}

export const secApi = {
  /** GET /health */
  health: () => json<SECHealth>("/health"),

  /**
   * GET /filings/recent?days_back=7&form_types=8-K,10-Q
   * Returns an array of FilingSignal.
   */
  recentFilings: (daysBack = 7, formTypes = "8-K,10-Q") =>
    json<FilingSignal[]>(
      `/filings/recent?days_back=${daysBack}&form_types=${encodeURIComponent(formTypes)}`,
    ),

  /**
   * GET /filings/{ticker}?days_back=30
   */
  filingsByTicker: (ticker: string, daysBack = 30) =>
    json<FilingSignal[]>(
      `/filings/${encodeURIComponent(ticker)}?days_back=${daysBack}`,
    ),

  /** GET /institutional/positions */
  institutionalPositions: () =>
    json<InstitutionalPositionsResponse>("/institutional/positions"),

  /** GET /signals/regulatory */
  regulatorySignals: () =>
    json<RegulatorySignalsResponse>("/signals/regulatory"),

  /** POST /analyze */
  analyze: (text: string, context = "") =>
    json<{ sentiment: Record<string, unknown>; entities: Record<string, unknown> }>(
      "/analyze",
      {
        method: "POST",
        body: JSON.stringify({ text, context }),
      },
    ),
};
