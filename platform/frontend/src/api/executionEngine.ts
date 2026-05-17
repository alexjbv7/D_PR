/**
 * Execution-engine REST client.
 *
 * Mirrors the FastAPI endpoints in platform/services/execution-engine/app/main.py.
 * Base URL comes from VITE_EXECUTION_URL (defaults to http://localhost:8010).
 */
import type {
  ExecutionAccount,
  ExecutionHealth,
  ExecutionOrderResult,
  ExecutionPosition,
} from "../types";

const BASE_URL =
  import.meta.env.VITE_EXECUTION_URL ?? "http://localhost:8010";

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`execution-engine ${path} → ${res.status}: ${body}`);
  }
  return (await res.json()) as T;
}

export const executionApi = {
  /** GET /health — counters + venues + kill-switch state. */
  health: () => json<ExecutionHealth>("/health"),

  /** GET /api/positions[?venue=...] — internal position snapshot. */
  positions: (venue?: string) =>
    json<{ count: number; positions: ExecutionPosition[] }>(
      `/api/positions${venue ? `?venue=${encodeURIComponent(venue)}` : ""}`,
    ),

  /** GET /api/orders/recent?limit=N — most recent OrderResults. */
  recentOrders: (limit = 50) =>
    json<{ count: number; orders: ExecutionOrderResult[] }>(
      `/api/orders/recent?limit=${limit}`,
    ),

  /** GET /api/account/{venue} — broker account snapshot. */
  account: (venue: string) =>
    json<ExecutionAccount>(`/api/account/${encodeURIComponent(venue)}`),

  /** POST /api/kill_switch/trip — halt new signal processing. */
  tripKillSwitch: () =>
    json<{ kill_switch: boolean }>("/api/kill_switch/trip", { method: "POST" }),

  /** POST /api/kill_switch/reset — resume signal processing. */
  resetKillSwitch: () =>
    json<{ kill_switch: boolean }>("/api/kill_switch/reset", { method: "POST" }),
};
