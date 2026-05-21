/**
 * useWebSocket — Hook reactivo para conexión WS con reconexión automática.
 *
 * Features:
 *   - Reconexión automática con backoff exponencial
 *   - Ping/pong keepalive cada 25s
 *   - Typed message parsing (WSMessage)
 *   - Estado de conexión observable
 *   - Callbacks tipados por tipo de evento
 */
import { useCallback, useEffect, useRef, useState } from "react";
import type { WSMessage, WSEventType } from "../types";

type WsStatus = "connecting" | "connected" | "disconnected" | "error";

interface UseWebSocketOptions {
  /** WebSocket URL — e.g. ws://localhost:8005/ws */
  url: string;
  /** Called when any message is received */
  onMessage?: (msg: WSMessage) => void;
  /** Filter by event type */
  onEvent?: Partial<Record<WSEventType, (data: unknown) => void>>;
  /** Whether to auto-reconnect (default: true) */
  autoReconnect?: boolean;
  /** Max reconnect delay in ms (default: 30000) */
  maxReconnectDelay?: number;
  /** Enable console logging (default: false) */
  debug?: boolean;
}

interface UseWebSocketReturn {
  status:    WsStatus;
  sendPing:  () => void;
  lastEvent: WSMessage | null;
  latency:   number | null;   // ms, calculado via ping/pong
}

export function useWebSocket({
  url,
  onMessage,
  onEvent,
  autoReconnect = true,
  maxReconnectDelay = 30_000,
  debug = false,
}: UseWebSocketOptions): UseWebSocketReturn {
  const wsRef         = useRef<WebSocket | null>(null);
  const retryCountRef = useRef(0);
  const pingTimerRef  = useRef<ReturnType<typeof setInterval> | null>(null);
  const pingSentAt    = useRef<number | null>(null);

  const [status,    setStatus]    = useState<WsStatus>("disconnected");
  const [lastEvent, setLastEvent] = useState<WSMessage | null>(null);
  const [latency,   setLatency]   = useState<number | null>(null);

  const log = useCallback(
    (...args: unknown[]) => { if (debug) console.log("[WS]", ...args); },
    [debug]
  );

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setStatus("connecting");
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus("connected");
      retryCountRef.current = 0;
      log("Connected", url);

      // Start keepalive ping
      pingTimerRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          pingSentAt.current = Date.now();
          ws.send("ping");
        }
      }, 25_000);
    };

    ws.onmessage = (ev) => {
      const raw: string = ev.data;

      // Handle pong
      if (raw === "pong") {
        if (pingSentAt.current) {
          setLatency(Date.now() - pingSentAt.current);
          pingSentAt.current = null;
        }
        return;
      }

      try {
        const msg = JSON.parse(raw) as WSMessage;
        setLastEvent(msg);
        onMessage?.(msg);

        // Dispatch to typed handler
        if (onEvent && msg.type) {
          const handler = onEvent[msg.type as WSEventType];
          handler?.(msg.data);
        }
      } catch {
        log("Parse error:", raw.slice(0, 80));
      }
    };

    ws.onclose = (ev) => {
      setStatus("disconnected");
      if (pingTimerRef.current) clearInterval(pingTimerRef.current);
      log("Closed", ev.code, ev.reason);

      if (autoReconnect && ev.code !== 1000) {
        const delay = Math.min(
          500 * 2 ** retryCountRef.current,
          maxReconnectDelay
        );
        retryCountRef.current++;
        log(`Reconnect in ${delay}ms (attempt ${retryCountRef.current})`);
        setTimeout(connect, delay);
      }
    };

    ws.onerror = (err) => {
      setStatus("error");
      log("Error", err);
    };
  }, [url, autoReconnect, maxReconnectDelay, onMessage, onEvent, log]);

  useEffect(() => {
    connect();
    return () => {
      autoReconnect && (retryCountRef.current = Infinity); // Stop auto-reconnect on unmount
      if (pingTimerRef.current) clearInterval(pingTimerRef.current);
      wsRef.current?.close(1000, "unmount");
    };
  }, [connect]);

  const sendPing = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      pingSentAt.current = Date.now();
      wsRef.current.send("ping");
    }
  }, []);

  return { status, sendPing, lastEvent, latency };
}
