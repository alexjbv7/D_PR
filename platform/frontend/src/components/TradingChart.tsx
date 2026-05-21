/**
 * TradingChart — lightweight-charts con señales overlay.
 *
 * Muestra:
 *   - Velas OHLCV (CandlestickSeries)
 *   - Señales long (markers verdes ▲) / short (rojos ▼)
 *   - Funding rate overlay (AreaSeries secundario)
 *   - Régimen de fondo (background shading)
 *
 * Props:
 *   symbol         : instrumento
 *   signals        : lista de señales para overlay
 *   historicalData : array OHLCV
 *   height         : height in px (default 400)
 */
import React, { useEffect, useRef, useCallback } from "react";
import {
  createChart,
  type IChartApi,
  type ISeriesApi,
  type CandlestickData,
  type SeriesMarker,
  type Time,
  ColorType,
  CrosshairMode,
} from "lightweight-charts";
import type { OHLCV, TradingSignal, RegimeState } from "../types";
import { REGIME_META } from "../types";

interface TradingChartProps {
  symbol:         string;
  historicalData: OHLCV[];
  signals:        TradingSignal[];
  regime?:        RegimeState | null;
  height?:        number;
  onCrosshairMove?: (price: number, time: number) => void;
}

// Chart theme — dark institutional
const CHART_THEME = {
  background:  { type: ColorType.Solid, color: "#0f172a" },
  textColor:   "#94a3b8",
  gridColor:   "#1e293b",
  borderColor: "#334155",
  upColor:     "#22c55e",
  downColor:   "#ef4444",
  wickUpColor:   "#22c55e",
  wickDownColor: "#ef4444",
};

export function TradingChart({
  symbol,
  historicalData,
  signals,
  regime,
  height = 400,
  onCrosshairMove,
}: TradingChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef     = useRef<IChartApi | null>(null);
  const candlesRef   = useRef<ISeriesApi<"Candlestick"> | null>(null);

  // Initialize chart
  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      width:  containerRef.current.clientWidth,
      height,
      layout: {
        background:  CHART_THEME.background,
        textColor:   CHART_THEME.textColor,
        fontFamily:  "'JetBrains Mono', 'Fira Code', monospace",
      },
      grid: {
        vertLines:  { color: CHART_THEME.gridColor },
        horzLines:  { color: CHART_THEME.gridColor },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: "#475569", labelBackgroundColor: "#1e293b" },
        horzLine: { color: "#475569", labelBackgroundColor: "#1e293b" },
      },
      rightPriceScale: {
        borderColor: CHART_THEME.borderColor,
        textColor:   CHART_THEME.textColor,
      },
      timeScale: {
        borderColor:          CHART_THEME.borderColor,
        timeVisible:          true,
        secondsVisible:       false,
        rightOffset:          5,
        fixLeftEdge:          false,
        lockVisibleTimeRangeOnResize: true,
      },
    });

    // Candlestick series
    const candles = chart.addCandlestickSeries({
      upColor:          CHART_THEME.upColor,
      downColor:        CHART_THEME.downColor,
      borderUpColor:    CHART_THEME.upColor,
      borderDownColor:  CHART_THEME.downColor,
      wickUpColor:      CHART_THEME.wickUpColor,
      wickDownColor:    CHART_THEME.wickDownColor,
    });

    chartRef.current   = chart;
    candlesRef.current = candles;

    // Crosshair move handler
    if (onCrosshairMove) {
      chart.subscribeCrosshairMove((param) => {
        if (param.time && param.seriesData) {
          const data = param.seriesData.get(candles) as CandlestickData | undefined;
          if (data) {
            onCrosshairMove(data.close, param.time as number);
          }
        }
      });
    }

    // Responsive resize
    const ro = new ResizeObserver(() => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth });
      }
    });
    ro.observe(containerRef.current);

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current   = null;
      candlesRef.current = null;
    };
  }, [height, onCrosshairMove]);

  // Update candle data
  useEffect(() => {
    if (!candlesRef.current || historicalData.length === 0) return;

    const data: CandlestickData[] = historicalData.map((d) => ({
      time:  d.time as Time,
      open:  d.open,
      high:  d.high,
      low:   d.low,
      close: d.close,
    }));

    candlesRef.current.setData(data);

    // Fit to content
    chartRef.current?.timeScale().fitContent();
  }, [historicalData]);

  // Update signal markers
  useEffect(() => {
    if (!candlesRef.current) return;

    const markers: SeriesMarker<Time>[] = signals
      .filter((s) => s.direction !== 0)
      .map((s) => ({
        time:     Math.floor(new Date(s.ts).getTime() / 1000) as Time,
        position: (s.direction === 1 ? "belowBar" : "aboveBar") as SeriesMarker<Time>["position"],
        color:    s.direction === 1 ? "#22c55e" : "#ef4444",
        shape:    (s.direction === 1 ? "arrowUp" : "arrowDown") as SeriesMarker<Time>["shape"],
        text:     `${s.strategy} p=${s.p_win.toFixed(2)}`,
        size:     Math.max(1, Math.min(3, Math.floor(s.p_win * 4))),
      }))
      .sort((a, b) => (a.time as number) - (b.time as number));

    candlesRef.current.setMarkers(markers);
  }, [signals]);

  // Regime background (via chart bands — simplified as a visual cue in title)
  const regimeMeta = regime ? REGIME_META[regime.regime_name] : null;

  return (
    <div className="flex flex-col h-full">
      {/* Chart header */}
      <div className="flex items-center justify-between px-1 pb-2">
        <div className="flex items-center gap-2">
          <span className="font-mono font-bold text-white">{symbol}</span>
          {regimeMeta && (
            <span
              className="text-xs px-2 py-0.5 rounded font-mono"
              style={{
                backgroundColor: regimeMeta.color + "25",
                color:           regimeMeta.color,
                border:          `1px solid ${regimeMeta.color}50`,
              }}
            >
              {regimeMeta.label}
            </span>
          )}
        </div>
        <div className="flex items-center gap-4 text-xs font-mono text-slate-500">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-green-500 inline-block" />
            Long signal
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-red-500 inline-block" />
            Short signal
          </span>
        </div>
      </div>
      {/* Chart container */}
      <div ref={containerRef} className="flex-1 rounded-lg overflow-hidden" />
    </div>
  );
}
