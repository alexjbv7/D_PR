"""
LiquidityScanner — Escaneo de liquidez on-chain y de derivados.

Monitora:
  1. Open Interest (OI) — crecimiento = apalancamiento acumulado
  2. Liquidaciones — clusters de liquidaciones = nivel de precio "imán"
  3. Funding rate extremos — carry trade saturation
  4. Exchange reserves — BTC/ETH en exchanges (diminución = acumulación)

Todas las métricas normalizadas a z-score rolling (30d) para comparabilidad.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class LiquiditySnapshot:
    ts:                   str
    symbol:               str
    open_interest_usd:    float
    oi_z_score:           float
    long_liquidations_1h: float     # USD
    short_liquidations_1h: float    # USD
    liq_imbalance:        float     # +1 = all longs, -1 = all shorts
    exchange_reserve:     float     # BTC or ETH balance on exchanges
    reserve_z_score:      float
    funding_z:            float
    squeeze_risk:         str       # none | low | medium | high | critical

    def to_dict(self) -> dict:
        return self.__dict__


class LiquidityScanner:
    """
    Processes derivatives and on-chain data into liquidity risk metrics.

    Usage
    -----
    scanner = LiquidityScanner()
    snapshot = scanner.process(raw_data)
    """

    def __init__(self, window: int = 30 * 24):  # 30d in hourly bars
        self._window = window
        # Rolling buffers
        self._oi:      dict[str, deque[float]] = {}
        self._reserve: dict[str, deque[float]] = {}
        self._funding: dict[str, deque[float]] = {}

    def process(self, data: dict) -> LiquiditySnapshot:
        """
        Parameters
        ----------
        data : dict with keys:
            symbol, open_interest_usd, long_liq_1h, short_liq_1h,
            exchange_reserve, funding_rate
        """
        symbol   = data.get("symbol", "BTCUSDT")
        oi       = float(data.get("open_interest_usd", 0))
        long_liq = float(data.get("long_liq_1h", 0))
        shrt_liq = float(data.get("short_liq_1h", 0))
        reserve  = float(data.get("exchange_reserve", 0))
        funding  = float(data.get("funding_rate", 0))

        # Ensure buffers
        for buf_dict in (self._oi, self._reserve, self._funding):
            if symbol not in buf_dict:
                buf_dict[symbol] = deque(maxlen=self._window)

        # Update buffers
        self._oi[symbol].append(oi)
        self._reserve[symbol].append(reserve)
        self._funding[symbol].append(funding)

        # Z-scores
        oi_z  = self._z_score(self._oi[symbol])
        res_z = self._z_score(self._reserve[symbol])

        # Funding z-score (rolling 7d = 168 bars)
        fund_buf = list(self._funding[symbol])
        if len(fund_buf) >= 10:
            fund_mu = np.mean(fund_buf[:-1])
            fund_sd = np.std(fund_buf[:-1]) + 1e-9
            fund_z  = (funding - fund_mu) / fund_sd
        else:
            fund_z = 0.0

        # Liquidation imbalance: +1=all longs liquidated, -1=all shorts
        total_liq = long_liq + shrt_liq
        liq_imbal = (long_liq - shrt_liq) / (total_liq + 1e-9) if total_liq > 0 else 0.0

        # Squeeze risk
        squeeze = self._classify_squeeze(
            oi_z=float(oi_z),
            fund_z=float(fund_z),
            liq_imbal=float(liq_imbal),
            reserve_z=float(res_z),
        )

        snap = LiquiditySnapshot(
            ts=datetime.now(timezone.utc).isoformat(),
            symbol=symbol,
            open_interest_usd=round(oi, 0),
            oi_z_score=round(float(oi_z), 3),
            long_liquidations_1h=round(long_liq, 0),
            short_liquidations_1h=round(shrt_liq, 0),
            liq_imbalance=round(float(liq_imbal), 4),
            exchange_reserve=round(reserve, 2),
            reserve_z_score=round(float(res_z), 3),
            funding_z=round(float(fund_z), 3),
            squeeze_risk=squeeze,
        )

        if squeeze in ("high", "critical"):
            logger.warning("liquidity.squeeze_risk",
                           symbol=symbol, level=squeeze,
                           oi_z=snap.oi_z_score, fund_z=snap.funding_z)

        return snap

    @staticmethod
    def _z_score(buf: deque) -> float:
        arr = np.array(buf)
        if len(arr) < 2:
            return 0.0
        mu, sd = arr[:-1].mean(), arr[:-1].std()
        return float((arr[-1] - mu) / (sd + 1e-9))

    @staticmethod
    def _classify_squeeze(
        oi_z:      float,
        fund_z:    float,
        liq_imbal: float,
        reserve_z: float,
    ) -> str:
        score = 0.0
        if oi_z > 2.0:   score += 1.0
        if oi_z > 3.0:   score += 1.0
        if abs(fund_z) > 2.5: score += 1.0
        if abs(fund_z) > 4.0: score += 1.0
        if abs(liq_imbal) > 0.7: score += 0.5
        if reserve_z < -2.0: score += 0.5   # reserves leaving → accumulation

        if score >= 4.0:   return "critical"
        elif score >= 3.0: return "high"
        elif score >= 2.0: return "medium"
        elif score >= 1.0: return "low"
        return "none"
