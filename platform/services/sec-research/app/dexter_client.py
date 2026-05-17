"""
DexterClient — Cliente para la API de Dexter (SEC filings NLP).

Dexter es un servicio de análisis de filings de la SEC que provee:
  - Sentiment analysis de 10-K, 10-Q, 8-K filings
  - Entity extraction (companies, executives, products)
  - Key risk factor changes between filings
  - Earnings surprise signals
  - Institutional ownership changes (13F)

Relevancia para crypto trading:
  - 8-K de exchanges listados (Coinbase, Robinhood) → regulatory signals
  - 13F de fondos institucionales → smart money BTC/ETH positions
  - 10-K risk factor changes → emerging regulatory language

API Base: https://api.dexter.io/v2  (o configurable via env)
Auth: Bearer token en header X-Api-Key
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

_BASE_URL = os.getenv("DEXTER_API_URL", "https://api.dexter.io/v2")
_API_KEY  = os.getenv("DEXTER_API_KEY", "")


@dataclass
class FilingSentiment:
    filing_id:    str
    ticker:       str
    form_type:    str       # 8-K | 10-K | 10-Q | 13F | etc.
    filed_at:     str       # ISO date
    sentiment:    str       # positive | negative | neutral | mixed
    score:        float     # -1 to +1
    confidence:   float
    summary:      str
    risk_delta:   Optional[float] = None   # risk factor change vs prev
    key_topics:   list[str] = None         # extracted topics

    def __post_init__(self):
        if self.key_topics is None:
            self.key_topics = []

    def to_dict(self) -> dict:
        return {
            "filing_id":  self.filing_id,
            "ticker":     self.ticker,
            "form_type":  self.form_type,
            "filed_at":   self.filed_at,
            "sentiment":  self.sentiment,
            "score":      round(self.score, 4),
            "confidence": round(self.confidence, 3),
            "summary":    self.summary[:200],
            "risk_delta": self.risk_delta,
            "key_topics": self.key_topics,
        }


@dataclass
class InstitutionalPosition:
    """From 13F filings — tracks institutional BTC/ETH related holdings."""
    institution:  str
    ticker:       str       # e.g. GBTC, IBIT, FBTC, COIN
    shares:       int
    value_usd:    float
    period:       str       # YYYY-QN
    change_pct:   float     # vs previous period


class DexterClient:
    """
    Async HTTP client for Dexter SEC filing analysis API.

    Usage
    -----
    client = DexterClient(api_key="...")
    await client.connect()
    sentiment = await client.get_filing_sentiment("COIN", form_type="8-K")
    """

    # Crypto-relevant tickers to track
    CRYPTO_TICKERS = ["COIN", "HOOD", "MSTR", "MARA", "RIOT", "CLSK", "BTBT"]
    # Bitcoin ETF tickers
    BTC_ETF_TICKERS = ["IBIT", "FBTC", "BTCO", "BITB", "ARKB", "HODL", "BRRR", "GBTC"]

    def __init__(
        self,
        api_key:  str = _API_KEY,
        base_url: str = _BASE_URL,
        timeout:  int = 30,
    ):
        self._api_key  = api_key
        self._base_url = base_url
        self._timeout  = timeout
        self._session  = None

    async def connect(self) -> None:
        try:
            import aiohttp
            self._session = aiohttp.ClientSession(
                headers={"X-Api-Key": self._api_key, "Accept": "application/json"},
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            )
            logger.info("dexter_client.connected", base_url=self._base_url)
        except ImportError:
            logger.warning("dexter_client.aiohttp_not_installed")

    async def close(self) -> None:
        if self._session:
            await self._session.close()

    async def get_recent_filings(
        self,
        tickers: list[str] | None = None,
        form_types: list[str] | None = None,
        days_back: int = 7,
    ) -> list[FilingSentiment]:
        """
        Fetch and parse recent filings for given tickers.
        Returns empty list if API unavailable (graceful degradation).
        """
        tickers    = tickers    or self.CRYPTO_TICKERS
        form_types = form_types or ["8-K", "10-Q", "10-K"]

        results = []
        for ticker in tickers:
            for form_type in form_types:
                try:
                    filings = await self._fetch_filings(ticker, form_type, days_back)
                    results.extend(filings)
                except Exception as e:
                    logger.debug("dexter_client.ticker_error",
                                 ticker=ticker, form=form_type, error=str(e))
        logger.info("dexter_client.filings_fetched", count=len(results))
        return results

    async def get_institutional_positions(
        self,
        period: str | None = None,
    ) -> list[InstitutionalPosition]:
        """
        Fetch 13F institutional positions in BTC ETF and crypto-related tickers.
        """
        if not self._session or not self._api_key:
            return self._mock_institutional_positions()

        try:
            url = f"{self._base_url}/institutional/positions"
            params = {
                "tickers": ",".join(self.BTC_ETF_TICKERS + self.CRYPTO_TICKERS),
                "period":  period or "latest",
            }
            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.warning("dexter_client.positions_error", status=resp.status)
                    return []
                data = await resp.json()
                return self._parse_positions(data)
        except Exception as e:
            logger.error("dexter_client.positions_fetch_error", error=str(e))
            return []

    # ------------------------------------------------------------------

    async def _fetch_filings(
        self,
        ticker: str,
        form_type: str,
        days_back: int,
    ) -> list[FilingSentiment]:
        if not self._session or not self._api_key:
            return self._mock_filings(ticker, form_type)

        url = f"{self._base_url}/filings/sentiment"
        params = {
            "ticker":    ticker,
            "form_type": form_type,
            "days_back": days_back,
            "limit":     5,
        }
        try:
            async with self._session.get(url, params=params) as resp:
                if resp.status == 404:
                    return []
                if resp.status != 200:
                    logger.warning("dexter_client.api_error",
                                   status=resp.status, ticker=ticker)
                    return []
                data = await resp.json()
                return self._parse_filings(ticker, data)
        except asyncio.TimeoutError:
            logger.warning("dexter_client.timeout", ticker=ticker)
            return []

    def _parse_filings(self, ticker: str, data: dict | list) -> list[FilingSentiment]:
        items = data if isinstance(data, list) else data.get("results", data.get("data", []))
        results = []
        for item in items:
            try:
                results.append(FilingSentiment(
                    filing_id=str(item.get("id", item.get("filing_id", ""))),
                    ticker=ticker,
                    form_type=item.get("form_type", ""),
                    filed_at=item.get("filed_at", item.get("date", "")),
                    sentiment=item.get("sentiment", "neutral"),
                    score=float(item.get("score", item.get("sentiment_score", 0.0))),
                    confidence=float(item.get("confidence", 0.7)),
                    summary=item.get("summary", item.get("excerpt", "")),
                    risk_delta=item.get("risk_delta"),
                    key_topics=item.get("topics", item.get("key_topics", [])),
                ))
            except (KeyError, TypeError, ValueError) as e:
                logger.debug("dexter_client.parse_skip", error=str(e))
        return results

    def _parse_positions(self, data: dict | list) -> list[InstitutionalPosition]:
        items = data if isinstance(data, list) else data.get("data", [])
        results = []
        for item in items:
            try:
                results.append(InstitutionalPosition(
                    institution=item.get("institution", ""),
                    ticker=item.get("ticker", ""),
                    shares=int(item.get("shares", 0)),
                    value_usd=float(item.get("value_usd", 0)),
                    period=item.get("period", ""),
                    change_pct=float(item.get("change_pct", 0)),
                ))
            except (KeyError, TypeError, ValueError):
                pass
        return results

    # ------------------------------------------------------------------
    # Mock data for development without API key

    @staticmethod
    def _mock_filings(ticker: str, form_type: str) -> list[FilingSentiment]:
        return [FilingSentiment(
            filing_id=f"mock-{ticker}-{form_type}",
            ticker=ticker,
            form_type=form_type,
            filed_at=datetime.utcnow().isoformat(),
            sentiment="neutral",
            score=0.0,
            confidence=0.0,
            summary="[Mock — no Dexter API key configured]",
        )]

    @staticmethod
    def _mock_institutional_positions() -> list[InstitutionalPosition]:
        return [
            InstitutionalPosition("BlackRock", "IBIT",  350_000_000, 15_000_000_000, "2025-Q1", 12.5),
            InstitutionalPosition("Fidelity",  "FBTC",  180_000_000,  7_800_000_000, "2025-Q1", 8.2),
            InstitutionalPosition("ARK Invest","ARKB",   45_000_000,  1_950_000_000, "2025-Q1", -3.1),
        ]
