"""
PolymarketParser — Parseo del CLOB de Polymarket para mercados cripto.

Polymarket opera un Central Limit Order Book (CLOB) on-chain (Polygon).
Cada mercado tiene dos tokens: YES y NO. El precio del token YES
(0-1 USDC) = probabilidad implícita del mercado de que ocurra el evento.

Casos de uso para trading:
  - P(BTC > X por fecha Y) como prior para señales direccionales
  - P(mercado en recesión) para ajustar macro bias
  - P(evento regulatorio) para gestión de riesgo event-driven

API v2 endpoints usados:
  GET /markets          — lista de mercados activos
  GET /markets/{id}     — detalle + orderbook
  GET /prices-history   — serie temporal de probabilidades
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


# Palabras clave para clasificar mercados por asset subyacente
_ASSET_PATTERNS: dict[str, list[str]] = {
    "BTC":  ["bitcoin", "btc"],
    "ETH":  ["ethereum", "eth"],
    "SOL":  ["solana", "sol"],
    "BNB":  ["bnb", "binance coin"],
    "XRP":  ["xrp", "ripple"],
    "AVAX": ["avalanche", "avax"],
    "MACRO": ["fed", "federal reserve", "recession", "inflation", "interest rate",
              "cpi", "fomc", "gdp"],
    "REGULATORY": ["sec", "cftc", "etf", "spot etf", "ban", "regulation",
                   "congress", "senate"],
}

# Directionality keywords
_DIRECTION_PATTERNS: dict[str, list[str]] = {
    "bullish": ["above", "over", "exceed", "higher", "rally", "bull", "reach",
                "hit", "surpass", "break", "ath"],
    "bearish": ["below", "under", "drop", "lower", "bear", "fall", "crash",
                "decline", "dump"],
}


@dataclass
class PolymarketEvent:
    market_id:      str
    slug:           str
    question:       str
    end_date:       Optional[str]
    yes_price:      float          # 0-1 probability
    no_price:       float
    volume_24h:     float          # USDC
    open_interest:  float          # USDC
    category:       str            # crypto | macro | regulatory | other
    asset:          Optional[str]  # BTC | ETH | SOL | etc.
    direction:      Optional[str]  # bullish | bearish | neutral
    liquidity:      str            # low | medium | high

    @property
    def probability(self) -> float:
        """YES token price = market-implied probability."""
        return self.yes_price

    @property
    def is_tradeable(self) -> bool:
        """Only use markets with sufficient liquidity."""
        return self.open_interest >= 10_000 and self.liquidity != "low"

    def to_dict(self) -> dict:
        return {
            "market_id":     self.market_id,
            "slug":          self.slug,
            "question":      self.question,
            "end_date":      self.end_date,
            "probability":   round(self.yes_price, 4),
            "volume_24h":    round(self.volume_24h, 2),
            "open_interest": round(self.open_interest, 2),
            "category":      self.category,
            "asset":         self.asset,
            "direction":     self.direction,
            "liquidity":     self.liquidity,
            "is_tradeable":  self.is_tradeable,
        }


class PolymarketParser:
    """
    Parses Polymarket REST API responses into PolymarketEvent objects.

    Usage
    -----
    parser = PolymarketParser()
    events = parser.parse_markets(raw_response)
    """

    def parse_markets(self, raw_response: dict | list) -> list[PolymarketEvent]:
        """
        Parse /markets or /markets?active=true response.

        Returns list of PolymarketEvent, filtered to crypto/macro relevance.
        """
        if isinstance(raw_response, list):
            items = raw_response
        else:
            items = raw_response.get("data", raw_response.get("markets", []))

        events = []
        for item in items:
            try:
                evt = self._parse_single(item)
                if evt and (evt.category in ("crypto", "macro", "regulatory")):
                    events.append(evt)
            except Exception as e:
                logger.debug("polymarket.parse_skip", error=str(e))

        logger.info("polymarket.parsed", total=len(items), relevant=len(events))
        return events

    def parse_single_market(self, raw: dict) -> Optional[PolymarketEvent]:
        """Parse a single /markets/{id} response."""
        try:
            return self._parse_single(raw)
        except Exception as e:
            logger.error("polymarket.parse_error", error=str(e))
            return None

    # ------------------------------------------------------------------

    def _parse_single(self, item: dict) -> Optional[PolymarketEvent]:
        market_id = str(item.get("id", item.get("market_id", "")))
        if not market_id:
            return None

        slug     = item.get("slug", "")
        question = item.get("question", item.get("title", ""))
        end_date = item.get("end_date", item.get("endDate"))

        # Tokens: YES token is first by convention
        tokens = item.get("tokens", [])
        yes_price, no_price = self._extract_prices(item, tokens)

        # Volume and OI
        vol_24h = float(item.get("volume24hr", item.get("volume_24h", 0)) or 0)
        oi      = float(item.get("openInterest", item.get("open_interest", 0)) or 0)

        # Classify
        question_lower = question.lower()
        category  = self._classify_category(question_lower)
        asset     = self._classify_asset(question_lower)
        direction = self._classify_direction(question_lower)
        liquidity = self._classify_liquidity(oi)

        return PolymarketEvent(
            market_id=market_id,
            slug=slug,
            question=question,
            end_date=end_date,
            yes_price=round(yes_price, 4),
            no_price=round(no_price, 4),
            volume_24h=vol_24h,
            open_interest=oi,
            category=category,
            asset=asset,
            direction=direction,
            liquidity=liquidity,
        )

    @staticmethod
    def _extract_prices(item: dict, tokens: list[dict]) -> tuple[float, float]:
        """Extract YES/NO prices from various API response formats."""
        # Format 1: outcomes array with prices
        outcomes = item.get("outcomes", [])
        outcome_prices = item.get("outcomePrices", [])
        if outcomes and outcome_prices:
            try:
                for i, o in enumerate(outcomes):
                    if str(o).upper() == "YES":
                        yes = float(outcome_prices[i])
                        no  = 1.0 - yes
                        return yes, no
            except (IndexError, TypeError, ValueError):
                pass

        # Format 2: tokens array with price field
        if tokens:
            yes_token = next(
                (t for t in tokens if str(t.get("outcome", "")).upper() == "YES"),
                tokens[0] if tokens else None
            )
            no_token = next(
                (t for t in tokens if str(t.get("outcome", "")).upper() == "NO"),
                tokens[1] if len(tokens) > 1 else None
            )
            yes_p = float(yes_token.get("price", 0.5)) if yes_token else 0.5
            no_p  = float(no_token.get("price", 1.0 - yes_p)) if no_token else 1.0 - yes_p
            return yes_p, no_p

        # Format 3: best_ask / best_bid
        yes_p = float(item.get("bestAsk", item.get("best_ask", 0.5)) or 0.5)
        return yes_p, 1.0 - yes_p

    @staticmethod
    def _classify_category(question: str) -> str:
        crypto_terms   = ["bitcoin", "btc", "ethereum", "eth", "crypto", "altcoin",
                          "solana", "bnb", "xrp", "defi", "nft", "blockchain", "usdt"]
        macro_terms    = ["fed", "recession", "inflation", "interest rate", "gdp",
                          "fomc", "cpi", "unemployment", "rate cut", "rate hike"]
        reg_terms      = ["sec", "cftc", "etf", "regulation", "ban", "legal",
                          "congress", "senate", "government"]

        if any(t in question for t in crypto_terms):
            return "crypto"
        if any(t in question for t in macro_terms):
            return "macro"
        if any(t in question for t in reg_terms):
            return "regulatory"
        return "other"

    @staticmethod
    def _classify_asset(question: str) -> Optional[str]:
        for asset, keywords in _ASSET_PATTERNS.items():
            if any(kw in question for kw in keywords):
                return asset
        return None

    @staticmethod
    def _classify_direction(question: str) -> Optional[str]:
        for direction, keywords in _DIRECTION_PATTERNS.items():
            if any(kw in question for kw in keywords):
                return direction
        return "neutral"

    @staticmethod
    def _classify_liquidity(oi: float) -> str:
        if oi >= 500_000:
            return "high"
        if oi >= 50_000:
            return "medium"
        return "low"
