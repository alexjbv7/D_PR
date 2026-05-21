"""
SEC Research Service — FastAPI app.

Analiza filings de la SEC relevantes para crypto trading y publica
señales regulatorias y de sentimiento institucional.

Endpoints:
  GET  /health
  GET  /filings/recent            — últimos filings con sentimiento
  GET  /filings/{ticker}          — filings por ticker
  GET  /institutional/positions   — posiciones 13F en ETFs cripto
  GET  /signals/regulatory        — señales regulatorias activas
  POST /analyze                   — analizar texto arbitrario
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import structlog
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .dexter_client import DexterClient, FilingSentiment
from .entity_extractor import EntityExtractor
from .sentiment_classifier import SentimentClassifier

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

dexter_client  = DexterClient()
sentiment_clf  = SentimentClassifier(use_finbert=os.getenv("USE_FINBERT", "false").lower() == "true")
entity_extr    = EntityExtractor()
_producer      = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _producer

    await dexter_client.connect()
    await sentiment_clf.load()
    await entity_extr.load()

    # Kafka producer for publishing SEC signals
    try:
        from aiokafka import AIOKafkaProducer
        _producer = AIOKafkaProducer(
            bootstrap_servers=os.getenv("KAFKA_SERVERS", "kafka:9092"),
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        await _producer.start()
        logger.info("sec_research.kafka_connected")
    except Exception as e:
        logger.warning("sec_research.kafka_unavailable", error=str(e))

    # Background polling
    asyncio.create_task(_poll_loop())
    logger.info("sec_research.started")
    yield

    await dexter_client.close()
    if _producer:
        await _producer.stop()


app = FastAPI(
    title="Los Ojos — SEC Research",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    text: str
    context: str = ""  # e.g. "sec_filing" | "news" | "tweet"


class FilingSignal(BaseModel):
    event_id:     str
    event_type:   str = "SECFilingSignal"
    ts:           str
    ticker:       str
    form_type:    str
    sentiment:    str
    score:        float
    confidence:   float
    signals:      list[str]
    is_market_moving: bool
    summary:      str
    regulators:   list[str]
    crypto_assets: list[str]
    amounts_usd:  list[float]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "service": "sec-research", "ts": datetime.now(timezone.utc).isoformat()}


@app.get("/filings/recent", response_model=list[FilingSignal])
async def get_recent_filings(
    days_back: int = Query(7, ge=1, le=30),
    form_types: str = Query("8-K,10-Q", description="Comma-separated form types"),
):
    """Fetch and analyze recent SEC filings for crypto-relevant companies."""
    types  = [t.strip() for t in form_types.split(",")]
    filings = await dexter_client.get_recent_filings(
        tickers=dexter_client.CRYPTO_TICKERS,
        form_types=types,
        days_back=days_back,
    )
    return [_filing_to_signal(f) for f in filings]


@app.get("/filings/{ticker}", response_model=list[FilingSignal])
async def get_filings_by_ticker(
    ticker: str,
    days_back: int = Query(30, ge=1, le=90),
):
    """Get filings for a specific ticker."""
    ticker = ticker.upper()
    filings = await dexter_client.get_recent_filings(
        tickers=[ticker],
        form_types=["8-K", "10-K", "10-Q"],
        days_back=days_back,
    )
    if not filings:
        raise HTTPException(status_code=404, detail=f"No filings found for {ticker}")
    return [_filing_to_signal(f) for f in filings]


@app.get("/institutional/positions")
async def get_institutional_positions():
    """Get 13F institutional positions in BTC ETFs and crypto equities."""
    positions = await dexter_client.get_institutional_positions()
    return {
        "positions": [p.__dict__ for p in positions],
        "count": len(positions),
        "ts": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/signals/regulatory")
async def get_regulatory_signals():
    """Get active regulatory signals from recent filings."""
    filings = await dexter_client.get_recent_filings(days_back=14)
    signals = [
        _filing_to_signal(f) for f in filings
        if f.sentiment in ("negative",) or "regulatory_action" in (f.key_topics or [])
    ]
    return {
        "signals": signals,
        "count":   len(signals),
        "has_active_regulatory_risk": len(signals) > 0,
    }


@app.post("/analyze")
async def analyze_text(req: AnalyzeRequest):
    """Analyze arbitrary text for financial sentiment and entities."""
    sentiment = sentiment_clf.classify(req.text)
    entities  = entity_extr.extract(req.text)
    return {
        "sentiment": sentiment.to_dict(),
        "entities":  entities.to_dict(),
    }


# ---------------------------------------------------------------------------
# Background tasks
# ---------------------------------------------------------------------------

async def _poll_loop():
    """Poll Dexter API every 15 minutes and publish signals to Kafka."""
    await asyncio.sleep(60)  # initial delay
    while True:
        try:
            filings = await dexter_client.get_recent_filings(days_back=1)
            market_moving = [f for f in filings
                             if f.sentiment in ("positive", "negative")
                             and f.confidence > 0.6]
            for filing in market_moving:
                sig = _filing_to_signal(filing)
                await _publish_signal(sig)
            if filings:
                logger.info("sec_research.poll_complete",
                            total=len(filings),
                            market_moving=len(market_moving))
        except Exception as e:
            logger.error("sec_research.poll_error", error=str(e))
        await asyncio.sleep(900)  # 15 min


async def _publish_signal(signal: FilingSignal) -> None:
    if _producer:
        try:
            await _producer.send("los_ojos.sec.signal", value=signal.dict())
        except Exception as e:
            logger.error("sec_research.publish_error", error=str(e))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filing_to_signal(filing: FilingSentiment) -> FilingSignal:
    # Enrich with NLP
    text     = f"{filing.form_type} {filing.ticker}: {filing.summary}"
    sent_res = sentiment_clf.classify(text)
    entities = entity_extr.extract(filing.summary)

    return FilingSignal(
        event_id=str(uuid.uuid4()),
        ts=datetime.now(timezone.utc).isoformat(),
        ticker=filing.ticker,
        form_type=filing.form_type,
        sentiment=filing.sentiment or sent_res.label,
        score=filing.score or sent_res.score,
        confidence=filing.confidence,
        signals=sent_res.signals + (filing.key_topics or []),
        is_market_moving=sent_res.is_market_moving,
        summary=filing.summary[:300],
        regulators=entities.regulators,
        crypto_assets=entities.crypto_assets,
        amounts_usd=entities.amounts_usd[:5],
    )
