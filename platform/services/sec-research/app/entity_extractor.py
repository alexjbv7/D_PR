"""
EntityExtractor — Extracción de entidades financieras de filings SEC.

Extrae:
  - Compañías mencionadas (con ticker si conocido)
  - Montos en USD (>= $1M)
  - Fechas clave (expected actions, deadlines)
  - Ejecutivos (CEO, CFO, Chairman)
  - Reguladores (SEC, CFTC, DOJ, OFAC, FRB, OCC)
  - Productos/assets cripto (Bitcoin, Ethereum, stablecoin names)

Usa SpaCy si disponible; fallback a regex-based extraction.
Diseño sin hard dependency en SpaCy (gran modelo).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Known entity maps
# ---------------------------------------------------------------------------

_CRYPTO_ASSETS = {
    "bitcoin": "BTC", "btc": "BTC",
    "ethereum": "ETH", "ether": "ETH", "eth": "ETH",
    "solana": "SOL", "sol": "SOL",
    "binance coin": "BNB", "bnb": "BNB",
    "ripple": "XRP", "xrp": "XRP",
    "cardano": "ADA", "ada": "ADA",
    "avalanche": "AVAX", "avax": "AVAX",
    "polygon": "MATIC", "matic": "MATIC",
    "usd coin": "USDC", "usdc": "USDC",
    "tether": "USDT", "usdt": "USDT",
    "chainlink": "LINK",
    "dogecoin": "DOGE",
}

_REGULATORS = {
    "sec": "SEC", "securities and exchange commission": "SEC",
    "cftc": "CFTC", "commodity futures trading commission": "CFTC",
    "doj": "DOJ", "department of justice": "DOJ",
    "ofac": "OFAC", "office of foreign assets control": "OFAC",
    "occ": "OCC", "federal reserve": "FRB", "frb": "FRB",
    "fincen": "FinCEN", "financial crimes enforcement": "FinCEN",
    "irs": "IRS",
}

_CRYPTO_COMPANIES = {
    "coinbase": "COIN", "binance": None, "kraken": None,
    "ripple labs": "XRP", "tether": "USDT", "circle": "USDC",
    "microstrategy": "MSTR", "marathon digital": "MARA",
    "riot platforms": "RIOT", "cleanspark": "CLSK",
    "galaxy digital": "GLXY", "grayscale": "GBTC",
    "blackrock": None, "fidelity": None, "ark invest": None,
}

_EXEC_TITLES = {
    "chief executive officer", "ceo",
    "chief financial officer", "cfo",
    "chief operating officer", "coo",
    "chairman", "president",
    "chief legal officer", "clo",
    "general counsel",
}


@dataclass
class ExtractedEntities:
    companies:   list[dict] = field(default_factory=list)  # {name, ticker, context}
    amounts_usd: list[float] = field(default_factory=list)  # $ amounts >= 1M
    dates:       list[str] = field(default_factory=list)
    executives:  list[dict] = field(default_factory=list)  # {name, title}
    regulators:  list[str] = field(default_factory=list)   # e.g. ["SEC", "CFTC"]
    crypto_assets: list[str] = field(default_factory=list) # e.g. ["BTC", "ETH"]
    has_regulatory_action: bool = False
    has_crypto_mention:    bool = False

    def to_dict(self) -> dict:
        return {
            "companies":   self.companies,
            "amounts_usd": [round(a) for a in self.amounts_usd[:10]],
            "dates":       self.dates[:10],
            "executives":  self.executives[:5],
            "regulators":  self.regulators,
            "crypto_assets": self.crypto_assets,
            "has_regulatory_action": self.has_regulatory_action,
            "has_crypto_mention":    self.has_crypto_mention,
        }


class EntityExtractor:
    """
    Extracts financial entities from SEC filing text.

    Usage
    -----
    extractor = EntityExtractor()
    await extractor.load()
    entities = extractor.extract(filing_text)
    """

    # Regex patterns
    _USD_PATTERN  = re.compile(
        r'\$\s*([\d,]+(?:\.\d+)?)\s*(billion|million|thousand|B|M|K)?',
        re.IGNORECASE
    )
    _DATE_PATTERN = re.compile(
        r'\b(?:January|February|March|April|May|June|July|August|September|'
        r'October|November|December)\s+\d{1,2},?\s+\d{4}\b'
        r'|\b\d{1,2}/\d{1,2}/\d{2,4}\b'
        r'|\b\d{4}-\d{2}-\d{2}\b',
        re.IGNORECASE
    )
    _EXEC_PATTERN = re.compile(
        r'([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+),?\s+'
        r'(?:our\s+|the\s+)?(?:' + '|'.join(_EXEC_TITLES) + r')',
        re.IGNORECASE
    )

    def __init__(self, use_spacy: bool = False):
        self._use_spacy = use_spacy
        self._nlp       = None

    async def load(self) -> None:
        if self._use_spacy:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
                logger.info("entity_extractor.spacy_loaded")
            except Exception as e:
                logger.warning("entity_extractor.spacy_unavailable", error=str(e))

    def extract(self, text: str) -> ExtractedEntities:
        text_lower = text.lower()
        entities   = ExtractedEntities()

        # Companies + tickers
        entities.companies = self._extract_companies(text_lower)

        # USD amounts
        entities.amounts_usd = self._extract_amounts(text)

        # Dates
        entities.dates = self._DATE_PATTERN.findall(text)[:10]

        # Executives
        entities.executives = self._extract_executives(text)

        # Regulators
        entities.regulators = self._extract_regulators(text_lower)

        # Crypto assets
        entities.crypto_assets = self._extract_crypto(text_lower)

        # Flags
        entities.has_crypto_mention    = len(entities.crypto_assets) > 0
        entities.has_regulatory_action = self._detect_regulatory_action(text_lower)

        # Enhance with SpaCy if available
        if self._nlp:
            self._enhance_with_spacy(text, entities)

        return entities

    # ------------------------------------------------------------------

    @staticmethod
    def _extract_companies(text_lower: str) -> list[dict]:
        found = []
        for name, ticker in _CRYPTO_COMPANIES.items():
            if name in text_lower:
                # Find context (30 chars before/after)
                idx = text_lower.find(name)
                context = text_lower[max(0, idx - 30): idx + len(name) + 30]
                found.append({"name": name.title(), "ticker": ticker, "context": context.strip()})
        return found

    def _extract_amounts(self, text: str) -> list[float]:
        amounts = []
        for match in self._USD_PATTERN.finditer(text):
            try:
                num_str   = match.group(1).replace(",", "")
                multiplier_str = (match.group(2) or "").lower()
                multiplier = {"billion": 1e9, "b": 1e9, "million": 1e6, "m": 1e6,
                              "thousand": 1e3, "k": 1e3}.get(multiplier_str, 1)
                amount = float(num_str) * multiplier
                if amount >= 1_000_000:   # filter < $1M
                    amounts.append(amount)
            except ValueError:
                pass
        return sorted(set(amounts), reverse=True)[:20]

    def _extract_executives(self, text: str) -> list[dict]:
        results = []
        for match in self._EXEC_PATTERN.finditer(text):
            name  = match.group(1).strip()
            # Extract title from the matched context
            after = text[match.start():match.end()].lower()
            title = next((t for t in _EXEC_TITLES if t in after), "executive")
            results.append({"name": name, "title": title})
        # Deduplicate
        seen = set()
        unique = []
        for e in results:
            k = e["name"].lower()
            if k not in seen:
                seen.add(k)
                unique.append(e)
        return unique

    @staticmethod
    def _extract_regulators(text_lower: str) -> list[str]:
        found = set()
        for pattern, abbrev in _REGULATORS.items():
            if pattern in text_lower:
                found.add(abbrev)
        return sorted(found)

    @staticmethod
    def _extract_crypto(text_lower: str) -> list[str]:
        found = set()
        for term, symbol in _CRYPTO_ASSETS.items():
            if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
                found.add(symbol)
        return sorted(found)

    @staticmethod
    def _detect_regulatory_action(text_lower: str) -> bool:
        action_terms = [
            "investigation", "subpoena", "wells notice", "enforcement",
            "charges", "lawsuit", "litigation", "injunction", "fine",
            "penalty", "consent order", "cease and desist",
        ]
        return any(t in text_lower for t in action_terms)

    def _enhance_with_spacy(self, text: str, entities: ExtractedEntities) -> None:
        """Add SpaCy NER results (organizations, persons) to entities."""
        try:
            doc = self._nlp(text[:10_000])  # limit to 10k chars
            orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
            persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            # Add new orgs not already found
            known = {c["name"].lower() for c in entities.companies}
            for org in set(orgs):
                if org.lower() not in known:
                    entities.companies.append({"name": org, "ticker": None, "context": "spacy"})
        except Exception as e:
            logger.debug("entity_extractor.spacy_error", error=str(e))
