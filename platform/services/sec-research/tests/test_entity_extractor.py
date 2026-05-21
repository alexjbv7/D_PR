"""Tests for EntityExtractor — financial entity extraction."""
import pytest
from app.entity_extractor import EntityExtractor, ExtractedEntities


@pytest.fixture
def extractor():
    e = EntityExtractor(use_spacy=False)
    return e


class TestExtract:
    def test_returns_entities(self, extractor):
        result = extractor.extract("Coinbase announced a partnership with Bitcoin.")
        assert isinstance(result, ExtractedEntities)

    def test_crypto_company_detected(self, extractor):
        result = extractor.extract("Coinbase reported record quarterly revenue.")
        companies = [c["name"].lower() for c in result.companies]
        assert "coinbase" in companies

    def test_crypto_asset_bitcoin(self, extractor):
        result = extractor.extract("Bitcoin reached an all-time high of $100,000.")
        assert "BTC" in result.crypto_assets
        assert result.has_crypto_mention is True

    def test_crypto_asset_ethereum(self, extractor):
        result = extractor.extract("Ethereum developers released the next upgrade.")
        assert "ETH" in result.crypto_assets

    def test_multiple_assets(self, extractor):
        result = extractor.extract("Bitcoin and Ethereum led the market rally today.")
        assert "BTC" in result.crypto_assets
        assert "ETH" in result.crypto_assets

    def test_regulator_sec(self, extractor):
        result = extractor.extract(
            "The Securities and Exchange Commission filed charges against the exchange."
        )
        assert "SEC" in result.regulators

    def test_regulator_cftc(self, extractor):
        result = extractor.extract(
            "CFTC investigation into derivatives trading practices."
        )
        assert "CFTC" in result.regulators

    def test_regulatory_action_flag(self, extractor):
        result = extractor.extract(
            "The company received a Wells Notice and faces SEC investigation."
        )
        assert result.has_regulatory_action is True

    def test_no_regulatory_action_plain_text(self, extractor):
        result = extractor.extract("Revenue increased 20% year over year.")
        assert result.has_regulatory_action is False

    def test_usd_amount_million(self, extractor):
        result = extractor.extract("The company raised $500 million in funding.")
        assert 500_000_000 in result.amounts_usd or \
               any(abs(a - 500_000_000) < 1 for a in result.amounts_usd)

    def test_usd_amount_billion(self, extractor):
        result = extractor.extract("Assets under management reached $2.5 billion.")
        assert any(abs(a - 2_500_000_000) < 1 for a in result.amounts_usd)

    def test_small_amounts_filtered(self, extractor):
        result = extractor.extract("The fine was $50,000.")
        # $50k < $1M threshold → should not appear
        assert not any(a < 1_000_000 for a in result.amounts_usd)

    def test_date_extraction(self, extractor):
        result = extractor.extract("The decision is expected by January 15, 2027.")
        assert len(result.dates) >= 1
        assert any("2027" in d or "January" in d for d in result.dates)

    def test_to_dict_serializable(self, extractor):
        import json
        result = extractor.extract("Coinbase SEC investigation Bitcoin ETF.")
        d = result.to_dict()
        json.dumps(d)  # must not raise

    def test_empty_text(self, extractor):
        result = extractor.extract("")
        assert result.companies == []
        assert result.crypto_assets == []
        assert result.has_regulatory_action is False

    def test_microstrategy_detected(self, extractor):
        result = extractor.extract("MicroStrategy increased its Bitcoin holdings.")
        companies = [c["name"].lower() for c in result.companies]
        assert "microstrategy" in companies
        assert "BTC" in result.crypto_assets
