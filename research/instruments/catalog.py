"""
Instrument Catalog
==================
Especificaciones pre-definidas de los instrumentos más operados.

DISCLAIMER:
Los valores de comisión, spread y margin son APROXIMADOS y pueden variar por
broker, condiciones de mercado y nivel de cuenta. Antes de operar dinero real,
verifica los valores con tu broker.

Fuentes:
- CME Group spec sheets (públicos)
- Brokers retail típicos (IC Markets, IBKR, AMP Clearing, NinjaTrader)
- Datos de mid-2024 (verificar antes de live trading)
"""
from __future__ import annotations

from .specs import AssetClass, ForexSpec, FutureSpec


# =====================================================================
# FOREX MAJORS
# =====================================================================

# Spreads y swaps típicos asumiendo broker ECN (IC Markets / Pepperstone Razor / similar)
# Comisión ECN: ~$3.50/lado/std-lot (= $7 round-turn por std lot)
ECN_COMMISSION_PER_STD_LOT_PER_SIDE = 3.50

EURUSD = ForexSpec(
    symbol="EURUSD",
    display_name="EUR/USD",
    asset_class=AssetClass.FOREX,
    tick_or_pip_size=0.0001,
    usd_per_unit_per_price_point=100_000.0,  # 1 std lot × 1.0 price unit = $100k
    min_size_increment=0.01,                  # micro lot
    typical_spread_in_price=0.0001,           # 1 pip on standard, ~0.2 ECN; conservador
    commission_per_unit_per_side_usd=ECN_COMMISSION_PER_STD_LOT_PER_SIDE,  # por std lot
    swap_long_usd_per_unit_per_day=-2.50,     # depende de tasa de interés diferencial
    swap_short_usd_per_unit_per_day=-1.00,
    base_currency="EUR",
    quote_currency="USD",
    standard_lot_size=100_000.0,
)

GBPUSD = ForexSpec(
    symbol="GBPUSD",
    display_name="GBP/USD",
    asset_class=AssetClass.FOREX,
    tick_or_pip_size=0.0001,
    usd_per_unit_per_price_point=100_000.0,
    min_size_increment=0.01,
    typical_spread_in_price=0.00015,          # ~1.5 pips standard
    commission_per_unit_per_side_usd=ECN_COMMISSION_PER_STD_LOT_PER_SIDE,
    swap_long_usd_per_unit_per_day=-3.00,
    swap_short_usd_per_unit_per_day=-1.50,
    base_currency="GBP",
    quote_currency="USD",
    standard_lot_size=100_000.0,
)

USDJPY = ForexSpec(
    symbol="USDJPY",
    display_name="USD/JPY",
    asset_class=AssetClass.FOREX,
    tick_or_pip_size=0.01,                    # JPY pairs: pip = 0.01
    usd_per_unit_per_price_point=100_000.0,   # ¡pero hay que multiplicar por 1/USDJPY_price!
    min_size_increment=0.01,
    typical_spread_in_price=0.015,            # ~1.5 pips
    commission_per_unit_per_side_usd=ECN_COMMISSION_PER_STD_LOT_PER_SIDE,
    swap_long_usd_per_unit_per_day=4.00,      # USD/JPY long: cobras carry (positivo) si JPY pays less
    swap_short_usd_per_unit_per_day=-6.00,
    base_currency="USD",
    quote_currency="JPY",
    standard_lot_size=100_000.0,
)

AUDUSD = ForexSpec(
    symbol="AUDUSD",
    display_name="AUD/USD",
    asset_class=AssetClass.FOREX,
    tick_or_pip_size=0.0001,
    usd_per_unit_per_price_point=100_000.0,
    min_size_increment=0.01,
    typical_spread_in_price=0.00015,
    commission_per_unit_per_side_usd=ECN_COMMISSION_PER_STD_LOT_PER_SIDE,
    swap_long_usd_per_unit_per_day=-2.00,
    swap_short_usd_per_unit_per_day=-2.00,
    base_currency="AUD",
    quote_currency="USD",
    standard_lot_size=100_000.0,
)


# =====================================================================
# E-MINI INDEX FUTURES (CME)
# =====================================================================

# Comisión típica retail: $0.50 - $2.50 per side. Usamos $1.50 como middle.
FUTURES_COMMISSION_DEFAULT = 1.50

ES = FutureSpec(
    symbol="ES",
    display_name="E-mini S&P 500",
    asset_class=AssetClass.INDEX_FUTURE,
    tick_or_pip_size=0.25,
    usd_per_unit_per_price_point=50.0,        # = contract_multiplier
    min_size_increment=1.0,                   # contratos enteros
    typical_spread_in_price=0.25,             # 1 tick durante RTH
    commission_per_unit_per_side_usd=FUTURES_COMMISSION_DEFAULT,
    swap_long_usd_per_unit_per_day=0.0,       # futuros no tienen swap
    swap_short_usd_per_unit_per_day=0.0,
    contract_multiplier=50.0,
    tick_value_usd=12.50,
    exchange="CME",
    initial_margin_per_unit_usd=13_200.0,     # aprox; CME ajusta periódicamente
    maintenance_margin_per_unit_usd=12_000.0,
)

NQ = FutureSpec(
    symbol="NQ",
    display_name="E-mini Nasdaq-100",
    asset_class=AssetClass.INDEX_FUTURE,
    tick_or_pip_size=0.25,
    usd_per_unit_per_price_point=20.0,
    min_size_increment=1.0,
    typical_spread_in_price=0.25,
    commission_per_unit_per_side_usd=FUTURES_COMMISSION_DEFAULT,
    contract_multiplier=20.0,
    tick_value_usd=5.00,
    exchange="CME",
    initial_margin_per_unit_usd=17_600.0,
    maintenance_margin_per_unit_usd=16_000.0,
)

YM = FutureSpec(
    symbol="YM",
    display_name="E-mini Dow Jones",
    asset_class=AssetClass.INDEX_FUTURE,
    tick_or_pip_size=1.0,
    usd_per_unit_per_price_point=5.0,
    min_size_increment=1.0,
    typical_spread_in_price=1.0,
    commission_per_unit_per_side_usd=FUTURES_COMMISSION_DEFAULT,
    contract_multiplier=5.0,
    tick_value_usd=5.00,
    exchange="CBOT",
    initial_margin_per_unit_usd=9_000.0,
    maintenance_margin_per_unit_usd=8_200.0,
)

RTY = FutureSpec(
    symbol="RTY",
    display_name="E-mini Russell 2000",
    asset_class=AssetClass.INDEX_FUTURE,
    tick_or_pip_size=0.10,
    usd_per_unit_per_price_point=50.0,
    min_size_increment=1.0,
    typical_spread_in_price=0.10,
    commission_per_unit_per_side_usd=FUTURES_COMMISSION_DEFAULT,
    contract_multiplier=50.0,
    tick_value_usd=5.00,
    exchange="CME",
    initial_margin_per_unit_usd=8_500.0,
    maintenance_margin_per_unit_usd=7_700.0,
)


# =====================================================================
# REGISTRY
# =====================================================================

_CATALOG: dict[str, "InstrumentSpec"] = {  # type: ignore
    "EURUSD": EURUSD,
    "GBPUSD": GBPUSD,
    "USDJPY": USDJPY,
    "AUDUSD": AUDUSD,
    "ES": ES,
    "NQ": NQ,
    "YM": YM,
    "RTY": RTY,
}


def get_instrument(symbol: str):
    """Retorna la spec del instrumento por símbolo (case-insensitive)."""
    key = symbol.upper().replace("/", "")
    if key not in _CATALOG:
        raise KeyError(
            f"Instrumento '{symbol}' no está en el catálogo. "
            f"Disponibles: {sorted(_CATALOG.keys())}"
        )
    return _CATALOG[key]


def list_instruments() -> list[str]:
    """Lista todos los instrumentos disponibles en el catálogo."""
    return sorted(_CATALOG.keys())
