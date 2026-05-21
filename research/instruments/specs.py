"""
Instrument Specifications
=========================
Base abstraction para describir instrumentos financieros con la microstructure
necesaria para backtesting realista.

DESIGN PRINCIPLE:
Toda la lógica de "cuánto vale 1 pip / 1 tick en USD", "cuánto cuesta el spread",
"cómo se redondea el size a unidades válidas" vive AQUÍ, no en el backtester.
El backtester solo opera con `n_units` (lots o contratos).

NOTA SOBRE MONEDAS:
Asumimos cuenta denominada en USD. Para pairs USD-quoted (EUR/USD, GBP/USD,
AUD/USD) la conversión a USD es directa. Para pairs cross o non-USD-quoted
(USD/JPY, EUR/GBP, etc.) la conversión requiere el precio actual del pair de
conversión. El campo `usd_conversion_factor` en `pnl_usd()` permite parametrizar.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AssetClass(str, Enum):
    FOREX = "forex"
    INDEX_FUTURE = "index_future"
    EQUITY_FUTURE = "equity_future"
    COMMODITY_FUTURE = "commodity_future"


# =====================================================================
# BASE
# =====================================================================

@dataclass(frozen=True)
class InstrumentSpec:
    """
    Base spec. Subclase via ForexSpec o FutureSpec. No usar directamente.

    Atributos requeridos para que el motor funcione:
      - symbol: 'EURUSD', 'ES', etc.
      - asset_class
      - tick_or_pip_size: incremento mínimo de precio (0.0001 EURUSD, 0.25 ES)
      - usd_per_unit_per_price_point: PnL en USD de mover 1 unidad de precio
            con 1 unit de instrumento. Para 1 lot EURUSD = 100_000.
            Para 1 contrato ES = 50.
      - min_size_increment: tamaño mínimo (0.01 lots para micro, 1 contrato).
      - typical_spread_in_price: spread típico en unidades de precio.
      - commission_per_unit_per_side_usd: comisión por unidad por lado.
      - swap_long_usd_per_unit_per_day: financiación diaria long (negativo = paga).
      - swap_short_usd_per_unit_per_day: financiación diaria short.
    """
    symbol: str
    asset_class: AssetClass

    # Microstructure
    tick_or_pip_size: float
    usd_per_unit_per_price_point: float
    min_size_increment: float

    # Costs
    typical_spread_in_price: float
    commission_per_unit_per_side_usd: float = 0.0
    swap_long_usd_per_unit_per_day: float = 0.0
    swap_short_usd_per_unit_per_day: float = 0.0

    # Margin (informational; futuros)
    initial_margin_per_unit_usd: Optional[float] = None
    maintenance_margin_per_unit_usd: Optional[float] = None

    # Display
    display_name: str = ""

    def pnl_usd(
        self,
        n_units: float,
        price_entry: float,
        price_exit: float,
        usd_conversion_factor: float = 1.0,
    ) -> float:
        """
        P&L en USD para una posición de n_units (signed) entre entry y exit.

        Para un long: pnl > 0 si exit > entry.
        Para un short (n_units negativo): pnl > 0 si exit < entry.

        Parameters
        ----------
        usd_conversion_factor : float
            Factor multiplicativo si la moneda quote no es USD.
            Para EURUSD (USD quote): 1.0
            Para USD/JPY: 1.0 / current_USDJPY_price (porque pip_value en USD
                                                       depende del precio JPY)
            Para crosses (EUR/GBP) en cuenta USD: GBP/USD price.
        """
        return (
            n_units
            * (price_exit - price_entry)
            * self.usd_per_unit_per_price_point
            * usd_conversion_factor
        )

    def round_to_min_increment(self, n_units: float) -> float:
        """Redondea HACIA CERO al múltiplo válido más cercano."""
        if self.min_size_increment <= 0:
            return n_units
        sign = 1 if n_units >= 0 else -1
        magnitude = abs(n_units)
        rounded = (magnitude // self.min_size_increment) * self.min_size_increment
        return sign * rounded

    @property
    def is_forex(self) -> bool:
        return self.asset_class == AssetClass.FOREX

    @property
    def is_future(self) -> bool:
        return self.asset_class in (
            AssetClass.INDEX_FUTURE,
            AssetClass.EQUITY_FUTURE,
            AssetClass.COMMODITY_FUTURE,
        )


# =====================================================================
# FOREX
# =====================================================================

@dataclass(frozen=True)
class ForexSpec(InstrumentSpec):
    """
    Forex pair specification.

    Convención: pip_size = 0.0001 para mayoría, 0.01 para pares JPY.
    Standard lot = 100,000 unidades de la base currency.
    Mini lot = 10,000. Micro lot = 1,000.

    Para pairs USD-quoted (EUR/USD, GBP/USD, AUD/USD, NZD/USD):
      pip_value_per_standard_lot_usd = 100_000 * pip_size = $10 (siempre).

    Para pairs USD-base (USD/JPY, USD/CHF, USD/CAD):
      pip_value en USD depende del precio actual del pair → usar
      usd_conversion_factor en pnl_usd().
    """
    base_currency: str = ""
    quote_currency: str = ""
    standard_lot_size: float = 100_000.0      # unidades de base currency

    # Para conveniencia de display:
    @property
    def pip_size(self) -> float:
        return self.tick_or_pip_size

    @property
    def is_jpy_pair(self) -> bool:
        return "JPY" in (self.base_currency, self.quote_currency)


# =====================================================================
# FUTURES
# =====================================================================

@dataclass(frozen=True)
class FutureSpec(InstrumentSpec):
    """
    Futures contract specification.

    Atributos clave:
    - tick_size: incremento mínimo de precio (0.25 para ES, NQ; 1.0 para YM)
    - tick_value_usd: $ por tick por contrato (12.50 ES, 5.00 NQ)
    - contract_multiplier: $ por punto de índice ($50 ES, $20 NQ)

    Relación: tick_value_usd = contract_multiplier × tick_size

    Para futuros, n_units = número de contratos (siempre entero).
    """
    contract_multiplier: float = 0.0
    tick_value_usd: float = 0.0
    exchange: str = "CME"

    # Para backtests realistas:
    typical_session_hours: tuple[str, str] = ("09:30", "16:00")  # NYSE RTH

    def __post_init__(self):
        # Validación: tick_value_usd debe = multiplier × tick_size
        # No podemos modificar campos en frozen dataclass, pero podemos validar
        expected_tick_value = self.contract_multiplier * self.tick_or_pip_size
        if abs(expected_tick_value - self.tick_value_usd) > 1e-6:
            # Solo warning, no error: algunos contratos pueden tener variantes
            import warnings
            warnings.warn(
                f"Spec inconsistency for {self.symbol}: "
                f"tick_value_usd={self.tick_value_usd}, but "
                f"multiplier × tick_size = {expected_tick_value}",
                stacklevel=2,
            )

    @property
    def tick_size(self) -> float:
        return self.tick_or_pip_size
