"""
Contratos de la arquitectura Alfa → Cartera → Ejecución (ADR-042).

Define las INTERFACES (Protocols) que desacoplan las capas:

    AlphaAgent → TradeSignal → CapitalAllocator → PositionSizer → RiskGate → Execution

Un ``AlphaAgent`` produce una señal (``TradeSignal``, reusada de
``quant_shared.schemas.signals``); la capa de cartera la ensambla, asigna capital y
dimensiona; el ``RiskGate`` impone límites duros; la ejecución la convierte en
órdenes. ``Explainer`` (interpretabilidad) y ``PromotionGate`` (walk-forward + DSR)
son transversales.

Regla dura (ADR-042 §3, ADR-009): **un ``AlphaAgent`` NUNCA decide tamaño ni riesgo
— solo señal.** El sizing vive en ``PositionSizer``/``CapitalAllocator``; los límites
en ``RiskGate``.

Este módulo es SOLO contratos (tipos + Protocols): cero lógica, cero dependencias
pesadas (sin torch/sklearn/pandas en las firmas públicas). Las implementaciones
viven en sus capas (``alpha/``, ``portfolio/``, ``risk/`` …) y se prueban contra
estas interfaces. Es aditivo: nada en producción lo importa todavía.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from quant_shared.schemas.signals import TradeSignal

# ---------------------------------------------------------------------------
# Taxonomía: estilo × clase de activo (catálogo de hipótesis de alfa)
# ---------------------------------------------------------------------------


class AssetClass(str, Enum):
    CRYPTO = "crypto"
    STOCK = "stock"
    FX = "fx"


class TradingStyle(str, Enum):
    INTRADAY = "intraday"
    SWING = "swing"
    POSITION = "position"
    ARBITRAGE = "arbitrage"


class Benchmark(str, Enum):
    """Benchmark del gate — cambia según el estilo (CATALOGO §plantilla)."""
    BUY_AND_HOLD = "buy_and_hold"   # estrategias direccionales
    ZERO = "zero"                   # market-neutral / event-driven / arbitraje


@dataclass(frozen=True)
class AlphaHypothesis:
    """
    Identidad falsable de un agente de alfa (ver CATALOGO_ALPHA_HIPOTESIS.md).

    El agente se especializa por HIPÓTESIS, no por el producto cartesiano
    activo×estrategia×modelo (DIAGNOSTICO §1.2).
    """

    id: str                      # "{clase}.{estilo}.{nombre}", p.ej. "crypto.swing.funding_reversion"
    asset_class: AssetClass
    style: TradingStyle
    thesis: str                  # 1 frase falsable
    horizon_bars: int            # horizonte típico en barras
    benchmark: Benchmark         # contra qué se mide en el gate
    invalidation: str            # qué resultado MATA la hipótesis (obligatorio)


@dataclass(frozen=True)
class FeeModel:
    """
    Modelo de costos PROPIO del módulo — específico de su clase/venue. Un agente
    crypto-perp y uno de equities tienen costos distintos; cada uno trae el suyo.
    """

    maker_bps: float = 0.0
    taker_bps: float = 0.0
    slippage_bps: float = 0.0
    funding: bool = False        # crypto perps pagan/cobran funding
    borrow_bps: float = 0.0      # short borrow (equities)


@dataclass(frozen=True)
class StrategyConfig:
    """
    Parámetros INTRÍNSECOS de la estrategia — propios de cada agente/módulo (es una
    librería: cada módulo es autocontenido). Definen la estrategia (un breakout sin
    su stop NO es la estrategia) y el modelo de costos de SU mercado.

    **No incluye** sizing de capital ni límites de firma: eso es ``PositionSizer`` /
    ``CapitalAllocator`` / ``RiskGate`` (ADR-009). Aquí va el riesgo *intrínseco*
    (stop/target/holding que definen la estrategia), NO el riesgo de firma.
    """

    fees: FeeModel = field(default_factory=FeeModel)
    intrinsic_stop_pct: float = 0.0     # stop propio de la estrategia (advisory)
    intrinsic_target_pct: float = 0.0   # target propio
    max_holding_bars: int = 0           # 0 = sin límite
    bar_size: str = "1d"                # el horizonte/barra de ESTA estrategia
    params: Mapping[str, float] = field(default_factory=dict)  # umbrales, ventanas…


# ---------------------------------------------------------------------------
# Tipos de apoyo (no acoplan el contrato a pandas/numpy)
# ---------------------------------------------------------------------------

FeatureVector = Mapping[str, float]   # conveniencia tabular: nombre_feature -> valor


@dataclass(frozen=True)
class PortfolioState:
    """Estado del portfolio que el agente puede mirar (no modificar)."""

    position: float = 0.0          # posición actual, signo = dirección
    equity: float = 1.0
    unrealized_pnl: float = 0.0
    holding_bars: int = 0


@dataclass(frozen=True)
class MarketContext:
    """
    Contexto que recibe un ``AlphaAgent`` en ``predict``. Es deliberadamente
    GENÉRICO para que CUALQUIER familia de modelo pueda extraer lo que necesita:

    - tabular (XGBoost, MLP) → usa ``features`` (FeatureVector plano)
    - secuencial (LSTM, TFT, TCN) → usa ``window`` (ventana de barras/features)
    - grafo (GNN) → construye su grafo desde ``window`` / fuentes externas

    Cada agente **OWNS su feature extraction**: el contrato no asume tabular. Por
    eso añadir un modelo de otra familia NO cambia esta interfaz (ADR-042 §3.1).
    """

    symbol: str
    features: FeatureVector                 # vista tabular (puede estar vacía para seq/graph)
    window: Any = None                      # ventana cruda de barras/features (model-agnóstica)
    portfolio: PortfolioState = field(default_factory=PortfolioState)
    timestamp: Any = None


@dataclass(frozen=True)
class AllocationDecision:
    weights: Mapping[str, float]       # hypothesis_id -> peso [0,1]; suma <= 1
    n_trials_searched: int             # entrada para la deflación honesta del DSR
    reason: str


@dataclass(frozen=True)
class SizeDecision:
    target_weight: float               # fracción de equity; el signo lleva la dirección
    method: str                        # "vol_target" | "frac_kelly" | "cvar" | stack
    reason: str


class RiskAction(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REDUCE = "reduce"


@dataclass(frozen=True)
class RiskDecision:
    action: RiskAction
    max_weight: float
    reason: str


@dataclass(frozen=True)
class Explanation:
    method: str                        # "shap" | "xrl_policy" | "attention" | ...
    summary: str
    detail: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Contratos (Protocols) — las plug-points de la arquitectura
# ---------------------------------------------------------------------------


@runtime_checkable
class AlphaAgent(Protocol):
    """
    Productor de señal. **NUNCA decide tamaño ni riesgo** (ADR-042 §3 / ADR-009).

    **Modular por diseño (ADR-042 §3.1):** cualquier familia de modelo —tabular
    (XGBoost), RL (DQN/PPO/SAC), secuencial (LSTM/TFT/TCN) o grafo (GNN)— se
    integra escribiendo UNA clase que implemente este Protocol. Recibe el mismo
    ``MarketContext`` genérico y devuelve el mismo ``TradeSignal``; el agente
    extrae internamente el input que su modelo necesita. Añadir un modelo nuevo
    NO toca ``portfolio/``, ``risk/``, ``execution/`` ni el gate.

    **Autocontenido (es una librería):** cada agente trae su propia ``hypothesis``,
    su ``config`` (``StrategyConfig``: fees/rates de su mercado + stops/targets
    intrínsecos + parámetros de la estrategia). El riesgo *intrínseco* (stop/target)
    vive con el agente; el riesgo de *firma* (caps, kill switch, sizing de capital)
    es externo (``RiskGate``/``Allocator``, ADR-009). El agente PROPONE su stop/
    target en la señal; la firma DISPONE.

    Cada agente estampa ``hypothesis.id`` en ``TradeSignal.strategy`` para trazabilidad.
    """

    hypothesis: AlphaHypothesis
    config: StrategyConfig

    def predict(self, context: MarketContext) -> TradeSignal: ...


@runtime_checkable
class FeatureProvider(Protocol):
    """Construye el state space causal/anti-leakage (GMM por fold, ADR-040)."""

    def build(self, raw: Any, train_idx: Any) -> Any: ...


@runtime_checkable
class CapitalAllocator(Protocol):
    """
    Asigna capital entre agentes con consciencia de **correlación** Y de
    **deflación de selección** (multiple testing). Núcleo del fondo multiagente.
    """

    def allocate(
        self,
        signals: Sequence[TradeSignal],
        cov: Any,
        track: Mapping[str, Any],
    ) -> AllocationDecision: ...


@runtime_checkable
class PositionSizer(Protocol):
    """Dimensiona con el stack vol-target · frac-Kelly · CVaR · régimen (DIAGNOSTICO §4)."""

    def size(
        self,
        weight: float,
        vol_forecast: float,
        regime: str,
        edge_posterior: Any,
    ) -> SizeDecision: ...


@runtime_checkable
class RiskGate(Protocol):
    """Límites duros + kill switch. **Externo al agente** (ADR-009)."""

    def check(self, signal: TradeSignal, size: SizeDecision) -> RiskDecision: ...


@runtime_checkable
class Explainer(Protocol):
    """
    Interpretabilidad transversal; la implementación se elige por tipo de modelo
    (tabular→SHAP, policy→XRL, transformer→attention; DIAGNOSTICO §3).
    """

    def explain(self, model: Any, state: FeatureVector) -> Explanation: ...


@runtime_checkable
class PromotionGate(Protocol):
    """
    Walk-forward + DSR deflactado (ADR-040). Único árbitro de promoción. Devuelve
    un ``GateResult`` (definido en la capa de research para no acoplar shared).
    """

    def evaluate(self, agent: AlphaAgent, data: Any) -> Any: ...
