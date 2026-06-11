"""
FirmRiskGate — límites de firma + kill switch (contrato ADR-042, ADR-009).

Implementa ``quant_shared.contracts.RiskGate``: la ÚLTIMA palabra sobre lo que
propusieron el agente (señal con riesgo intrínseco) y el sizer (peso). Puede
ALLOW / REDUCE / DENY; nunca dimensiona ni genera señal.

Es la versión research/ del gate de producción
(``platform/services/execution-engine/app/risk_gate.py`` — solo referencia de
reglas, NO tocado): mismo principio de cortocircuito ordenado y kill switch
pegajoso, pero operando sobre pesos (``SizeDecision``) en vez de
``OrderIntent``/notional, para el loop research → portfolio.

Orden de evaluación (el orden importa; primera brecha corta)
------------------------------------------------------------
0. **Kill switch** — si está tripped, DENY para CUALQUIER señal sin evaluar
   nada más (gap P1-002). El riesgo de firma SIEMPRE gana sobre el alfa.
1. **Drawdown diario** > límite → trip del kill switch + DENY (CLAUDE.md §12.2).
2. **Drawdown mensual** > límite → freeze: DENY todo (incluso cierres).
   Se evalúa antes que el semanal porque el freeze total domina sobre el
   modo solo-cierre — un cierre permitido por la regla semanal no puede
   saltarse un freeze mensual.
3. **Drawdown semanal** > límite → solo-cierre: DENY aperturas/incrementos;
   los cierres y reducciones pasan (capados a la exposición actual).
4. **Cap por símbolo** (§12.1) → |peso| > cap → REDUCE al cap.
5. **Cap de leverage bruto** (§12.6) → si ``gross_resto + peso`` rompe el cap,
   REDUCE al headroom disponible o DENY si no hay.

Semántica de pesos: ``size.target_weight`` es el peso OBJETIVO total de la
posición en el símbolo (reemplaza su exposición actual, no se suma); la
exposición bruta propuesta es ``Σ|exposición otros símbolos| + |peso|``.

Estado: el gate se construye con ``RiskLimits`` y se alimenta con
``update(AccountSnapshot)``; ``check`` usa el último snapshot. El kill switch
es PEGAJOSO: se dispara por snapshot (``kill_switch_tripped=True``), por
drawdown diario, o manualmente (``trip_kill_switch``); solo se rearma con
``reset_kill_switch()`` explícito — un snapshot posterior "limpio" NO lo
resetea (decisión humana, §12.7).
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

from quant_shared.contracts import RiskAction, RiskDecision, SizeDecision
from quant_shared.schemas.signals import SignalDirection, TradeSignal

logger = logging.getLogger(__name__)

_DIRECTION_SIGN: dict[SignalDirection, int] = {
    SignalDirection.LONG: 1,
    SignalDirection.SHORT: -1,
    SignalDirection.FLAT: 0,
}


@dataclass(frozen=True)
class RiskLimits:
    """
    Límites de firma — config con defaults, sin magic numbers (§20.6).
    Fracciones de equity (0.05 = 5 %).

    ``leverage_cap``: 1.0 equities (default); 2.0 FX; 3.0 crypto (§12.6).
    """

    daily_dd_kill: float = 0.03        # §12.2: DD diario → kill switch
    weekly_dd_close_only: float = 0.07  # §12.2: DD semanal → solo cierre
    monthly_dd_freeze: float = 0.12     # §12.2: DD mensual → freeze total
    per_symbol_cap: float = 0.05        # §12.1: max |peso| por símbolo
    leverage_cap: float = 1.0           # §12.6: exposición bruta total

    def __post_init__(self) -> None:
        for name in (
            "daily_dd_kill", "weekly_dd_close_only",
            "monthly_dd_freeze", "per_symbol_cap",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} debe estar en [0, 1], recibido {value}")
        if self.leverage_cap <= 0.0:
            raise ValueError(f"leverage_cap debe ser > 0, recibido {self.leverage_cap}")


@dataclass(frozen=True)
class AccountSnapshot:
    """
    Foto de cuenta que alimenta al gate vía ``update``.

    Drawdowns como fracciones positivas (0.04 = 4 % bajo el peak).
    ``exposure_by_symbol``: peso firmado actual por símbolo (fracción de
    equity); el signo es la dirección.
    """

    equity: float = 1.0
    drawdown_daily: float = 0.0
    drawdown_weekly: float = 0.0
    drawdown_monthly: float = 0.0
    exposure_by_symbol: Mapping[str, float] = field(default_factory=dict)
    kill_switch_tripped: bool = False


class FirmRiskGate:
    """
    ``RiskGate`` stateful de firma.

    Examples
    --------
    >>> gate = FirmRiskGate(RiskLimits())
    >>> gate.update(AccountSnapshot(equity=100_000.0))
    >>> decision = gate.check(signal, size)
    >>> decision.action
    <RiskAction.ALLOW: 'allow'>
    """

    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()
        self._snapshot = AccountSnapshot()
        self._kill_switch_tripped = False

    # ------------------------------------------------------------------
    # Estado
    # ------------------------------------------------------------------

    def update(self, snapshot: AccountSnapshot) -> None:
        """
        Registra el snapshot más reciente. Si trae ``kill_switch_tripped``,
        dispara el switch (pegajoso: un snapshot posterior en False NO lo
        rearma — solo ``reset_kill_switch``).
        """
        self._snapshot = snapshot
        if snapshot.kill_switch_tripped:
            self.trip_kill_switch()

    def trip_kill_switch(self) -> None:
        """Bloquea toda señal nueva. Idempotente."""
        if not self._kill_switch_tripped:
            logger.critical("firm_risk_gate.kill_switch.tripped")
        self._kill_switch_tripped = True

    def reset_kill_switch(self) -> None:
        """Rearmado EXPLÍCITO (humano, §12.7). Idempotente."""
        if self._kill_switch_tripped:
            logger.warning("firm_risk_gate.kill_switch.reset")
        self._kill_switch_tripped = False

    @property
    def kill_switch_tripped(self) -> bool:
        return self._kill_switch_tripped

    # ------------------------------------------------------------------
    # Contrato
    # ------------------------------------------------------------------

    def check(self, signal: TradeSignal, size: SizeDecision) -> RiskDecision:
        """
        Evalúa la propuesta agente+sizer contra los límites de firma.

        Returns
        -------
        RiskDecision
            ``max_weight`` = |peso| máximo permitido; ``reason`` nombra el
            límite que disparó.
        """
        lim = self.limits
        snap = self._snapshot

        # STEP-0 — kill switch: gana SIEMPRE, sin evaluar nada más (P1-002)
        if self._kill_switch_tripped:
            return RiskDecision(
                action=RiskAction.DENY, max_weight=0.0,
                reason="kill_switch: tripped — toda señal denegada hasta reset explícito",
            )

        # 1. DD diario → trip + DENY (§12.2)
        if snap.drawdown_daily > lim.daily_dd_kill:
            self.trip_kill_switch()
            return RiskDecision(
                action=RiskAction.DENY, max_weight=0.0,
                reason=(
                    f"drawdown_daily: {snap.drawdown_daily:.2%} > "
                    f"{lim.daily_dd_kill:.2%} — kill switch tripped"
                ),
            )

        # 2. DD mensual → freeze total (domina sobre el solo-cierre semanal)
        if snap.drawdown_monthly > lim.monthly_dd_freeze:
            return RiskDecision(
                action=RiskAction.DENY, max_weight=0.0,
                reason=(
                    f"drawdown_monthly: {snap.drawdown_monthly:.2%} > "
                    f"{lim.monthly_dd_freeze:.2%} — freeze, revisión humana (§12.2)"
                ),
            )

        proposed = abs(size.target_weight)
        dir_sign = _DIRECTION_SIGN[signal.direction]
        current = float(snap.exposure_by_symbol.get(signal.symbol, 0.0))

        # 3. DD semanal → modo solo-cierre (§12.2)
        if snap.drawdown_weekly > lim.weekly_dd_close_only:
            reason_base = (
                f"drawdown_weekly: {snap.drawdown_weekly:.2%} > "
                f"{lim.weekly_dd_close_only:.2%} — modo solo-cierre"
            )
            opens_or_adds = dir_sign != 0 and (
                current == 0.0 or (current > 0) == (dir_sign > 0)
            )
            if opens_or_adds:
                return RiskDecision(
                    action=RiskAction.DENY, max_weight=0.0,
                    reason=f"{reason_base}: apertura/incremento denegado",
                )
            if dir_sign == 0:
                return RiskDecision(
                    action=RiskAction.ALLOW, max_weight=0.0,
                    reason=f"{reason_base}: cierre permitido",
                )
            # Dirección opuesta: reducción permitida hasta la exposición actual
            # (más allá sería flip = apertura del lado contrario)
            if proposed > abs(current):
                return RiskDecision(
                    action=RiskAction.REDUCE, max_weight=abs(current),
                    reason=f"{reason_base}: reducción capada a la exposición actual",
                )
            return RiskDecision(
                action=RiskAction.ALLOW, max_weight=proposed,
                reason=f"{reason_base}: reducción permitida",
            )

        # Señal de cierre en condiciones normales: nada que capar
        if dir_sign == 0 or proposed == 0.0:
            return RiskDecision(
                action=RiskAction.ALLOW, max_weight=0.0,
                reason="ok: señal de cierre/flat — sin exposición nueva",
            )

        # 4. Cap por símbolo (§12.1)
        allowed = proposed
        binding = ""
        if allowed > lim.per_symbol_cap:
            allowed = lim.per_symbol_cap
            binding = (
                f"per_symbol_cap: |peso| {proposed:.2%} > {lim.per_symbol_cap:.2%} "
                f"({signal.symbol})"
            )

        # 5. Leverage bruto (§12.6) — el peso objetivo reemplaza la exposición
        # actual del símbolo
        gross_others = sum(
            abs(w) for s, w in snap.exposure_by_symbol.items() if s != signal.symbol
        )
        headroom = lim.leverage_cap - gross_others
        if headroom <= 0.0:
            return RiskDecision(
                action=RiskAction.DENY, max_weight=0.0,
                reason=(
                    f"leverage_cap: sin headroom — gross resto {gross_others:.2%} "
                    f">= cap {lim.leverage_cap:.2%}"
                ),
            )
        if allowed > headroom:
            allowed = headroom
            binding = (
                f"leverage_cap: gross {gross_others + proposed:.2%} > "
                f"{lim.leverage_cap:.2%}, headroom {headroom:.2%}"
            )

        if allowed < proposed:
            return RiskDecision(
                action=RiskAction.REDUCE, max_weight=allowed, reason=binding,
            )
        return RiskDecision(
            action=RiskAction.ALLOW, max_weight=proposed,
            reason="ok: dentro de límites de firma",
        )
