# ADR-036 — DRL-First: Deep Reinforcement Learning como arquitectura primaria

**Status**: Accepted  
**Date**: 2026-06-03  
**Supercede**: ADR-006 (Q-learning tabular como MVP — se mantiene, pero como paso 1 de la escalera DRL, no como objetivo final)

---

## Contexto

Hasta esta decisión, el sistema usaba **ML supervisado** como arquitectura primaria:

1. Extraer features de mercado
2. Labelear con Triple-Barrier (TP/SL/timeout)
3. Entrenar un clasificador (XGBoost / ResMLP) para predecir {-1, 0, +1}
4. Convertir la probabilidad en señal → sizing Kelly → orden

Este enfoque tiene limitaciones estructurales que se vuelven críticas en producción:

**Problema 1 — El objetivo está desalineado.**  
El clasificador optimiza *accuracy de predicción de dirección*, no *P&L ajustado por riesgo*. Un modelo que predice bien pero entra en momentos de alta volatilidad o costos elevados puede ser rentable en backtest y ruinoso en producción.

**Problema 2 — Los labels son proxy, no ground truth.**  
Triple-Barrier requiere decidir upfront `tp_pct`, `sl_pct` y `timeout`. Estas decisiones son arbitrarias y sesgan el modelo hacia patrones que funcionan con *esos* parámetros específicos. Pequeños cambios en los barriers cambian radicalmente los labels y por tanto el modelo.

**Problema 3 — No hay feedback loop.**  
El modelo supervisado no aprende de sus propias acciones. No sabe que ejecutar una orden grande mueve el precio (market impact), ni que entrar en ciertos momentos del día tiene costos distintos. Aprende patrones históricos estáticos.

**Problema 4 — Regime shifts.**  
Un clasificador entrenado en régimen de baja volatilidad falla en régimen de alta volatilidad aunque las features técnicas sean similares. El sistema necesita retraining manual cada vez que el régimen cambia.

---

## Decisión

**El sistema adopta Deep Reinforcement Learning (DRL) como arquitectura primaria de toma de decisiones de trading.**

El agente DRL:
- Observa el **estado** del mercado y del portfolio
- Elige una **acción** (dirección + sizing)
- Recibe un **reward** basado en P&L real ajustado por riesgo y costos
- Aprende a maximizar el retorno acumulado a largo plazo

El objetivo de optimización es directamente lo que importa en producción, no un proxy.

### Escalera de implementación

```
Paso 1 [ACTUAL]   Q-learning tabular
                  Estado discretizado, Q-table explícita
                  Sin red neuronal — válido como sanity check
                  research/models/rl_agent.py

Paso 2 [PRÓXIMO]  DQN (Deep Q-Network)
                  Red neuronal (ResMLP) sustituye la Q-table
                  Experience replay buffer + target network
                  Acción discreta: {-1, 0, +1}

Paso 3            PPO (Proximal Policy Optimization)
                  Actor-critic con redes separadas
                  Acción continua ∈ [-1, 1] → sizing fraccional directo
                  On-policy, estable, fácil de debuggear

Paso 4            SAC (Soft Actor-Critic)
                  Maximum entropy → mejor exploración
                  Off-policy → más sample efficient que PPO
                  Objetivo: trading en producción
```

### XGBoost como baseline permanente

XGBoost **no se elimina**. Se mantiene como:
1. **Baseline de comparación**: el agente DRL debe superar a XGBoost en DSR y Sharpe OOS para ser promovido a staging. Si no lo supera, hay un problema en el diseño del reward o el estado.
2. **Señal auxiliar**: la probabilidad calibrada de XGBoost puede entrar como feature en el estado del agente DRL (meta-información sobre el mercado).
3. **Fallback de emergencia**: si el agente DRL entra en modo degradado, XGBoost puede generar señales mientras el agente se recupera.

---

## Alternativas consideradas y descartadas

### A. Mantener ML supervisado + mejorar labels

**Descartado porque**: los problemas 1 (objetivo desalineado) y 3 (sin feedback loop) son estructurales y no se resuelven mejorando los labels. Podríamos hacer mejor Triple-Barrier, pero seguiría siendo un proxy del objetivo real.

### B. ML supervisado + capa de RL encima (híbrido)

**Parcialmente válido**: usar XGBoost para filtrar señales y RL para sizing. Esto reduce el espacio de exploración del agente. Se mantiene como opción táctica (XGBoost como feature de estado), pero el agente DRL toma la decisión final.

### C. Transformers para series temporales (Temporal Fusion Transformer)

**Descartado por ahora**: TFT es state-of-the-art para predicción supervisada de series temporales, no para control de trading. Podría entrar como encoder del estado en el agente DRL (futuro), pero no como arquitectura primaria de decisión.

### D. Imitation Learning (aprender de un trader experto)

**Descartado**: no tenemos trazas de un trader experto de calidad suficiente. Podría ser relevante en el futuro si hay datos de órdenes institucionales.

---

## Consecuencias

### Positivas

- **Alineación objetivo-métrica**: el agente optimiza directamente P&L ajustado por riesgo
- **Adaptación continua**: el agente puede actualizar su política online sin retraining completo
- **Market impact implícito**: el reward incluye costos reales → el agente aprende a no sobre-operar
- **Sin labeling manual**: se elimina la dependencia de Triple-Barrier y sus parámetros arbitrarios
- **Generalización de régimen**: un agente bien entrenado generaliza mejor a regímenes no vistos

### Negativas / trade-offs

- **Sample inefficiency**: DRL necesita muchas más interacciones que ML supervisado para aprender. Mitigación: empezar con simulación histórica acelerada, luego paper trading.
- **Inestabilidad de entrenamiento**: DRL puede colapsar o divergir. Mitigación: PPO con clip ratio conservador, early stopping en DSR, circuit breaker externo.
- **Reward hacking**: el agente puede encontrar formas de maximizar el reward que no son trading real (e.g., nunca operar para evitar drawdown). Mitigación: diseño cuidadoso del reward (ver ADR-037).
- **Caja negra**: las decisiones del agente son menos interpretables que XGBoost. Mitigación: logging de attention weights / SHAP sobre el estado, baseline comparison siempre activa.
- **El agente DRL NO decide risk limits**: los límites de riesgo (kill switch, DD máximo, per-symbol cap) siguen siendo responsabilidad del `RiskGate` externo. Esto es no negociable (reafirma ADR-009).

---

## Restricciones de diseño (no negociables)

1. **El agente DRL recibe el estado DESPUÉS del RiskGate** — nunca puede bypassear el risk management.
2. **El reward no incluye leverage implícito** — la acción del agente es una fracción de Kelly, no un múltiplo del equity.
3. **Validación walk-forward obligatoria** — el agente se evalúa OOS con `WalkForwardRunner` igual que cualquier modelo supervisado. No hay métricas IS.
4. **Comparación vs XGBoost baseline en cada promoción** — un agente que no supera el baseline no se promueve a staging.
5. **Paper trading mínimo 30 días** antes de cualquier capital real.

---

## Próximo ADR

**ADR-037**: Diseño del Environment DRL — especificación formal de estado, acción y función de reward.

---

**Maintainer**: Alex / Claude  
**Revisado**: 2026-06-03
