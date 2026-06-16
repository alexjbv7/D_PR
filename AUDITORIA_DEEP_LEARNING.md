# Auditoría técnica — Módulo de Deep Learning

> Auditoría del módulo de DL del bot, tratando la **generación de señales como fase
> independiente**. Sin concesiones. Aterrizada en el código real y en la evidencia
> empírica de la sesión (3 gates fallidos). Fecha: 2026-06-10.

---

## HALLAZGO CENTRAL (léelo primero)

**La maquinaria de DL es correcta y hasta sofisticada; el problema no es técnico, es
de hipótesis.** La función de costo existe, es explícita, está documentada, se
optimiza bien y —tras ADR-041— está alineada con el objetivo financiero. El gate de
validación es riguroso. **Lo que falla es la ALFA**: un DQN direccional sobre índices
líquidos diarios no tiene edge (Sharpe 0.30–0.55 vs buy-and-hold 0.8–1.3 en 3 runs).
Confundir "la loss está rota" con "la estrategia no tiene edge" sería un error de
diagnóstico. Son ortogonales: el motor está bien; el destino estaba mal.

---

## 1. Flujo completo del pipeline de decisión

```
Datos → Features → Deep Learning → Señales → Riesgo → Sizing → Ejecución
```

| Fase | Qué hace | Depende de | Archivo |
|------|----------|-----------|---------|
| **Datos** | OHLCV point-in-time (Alpaca IEX/crypto) | broker/feed | `data/drl_dataset.py`, `data/alpaca_bars.py` |
| **Features** | 42-dim: 14 técnicas + 7 régimen (GMM) + 5 portfolio + reserva | datos; GMM por fold (anti-leakage) | `data/drl_dataset.py`, `features/regime_gmm.py` |
| **Deep Learning** | DQN: estado→Q-values→acción | features | `models/drl/{dqn,dqn_trainer,backbone}.py` |
| **Señales** | acción greedy {-1,0,+1} → `TradeSignal` | DL; contrato `AlphaAgent` | `alpha/agents/dqn_agent.py` |
| **Riesgo** | límites duros, kill switch (externo) | señal | `risk/gate.py` (RiskGate) |
| **Sizing** | vol-target·Kelly·CVaR | señal + riesgo | `portfolio/sizing.py` |
| **Ejecución** | order routing → fills | sizing | `platform/services/execution-engine/` |

**Validación transversal (gatekeeper):** walk-forward + DSR deflactado
(`models/drl/dsr_gate.py`). Nada se promueve sin pasarlo.

---

## 2. El módulo de Deep Learning en aislamiento

- **Problema que resuelve**: **Aprendizaje por Refuerzo** (control secuencial), NO
  clasificación/regresión/forecasting. El agente aprende una **política** estado→acción
  maximizando un reward acumulado. (La detección de régimen es GMM, un input, no DL.)
- **Arquitectura**: **TradingResMLP** — MLP **residual feedforward** (ResBlock +
  SwiGLU + LayerNorm, ADR-038). **NO** es LSTM, Transformer, TFT, CNN-TS, TCN, ni GNN.
  → *Hallazgo*: para un problema **temporal**, el modelo es **feedforward sin memoria
  aprendida**. La "temporalidad" vive solo en los features hechos a mano (retornos a
  1/5/20, vol, etc.), no en la arquitectura. Justificable por la muestra chica
  (~1.400 barras diarias), pero es una limitación explícita.
- **Entradas**: vector de 42-dim (estado del env, ADR-037).
- **Salidas**: 3 Q-values → `argmax` (acción greedy) → `{-1,0,+1}`.
- **Interacción con señales**: el `DqnAlphaAgent` (ADR-042) mapea la acción a un
  `TradeSignal` (`direction`, `p_win`, stops intrínsecos), **sin** decidir capital.

---

## 3. ¿Existe función de costo integrada?

**Sí — y hay que distinguir las DOS de un sistema RL:**

| Nivel | Qué es | ¿Definida? | ¿Explícita? | ¿Documentada? | ¿Optimizada? |
|-------|--------|-----------|------------|---------------|--------------|
| **Loss de optimización** | Error TD (Huber/`smooth_l1`) entre `Q(s,a)` y `r+γ·maxQ'` | ✅ | ✅ `dqn_trainer.py:182-183` | ✅ docstrings + ADR-038 | ✅ Double-DQN, target net, grad-clip, Adam |
| **Objetivo financiero (reward)** | `r = pos·ret_MTM − costos − λ·DD − λ_vol·vol − idle` | ✅ | ✅ `envs/trading_env.py::compute_reward_mtm` | ✅ ADR-037/041 | ✅ es lo que el agente maximiza |

- **¿Alineada con el objetivo financiero?** **Sí, tras ADR-041.** Antes había un
  desalineamiento (el reward usaba P&L **realizado** mientras el gate medía
  **mark-to-market**); se corrigió a retorno MTM por barra. Hoy lo que el agente
  entrena ≈ lo que el examen mide. **Esto es de lo mejor del módulo.**
- **Veredicto §3/§4**: la función de costo **NO es una debilidad** — existe, es
  explícita, documentada, se optimiza correctamente y está alineada. **No aplica
  proponer alternativas por ausencia.** Mejoras *opcionales* (no urgentes):

| Mejora posible del reward | Cuándo | Riesgo |
|---------------------------|--------|--------|
| Reward basado en **Sharpe/Sortino diferencial** | si se quiere optimizar risk-adjusted directo | reward hacking; varianza alta |
| **Calmar / drawdown penalization** afinada | si el objetivo pasa a control de DD | sobre-penaliza, vuelve al agente flat |
| **Utility-based (CRRA)** | marco unificador de aversión al riesgo | sensible a la spec de utilidad |

---

## 5. Coherencia features → modelo → loss → señales → métricas

| Eslabón | Estado | Nota |
|---------|--------|------|
| Features → Modelo | **Coherente** | features tabulares ↔ MLP tabular. Pero sin modelado temporal (gap, no incoherencia). |
| Modelo → Loss/Reward | **Coherente (post-ADR-041)** | TD + reward MTM alineado con la métrica del gate. |
| Loss → Señales | **Parcial** | la acción greedy es coherente; pero el `p_win` = softmax de Q-values **NO es probabilidad calibrada** (Q-values son retornos, no log-odds). La señal *afirma* un `p_win` que no es frecuencia-calibrada. |
| Señales → Métricas | **Coherente y riguroso** | DSR walk-forward deflactado vs buy-and-hold + XGBoost. La deflación frena el sesgo de selección. |

**Riesgo de sobreajuste**: alto por construcción (muestra ~1.400 barras vs MLP
profunda) — **pero correctamente mitigado** por el walk-forward + DSR. De hecho, el
gate **hizo su trabajo**: detectó que no hay edge en vez de reportar un alfa fantasma.
Eso es una fortaleza del diseño de validación, no un fallo.

**Supuesto incorrecto detectado**: tratar el `p_win` del DQN como probabilidad
calibrada aguas abajo (`is_actionable(min_p_win=0.52)`). El softmax de Q-values es
**ordinal**, no calibrado → un umbral de 0.52 sobre él no significa "52% de acierto".

---

## 6. La fase de señales como módulo independiente

- **Tipos de señal**: direccional `{-1,0,+1}` (SHORT/FLAT/LONG). All-or-nothing, sin
  sizing (el sizing es externo, `PositionSizer`).
- **Umbrales**: el DQN usa `argmax` (sin umbral propio). Downstream, `TradeSignal.
  is_actionable(min_p_win=0.52, min_confidence=0.5)` impone umbrales — pero sobre un
  `p_win` no calibrado (ver §5).
- **Probabilidades → decisión**: `argmax_a Q(s,a)` (greedy); `p_win = softmax(Q)[a]`.
- **Filtros adicionales**: existen en el toolkit (`models/meta_labeler.py`,
  `models/entry_filter.py`) pero **NO están cableados** en el path del DQN actual.
- **Incertidumbre**: la `TradeSignal` tiene campos (`confidence`, `p_win_raw`) pero el
  DQN **no produce incertidumbre genuina** (no hay ensemble/Bayesiano; el softmax no
  es incertidumbre).
- **Calibración probabilística**: el sistema **tiene** la infra (`IsotonicCalibrator`,
  `TemperatureScaling`) pero **no se aplica al DQN** por defecto (el propio
  `DqnAlphaAgent` lo documenta y deja un hook `calibrator`, sin usar). → **Gap**.

---

## 7. Diagnóstico crudo (clasificación por componente)

| Componente | Clasificación | Justificación |
|-----------|---------------|---------------|
| Loss de optimización (TD/Huber) | **Correctamente diseñado** | Double-DQN, target net, grad-clip; explícito y optimizado. |
| Reward (objetivo financiero) | **Correctamente diseñado** | MTM alineado con el gate tras ADR-041. |
| Desacople de la señal (contrato `AlphaAgent`) | **Correctamente diseñado** | Lo mejor del sistema; señal = artefacto de primera clase (ADR-042). |
| Validación (DSR walk-forward) | **Correctamente diseñado** | Anti-leakage, deflación, benchmark por estilo. |
| Modelado temporal | **Ausente** | Feedforward; sin LSTM/TFT/TCN. Justificado por datos, pero limitación. |
| Calibración del `p_win` del DQN | **Deficiente** | infra existe, no se aplica; umbrales sobre proba no calibrada. |
| Incertidumbre en la señal | **Mejorable/Ausente** | sin ensemble/Bayesiano; campos presentes pero no poblados con incertidumbre real. |
| Filtros (meta-labeler/entry) en el path DQN | **Ausente (no cableado)** | existen en el repo, no se usan en el flujo actual. |
| **Edge de la alfa (hipótesis)** | **Crítico** | 3 gates fallidos; el direccional-diario sobre índices líquidos no tiene alfa explotable. *Es el problema real.* |

---

## Respuestas finales

**¿El Deep Learning está correctamente integrado?**
Técnicamente **sí** (gate riguroso + contrato `AlphaAgent` + reward alineado), con
**dos gaps reales**: arquitectura feedforward (sin temporalidad aprendida) y `p_win`
sin calibrar. Pero la integración no es el cuello de botella.

**¿La función de costo existe y está alineada con el objetivo financiero?**
**Sí, ambas.** Loss TD explícita y bien optimizada; reward financiero MTM alineado con
la métrica del gate tras ADR-041. **No es una debilidad — es una fortaleza.** Quien
sospeche que "falta la loss" está mirando el lugar equivocado.

**¿La fase de señales está bien desacoplada del modelo?**
**Sí, excelentemente** (ADR-042): el `AlphaAgent` produce `TradeSignal`; sizing/riesgo/
ejecución son capas separadas. Es lo mejor diseñado del sistema. *Único pero*: la señal
viaja con un `p_win` no calibrado.

**¿Qué cambios producirían la mayor mejora?** (ordenados)
1. **Aceptar que el problema es la ALFA, no la loss** → pivotar de direccional a
   **market-neutral** (stat-arb / funding reversion). *Ya en curso.* Mayor impacto, lejos.
2. **Calibrar el `p_win`** (isotónica/temperature sobre outcomes OOS) — convierte la
   señal en una probabilidad usable; cablear el hook que ya existe.
3. **Cablear meta-labeler/entry-filter** en el path productivo (ya existen).
4. Si se insiste en direccional: **modelar temporalidad** (TCN/LSTM) — pero solo con
   más datos (la muestra IEX no lo justifica hoy).

**¿Cómo rediseñarías el módulo (estándar institucional)?**
Exactamente la arquitectura ya diseñada (ADR-042): el agente DL como **una fuente de
alfa enchufable** detrás de un contrato `Signal` **calibrado**; complejidad del modelo
**emparejada a la SNR/datos** (GBM/reglas por defecto, DL solo donde lo justifique el
dato — diagnóstico F.4); el reward financiero como objetivo (ya está); el **gate DSR
walk-forward como árbitro único** (ya está). Y la regla de oro del auditor: separar
**"la máquina es correcta"** (lo es) de **"encontrar una hipótesis de alfa con edge"**
(el problema difícil, abierto). El DL no necesita reconstrucción; necesita una **alfa
que valga la pena predecir** — y por eso el pivote a market-neutral es el cambio de
mayor impacto, no tocar la red ni la loss.
