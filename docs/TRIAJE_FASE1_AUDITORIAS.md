# INFORME DE TRIAJE — FASE 1
## Auditoría DL (AUDITORIA_DEEP_LEARNING.md) + Diagnóstico Arquitectura (DIAGNOSTICO_FABLE5_ARQUITECTURA.md)

**Fecha de triaje:** 2026-07-17
**Estado:** Fase 1 completada — CERO cambios aplicados. Esperando aprobación de alcance.

---

## 0. Insumos, límites y metodología adaptada

**Recibido:** los dos documentos de análisis (ambos fechados 2026-06-10).
**NO recibido:** (a) código fuente de `quant_bot`; (b) documento original de la propuesta FABLE 5 que el diagnóstico critica; (c) logs de los 3 runs/gates fallidos; (d) los ADRs referenciados (037, 038, 040, 041, 042).

**Jerarquía de evidencia usada** (de mayor a menor):
1. **Código fuente** — no disponible → nada alcanza el nivel máximo de confirmación.
2. **Hecho matemático/lógico autocontenido** — verificable sin código (p. ej., "softmax de Q-values no es probabilidad calibrada" es cierto por construcción).
3. **Corroboración cruzada entre los dos documentos** — con la advertencia ⚠ de abajo.
4. **Afirmación única de un documento** — mínima confianza.

**⚠ Advertencia de procedencia (léela antes que la tabla):** el diagnóstico de arquitectura escribe en primera persona sobre el fix de ADR-041 (*"exactamente el bug que **arreglamos** con el reward mark-to-market"*, §F.1-F.3). Esto indica que ambos documentos comparten autor/sesión con quien implementó ese fix. Consecuencias: (1) la corroboración cruzada entre ellos es **más débil** que entre fuentes independientes; (2) el elogio de la auditoría DL a ADR-041 (*"es de lo mejor del módulo"*) es potencialmente **el optimizador calificando su propio trabajo** — el anti-patrón que tu proceso prohíbe. Registrado como hallazgo X-004.

---

## 1. TABLA DE TRIAJE

Clasificación: **C**=CONFIRMADO · **FP**=FALSO POSITIVO · **SV**=SOBREVALORADO · **NV**=NO VERIFICABLE.
Severidad recalibrada: S1 crítico · S2 alto · S3 medio · S4 bajo. Esfuerzo: S/M/L.
Tipo de fix: **D**=decisión/proceso (ejecutable sin código) · **K**=código (requiere repo).

### Hallazgos de la Auditoría DL (A-)

| ID | Hallazgo | Clas. | Sev. doc → recal. | Evidencia y notas | Verificación pendiente | Esf. | Tipo |
|----|----------|-------|-------------------|--------------------|------------------------|------|------|
| A-001 | **La alfa no tiene edge**: DQN direccional diario sobre índices líquidos falla 3 gates (Sharpe 0.30–0.55 vs. buy-and-hold 0.8–1.3) | **C** | Crítico → **S1** | Evidencia empírica reportada en ambos docs de forma consistente; el gate hizo su trabajo. El fix NO es código: es matar/pivotar la hipótesis formalmente (pivote a market-neutral ya declarado "en curso") | Logs de los 3 runs para cerrar el expediente | S | **D** |
| A-002 | `p_win` = softmax(Q-values) **no es probabilidad calibrada**, pero aguas abajo se umbraliza como si lo fuera (`is_actionable(min_p_win=0.52)`) | **C** (mecanismo) | — → **S2** | El fundamento es un hecho matemático (nivel 2 de evidencia): Q-values son retornos esperados, su softmax es ordinal, no frecuencia-calibrada. Corroborado ×2 docs: el contrato objetivo del diagnóstico exige `p_win (calibrada)` — el estado actual viola el contrato de diseño | `dqn_agent.py`, clase `TradeSignal`, call sites de `is_actionable` para confirmar el uso exacto | M | K |
| A-003 | Infra de calibración existe (`IsotonicCalibrator`, `TemperatureScaling`) pero **no cableada** al DQN; hook `calibrator` sin usar | **NV** (corrob. ×2 — prioridad alta) | — → **S2** si cierto | Afirmación sobre estado del repo; ambos docs coinciden pero comparten procedencia (⚠) | `dqn_agent.py`, módulo de calibración, tests | M | K |
| A-004 | Sin modelado temporal aprendido (ResMLP feedforward, no LSTM/TCN/TFT) | **SV** | "Ausente" → **S4** | La propia auditoría lo justifica por N (~1.400 barras) y el diagnóstico (§F.4) confirma que DL secuencial en diario data-poor es incoherente. No es defecto: es la elección correcta para los datos. El problema real es A-001 | — (revisar solo si cambia la granularidad de datos) | — | — |
| A-005 | Sin incertidumbre genuina en la señal (no ensemble/Bayesiano; campos `confidence`/`p_win_raw` presentes pero no poblados con incertidumbre real) | **C** (conceptual) / NV (estado exacto) | — → **S3** | Coherente con A-002/A-003; subordinado al pivote de alfa (poblar incertidumbre de una señal sin edge no aporta) | Código de `TradeSignal` y del agente | M | K |
| A-006 | Meta-labeler y entry-filter existen en el repo pero **no cableados** en el path del DQN | **NV** | — → **S3** | Afirmación de estado del repo, un solo doc la detalla. Nota: cablearlos **cambia la lógica de estrategia** → bandera roja del protocolo: exigiría re-validación completa. Sin sentido antes del pivote de alfa | `models/meta_labeler.py`, `models/entry_filter.py`, wiring del path productivo | M | K |
| A-007 | Riesgo de sobreajuste alto por construcción (MLP profunda vs. ~1.400 muestras) | **C** con control existente | — → **S3** (monitoreo) | Mitigado por WFA+DSR según ambos docs; la evidencia de que el control funciona es que bloqueó 3 promociones. Sin acción correctiva; mantener el gate intocable | Verificar en código que el gate sigue en el path obligatorio de promoción | — | — |
| A-008 | Desalineamiento histórico reward (P&L realizado) vs. gate (MTM) — **reportado como corregido** por ADR-041 | **NV** ⚠ | — → **S1 si el fix no está** | El fix es reclamado y luego auto-elogiado por la misma cadena documental (ver X-004). Exactamente el tipo de afirmación que el protocolo exige verificar en código, no en prosa | `envs/trading_env.py::compute_reward_mtm`, ADR-041, y que exista test de regresión del alineamiento | S (verificar) | K |

### Hallazgos del Diagnóstico de Arquitectura (D-)

| ID | Hallazgo | Clas. | Sev. doc → recal. | Evidencia y notas | Verificación pendiente | Esf. | Tipo |
|----|----------|-------|-------------------|--------------------|------------------------|------|------|
| D-001 | **Prioridades invertidas**: diseñar plataforma multiagente (~168 agentes, 5–6 librerías) antes de tener UN agente con edge demostrado | **C** | Grave → **S1** (proyecto) | Se sostiene sobre A-001 (confirmado) + lógica: es el riesgo dominante declarado ("nunca enviar nada"). Coincide con tu plan maestro (scope creep, F1: un mercado primero) | — | S | **D** |
| D-002 | Taxonomía Activo×Estrategia×Modelo malformada (ejes no ortogonales; "momentum no es un modelo"; celdas absurdas) | **NV** (argumento sólido) | — → **S2** | El argumento lógico es correcto por sí mismo, pero **no tengo el documento de la propuesta FABLE 5** para verificar que la propuesta realmente contiene esa matriz | Documento original de la propuesta FABLE 5 | M | **D** |
| D-003 | Falta la capa de **capital allocation + deflación de selección entre agentes** (meta-overfitting): el problema central de un fondo multiagente, listado pero no diseñado | **C** (vacío de diseño) | Crítico → **S1 condicional** (bloquea Nivel 2, no el estado actual con 1 agente) | Consistente entre docs y con tu plan maestro (§17.16). Con un solo agente hoy no es riesgo operativo; se convierte en gate obligatorio antes de escalar a N agentes | — | L | D+K |
| D-004 | Falta capa point-in-time / survivorship / corporate actions en el diseño multi-asset | **NV** con tensión | — → **S2** | ⚠ Tensión: la auditoría DL afirma "OHLCV point-in-time (Alpaca IEX/crypto)" para el path actual. Posible reconciliación: existe para 1 activo, ausente como diseño multi-asset. Requiere verificación | `data/drl_dataset.py`, `data/alpaca_bars.py`; diseño de datos de la propuesta | L | K |
| D-005 | Sizing: "falta el vol-target como base y el CVaR como overlay" (repo tiene `kelly.py`, `bayesian_sizer.py`, `dynamic_rr.py`) | **NV** — **CONTRADICE** a la auditoría DL | — → **S1 potencial** | La auditoría DL lista el sizing como "vol-target·Kelly·CVaR" ya operativo (`portfolio/sizing.py`). **Los dos documentos no pueden ser ciertos a la vez.** Ver X-001 | `portfolio/sizing.py` y módulos citados | M | K |
| D-006 | Interpretabilidad mal concebida (como "librería"; SHAP post-hoc sobre política RL ≈ teatro); falta XRL de comportamiento (acciones por régimen, visitación de estados, counterfactual rollouts, reward decomposition) | **C** (conceptual) | — → **S3** | Argumento técnico defendible por sí mismo; propuesta concreta y accionable (base: `error_analysis.py`) | Confirmar estado de `error_analysis.py` | M | K |
| D-007 | Online learning en el path de riesgo = peligroso (si la propuesta lo contempla) | **NV** (propuesta no disponible) | — → **S2 preventivo** | Independientemente de la propuesta, conviene adoptarlo como regla de gobernanza: prohibido aprender en vivo en el path que puede quebrar la cuenta | Documento de propuesta | S | **D** |
| D-008 | AutoML/NAS/meta-learning en baja SNR = máquina de falsos descubrimientos; Optuna solo con budget capado y deflación por nº de configs | **C** (metodológico) | — → **S2** | Coincide 1:1 con tu plan maestro (presupuesto de experimentos → DSR/PBO). Acción: regla de gobernanza escrita, no código | — | S | **D** |
| D-009 | Vacíos de roadmap: monitoreo de alpha decay, atribución de P&L por agente, capacidad/liquidez/impacto | **C** (vacíos declarados) | — → **S2** | Coinciden con tu plan maestro (§9, §17.14, §18.14); no operativos hoy con 1 agente, obligatorios antes de Nivel 2–3 | — | L | D+K |
| D-010 | Model governance / MLOps ausente (registry, drift, retraining gated) | **C** (vacío declarado) | Alto → **S3** | Real pero subordinado a tener un agente con edge; secuenciar en Nivel 2 | — | L | K |

### Hallazgos NUEVOS del triaje (X-) — no reportados por ningún documento

| ID | Hallazgo | Clas. | Sev. | Detalle | Verificación pendiente |
|----|----------|-------|------|---------|------------------------|
| X-001 | **Contradicción inter-documento sobre el estado del sizing**: la auditoría DL lo describe como "vol-target·Kelly·CVaR" implementado; el diagnóstico afirma que vol-target y CVaR **faltan** | Nuevo — NV | **S1 potencial** | Al menos uno de los dos documentos describe mal el sistema real. Cualquier decisión tomada sobre el documento equivocado hereda el error. Es el argumento definitivo de por qué este triaje necesita el código | `portfolio/sizing.py` |
| X-002 | **Posible acoplamiento p_win no calibrado → Kelly**: si el sizing fraccional-Kelly consume `p_win` como estimador del edge, probabilidades no calibradas (A-002) distorsionan sistemáticamente el tamaño de posición | Nuevo — NV | **S1 potencial** | Ningún documento examina este eslabón: la auditoría se detiene en "la señal viaja con p_win no calibrado" y no sigue el dato hasta el sizing. Si el acoplamiento existe, A-002 sube de S2 a S1 (riesgo directo de capital por sobre-sizing) | `portfolio/sizing.py`, interfaz señal→sizer |
| X-003 | **Inconsistencia numérica en el criterio del gate**: el diagnóstico reporta "DSR 0.85 que aún no supera buy-and-hold" y fija salida de Nivel 1 en "DSR deflactado > 0.4"; la auditoría reporta 3 gates fallidos. Si DSR=0.85 > 0.4, ¿qué criterio exacto falló y qué significa cada número (¿probabilidad? ¿ratio?)? | Nuevo — NV | **S2** | El criterio de promoción es el número más importante del sistema y los documentos lo reportan de forma irreconciliable sin definición operativa. El propio diagnóstico advierte: "no delegues el diseño del criterio de promoción" | `models/drl/dsr_gate.py` + logs de runs + definición escrita del criterio |
| X-004 | **Cadena documental auto-validada**: el diagnóstico escribe "el bug que **arreglamos**" (ADR-041) y la auditoría elogia ese mismo fix como "de lo mejor del módulo" — indicio de autoría/sesión compartida validando trabajo propio | Nuevo — C (evidencia textual) | **S2** (proceso) | Viola el principio "el optimizador no califica su propio trabajo". No invalida los documentos, pero degrada la corroboración cruzada y exige que A-008 se verifique en código con test independiente | Confirmar procedencia de ambos docs; verificar A-008 en código |

---

## 2. RESUMEN CUANTITATIVO

| Clasificación | IDs | Total |
|---------------|-----|-------|
| CONFIRMADO (incl. parciales/conceptuales) | A-001, A-002, A-005, A-007, D-001, D-003, D-006, D-008, D-009, D-010, X-004 | **11** |
| SOBREVALORADO | A-004 | **1** |
| FALSO POSITIVO | — (ver nota) | **0** |
| NO VERIFICABLE | A-003, A-006, A-008, D-002, D-004, D-005, D-007, X-001, X-002, X-003 | **10** |
| **Total (18 originales + 4 nuevos)** | | **22** |

**Tasa de inflación de los informes: baja (1/18 sobrevalorado, 0 falsos positivos detectables).** Estos documentos NO exhiben el sesgo adversarial clásico — exhiben el **sesgo opuesto**: son diagnósticos mixtos que validan generosamente ("excelentemente desacoplado", "de lo mejor del módulo") con posible auto-calificación (X-004). Sin código, no puedo descartar que alguna de esas *fortalezas afirmadas* sea el verdadero falso positivo. La verificación de Fase 3 debe apuntar tanto a los defectos como a las fortalezas reclamadas.

---

## 3. BACKLOG PRIORIZADO

| Prio | ID(s) | Acción | Tipo | Bloqueante |
|------|-------|--------|------|-----------|
| 1 | A-001 + D-001 | **Formalizar la muerte/pivote de la hipótesis direccional** (entrada en cementerio de hipótesis + Decision Log) y **congelar el build multiagente** hasta el criterio de salida de Nivel 1 | Decisión | Sí — ordena todo lo demás |
| 2 | X-001 + D-005 | Verificar en código el estado real del sizing (¿vol-target y CVaR existen?) y resolver qué documento miente | Código | Sí — S1 potencial |
| 3 | X-002 | Rastrear si `p_win` alimenta el Kelly del sizer | Código | Sí — S1 potencial |
| 4 | A-008 + X-004 | Verificar ADR-041 en código con test de regresión independiente (no aceptar la prosa) | Código | Sí — S1 potencial |
| 5 | X-003 | Escribir la definición operativa del criterio del gate y reconciliar los números (0.85 / 0.4 / 3 fallos) | Decisión + código | Sí — gobierna promociones futuras |
| 6 | A-002 + A-003 | Cablear calibración (isotónica/temperature sobre outcomes OOS) al hook existente; bandera roja: cambia comportamiento de señal ⇒ re-validación | Código | No (la estrategia está muerta; aplica a la siguiente alfa) |
| 7 | D-007 + D-008 | Adoptar reglas de gobernanza: no-online-learning en risk path; Optuna con budget capado + deflación por nº configs | Decisión | No |
| 8 | D-003 + D-009 | Diseñar capa de allocation + deflación de selección + alpha decay + atribución como **gate de entrada a Nivel 2** | Diseño | Antes de N agentes |
| 9 | D-006 | Interfaz `Explainer` + XRL de comportamiento de política | Código | No |
| 10 | A-005, A-006, D-010, D-002, D-004 | Subordinados al pivote de alfa y al Nivel 2; re-triar tras decisión de Prio 1 | Mixto | No |
| — | A-004, A-007 | **Sin acción** (elección justificada / control funcionando) | — | — |

---

## 4. VEREDICTO DE DESPLIEGUE PRELIMINAR

# 🔴 BLOQUEADO

- **La estrategia actual (DQN direccional):** bloqueada por **A-001 (S1)** — sin edge, 3 gates fallidos. Nota a favor del sistema: el bloqueo ya lo ejerce el propio gate DSR; el proceso funcionó.
- **El sistema como plataforma:** bloqueado adicionalmente por **tres S1 potenciales sin verificar** (X-001 estado del sizing, X-002 acoplamiento p_win→Kelly, A-008 fix ADR-041 no verificado) — todos requieren el código.
- **El build multiagente FABLE 5:** bloqueado por **D-001** hasta cumplir el criterio de salida de Nivel 1 (que a su vez requiere resolver X-003: hoy ese criterio está numéricamente indefinido).

**Condiciones para levantar el bloqueo:** cerrar Prio 1–5 del backlog. Las Prio 2–4 son inviables sin subir el código fuente.

---

## >>> CHECKPOINT — FASE 1 COMPLETADA. CERO CAMBIOS APLICADOS.

Esperando una de:
- **Lista de IDs** a ejecutar (p. ej., `Prio 1, 5 y 7` — las decisiones ejecutables sin código), o
- **`EJECUTAR TODO`** (procede solo con S1/S2 confirmados de tipo Decisión; los de tipo Código quedarán bloqueados hasta recibir el repo), o
- **Subir el código** (`quant_bot` en zip) para desbloquear Prio 2–4 y re-triar los 10 NO VERIFICABLES.

---
---

# FASE 1B — Resultados de verificación en código

**Fecha 1B:** 2026-07-17  
**Fuente de verdad:** repo `alexjbv7/D_PR` @ `main` (SHA `68f9b59…` al momento de la lectura).  
**Método:** solo lectura vía GitHub API/raw. **Cero cambios de código.** Única escritura: este documento.  
**Regla X-004 aplicada:** ninguna afirmación de AUDITORIA_DEEP_LEARNING.md ni DIAGNOSTICO_FABLE5 se aceptó sin contraparte en código/tests/logs.

### Credibilidad de documentos fuente (X-001 como litmus)

| Documento | Veredicto sobre estado del sizing | Credibilidad residual |
|-----------|-----------------------------------|------------------------|
| **AUDITORIA_DEEP_LEARNING.md** | Describe sizing como `vol-target·Kelly·CVaR` en `portfolio/sizing.py` | **Más cerca del código** para el stack de posición. Sigue sujeta a X-004 (auto-elogio ADR-041). |
| **DIAGNOSTICO_FABLE5…** | Afirma que “falta el vol-target como base y el CVaR como overlay” | **Desactualizado / incorrecto** respecto a `research/portfolio/sizing.py` actual. Sus afirmaciones de “estado del sistema” bajan de credibilidad; sus argumentos de diseño (taxonomía, allocation multiagente) no se invalidan. |

**Matices que ninguno de los dos clava del todo:**
- CVaR implementado es **CVaR-lite por posición** (gaussiano), no CVaR de cartera con correlaciones (eso el propio sizer lo delega a un `CapitalAllocator` que no se verificó como completo).
- `bayesian_sizer.py` existe pero **no** es invocado por `StackedPositionSizer`; solo se acepta un posterior si el caller lo pasa a `edge_posterior`.

---

## 1. Verificaciones bloqueantes (orden obligatorio)

### 1.1 X-001 — Estado real del sizing

| Pregunta | Respuesta | Evidencia |
|----------|-----------|-----------|
| ¿Vol-targeting como escalar base? | **SÍ** | `research/portfolio/sizing.py` L204–205: `vol_scale = cfg.vol_target / max(vol_forecast, cfg.vol_eps)`; default `vol_target=0.15` (L87). |
| ¿Overlay/constraint CVaR? | **SÍ, lite por posición** | L213–216: si `magnitude * es_unit > cvar_budget` recorta; `_expected_shortfall_unit` L261–272. Método declarado `_METHOD = "vol_target·frac_kelly·regime·cvar"` (L52). |
| ¿Kelly + R:R dinámico? | **SÍ** | L177–185: `compute_dynamic_rr` → `kelly_fraction_binary`. |
| ¿Cuál documento describía bien? | **Auditoría DL (parcialmente)**; **Diagnóstico mentía / estaba desfasado** al decir que vol-target y CVaR faltaban. |

**Clasificación 1B:** CONFIRMADO (contradicción resuelta a favor del código = stack existe).  
**Sev. recalibrada:** **S4** como “sizing ausente” (era falso). Residual: **S3** — no hay CVaR de cartera multi-activo cableado; sizer research ≠ path execution paper.

---

### 1.2 X-002 — Acoplamiento p_win → Kelly

**Cadena A — research stack (con guard):**

1. `DqnAlphaAgent.predict` emite `p_win` = softmax(Q)[a], `p_win_calibrated = (calibrator is not None)` — default **False**  
   (`research/alpha/agents/dqn_agent.py` ~L186–201).  
2. `kelly_fraction` y `size_usd` del agente = **siempre 0.0** (mismo archivo ~L202–206).  
3. Único punto autorizado a Kelly desde señal:  
   `edge_posterior_from_signal` → `require_calibrated_signal` → si no calibrado **lanza**  
   (`research/portfolio/sizing.py` L55–75; `shared/quant_shared/schemas/signals.py` L27–41).  
4. `StackedPositionSizer.size(edge_posterior=…)` usa el float/posterior en Kelly (L177–185) **sin** re-chequear calibración si le pasan un float crudo.

**Cadena B — platform (sin guard de calibración):**

1. `AllocationEngine.position_size_from_signal(p_win, stop_loss_pct, …)` aplica Kelly fraccional **directo sobre `p_win`** sin `p_win_calibrated`  
   (`platform/services/strategy-orchestrator/app/allocation_engine.py` L127–149).  
2. `signal_translator.translate_signal` toma `position_size` como Kelly y construye `OrderIntent` sin calibración  
   (`platform/services/execution-engine/app/signal_translator.py` L89–117).

**Cadena C — gate de promoción:** no usa Kelly; solo posiciones greedy → `positions_to_returns` (`dsr_gate.py`).

**Call sites de `is_actionable`:** solo la **definición** en `signals.py` L90–96. **Ningún consumidor** en el repo (scan de paths signal/orchestr/sizing/agent).

**Veredicto X-002:**  
- **CONFIRMADO** como **riesgo de API / path platform**: existe acoplamiento p_win→Kelly sin calibración en `AllocationEngine` y vía `position_size` en execution.  
- **NO CONFIRMADO como S1 en el path default del DQN research**: el agente no dimensiona; el guard research bloquea Kelly sin calibrar.  
**Sev.:** **S2** (fallo funcional / breach de política R-02 en platform si se alimenta p_win crudo). **No sube A-002 a S1** en el path DQN actual.

---

### 1.3 A-008 — Fix ADR-041 (reward MTM)

| Check | Resultado | Evidencia |
|-------|-----------|-----------|
| ¿Reward MTM por barra neto de costos? | **SÍ** | `compute_reward_mtm` en `research/envs/trading_env.py` (~L238–296): `w_ret·pos_{t-1}·ret − w_cost·(fee_bps/1e4)·|Δpos| − …`. Default `reward_mode="mtm"` (~L152–153). |
| ¿Alineado con definición del gate? | **SÍ (con w_dd=w_vol=w_idle=0)** | Contrato documentado + test bit-a-bit. |
| ¿Test de regresión que fije el alineamiento? | **SÍ** | `research/tests/test_reward_shaping.py` → `test_reward_matches_gate_return_def` (comentario “Without penalties, env rewards == dsr_gate.positions_to_returns”); también tests hold win/lose, costo en unidades de retorno, idle, A/B mode, no leakage Optuna. |

**Veredicto A-008:** CONFIRMADO — fix **implementado y protegido por tests**.  
**Sev.:** baja de “S1 si no está” a **cerrado / S4 residual** (ver Y-002: término vol del reward está muerto por bug separado).

---

### 1.4 X-003 — Criterio operativo del gate DSR

**Definición unívoca en código** (`research/models/drl/dsr_gate.py` + `walk_forward_runner.py`):

| Elemento | Valor en código |
|----------|-----------------|
| Retorno por barra | `r_t = pos_{t-1}·(close_t/close_{t-1}−1) − fee_bps/1e4·\|Δpos_t\|` (`positions_to_returns`) |
| PSR | `P(SR_real > SR*)` con SE de Mertens (`walk_forward_runner.py` L71–114). Rango **[0, 1]**. |
| DSR | `PSR(E[max SR \| n_trials])` Bailey & LdP (`L117–148`). Con **`n_trials ≤ 1` → DSR ≡ PSR(SR*=0)**. |
| Condiciones de promoción (AND) | (1) `dsr_agent > dsr_threshold` (default **0.4**); (2) `sharpe_agent > sharpe_buyhold`; (3) `dsr_agent > dsr_xgb` (`evaluate_drl_gate` ~L564–588). |
| Anti-leakage | embargo ≥ 60; GMM re-fit por fold; OOS greedy ε=0. |

**Qué significa “DSR 0.85”:** **probabilidad ~0.85** de que el Sharpe real del agente sea > 0 (con n_trials=1), **no** un ratio de Sharpe. Por eso puede ser 0.85 y aun así fallar el gate si no bate buy-and-hold.

**Reconciliación de los 3 gates fallidos (logs en repo):**

| Run | Log | dsr_agent | sharpe_agent | sharpe_B&H | Condición que falló |
|-----|-----|-----------|--------------|------------|---------------------|
| SPY largo | `research/gate_run.txt` | ≈0.848 | 0.553 | 1.325 | **(2) B&H** (DSR>0.4 se cumpliría) |
| SPY post | `research/gate_spy_post.txt` | ≈0.708 | 0.300 | 1.266 | **(2) B&H** |
| BTC/USD | `research/gate_btc.txt` | ≈0.843 | 0.443 | 0.818 | **(2) B&H** |

**Veredicto X-003:** CONFIRMADO como inconsistencia **documental** (números mal interpretados en prosa); el criterio en código **es unívoco** en `evaluate_drl_gate` + defaults CLI.  
**Sev.:** **S3** gobernanza/docs (no ambigüedad runtime del gate DQN). Residual: criterio de Nivel 1 del diagnóstico (“DSR deflactado > 0.4”) es **incompleto** frente al código (faltan B&H y XGB).

---

## 2. Resto de NO VERIFICABLES (re-clasificados)

| ID | Veredicto 1B | Sev. | Evidencia resumida |
|----|--------------|------|--------------------|
| **A-003** | **CONFIRMADO** | **S2** | Hook `calibrator` en `DqnAlphaAgent.__init__` / `predict`; default `None` → `p_win_calibrated=False`. Infra OOS: `research/alpha/agents/dqn_calibration.py`. Gate no cablea calibrador. Test `test_calibrator_is_applied` en `test_alpha_agents.py`. |
| **A-002** | **CONFIRMADO** (mecanismo) + **SOBREVALORADO** el riesgo aguas abajo *hoy* | **S2→S3** operativo | Softmax no calibrado: sí (`dqn_agent`). `is_actionable(min_p_win=0.52)` existe (`signals.py` L90–96) pero **cero call sites**. `confidence` default 0 → aunque se llamara, fallaría por confidence, no por p_win. |
| **A-006** | **CONFIRMADO** | **S3** | `models/entry_filter.py`, `models/meta_labeler.py` existen; path DQN/gate no los importa. |
| **A-005** | **CONFIRMADO** | **S3** | `p_win_raw` = softmax real; `confidence` no lo setea DQN (queda 0.0). Sin ensemble. |
| **D-004** | **CONFIRMADO** (gap) | **S2** multi-asset / **S3** single-asset | `drl_dataset.py`: features causales + GMM solo en train. `alpaca_bars.py`: OHLCV crudo, **sin** corporate actions / survivorship / PIT adjustments. La etiqueta “point-in-time” de la auditoría DL es **exagerada**. |
| **A-007** | **CONFIRMADO control en DQN** / **FALSO** “intocable en todo el sistema” | **S2** | DQN: `train_drl.py` exit `2` si gate FAIL (L~270–275). **PPO/SAC: `edge = True` hardcodeado, sin gate** (mismos archivos ~L277–297) → promoción posible sin DSR. |
| **D-005** | resuelto vía X-001 | — | Ver §1.1. |
| **D-002** | sigue **NV** (sin documento FABLE 5 original) | S2 conceptual | Sin cambio. |
| **D-007** | sigue **NV** de propuesta; regla preventiva válida | S2 preventivo | Sin cambio de código. |

---

## 3. Hallazgos NUEVOS (Y-) no reportados por los documentos

| ID | Hallazgo | Clas. | Sev. | Evidencia |
|----|----------|-------|------|-----------|
| **Y-001** | PPO/SAC en `train_drl` marcan éxito sin gate DSR | CONFIRMADO | **S2** | `research/cli/train_drl.py` ~L277–297: `edge = True` + TODO. |
| **Y-002** | Término de vol del reward MTM siempre 0 | CONFIRMADO | **S2** 🔴 estrategia | `trading_env.py` ~L395–396: `vol_target = vol_realized` ⇒ `max(0, vol−vol)=0`. `w_vol` es dead. **Cambia reward si se corrige → re-gate.** |
| **Y-003** | Kelly en orchestrator sin guard de calibración | CONFIRMADO | **S2** (S1 si entra capital paper/live con p_win crudo) | `allocation_engine.py` L127–149. |
| **Y-004** | Translator de ejecución confía en `position_size` como Kelly | CONFIRMADO | **S2** | `signal_translator.py` L89–117. |
| **Y-005** | Train/serve: `daily_pnl_pct=0.0` hardcode en serve | CONFIRMADO | **S3** | `dqn_agent.py` ~L226–232; documentado como limitación. |

---

## 4. Fortalezas afirmadas por docs — verificación escéptica

| Fortaleza reclamada | ¿Cierta en código? | Nota |
|---------------------|--------------------|------|
| “Gate riguroso” | **Parcial** | Sí para DQN CLI; no para PPO/SAC (Y-001). Criterio AND de 3 condiciones real. |
| “Reward alineado post-ADR-041” | **Sí** | Tests de regresión lo fijan (A-008). |
| “Desacople excelente AlphaAgent” | **Sí en research** | DQN/XGB `kelly_fraction=0`. Platform vuelve a Kelly en otro sitio (Y-003/Y-004). |
| “Sizing vol-target·Kelly·CVaR operativo” | **Sí en research sizer** | No probado que sea el path paper trading. |
| “OHLCV point-in-time” | **Sobrevalorado** | Causal features ≠ PIT/survivorship institucional (D-004). |

---

## 5. Tabla maestra post-1B (IDs de la Fase 1 + Y)

| ID | Clas. post-1B | Sev. | Tipo | Notas |
|----|---------------|------|------|-------|
| A-001 | C | **S1** (estrategia) | D | 3 gates FAIL B&H; no bug de motor. |
| A-002 | C (math) / SV (uso aguas abajo hoy) | **S3** | K | Softmax no calibrado; `is_actionable` sin consumidores. |
| A-003 | **C** (era NV) | **S2** | K | Hook sin default productivo. |
| A-004 | SV | S4 | — | Sin cambio. |
| A-005 | C | S3 | K | confidence vacío. |
| A-006 | **C** (era NV) | S3 | K | Filtros fuera del path. |
| A-007 | C parcial | **S2** | K | Gate DQN OK; PPO/SAC no. |
| A-008 | **C fix OK** (era NV/S1) | **cerrado** | — | Tests protegen MTM. |
| D-001 | C | S1 proyecto | D | Sin cambio. |
| D-002 | NV | S2 | D | Sin propuesta original. |
| D-003 | C | S1 condicional N>1 | D+K | Sin cambio. |
| D-004 | **C** (era NV) | S2/S3 | K | Sin CA/survivorship. |
| D-005 / X-001 | **C resuelto** | S4 (falso “falta”) | — | Diagnóstico erró en estado. |
| D-006…D-010 | C conceptuales | S2–S3 | mix | Sin re-litigar 1B. |
| X-002 | **C parcial** | **S2** | K | No S1 en path DQN default. |
| X-003 | **C (docs)** | **S3** | D | Código unívoco; prosa confunde DSR con Sharpe. |
| X-004 | C | S2 proceso | D | Sigue vigente. |
| Y-001 | C | S2 | K | Gate saltado en PPO/SAC. |
| Y-002 | C | S2 🔴 | K | Reward vol muerto. |
| Y-003 | C | S2 | K | Kelly sin calibración en orchestrator. |
| Y-004 | C | S2 | K | Translator. |
| Y-005 | C | S3 | K | Obs serve incompleta. |

---

## 6. VEREDICTO DE DESPLIEGUE actualizado (post-1B)

# 🔴 BLOQUEADO (sin cambio de color)

| Componente | Motivo post-1B |
|------------|----------------|
| Estrategia DQN direccional | **A-001 S1** — 3/3 gates fallan condición Sharpe ≤ buy-and-hold. El gate **sí** bloquea promoción DQN. |
| Fix ADR-041 / reward MTM | **Desbloqueado como riesgo de integridad** — A-008 verificado + tests. |
| Sizing “fantasma” (X-001) | **Desbloqueado** — stack existe; diagnóstico se equivocaba. |
| Acoplamiento p_win→Kelly (X-002) | **Rebaja a S2**: no es el default del DQN agent; **sí** es riesgo en platform (Y-003/Y-004). |
| Criterio del gate (X-003) | **Código OK**; docs incompletos. Fallos reales = B&H, no “DSR 0.85 vs 0.4”. |
| Platform paper path | **Sigue en riesgo S2** por Y-003/Y-004 si llegan señales con p_win/position_size no calibrados. |
| Multiagente FABLE | **D-001** — sin cambio. |

**Condiciones para levantar bloqueo de estrategia:** nueva hipótesis con gate PASS (las 3 condiciones), no parches de red.  
**Condiciones para endurecer platform:** cerrar Y-003/Y-004 (guards de calibración en path de capital).

---

## 7. Backlog de fixes re-priorizado (solo lectura → candidatos Fase 2)

| Prio | ID(s) | Acción | 🔴 ¿Afecta estrategia/validación estadística? | Tipo |
|------|-------|--------|-----------------------------------------------|------|
| 1 | A-001 + D-001 | Decision Log: matar/pivotar hipótesis direccional; congelar multiagente | No (gobernanza) | D |
| 2 | X-003 | ADR/doc único: criterio = DSR>θ **y** Sharpe>B&H **y** DSR>XGB; DSR ∈ [0,1] | No | D |
| 3 | Y-003 + Y-004 | Guard calibración / rechazar kelly no calibrado en orchestrator + translator | No (riesgo) | K |
| 4 | Y-001 | No devolver exit 0 en PPO/SAC sin gate (o cablear gate) | No (gobernanza promo) | K |
| 5 | A-003 | Cablear calibrador OOS en path de serve (no en TEST del gate) | **SÍ** — re-validar señales | K |
| 6 | Y-002 | `vol_target` fijo o forecast ≠ realized del mismo bar | **SÍ** — reentrenar + re-gate | K |
| 7 | Y-005 | Extender PortfolioState o enmascarar dim en train | Posible SÍ | K |
| 8 | D-007 + D-008 | Reglas gobernanza (no online-learning risk path; budget Optuna) | No | D |
| 9 | A-006, A-005, D-004, D-003… | Tras pivote de alfa / Nivel 2 | variable | mix |
| — | A-008, X-001-como-ausencia | **Sin fix** (ya correctos o falso positivo de docs) | — | — |

---

## >>> CHECKPOINT FASE 1B — COMPLETADA. CERO FIXES APLICADOS.

Esperando:
- Lista de IDs a corregir en **Fase 2**, o  
- **`EJECUTAR TODO`** (= solo **S1 y S2 CONFIRMADOS** de tipo código/decisión listos: p.ej. **Y-003, Y-004, Y-001, A-003**, y decisiones **A-001/D-001/X-003/D-007/D-008**).  

**No incluidos en EJECUTAR TODO automático sin mención explícita:** Y-002 y A-003 cableado de calibración (marca roja de estrategia) — requieren aprobación consciente porque invalidan validación estadística vigente.
