# SEGUIMIENTO DEL BOT — Bitácora de proceso y decisiones

> **Para qué sirve este documento:** dar seguimiento al avance, entender **por qué**
> se tomó cada decisión, mantener el **objetivo** a la vista, y **no repetir
> errores**. Es un documento **vivo**: cada vez que pasa algo relevante, se añade
> una entrada. Mantenlo a la mano.
>
> Compañero: [ARQUITECTURA_Y_APRENDIZAJE.md](ARQUITECTURA_Y_APRENDIZAJE.md) explica
> *cómo* funciona; este documento registra *qué se hizo y por qué*.

---

## 1. OBJETIVO

**Construir un agente de trading (deep reinforcement learning) que demuestre un edge
real y robusto** — medido por DSR walk-forward fuera de muestra — **superando a
buy-and-hold y a un baseline XGBoost**, con gestión de riesgo estructural y sin
sobreajuste.

**Criterio de éxito (el "examen" a aprobar):**
`DSR_agente > 0.4` (meta 0.6) **Y** `Sharpe_agente > Sharpe_buyhold` **Y**
`DSR_agente > DSR_xgboost`, sobre OOS concatenado.

**No-objetivos:** no prometer rentabilidad; no confundir señal con suerte; no operar
en vivo hasta pasar el gate y shadow trading.

---

## 2. ESTADO ACTUAL (2026-07-17)

- **Peldaño:** DQN (escalera Q-table → DQN → PPO → SAC). PPO/SAC **no promocionan** sin gate (Y-001).
- **Infraestructura:** ✅ loader datos reales, ✅ env, ✅ entrenamiento, ✅ gate DSR
  walk-forward, ✅ reward MTM (ADR-041), ✅ guards Kelly calibrado (Y-003/Y-004),
  ✅ calibrador OOS en serve/train (A-003), ✅ `vol_target` del reward vivo (Y-002).
- **Último resultado serio (pre-ajustes de hoy):** DSR ~0.85 pero **FAIL** vs B&H
  (3 gates: SPY/BTC). Hipótesis direccional **formalmente muerta** (A-001).
- **Bloqueo / pendiente inmediato:** 🔴 **reentrenar DQN + re-gate** tras Y-002
  (reward cambió) y A-003 (p_win serve cambió). No promocionar checkpoints viejos.
- **Docs de auditoría:** `docs/TRIAJE_FASE1_AUDITORIAS.md`, `docs/FIXLOG_FASE2.md`,
  `docs/adr/044-promotion-gate-criteria.md`, `docs/governance/*`.

---

## 3. LÍNEA DE TIEMPO (hitos)

| Fecha | Hito |
|-------|------|
| 2026-06-09 | Driver de entrenamiento DRL (`train_drl.py`) + DQN/PPO/SAC cableados; fix de `TemperatureScaling`. |
| 2026-06-10 | Loader de datos reales Alpaca + 21 features (GMM régimen anti-leakage). |
| 2026-06-10 | Span de datos resuelto: IEX gratis = 2018-11 en adelante (~1.476 barras; sin SIP). |
| 2026-06-10 | **ADR-040**: gate DSR walk-forward (vs buy-and-hold + XGBoost). 7/7 tests. |
| 2026-06-10 | Paralelización de folds del gate (`--n-jobs`, spawn-safe, reproducible). |
| 2026-06-10 | **Primer run serio** SPY: DQN DSR=0.85, no bate buy-and-hold. |
| 2026-06-10 | **ADR-041**: reward mark-to-market (alinea train/eval). Implementado, 28 tests (commit `4447e07`). |
| **2026-07-17** | **Triaje + Fase 1B + Fase 2 (fixes)** — ver §3.1 con horas. |

### 3.1 Ajustes 2026-07-17 (horas locales UTC−05:00)

| Hora | ID / tema | Ajuste | Evidencia / commit |
|------|-----------|--------|--------------------|
| ~12:00–13:00 | Fase 1B | Verificación en código del triaje; X-001: sizing **sí** tiene vol-target·Kelly·CVaR-lite; diagnóstico FABLE desactualizado. X-003: DSR ∈ [0,1]; 3 FAIL = Sharpe ≤ B&H. | `docs/TRIAJE_FASE1_AUDITORIAS.md` |
| 13:42 | **Y-003** | Kelly en `AllocationEngine.position_size_from_signal` bloqueado si `p_win_calibrated=False`. | `ebf289b` |
| 13:42 | **Y-004** | `translate_signal` no acepta Kelly/`position_size` sin `p_win_calibrated`. | `5727bfd` |
| 13:42 | **Y-001** | PPO/SAC en `train_drl` salen con **exit 2** (no auto-promover sin gate). | `7017294` |
| 13:42 | **A-001, D-001, X-003** | Decision Log: hipótesis direccional muerta; congelar multiagente; **ADR-044** criterio gate unívoco. | `f669db7` |
| 13:42 | Docs | FIXLOG Fase 2 + triaje en `docs/`. | `0492ee8`, `5cb6dfb` |
| 13:51 | **Y-002** 🔴 | Reward: `vol_target` fijo en config (default 0.01); ya no `= vol_realized`. **Invalida validación previa.** | `dd05bef` |
| 13:51 | **A-003** 🔴 | Calibrador OOS cableado: sidecar + `from_checkpoint_calibrated`; `train_drl` guarda calibrador. **Cambia p_win serve.** | `5fdc375` |
| 13:51–13:52 | Docs | FIXLOG + Decision Log actualizados (marca roja Y-002/A-003). | `2118c71`, `657aea1` |
| 13:55+ | Push + SEGUIMIENTO | Push a GitHub + esta bitácora con horas. | este commit |

---

## 4. REGISTRO DE DECISIONES (qué y POR QUÉ)

> Resumen en cristiano de los ADR. El detalle técnico está en `ADR-0XX-*.md`.

**ADR-040 — Gate DSR walk-forward.**
*Qué:* reemplazar el criterio `reward_OOS > 0` por un juez estadístico riguroso.
*Por qué:* un solo split daba resultados inestables (−0.19 vs −11.26 según el año
que cayera en el examen). Necesitábamos un veredicto que no dependiera del azar de
la ventana. *Decisión clave:* GMM de régimen se re-ajusta **por fold** (si se
ajustara una vez sobre todo, habría fuga de información del futuro → edge fantasma).

**ADR-041 — Reward mark-to-market.**
*Qué:* premiar el retorno de la posición **cada barra**, no solo al cerrar.
*Por qué:* el agente tenía DSR 0.85 (señal real) pero perdía contra buy-and-hold
porque el reward viejo (solo P&L realizado) no lo premiaba por **montar
tendencias**, y encima lo penalizaba por sostener posiciones. *Matiz crítico:* esto
roza un anti-patrón prohibido (P&L no realizado acumulado incentiva sostener
perdedoras), pero el retorno MTM **por barra** es distinto — una perdedora sangra
puntos cada barra, así que sí presiona a salir. Anti-patrón aclarado, no borrado.

**Decisión de método:** Optuna (búsqueda de hiperparámetros) queda **subordinado**
al fix estructural. Primero arreglar el reward y re-medir; tunear solo si hace falta.
*Por qué:* tunear un reward mal diseñado desperdicia cómputo — el método más
eficiente es arreglar la causa raíz antes de optimizar los detalles.

**2026-07-17 — Gobernanza post-triaje (A-001 / D-001 / ADR-044).**
*Qué:* matar formalmente `stock.position.dqn_directional`; congelar multiagente
hasta un agente con gate PASS; documentar criterio de promoción (DSR > θ **y**
Sharpe > B&H **y** DSR > XGB; DSR es probabilidad ∈ [0,1]).
*Por qué:* 3 gates fallidos por B&H; construir 168 agentes sin edge es scope creep.

**2026-07-17 — Guards de capital (Y-003 / Y-004).**
*Qué:* no Kelly ni `position_size` sin `p_win_calibrated=True`.
*Por qué:* softmax de Q no es frecuencia; Kelly sobre pseudo-proba = sizing distorsionado.

**2026-07-17 — Y-002 / A-003 (marca roja).**
*Qué:* vol-target del reward vivo; calibrador OOS en path de serve/train.
*Por qué:* bugs/gaps reales en código. *Consecuencia:* hay que **reentrenar y
re-gate**; no reutilizar métricas de runs anteriores como evidencia de edge.

---

## 5. ERRORES COMETIDOS Y LECCIONES (para NO repetir)

| Error | Qué pasó | Lección / regla |
|-------|----------|-----------------|
| **Span de datos silencioso** | Pedimos SPY 2015-2024 pero IEX solo dio desde 2018-11; no nos dimos cuenta al principio. | Verificar SIEMPRE el rango real devuelto (`bars=`, fechas min/max), no asumir que el rango pedido = el recibido. |
| **Un solo split como criterio** | DSR de un split saltó de −0.19 a −11.26 solo por mover el eval. | Nunca decidir con un split único. Walk-forward multi-fold + DSR. |
| **Reward desalineado** | Se entrenaba con P&L realizado pero se evaluaba con mark-to-market → el agente no montaba tendencias. | El reward de entrenamiento debe medir **lo mismo** que el examen. Alinear objetivo y métrica. |
| **Token de GitHub expirado en Cursor Cloud** | El agente cloud no podía hacer pull (token `ghs_` caducado, embebido en `insteadOf`). | Los tokens de los background agents caducan (~1h). Si falla la auth, **agente nuevo**, no pelear con la config. |
| **Intérprete de Python equivocado** | `pip` instalaba en un Python y `python` corría en otro (venv hermes-agent). | Usar `python -m pip` para instalar en el MISMO intérprete que ejecuta. Verificar `python --version`. |
| **Credenciales placeholder** | Se corrió con `ALPACA_API_KEY="tu_key"` literal → 401. | Confirmar que las keys reales estén cargadas (`$env:...Substring(0,4)`) antes de lanzar. |
| **Perder el output al reiniciar Cursor** | La terminal de Cursor borra el scrollback al reiniciar; se perdió un run. | Correr en ventana aparte y/o `\| Tee-Object archivo.txt`. El código siempre está en git; solo se pierde la pantalla. |
| **Sync lag de OneDrive** | `py_compile` daba SyntaxErrors falsos en archivos recién editados. | El mount tarda en sincronizar; verificar con la herramienta de lectura autoritativa, no con compilaciones inmediatas. |
| **vol_target = vol_realized (Y-002)** | El término `w_vol` del reward era siempre 0. | Nunca anclar el target de una penalización al valor realizado de la misma barra. |
| **p_win sin calibrar en Kelly (Y-003/4)** | Platform podía dimensionar con softmax crudo. | Guard R-02: calibración OOS obligatoria antes de Kelly. |
| **PPO/SAC edge=True (Y-001)** | Entrenar sin gate se reportaba como éxito promocionable. | Sin juez OOS → exit 2 (no promover). |
| **Docs que se auto-elogian (X-004)** | Auditoría y diagnóstico compartían narrativa del fix ADR-041. | Código + tests independientes; no confiar solo en prosa. |

---

## 6. MÉTRICAS / RESULTADOS

| Run | Datos | Config | Resultado | Veredicto |
|-----|-------|--------|-----------|-----------|
| Stub | random-walk | DQN 500ep | OOS −8.8 | NO EDGE (esperado, datos falsos) |
| SPY single-split A | 2018-2024 | DQN 500ep | OOS −0.19 | Inestable (un split) |
| SPY single-split B | 2018-2026 | DQN 500ep | OOS −11.26 | Inestable (un split) |
| **SPY gate** | 2018-2026 | DQN 3 folds × 500ep | **DSR 0.848, Sharpe 0.553** vs buyhold 1.325, xgb 0.284 | **FAIL** (señal real, no bate pasivo) |
| SPY gate + reward MTM | 2018-2026 | DQN 3 folds × 500ep | *pendiente* | *pendiente* |
| BTC gate | ~2018-2026 | DQN 3 folds × 500ep | Sharpe 0.443 vs B&H 0.818, DSR≈0.84 | **FAIL** (B&H) |
| Post Y-002/A-003 | — | reward + calibrador nuevos | *no corrido* | **re-gate obligatorio** |

---

## 7. PRÓXIMOS PASOS

1. **[INMEDIATO / 🔴]** Reentrenar DQN y re-correr gate SPY (y/o BTC) **después de
   Y-002 + A-003**. Criterio = ADR-044 (3 condiciones). Checkpoints viejos no valen.
2. **[HIPÓTESIS]** No insistir en direccional-diario índices: pivote market-neutral /
   stat-arb (ADR-043) u otra tesis falsable.
3. **[SI HACE FALTA]** Optuna sobre pesos del reward solo **tras** un re-gate limpio.
4. **[DESPUÉS]** PPO/SAC solo cuando exista gate DSR para esos algos (hoy exit 2).

---

## 8. CÓMO MANTENER ESTE DOCUMENTO

Cada vez que pase algo relevante (un run, una decisión, un error, un cambio de
rumbo), añade:

- A la **línea de tiempo** (§3): una fila con fecha + hito.
- Si fue una **decisión** con tradeoff: a §4, con el **por qué**.
- Si fue un **error**: a §5, con la **lección**.
- Si fue un **run**: a §6, con números.
- Actualiza el **estado actual** (§2) y los **próximos pasos** (§7).

Regla de oro: escribe el **por qué**, no solo el qué. El "qué" lo cuenta el código;
el "por qué" se pierde si no se anota.
