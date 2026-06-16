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

## 2. ESTADO ACTUAL (2026-06-10)

- **Peldaño:** DQN (escalera Q-table → DQN → PPO → SAC).
- **Infraestructura:** ✅ loader datos reales, ✅ env, ✅ entrenamiento, ✅ gate DSR
  walk-forward paralelizado, ✅ reward mark-to-market (ADR-041).
- **Último resultado serio:** agente con **señal real (DSR 0.85)** que **no bate
  buy-and-hold** por reward desalineado → **arreglado en ADR-041**.
- **Bloqueo / pendiente inmediato:** re-correr el gate SPY con `reward_mode="mtm"`
  para comparar contra el `DSR 0.848 / Sharpe 0.553` anterior.

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

---

## 6. MÉTRICAS / RESULTADOS

| Run | Datos | Config | Resultado | Veredicto |
|-----|-------|--------|-----------|-----------|
| Stub | random-walk | DQN 500ep | OOS −8.8 | NO EDGE (esperado, datos falsos) |
| SPY single-split A | 2018-2024 | DQN 500ep | OOS −0.19 | Inestable (un split) |
| SPY single-split B | 2018-2026 | DQN 500ep | OOS −11.26 | Inestable (un split) |
| **SPY gate** | 2018-2026 | DQN 3 folds × 500ep | **DSR 0.848, Sharpe 0.553** vs buyhold 1.325, xgb 0.284 | **FAIL** (señal real, no bate pasivo) |
| SPY gate + reward MTM | 2018-2026 | DQN 3 folds × 500ep | *pendiente* | *pendiente* |

---

## 7. PRÓXIMOS PASOS

1. **[INMEDIATO]** Re-correr el gate SPY con `reward_mode="mtm"`. ¿Cierra la brecha
   contra buy-and-hold? (necesita un entorno con torch).
2. **[EN PARALELO]** Sonda multi-régimen: gate sobre BTC/USD (donde buy-and-hold no
   es un Sharpe 1.3) para ver si el edge aparece donde el pasivo no domina.
3. **[SI HACE FALTA]** Optuna sobre los pesos del reward (`w_ret, w_cost, ...`), con
   embudo proxy-barato → gate-caro y deflación honesta del DSR.
4. **[DESPUÉS]** Subir el peldaño a PPO (acción continua / sizing fraccional) si DQN
   se queda corto.

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
