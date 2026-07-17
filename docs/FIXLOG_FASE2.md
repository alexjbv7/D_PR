# FIXLOG — Fase 2 (EJECUTAR TODO)

**Fecha:** 2026-07-17  
**Alcance aprobado:** Y-003, Y-004, Y-001 + decisiones A-001, D-001, X-003  
**Excluidos (marca roja / no en listado EJECUTAR TODO):** Y-002, A-003 cableado de calibración productiva  
**Workspace:** `C:\Users\alexj\hermes-projects\D_PR` (clone de `alexjbv7/D_PR@68f9b59`)  
**Push:** no realizado (requiere orden explícita de Alex)

---

## Tabla de fixes

| ID | Causa raíz | Archivos | Cambio | Test (falló antes: sí/no) | Sensible | ¿Afecta estrategia? | Commit | Estado |
|----|------------|----------|--------|---------------------------|----------|---------------------|--------|--------|
| **Y-003** | Kelly en orchestrator aceptaba `p_win` sin flag de calibración | `platform/services/strategy-orchestrator/app/allocation_engine.py`, tests | `position_size_from_signal(..., p_win_calibrated=False)` lanza `UncalibratedSignalError` si no calibrado | sí (nuevo `test_uncalibrated_p_win_raises`; tests existentes actualizados con `p_win_calibrated=True`) | **SÍ** (sizing) | No (guard de capital, no cambia señales/alfa) | `ebf289b` | HECHO |
| **Y-004** | Translator trataba `position_size` como Kelly sin exigir calibración | `platform/services/execution-engine/app/signal_translator.py`, tests | Si `position_size>0` y `p_win_calibrated` no es True → `None` + log | sí (`test_uncalibrated_kelly_returns_none`; fixtures con flag True) | **SÍ** (órdenes) | No | `5727bfd` | HECHO |
| **Y-001** | PPO/SAC devolvían éxito de promoción sin gate DSR (`edge=True`) | `research/cli/train_drl.py`, `research/tests/test_train_drl_y001_no_promote.py` | Tras entrenar PPO/SAC: exit **2** + log NO PROMOTE | sí (test estructural: no existe `edge = True`) | No | No | `7017294` | HECHO |
| **A-001** | Hipótesis direccional falló 3 gates; no formalizada | `docs/governance/DECISION_LOG.md`, `docs/governance/CEMENTERIO_HIPOTESIS.md` | Decisión: hipótesis MUERTA / no promocionar | n/a (doc) | No | No (gobernanza) | `f669db7` | HECHO |
| **D-001** | Scope multiagente antes de 1 agente con edge | mismos + Decision Log | Congelar multiagente hasta Nivel 1 (ADR-044) | n/a | No | No | `f669db7` | HECHO |
| **X-003** | DSR mal interpretado en prosa | `docs/adr/044-promotion-gate-criteria.md`, Decision Log | Criterio unívoco: 3 AND; DSR ∈ [0,1]; reconcilia 3 FAIL = B&H | n/a | No | No | `f669db7` | HECHO |

---

## Doble justificación (sensibles Y-003 / Y-004)

| Fix | Por qué es correcto | Por qué no altera otro comportamiento legítimo |
|-----|---------------------|------------------------------------------------|
| Y-003 | R-02 / ADR-042: Kelly exige p_win frecuencia-calibrado; softmax de Q no lo es | Callers que ya calibran pasan `p_win_calibrated=True`; vol-target puro no usa esta función |
| Y-004 | Mismo principio en el path de ejecución paper/live | Señales con sizing legítimo deben marcar el flag; sin flag no se envía orden (fail-closed) |

---

## Resultados de tests (números reales)

Entorno: Anaconda Python 3.x, deps mínimas instaladas localmente.

| Suite | Resultado |
|-------|-----------|
| `strategy-orchestrator/tests/test_allocation_engine.py` | **15 passed** |
| `execution-engine/tests/test_signal_translator.py` + `TestSignalTranslator` | **22 passed** (14 translator + 8 pipeline translator) |
| `research/tests/test_train_drl_y001_no_promote.py` | **1 passed** |
| `research/tests/test_reward_shaping.py` + `test_dsr_gate.py` + Y-001 | **17 passed, 2 skipped** |
| `execution-engine/tests/test_risk_gate.py` | **no corrido** — falta `pandas_market_calendars` / stack async completo en este env |
| `TestNormaliseSymbol` / FastAPI main | **preexistente** — falta `fastapi` en env de triaje |

**No se afirma “suite monorepo 100%”**: el monorepo es multi-paquete con dependencias pesadas (torch, fastapi, kafka). Los tests **del alcance del fix** pasan.

### Demostración pre/post (guardas)

```
Y-003: position_size_from_signal(0.65, 0.02) sin flag → UncalibratedSignalError
Y-004: translate_signal sin p_win_calibrated → None
Y-001: train_drl.py no contiene "edge = True"
```

---

## Continuación — Y-002 + A-003 (aprobados 2026-07-17)

| ID | Causa raíz | Archivos | Cambio | Test (falló antes) | Sensible | ¿Afecta estrategia? | Commit | Estado |
|----|------------|----------|--------|--------------------|----------|---------------------|--------|--------|
| **Y-002** | `vol_target = vol_realized` anulaba `w_vol` | `research/envs/trading_env.py`, `test_reward_shaping.py` | `EnvironmentConfig.vol_target` (default 0.01 diario) usado en `step` | sí `test_y002_vol_penalty_not_dead` | **SÍ** | **🔴 SÍ** — cambia reward → **reentrenar DQN + re-correr gate** | `dd05bef` | HECHO |
| **A-003** | Hook calibrador no cableado en serve/train | `dqn_agent.py`, `train_drl.py`, `alpha/agents/__init__.py`, `test_a003_…` | `from_checkpoint_calibrated` + sidecar auto-load; train_drl guarda calibrador en TRAIN_calib | sí suite A-003 | **SÍ** (señales) | **🔴 SÍ** — cambia `p_win` en serve → **re-validar señales/Kelly** | `5fdc375` | HECHO |

### Tests (continuación)

| Suite | Resultado |
|-------|-----------|
| `test_reward_shaping` + A-003 + Y-001 | **14 passed** (incluye Y-002 + alineamiento gate) |

---

## PENDIENTES

| ID | Motivo |
|----|--------|
| Y-005 | train/serve daily_pnl_pct |
| Suite monorepo completa en CI con deps completas | entorno local incompleto |
| **Re-run gate ADR-040 post Y-002** | obligatorio tras cambio de reward |
| **Re-validar path de señales post A-003** | p_win calibrado cambia sizing waters-down |

## NO CORREGIDOS (con razón)

| ID | Razón |
|----|-------|
| A-001 código de red | Decisión de gobernanza, no parche de red |
| A-008 | Ya correcto en 1B |
| X-001 “sizing ausente” | Falso positivo documental |

---

## Commits locales previstos

Mensajes (atómicos al hacer `git commit`):

1. `fix(Y-003): block Kelly sizing on uncalibrated p_win in AllocationEngine`
2. `fix(Y-004): require p_win_calibrated before Kelly position_size in translator`
3. `fix(Y-001): PPO/SAC train_drl exit 2 until DSR gate exists`
4. `docs(A-001,D-001,X-003): decision log, hypothesis cemetery, ADR-044 gate criteria`
5. `docs: FIXLOG_FASE2 + TRIAJE_FASE1_AUDITORIAS`
