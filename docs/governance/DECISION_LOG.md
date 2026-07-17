# Decision Log — quant_bot / D_PR

Registro de decisiones de gobernanza (triaje Fase 1 / 1B).  
Formato: ID · fecha · decisión · justificación · consecuencias.

---

## 2026-07-17 — A-001 · Hipótesis direccional DQN marcada como muerta

| Campo | Valor |
|-------|--------|
| **ID** | A-001 |
| **Decisión** | La hipótesis `stock.position.dqn_directional` se declara **inválida / no promocionable** en su forma actual (DQN direccional diario sobre índices/crypto líquidos). |
| **Evidencia** | Tres runs de gate ADR-040 con `passed=False`, todos por `sharpe_agent <= sharpe_buyhold` (`research/gate_run.txt`, `gate_spy_post.txt`, `gate_btc.txt`). DSR a menudo > 0.4; el fallo no es “falta de DSR” sino falta de edge vs B&H. |
| **Consecuencias** | (1) No promover checkpoints de este agente a paper/live. (2) Cementerio de hipótesis: ver entrada en `docs/governance/CEMENTERIO_HIPOTESIS.md`. (3) Trabajo de alfa prioriza market-neutral / otras tesis (ADR-043) sobre re-tunear la red direccional. |
| **No implica** | Borrar el código del DQN ni el gate; el harness de validación se conserva. |

---

## 2026-07-17 — D-001 · Congelar build multiagente hasta Nivel 1

| Campo | Valor |
|-------|--------|
| **ID** | D-001 |
| **Decisión** | **Congelar** el build de plataforma multiagente tipo FABLE (matriz Activo×Estrategia×Modelo, N agentes, AutoML/NAS) hasta cumplir el **criterio de salida de Nivel 1** (ver ADR-044 / X-003). |
| **Justificación** | Prioridades invertidas: diseñar ~168 agentes antes de un agente con edge OOS demostrado. Riesgo dominante: nunca enviar nada productivo. |
| **Criterio de desbloqueo** | Al menos **un** agente (cualquier hipótesis) con gate ADR-040 PASS (las tres condiciones de ADR-044) en walk-forward documentado. |
| **Permitido mientras tanto** | Un agente, harness de validación, risk/execution, pivote a 1–2 hipótesis falsables (p.ej. stat-arb), tooling de research. |

---

## 2026-07-17 — Y-002 / A-003 · Fixes con invalidación de validación estadística

| Campo | Valor |
|-------|--------|
| **IDs** | Y-002, A-003 |
| **Decisión** | Aplicar fixes de software **y declarar inválida** cualquier validación estadística / paper performance obtenida con el reward o p_win previos. |
| **Y-002** | Reward MTM: `vol_target` configurable (default 0.01), ya no `= vol_realized`. Cambia la función de reward. |
| **A-003** | Path serve: calibrador OOS sidecar + `from_checkpoint_calibrated`; `train_drl` persiste calibrador. Cambia `p_win` emitido. |
| **Acción obligatoria** | Reentrenar DQN y re-correr gate ADR-040 (criterio ADR-044). No promover checkpoints pre-fix. |

---

## 2026-07-17 — X-003 · Criterio operativo del gate (referencia)

| Campo | Valor |
|-------|--------|
| **ID** | X-003 |
| **Decisión** | El único criterio de promoción de un agente DRL es el documentado en **ADR-044** (y implementado en `models.drl.dsr_gate.evaluate_drl_gate`). |
| **Aclaración numérica** | `dsr_agent` es un **PSR/DSR ∈ [0, 1]** (probabilidad), no un Sharpe. Un “DSR 0.85” no implica superar buy-and-hold. |
| **Doc canónico** | `docs/adr/044-promotion-gate-criteria.md` |
