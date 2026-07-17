# ADR-044 — Criterio operativo de promoción (gate DSR walk-forward)

**Estado:** Aceptado  
**Fecha:** 2026-07-17  
**Relaciona:** ADR-040, triaje X-003 / A-001  
**Código canónico:** `research/models/drl/dsr_gate.py` · CLI `research/cli/train_drl.py`

---

## Contexto

Los documentos de auditoría mezclaron “DSR 0.85” con el umbral “> 0.4” y con fallos de gate, sin definir qué es numéricamente el DSR. Tres runs fallaron; hacía falta una definición unívoca.

## Decisión

### 1. Definición de DSR / PSR

- **PSR** = `P(SR_real > SR* | datos)` ∈ **[0, 1]**  
  Implementación: `models.walk_forward_runner.probabilistic_sharpe_ratio` (SE de Mertens).
- **DSR** = `PSR(E[max SR | n_trials])` (Bailey & López de Prado).  
  Si `n_trials <= 1`, **DSR ≡ PSR con SR\* = 0**.
- **No** se interpreta DSR como un ratio de Sharpe anualizado.

### 2. Retorno por barra (única definición OOS)

```
r_t = pos_{t-1} * (close_t / close_{t-1} - 1) - (fee_bps / 1e4) * |Δpos_t|
```

Función: `models.drl.dsr_gate.positions_to_returns`.

### 3. Condiciones de promoción (AND — las tres)

Para un agente **direccional** (`evaluate_drl_gate`):

1. `dsr_agent > dsr_threshold` (default **0.4**)
2. `sharpe_agent > sharpe_buyhold` (mismo folds, misma definición de retorno)
3. `dsr_agent > dsr_xgb` (baseline XGBoost, `n_trials=1` en el baseline)

`passed = True` solo si las tres se cumplen.

Para agentes **market-neutral** (benchmark ZERO): ver `evaluate_zero_gate` (ADR-043).

### 4. Anti-leakage (no negociable)

- Embargo ≥ 60 barras (`MIN_EMBARGO_BARS`)
- GMM de régimen re-ajustado **por fold** solo en train
- Evaluación OOS greedy (ε = 0)

### 5. Códigos de salida CLI (`train_drl`)

| Código | Significado |
|--------|-------------|
| 0 | DQN entrenó **y** gate PASS |
| 1 | Error de configuración / datos |
| 2 | Gate FAIL **o** algo sin gate implementado (PPO/SAC — Y-001) |

### 6. Qué fallaron los tres runs documentados

| Run | Fallo |
|-----|--------|
| SPY largo (`gate_run.txt`) | Condición 2: Sharpe 0.553 ≤ B&H 1.325 (DSR ≈ 0.85 > 0.4) |
| SPY post (`gate_spy_post.txt`) | Condición 2: 0.300 ≤ 1.266 |
| BTC (`gate_btc.txt`) | Condición 2: 0.443 ≤ 0.818 |

El umbral DSR 0.4 **no** fue el cuello de botella reportado; lo fue batir buy-and-hold.

## Consecuencias

- Criterio de “Nivel 1” incompleto si solo dice “DSR > 0.4”: debe citar las **tres** condiciones.
- Documentación y dashboards deben etiquetar DSR como probabilidad, no como Sharpe.
- PPO/SAC no pueden salir con código 0 hasta extender el gate (Y-001).

## Alternativas rechazadas

- Usar solo OOS mean reward > 0 (heurística frágil pre-ADR-040).
- Tratar DSR como Sharpe deflactado en unidades de ratio (confunde operadores).
