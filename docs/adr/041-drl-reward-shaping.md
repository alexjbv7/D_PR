# ADR-041 — DRL Reward Shaping: Mark-to-Market Alignment

> **Spec de implementación para Fable 5 (esfuerzo alto/muy alto).**
> Rediseña la función de reward del `TradingEnvironment` para alinear el
> objetivo de entrenamiento con la métrica de evaluación del gate (ADR-040).
> **Status: Accepted** (implementado 2026-06-10; tests §8 verdes en CPU).
> Diseño blindado por Opus/Claude; implementación por Fable 5.
> Enmienda parcialmente el anti-patrón de reward de ADR-037 (ver §4).
>
> **Nota de implementación (2026-06-10)**: en modo `"mtm"` el episodio termina
> con la posición abierta (sin cierre forzoso) — el cierre forzoso del legacy
> cobraría una fee fantasma que `positions_to_returns` no ve, rompiendo la
> identidad bit-a-bit del §7. El gate da el mismo trato sin liquidación a
> buy-and-hold. El re-run del gate SPY con `reward_mode="mtm"` (§9) queda
> pendiente como paso operativo; Optuna (§5.2) no se lanza hasta medir ese run.

---

## 1. Motivación (qué nos dijo el primer run serio)

Run real SPY 2018-2026, DQN 3 folds × 500 eps (ADR-040 gate):

```
dsr_agent=0.848  sharpe_agent=0.553  sharpe_buyhold=1.325  dsr_xgb=0.284  passed=False
```

El agente tiene **señal real** (DSR 0.85 creíble, supera el umbral 0.4 y al baseline
XGBoost) pero **no le gana a buy-and-hold de SPY**. La causa es un bug de diseño del
reward, no de capacidad del modelo:

**Desalineamiento train/eval.** `envs/trading_env.py::compute_reward` usa:
```python
r_pnl = pnl_realized / equity   # SOLO P&L realizado
```
El agente solo "cobra" al **cerrar** una posición. Mantener un largo ganador en una
tendencia da **reward 0** hasta el cierre. Peor: `r_idle = -idle_penalty if
holding_bars > max_idle_bars` **penaliza sostener** una posición. Pero el gate (y
buy-and-hold) miden **mark-to-market** por barra (`pos·price_return`). O sea: el
agente se entrena para hacer trades cortos y salir, pero se evalúa contra alguien
que monta la tendencia → estructuralmente pierde contra el pasivo en un bull market.

**Bug secundario de escala.** `r_cost = -(fee_bps/1e4)·|Δpos|·price` escala por el
precio absoluto (~500 para SPY), mientras `r_pnl` está normalizado por equity. Son
**inconmensurables**: los fees dominan el reward y desincentivan operar del todo.

## 2. Decisión

Rediseñar el reward por paso para que sea **mark-to-market**, con costos en unidades
de retorno y penalización de inactividad que NO castigue sostener posiciones. El
RiskGate sigue externo (ADR-009); el reward nunca decide límites de riesgo.

## 3. Diseño del nuevo reward

```
r_t = w_ret  · (pos_{t-1} · price_return_t)              # MTM por barra
    − w_cost · (fee_bps / 1e4) · |Δpos_t|                # costo en unidades de retorno (sin ·price)
    − w_dd   · max(0, drawdown_t − dd_threshold)         # penalización de drawdown (se mantiene)
    − w_vol  · max(0, vol_realized_t − vol_target)        # penalización de vol (se mantiene)
    − w_idle · 1[pos_t == 0 y flat_bars > max_flat_bars]  # penaliza estar FLAT, no sostener
```

donde `price_return_t = close_t / close_{t-1} − 1`.

- **`reward_mode`**: añadir a `EnvironmentConfig` un campo `reward_mode ∈ {"mtm", "realized"}`,
  default `"mtm"`. El modo `"realized"` preserva el comportamiento legacy de ADR-037
  para A/B (shadow trading, igual que el rollout de ResMLP).
- **Pesos configurables** en `EnvironmentConfig` (`w_ret`, `w_cost`, `w_dd`, `w_vol`,
  `w_idle`) como dataclass con defaults — son el espacio de búsqueda de Optuna (§5).
- La definición de `pos·price_return` debe ser **idéntica** a la del gate
  (`dsr_gate.positions_to_returns`, ADR-040 §3.3) para que entrenamiento y evaluación
  midan lo mismo. Esto es el corazón del fix.

## 4. Enmienda al anti-patrón de ADR-037 (lo crítico — leer con cuidado)

CLAUDE.md §6.10 y ADR-037 prohíben: *"Reward con P&L no realizado — incentiva mantener
posiciones perdedoras."* El nuevo reward NO viola esto, y hay que documentar por qué:

| Patrón | Qué es | ¿Incentiva sostener perdedoras? |
|--------|--------|---------------------------------|
| ❌ P&L **no realizado acumulado** | Recompensa la ganancia en papel total de una posición abierta | **Sí** — el agente "disfruta" la ganancia latente y evita cerrar perdedoras esperando recuperación |
| ✅ Retorno MTM **por barra** (este ADR) | Recompensa el retorno real de cada barra: `pos·ret_t` | **No** — una posición perdedora sangra reward negativo **cada barra**, creando presión para salir |

El retorno MTM por barra es el reward estándar y teóricamente fundamentado en RL para
trading (es el retorno que la estrategia realmente gana). Castiga sostener perdedoras
correctamente. **Fable 5 debe actualizar el anti-patrón en CLAUDE.md/ADR-037 para
distinguir ambos, no borrarlo.**

## 5. Optuna — pero subordinado al fix (eficiencia)

> El usuario pidió el método más eficiente. **El fix estructural (§3) va primero; Optuna
> después, y solo si hace falta.** Tunear los pesos de un reward desalineado desperdicia
> cómputo. Secuencia obligatoria:

1. Implementar el reward MTM con pesos por defecto razonables. Re-correr el gate SPY.
   Medir cuánto del gap contra buy-and-hold cierra el fix **solo**.
2. Si aún no basta, Optuna sobre los pesos `(w_ret, w_cost, w_dd, w_vol, w_idle)` con
   **embudo barato → caro**:
   - **Proxy barato**: objetivo = Sharpe en un slice de validación de UN fold, episodios
     reducidos, con `MedianPruner` (CLAUDE.md §6.7). Poda configs malas rápido.
   - **Gate completo solo en finalistas**: top-K del proxy → DSR walk-forward completo.
   - **Budget**: ≤ 50 trials de proxy, ≤ 5 finalistas al gate.
3. **Deflación honesta del DSR**: `n_trials_searched` del gate = nº de configs evaluadas
   **en OOS** (los finalistas), no los trials del proxy. Inflar esto sería trampa
   (CLAUDE.md §6.10). El DSR final ya viene deflactado por la búsqueda.

Módulo opcional: `research/models/drl/reward_search.py` (Optuna), solo si se llega al
paso 2.

## 6. Anti-leakage (hereda de ADR-040, no negociable)

- El cambio es en `compute_reward` (env) — NO toca el pipeline de features, así que el
  re-fit del GMM por fold (ADR-040 §4.1) sigue intacto.
- Optuna: el proxy de validación debe usar un slice **disjunto** del train del fold, y
  **nunca** tocar el fold de test final usado para el DSR. Test anti-leakage obligatorio.
- Embargo ≥ 60 barras se mantiene.

## 7. Interfaces

- `envs/trading_env.py`: `compute_reward` gana `reward_mode` + pesos; o un
  `compute_reward_mtm(...)` separado seleccionado por `reward_mode`. `EnvironmentConfig`
  gana `reward_mode` y los `w_*` (con defaults; tipado estricto).
- `dsr_gate.positions_to_returns` es la fuente de verdad de la definición de retorno;
  el reward MTM debe coincidir bit-a-bit con ella en el caso sin penalizaciones.
- Sin dependencias nuevas salvo que se llegue a Optuna (ya está en `pyproject.toml`).

## 8. Tests de aceptación (Fable 5 escribe y deja verdes, CPU)

`research/tests/test_reward_shaping.py`:

1. `test_mtm_rewards_winning_hold`: posición larga sostenida en un tramo alcista conocido
   → reward positivo **cada barra** (no solo al cerrar).
2. `test_mtm_penalizes_losing_hold`: largo sostenido en tramo bajista → reward negativo
   cada barra (prueba que NO es el anti-patrón de P&L no realizado).
3. `test_cost_in_return_units`: un flip de 1 unidad de posición cuesta exactamente
   `fee_bps/1e4` (sin `·price`).
4. `test_idle_penalizes_flat_only`: la penalización se dispara con `pos==0` prolongado,
   NO al sostener una posición.
5. `test_reward_mode_ab`: mismo trayecto, `reward_mode="mtm"` vs `"realized"` dan rewards
   distintos (el switch funciona; legacy preservado).
6. `test_reward_matches_gate_return_def`: sin penalizaciones, el término MTM == 
   `dsr_gate.positions_to_returns` sobre el mismo path (consistencia train/eval).
7. `test_reward_finite_deterministic`: sin NaN; mismo seed → mismo reward.
8. (Si Optuna) `test_search_no_leakage`: el objetivo de búsqueda nunca ve el fold de test.

## 9. Definition of Done

- `compute_reward` con modo MTM + pesos configurables; legacy preservado para A/B.
- Anti-patrón de ADR-037/CLAUDE.md aclarado (per-bar MTM ≠ unrealized acumulado).
- Tests del §8 verdes en CPU.
- Re-run del gate SPY con `reward_mode="mtm"` y su `GateResult` (comparar vs el 0.848/0.553 actual).
- Optuna solo si el fix estructural no cierra el gap; con deflación honesta del DSR.
- Commit atómico (§20.7), un PR, co-autoría. Actualizar ADR table en CLAUDE.md.

## 10. Riesgos y limitaciones

- El reward MTM puede inducir más turnover si `w_cost` queda bajo → vigilar fees netos.
- Cerrar el gap contra buy-and-hold en un bull market puede requerir además permitir
  apalancamiento o cambiar el universo (fuera de alcance; ver sonda multi-asset/BTC).
- Si tras el fix el agente iguala pero no supera buy-and-hold, reconsiderar el criterio
  del gate (opción "reframe": Sortino/drawdown), que es un ADR aparte.
