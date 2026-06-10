# ADR-040 — DRL DSR Walk-Forward Promotion Gate

> **Spec de implementación para Fable 5 (esfuerzo alto/muy alto).**
> Reemplaza la heurística `edge = oos_reward > 0` de `research/cli/train_drl.py`
> por un criterio de promoción estadísticamente robusto: DSR del agente sobre OOS
> walk-forward concatenado, vs baselines buy-and-hold y XGBoost.
> **Status: Proposed.** Diseño blindado por Opus/Claude; implementación por Fable 5.

---

## 1. Motivación (por qué existe este ADR)

Un único split train/eval es frágil: el mismo agente DQN sobre SPY dio
`OOS reward = −0.19` (eval ≈ 2023-24) y `−11.26` (eval ≈ 2024-26) — el número
saltó 60x solo por mover la ventana. **Un split no es criterio.** Necesitamos
DSR sobre múltiples folds OOS concatenados, deflactado por el número de trials,
y comparado contra baselines. Solo así "edge" deja de depender del año que tocó.

## 2. NO reinventar — reusar esto (obligatorio)

| Necesidad | Reusar | Ubicación |
|-----------|--------|-----------|
| PSR (Mertens, skew/kurt) | `probabilistic_sharpe_ratio(returns, sr_benchmark, periods_per_year)` | `models/walk_forward_runner.py:71` |
| DSR (deflación por n_trials, Bailey-LdP) | `deflated_sharpe_ratio(returns, n_trials, periods_per_year)` | `models/walk_forward_runner.py:117` |
| Folds walk-forward | `WalkForwardSplitter(train_size, test_size, embargo)` | `models/validation.py:22` |
| Baseline supervisado | `XGBoostClassifier` | `models/zoo.py:143` |
| Loader de datos + features | `build_drl_dataset(...)` | `data/drl_dataset.py` |
| Trainers/agentes | `DQNTrainer/PPOTrainer/SACTrainer` + `TradingEnvironment` | `models/drl/`, `envs/` |

Si una de estas funciones no cubre un caso, **extiéndela, no la dupliques.**

## 3. Diseño

### 3.1 Módulo nuevo: `research/models/drl/dsr_gate.py`

```python
@dataclass(frozen=True)
class GateResult:
    dsr_agent: float
    psr_agent: float
    sharpe_agent: float
    sharpe_buyhold: float
    dsr_xgb: float
    n_trials: int
    n_oos_bars: int
    passed: bool
    reason: str   # explicación legible del veredicto

def walk_forward_oos_returns(
    agent_spec: AgentSpec,          # algo, config, episodes, seed
    raw_ohlcv: pd.DataFrame,        # OHLCV crudo (sin features); ver §4.2
    splitter: WalkForwardSplitter,
    env_cfg: EnvironmentConfig,
    *, seed: int = 42,
) -> np.ndarray:
    """Entrena el agente por fold en train_k, evalúa GREEDY (eps=0) en test_k,
    concatena los retornos por barra de los tramos test. Sin solape train/test."""

def buyhold_oos_returns(raw_ohlcv, splitter) -> np.ndarray: ...
def xgb_oos_returns(raw_ohlcv, splitter, ...) -> np.ndarray: ...

def evaluate_drl_gate(
    agent_returns, buyhold_returns, xgb_returns,
    n_trials: int, dsr_threshold: float = 0.4, periods_per_year: int = 252,
) -> GateResult: ...
```

### 3.2 Criterio de promoción (`passed`)

`passed = True` **solo si las tres condiciones**:

1. `dsr_agent > dsr_threshold` (default 0.4; KPI objetivo §1.1 = 0.6).
2. `sharpe_agent > sharpe_buyhold` (le gana a comprar y mantener SPY).
3. `dsr_agent > dsr_xgb` (supera el baseline supervisado — §6.10).

`reason` debe explicitar cuál condición falló.

### 3.3 Retornos por barra (definición única, usar en agente y baselines)

```
r_t = position_{t-1} · price_return_t − fee_bps/1e4 · |Δposition_t|
price_return_t = close_t / close_{t-1} − 1
```

- Agente: `position_t ∈ {−1,0,+1}` de la acción greedy.
- Buy-and-hold: `position_t = +1` constante (Δposition = 0 salvo entrada inicial).
- XGBoost: `position_t = sign(argmax_proba − 1)` mapeado a {−1,0,+1}.
- `fee_bps` = `EnvironmentConfig.fee_bps` (consistencia con el env).

### 3.4 Wiring en `cli/train_drl.py`

Reemplazar el bloque `edge = oos_reward > 0` por:
```python
result = evaluate_drl_gate(agent_r, buyhold_r, xgb_r, n_trials=args.n_trials_searched)
logger.info("GATE: %s", result.reason)
return 0 if result.passed else 2
```
Añadir flag `--wf-folds` (default 5) y `--n-trials-searched` (default 1).

## 4. Contrato anti-leakage (lo crítico — no negociable)

Un gate con leakage produce **edge fantasma** → riesgo de capital real. Por fold k:

1. **Regime GMM por fold.** `build_drl_dataset` hoy ajusta el GMM en un único
   `train_frac`. Para WF **eso es leakage**: el GMM vería folds futuros.
   **Refactor obligatorio:** exponer la construcción de features de modo que el
   GMM (`GMMRegimeDetector.fit`) se ajuste **solo con las barras de train_k** y
   transforme train_k+test_k. Cada fold tiene su propio régimen. Añadir test que
   verifique `len(close_visto_por_fit) == len(train_k)` por fold.
2. **Features técnicas:** ya son causales (ventanas sobre pasado). Mantener.
   Cualquier escalado/normalización nuevo se ajusta en train_k únicamente.
3. **Embargo:** `splitter.embargo ≥ 60` barras (la ventana más larga es
   `vol_z_60`). Separa train_k de test_k para matar look-ahead de borde.
4. **Greedy en test:** `epsilon=0`. Nada de exploración en OOS.
5. **El test set final (p.ej. último fold / año más reciente) se toca UNA vez**
   (§6.10). No iterar hiperparámetros mirando ese fold.
6. **Baselines bajo el mismo régimen de folds y embargo** que el agente — mismas
   ventanas test, misma definición de retorno (§3.3). Comparación justa.

## 5. Tests de aceptación (Fable 5 debe escribirlos y pasarlos)

`research/tests/test_dsr_gate.py`:

1. `test_dsr_matches_reference`: serie sintética de Sharpe conocido → `deflated_sharpe_ratio` da un valor en rango esperado (sanity vs cálculo independiente).
2. `test_no_leakage_gmm_per_fold`: spy sobre `GMMRegimeDetector.fit` confirma que en cada fold ve **solo** índices de train_k (mirror del patrón en `test_drl_dataset.py::test_gmm_fitted_on_train_slice_only`).
3. `test_buyhold_returns_equal_price_returns`: el baseline buy-and-hold reproduce exactamente `close.pct_change()` (menos el fee de entrada).
4. `test_gate_fails_when_below_threshold`: agente con retornos ~0 → `passed=False` con `reason` correcto.
5. `test_gate_fails_when_below_buyhold`: agente con Sharpe < buy-and-hold → `passed=False`.
6. `test_embargo_enforced`: `splitter.embargo >= 60`, y train_k ∩ test_k = ∅.
7. `test_end_to_end_stub`: corre el gate completo sobre datos stub (pocas folds, pocos episodios) y devuelve un `GateResult` válido sin NaN.

Todos CPU-only, rápidos (usar stub/sintético, no Alpaca). Marcar el e2e pesado con `importorskip("torch")`.

## 6. Notas de coste y alcance

- **Coste:** WF multi-fold = N folds × entrenamiento. Hacer `--wf-folds` y
  `episodes` configurables; para el gate basta N=3-5 y episodios reducidos.
  Documentar el wall-clock.
- **n_trials:** debe reflejar el nº real de configs/seeds probados (sesgo de
  selección). Si solo se probó 1 config, `n_trials=1` y DSR≡PSR. No inflar ni
  subreportar — es lo que hace honesto al DSR.
- **Fuera de alcance:** Optuna sobre el agente, multi-símbolo, PPO/SAC gating
  (el contrato del gate es agnóstico al algo vía `agent_spec`, pero validar
  primero con DQN). Iterar reward/arquitectura es el Paso 3, posterior.

## 7. Definition of Done

- `dsr_gate.py` + `test_dsr_gate.py` (7 tests verdes, CPU).
- `train_drl.py` usa el gate; `--wf-folds`, `--n-trials-searched` añadidos.
- Refactor del régimen por-fold con su test anti-leakage.
- Un run real (DQN, SPY, iex, N=3) que imprima el `GateResult` completo.
- Commit atómico (§20.7), un solo PR, mensaje descriptivo, co-autoría.
