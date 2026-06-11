# ADR-043 — Pivote market-neutral: stat-arb de pares cointegrados

> **Status: Accepted** (implementado 2026-06-11; tests §8 verdes en CPU).
> Diseño blindado por Opus; implementación por Fable 5.
> Primer agente **market-neutral** del sistema, tras el veredicto del Nivel 1: el
> DQN direccional no bate buy-and-hold en 3 runs serios (SPY 2018/2021, BTC). Un
> stat-arb de pares es long-short por construcción y se mide **vs ZERO**, no vs el
> pasivo — el encuadre donde el short deja de ser una acción perdedora.
>
> **Run real SPY/QQQ (2026-06-11, DoD §11)** — `python -m research.cli.run_pair_gate`:
> 1.476 barras IEX (2020-07-27 → 2026-06-10), splitter 600/150/embargo 60
> expanding, 5 folds. **Los 5 folds rechazan cointegración** (Engle-Granger
> p ∈ [0.39, 0.93] en train) → flat → `GateResult(passed=False, dsr=0.0,
> sharpe=0.0, n_oos=750, n_trials=1)`. FAIL esperado y válido (§9: "SPY/QQQ
> es un par fino; espera muchos FAIL") — el método anti-leakage queda
> validado; el siguiente paso es probar pares mejores (sustitutos
> sectoriales, dual-listed) con deflación honesta del n_trials acumulado.

---

## 1. Motivación

Tres runs serios: DQN direccional con señal real pero débil (Sharpe 0.30–0.55) que
**pierde contra buy-and-hold** (0.7–1.3). El reward MTM no cerró la brecha. Causa: el
encuadre "batir el pasivo con un agente direccional sobre un índice líquido" es
perdedor por construcción — en una tendencia, el short solo resta.

Un **stat-arb de pares** invierte el juego: opera el **spread** entre dos activos
cointegrados (largo el barato, corto el caro), es **dollar/beta-neutral**, y su
benchmark es **ZERO** (retorno absoluto), no el índice. El short es la mitad del edge.

## 2. Decisión

Implementar un agente **rule-based** de stat-arb de pares (NO DRL — la cointegración
+ z-score es un problema clásico; DRL sería sobreingeniería, diagnóstico F.4). Par
inicial: **SPY/QQQ** (líquido, datos diarios ya disponibles en Alpaca). Validarlo con
el gate walk-forward **vs ZERO**.

## 3. Especificación de la estrategia

Por fold, sobre **train_k** únicamente:
1. **Cointegración**: test Engle-Granger (`statsmodels.tsa.stattools.coint`) sobre
   los log-precios de las dos patas. Si no cointegran (p-value > α, default 0.05) →
   **par rechazado en ese fold** (no se opera).
2. **Hedge ratio β**: OLS `log(y) ~ log(x)` sobre train. `spread = log(y) − β·log(x)`.
3. **Normalización**: `mean`, `std` del spread sobre train. `z = (spread − mean)/std`.
4. **Half-life**: estimar la vida media de reversión (Ornstein-Uhlenbeck: regresión de
   `Δspread` sobre `spread_lag`). Si half-life > `max_half_life` (default 30 barras) o
   negativa → **par rechazado** (no revierte útilmente). *Criterio de falsación.*

Sobre **test_k** (usando β, mean, std **congelados del train** — anti-leakage):
5. **Señal** (long-short, usa AMBAS patas):
   - `z > +entry` → spread caro → **SHORT spread** (short y, long β·x).
   - `z < −entry` → spread barato → **LONG spread** (long y, short β·x).
   - `|z| < exit` → cerrar (reversión a la media).
6. Defaults: `entry = 2.0`, `exit = 0.5` (en desviaciones estándar).

## 4. Anti-leakage (LO CRÍTICO — no negociable)

| Qué | Dónde se ajusta | Si se ajusta mal |
|-----|-----------------|------------------|
| Test de cointegración | **train_k only** | "descubres" pares que cointegran por mirar el futuro |
| Hedge ratio β | **train_k only** (OLS) | el spread se construye sabiendo el co-movimiento futuro → revierte mágicamente |
| `mean`, `std` del spread (z-score) | **train_k only** | el z-score conoce la distribución futura del spread → señales tramposas |
| Half-life | **train_k only** | seleccionas pares por reversión futura |

En test SOLO se **aplican** los parámetros congelados del train. Embargo ≥ 60 barras
(consistente con ADR-040). Test obligatorio que verifique que `coint`/OLS/mean/std
ven **solo** índices de train por fold (espía, igual que ADR-040 §5.2).

## 5. Retorno market-neutral (definición única)

```
ret_spread_t = (ret_y_t − β·ret_x_t) / (1 + |β|)          # beta-hedged, normalizado
r_t          = position_t · ret_spread_t − costos
costos       = (fee_bps/1e4) · (|Δpos_y| + |Δpos_x|)       # DOS patas → doble fee
```
`position_t ∈ {−1, 0, +1}` (short/flat/long spread). El doble fee es clave: los edges
de stat-arb son finos y los costos de 2 patas los matan si no se modelan.

## 6. Gate vs ZERO (extensión de ADR-040)

Para `benchmark = Benchmark.ZERO` (ya en el contrato), el criterio de promoción cambia:
- ❌ NO se compara contra buy-and-hold (un market-neutral no intenta batir el índice).
- ✅ Promociona si **`DSR_agent > dsr_threshold`** (el DSR ya codifica "Sharpe
  creíblemente > 0") **Y** `sharpe_agent > 0` sobre OOS concatenado.
- Baseline de control: comparar contra un long-short **aleatorio** (z-score barajado) o
  XGBoost sobre el spread — para confirmar que el edge no es ruido.

Reusar `deflated_sharpe_ratio` / `probabilistic_sharpe_ratio` (walk_forward_runner) y
`WalkForwardSplitter`. La deflación honesta sigue: `n_trials` = nº de pares/configs
evaluados OOS (¡buscar entre muchos pares infla el sesgo de selección — clave!).

## 7. Interfaces y alcance

- `research/alpha/statarb/pairs.py`: `PairStatArb` con `fit(train) → params`,
  `signals(test, params) → positions`, `returns(test, positions) → np.ndarray`.
- Gate: extender `dsr_gate` con un path `benchmark=ZERO` que use el retorno del §5.
- **Alcance de este primer build**: validar el EDGE (¿pasa el gate vs zero?). La
  integración como `AlphaAgent`/`TradeSignal` de 2 patas (emite 2 órdenes) es un
  follow-up (Nivel 2) — el contrato `Signal` es por-símbolo y un par necesita una
  extensión (`PairSignal` o lista de señales). NO lo construyas aún; primero el edge.

## 8. Tests de aceptación (Fable 5, CPU verdes)

1. **Anti-leakage**: espía `coint`/OLS/mean/std → cada fold ve solo índices de train.
2. **Cointegración sintética**: con dos series cointegradas generadas, el spread fitted
   tiene media ~0 y el z-score revierte; con dos random-walks independientes, el par se
   **rechaza** (p-value alto).
3. **Half-life**: par con reversión rápida pasa el filtro; uno sin reversión se rechaza.
4. **Long-short usado**: las posiciones contienen tanto +1 como −1 (no colapsa a un lado).
5. **Doble fee**: el costo escala con `|Δpos_y| + |Δpos_x|`.
6. **Gate vs zero**: con un spph cointegrado-rentable sintético → `DSR > threshold`,
   `passed=True`; con ruido → `passed=False`.
7. **Embargo ≥ 60** entre train y test por fold.

## 9. Riesgos y limitaciones (honestos)

- **La cointegración se rompe** (regime shifts): muchos pares "cointegran" in-sample y
  fallan OOS. El gate walk-forward existe justo para rechazarlos — espera muchos FAIL.
- **SPY/QQQ es un par fino**: ambos siguen el mercado; el spread es pequeño y tras
  doble fee el edge puede no existir. Es el ejemplo didáctico para arrancar; pares
  mejores (sustitutos sectoriales, dual-listed) vienen después.
- **Muestra chica** (~1.400 barras IEX): β/mean/std por fold son ruidosos.
- **Sesgo de selección**: buscar entre N pares y quedarse con el mejor infla
  descubrimientos falsos — la deflación del DSR por `n_trials` es obligatoria.

## 10. Asignación

- Diseño (este ADR, anti-leakage, criterio vs-zero): **Opus + revisión humana**.
- Implementación (`pairs.py`, gate-vs-zero, tests): **Fable 5 · alto**.
- Verificación: el test anti-leakage (espía β/mean/std train-only) — donde un error
  da edge fantasma con dinero real.

## 11. Definition of Done

- `research/alpha/statarb/pairs.py` + extensión `dsr_gate` benchmark=ZERO.
- 7 tests del §8 verdes en CPU.
- Un run real SPY/QQQ con `GateResult` (vs zero). Esperar FAIL es un resultado válido
  (dice "este par no tiene edge tras costos"); lo importante es que el **método** sea
  correcto y anti-leakage.
- Commit atómico, CLAUDE.md §19-20.
