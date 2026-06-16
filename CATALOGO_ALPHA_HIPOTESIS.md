# Catálogo de Hipótesis de Alfa (por estilo × clase × edge)

> Backlog de investigación de `alpha/` (ADR-042). **No es una lista de deseos ni una
> orden de construir.** Cada entrada es una **hipótesis falsable y gateable**: una
> tesis con mecanismo, lógica, datos requeridos, criterio de invalidación y benchmark
> de validación. Una idea sin criterio de falsación **no entra** aquí.
>
> **Disciplina (vinculante):**
> 1. Nada se construye hasta tener el **dato** y pasar su **gate deflactado** (ADR-040).
> 2. El **benchmark del gate cambia según el estilo** (ver plantilla): direccional →
>    buy-and-hold; market-neutral / event-driven → retorno cero / Sharpe absoluto.
> 3. El modelo se **empareja a la SNR** (diagnóstico F.4): regla/GBM por defecto; RL
>    solo si el problema es control secuencial; DL solo con datos que lo justifiquen.
> 4. Prioriza por **factibilidad con el dato actual**, no por atractivo.

---

## Plantilla (toda entrada la sigue)

```
### {clase}.{estilo}.{nombre_corto}
- **Tesis** (1 frase falsable):
- **Mecanismo** (por qué existe el edge — estructural / conductual / microestructura):
- **Clase / Estilo / Horizonte**:
- **Lógica entrada/salida** (conceptual, no código):
- **Features requeridas** (✅ existe en state space / 🔵 falta):
- **Datos requeridos** (y si los tienes HOY):
- **Modelo sugerido** (regla / GBM / RL / híbrido — justificado por SNR):
- **Criterio de invalidación** (qué resultado MATA la hipótesis):
- **Benchmark del gate** (direccional→buy&hold; neutral→cero/Sharpe absoluto):
- **Factibilidad hoy**: ✅ ahora / 🔵 requiere dato / ⚪ experimental
- **Referencia** (si aplica):
```

---

## Ejemplos desarrollados (FIJAN LA VARA — Fable 5 replica este nivel)

### fx.intraday.london_open_breakout
- **Tesis**: el rango de los primeros 30 min de la sesión de Londres (~07:00–07:30 UTC) define un rango cuya ruptura tiene follow-through intradía estadísticamente significativo en majors (EUR/USD, GBP/USD).
- **Mecanismo**: la apertura de Londres concentra el mayor volumen/liquidez FX del día; el flujo institucional acumulado overnight se ejecuta, y la ruptura del rango inicial refleja un desequilibrio de orden que persiste por inercia hasta la apertura de NY.
- **Clase / Estilo / Horizonte**: FX / Intraday / minutos–horas (cierre antes del NY close).
- **Lógica entrada/salida**: fijar [high, low] del rango 07:00–07:30 UTC; entrar en ruptura confirmada (close fuera del rango + filtro de volumen/ATR); stop al lado opuesto del rango; target N×rango o trailing; **cierre forzoso a fin de sesión**.
- **Features**: flags de sesión, opening-range high/low, ancho de rango, `volume_z` intradía, ATR intradía, time-of-day. → 🔵 mayoría NO existen (falta dato intradía).
- **Datos**: barras 1m–5m de FX majors con timestamp de sesión. **Hoy: ❌** (sin feed FX intradía).
- **Modelo**: regla determinista como baseline; GBM/meta-labeler para filtrar falsos breakouts. **RL no aquí** (no es control secuencial al inicio).
- **Invalidación**: si el follow-through post-ruptura no supera spread+slippage OOS, o el edge no sobrevive walk-forward por sesión.
- **Benchmark del gate**: **NO buy-and-hold** (intradía, neutral overnight) → retorno cero / coin-flip de rupturas; Sharpe absoluto + DSR.
- **Factibilidad hoy**: 🔵 requiere datos intradía FX (Nivel 3).
- **Referencia**: literatura de *Opening Range Breakout* (ORB).

### macro.event.scheduled_news_magnitude
- **Tesis**: eventos macro programados (NFP, CPI, decisiones FOMC/ECB) producen un salto de **magnitud/volatilidad predecible** (aunque la dirección no lo sea), explotable vía estructuras de volatilidad o breakout post-release filtrado por sorpresa.
- **Mecanismo**: la incertidumbre se resuelve en un instante → repricing brusco; la magnitud es función de la sorpresa (actual vs consenso). La dirección es difícil; la magnitud no.
- **Clase / Estilo / Horizonte**: FX / índices STOCKS / event-driven intraday (ventana de minutos alrededor del release).
- **Lógica entrada/salida**: posición **no direccional** (straddle) pre-evento, o breakout direccional post-release con filtro de sorpresa; salida en minutos.
- **Features**: calendario económico (timestamp, consenso, actual, sorpresa), vol implícita pre-evento, time-to-event. → 🔵 ninguna existe.
- **Datos**: calendario económico con consensos + (para straddle) datos de opciones/IV. **Hoy: ❌**.
- **Modelo**: regla event-driven + GBM para **predecir magnitud** (regresión). RL no aporta (no es control secuencial).
- **Invalidación**: si el movimiento post-evento no cubre el coste del straddle, o el spread se ensancha tanto en el evento que come el edge.
- **Benchmark del gate**: retorno cero / Sharpe absoluto; **modelar el ensanchamiento de spread en eventos** (si no, edge fantasma).
- **Factibilidad hoy**: 🔵 requiere calendario macro + opciones (Nivel 3). *Nota: ya hay servicio `macroeconomic` (FRED) en `platform/` — base parcial.*

### crypto.swing.funding_reversion
- **Tesis**: cuando el funding rate de un perpetuo cripto alcanza extremos (z-score alto), el precio tiende a revertir, porque el posicionamiento apalancado unilateral se ve forzado a desapalancarse.
- **Mecanismo**: funding extremo = longs (o shorts) apalancados sobre-extendidos pagando carry; el coste fuerza el desapalancamiento → reversión del precio.
- **Clase / Estilo / Horizonte**: CRYPTO perps / Swing / horas–días.
- **Lógica entrada/salida**: entrar contra el lado del funding extremo (z > umbral); salir al normalizarse el funding o a target/stop por ATR.
- **Features**: `funding_z` (**¡ya reservado en el contrato del env!**), OI, basis spot-perp. → 🔵 `funding_z` existe como placeholder (en 0); falta el dato real.
- **Datos**: funding rate + OI de Binance/Bybit (WS). **Hoy: ❌ pero es el MÁS CERCANO** — ya hay colector Binance en `los_ojos`/`platform`.
- **Modelo**: regla/GBM (mean-reversion clásico); RL opcional.
- **Invalidación**: si la reversión post-funding-extremo no supera fees+funding OOS.
- **Benchmark del gate**: market-neutral → **Sharpe absoluto, NO buy-and-hold de BTC**.
- **Factibilidad hoy**: 🔵 el más alcanzable — conectar el feed de funding que ya colectas.

### stocks.position.earnings_drift (PEAD)
- **Tesis**: tras una sorpresa de earnings, el precio sigue derivando en la dirección de la sorpresa durante días/semanas (Post-Earnings Announcement Drift).
- **Mecanismo**: under-reaction conductual + difusión lenta de información entre inversores; anomalía documentada desde Ball & Brown (1968).
- **Clase / Estilo / Horizonte**: STOCKS / Position / días–semanas.
- **Lógica**: largo (corto) tras sorpresa positiva (negativa) de earnings; mantener N días; neutralizar por sector/beta.
- **Features**: sorpresa de earnings (actual vs estimado), revisiones de analistas, sector. → 🔵 no existen.
- **Datos**: fundamentals point-in-time + calendario de earnings con estimados. **Hoy: ❌** (necesita PIT financials).
- **Modelo**: GBM (clasif./regresión de drift); cross-sectional ranking. RL no.
- **Invalidación**: si el drift no supera costes + se desvanece OOS (la anomalía ha decaído mucho post-2000s — **vigilar alpha decay**).
- **Benchmark del gate**: market-neutral (neutralizado por sector/beta) → Sharpe absoluto.
- **Factibilidad hoy**: 🔵 requiere fundamentals PIT (Nivel 3).

---

## Matriz a poblar (Fable 5 rellena las celdas con el formato de la plantilla)

| Estilo ↓ \ Clase → | CRYPTO | STOCKS | FX |
|---|---|---|---|
| **Intraday** | liquidation-cascade scalp; basis micro-arb | ORB de apertura US; gap fill; VWAP reversion | london_open_breakout ✅(ejemplo); session momentum |
| **Swing** | funding_reversion ✅(ejemplo); OI divergence | sector rotation; mean-reversion sobreventa | carry intradía; trend por sesión |
| **Position** | trend on-chain (MVRV/SOPR); halving cycle | earnings_drift ✅(ejemplo); momentum 12-1; value/quality factor | carry trade (rate differential); trend macro |
| **Arbitraje** | cross-exchange; spot-perp basis; triangular | pairs/stat-arb cointegrado; ETF-NAV | triangular FX; covered-interest parity dev |

> Cada celda puede tener **varias** hipótesis. Fable 5 desarrolla cada una con la
> plantilla completa, marcando honestamente la **factibilidad con el dato actual**.

---

## Prompt para Fable 5

> Puebla `CATALOGO_ALPHA_HIPOTESIS.md` siguiendo **exactamente** la plantilla y el nivel
> de los 4 ejemplos desarrollados (London breakout, news magnitude, funding reversion,
> earnings drift). Para cada celda de la matriz (estilo × clase), desarrolla las
> hipótesis listadas y añade las que conozcas del estado del arte. Reglas no negociables:
>
> 1. **Cada hipótesis debe ser falsable** — incluye siempre el *criterio de invalidación*
>    (qué resultado la mata). Una idea sin criterio de falsación no entra.
> 2. **El benchmark del gate depende del estilo**: direccional → buy-and-hold;
>    market-neutral / event-driven / arbitraje → retorno cero / Sharpe absoluto. Nunca
>    compares un arbitraje contra buy-and-hold.
> 3. **Empareja el modelo a la SNR** (CLAUDE.md §6.3, diagnóstico F.4): regla/GBM por
>    defecto; RL solo para control secuencial; DL solo con datos que lo justifiquen. NO
>    propongas TFT/GNN para barras diarias.
> 4. **Marca la factibilidad honestamente**: la mayoría serán 🔵 (requieren datos que
>    hoy no existen — intradía/tick, LOB, funding, fundamentals PIT). Identifica
>    explícitamente las 3-5 hipótesis **más alcanzables con el dato actual o más cercano**
>    (el feed de funding crypto que ya se colecta es la frontera).
> 5. **No escribas código** — esto es el backlog de research. La implementación de
>    cualquier hipótesis espera a tener su dato y a que su agente pase el gate (ADR-040).
> 6. Vincula cada hipótesis a un `alpha_hypothesis_id` con el formato `{clase}.{estilo}.{nombre}`
>    (es el campo del contrato `Signal`, ADR-042 §3).
>
> Entrega: el catálogo poblado + una **tabla de priorización** al final ordenando todas
> las hipótesis por factibilidad-con-dato-actual × fuerza-esperada-del-edge.

---

## Nota de disciplina (no borrar)

Este catálogo es un **mapa del territorio**, no un plan de construcción. Tener 40
hipótesis hermosas no acerca el Nivel 1. La secuencia sigue siendo: **(1)** cerrar el
agente actual (gate deflactado > buy-and-hold), **(2)** construir el `portfolio/` core,
**(3)** recién entonces ir poblando `alpha/` con las hipótesis MÁS alcanzables de este
catálogo, una por una, cada una con su gate. Ir al revés es el error de escalar
prematuro que el diagnóstico marca como riesgo #1.
