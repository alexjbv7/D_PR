# Bugbot — reglas del monorepo quant_bot (PROJECT ML)

Estas reglas aplican a todo el repositorio. Bugbot las combina con reglas de equipo/repo
en el dashboard de Cursor. Ver: https://cursor.com/docs/bugbot

## Alcance del proyecto

- Monorepo: `research/` (ML, backtest, riesgo), `platform/` (microservicios FastAPI),
  `shared/` (`quant_shared` schemas y features).
- No es HFT: latencia objetivo 50–500 ms en el path crítico (no microsegundos).
- Broker objetivo en ejecución: Alpaca (paper primero) vía `execution-engine`.

## Severidad: bloquear merge si…

1. **Secretos**: API keys, tokens, `.env` reales, credenciales en código o logs.
2. **Dinero en float**: cualquier valor monetario, qty de broker, PnL o precio de orden
   persistido/calculado con `float` en paths de ejecución o schemas compartidos.
   Usar `Decimal` (ADR-010).
3. **IDs de órdenes**: `client_order_id` / `intent_id` debe ser UUID v7 (time-sortable),
   no UUID4 aleatorio ni enteros autoincrement para idempotencia de broker.
4. **Timestamps naive**: `datetime` sin timezone en eventos, órdenes o Kafka payloads.
   Siempre UTC con `timezone.utc`.
5. **Anti-leakage ML**: `fit()`, `calibrate()`, `WalkForwardSplitter`, GMM regime,
   `BayesianWinUpdater.fit`, PCA — solo sobre datos estrictamente anteriores al fold/test.
   Flaggear `fit` sobre el dataset completo antes del split temporal.
6. **Capas invertidas**: imports desde `execution` o `platform` hacia `research/models`
   para lógica de negocio; o `features` importando `risk`/`execution`. Orden válido:
   `features → models → risk → execution`.
7. **Duplicación de núcleo**: nuevo `WalkForwardRunner`, calibrador isotónico paralelo,
   Kelly desde cero, o triple-barrier duplicado en vez de extender
   `research/features/labeling.py` y módulos existentes.
8. **Métricas IS como éxito**: PRs que citan Sharpe/PSR solo in-sample como criterio
   de promoción sin OOS walk-forward / DSR.
9. **Borrado de tests** sin reemplazo equivalente en módulos públicos de `research/` o
   `platform/services/*/tests/`.
10. **Kill-switch / drawdown solo en memoria**: cambios que persisten estado de riesgo
    crítico solo en variables globales de proceso sin Redis/Postgres (regresión conocida).

## Severidad: bug no bloqueante (reportar igual)

- `print()` en servicios de producción (usar structlog/logging).
- `except Exception:` sin re-raise o métrica en paths de trading.
- TODO/FIXME sin owner y fecha (`# TODO(@user YYYY-MM-DD):`).
- Nuevas dependencias en `pyproject.toml` / `requirements.txt` sin uso en el mismo PR.
- Hardcoded `sqrt(8760)` u otros factores de anualización sin documentar timeframe.
- Platform `strategy-orchestrator` que emite `position_size` fijo sin pasar por
  Kelly/ATR cuando el PR toca sizing en vivo.

## Trading / Alpaca (execution-engine)

Si el PR toca `platform/services/execution-engine/`:

- Verificar mapeo de errores Alpaca a excepciones tipadas (`BrokerError` subclasses).
- `extended_hours` explícito cuando la orden pretende pre/post market.
- No asumir short en cuenta cash; no asumir fractional en tipos de orden que Alpaca
  rechaza (p. ej. algunos brackets).
- Reintentos: solo 429 y 5xx, backoff exponencial + jitter, máximo 3 intentos.
- Idempotencia: re-submit con mismo `client_order_id` no debe duplicar exposición.
- Reconciler: cambios en posiciones deben mantener loop ~60s y alertar discrepancias.

## Kafka / eventos

- No duplicar schemas de eventos: canónico en `shared/quant_shared/schemas/`.
  `platform/libs/shared/events.py` debe ser shim de re-export, no segunda fuente.
- Cambios breaking en topics o payloads requieren nota de migración en el PR.

## Tests esperados

| Área | Mínimo |
|------|--------|
| `research/risk/*`, `research/models/*` | pytest unitario |
| `execution-engine` brokers | `test_alpaca_*`, `test_broker_contract` |
| Schemas `quant_shared` | validación pydantic / roundtrip |
| Bugfix en servicio FastAPI | test de regresión o ampliación de test existente |

## Fuera de alcance de Bugbot (no exigir)

- Refactors cosméticos masivos no pedidos en el PR.
- Documentación `*.md` salvo que contradiga invariantes anteriores.
- Performance de notebooks en `research/notebooks/`.

## Comandos útiles en PR (humano)

- `cursor review` o `bugbot run` — revisión manual.
- `cursor review verbose=true` — diagnóstico con request ID.
- `@cursor remember [hecho]` — regla aprendida en dashboard (equipo).
