# Bugbot — capa platform (microservicios, Kafka, frontend)

## Servicios y responsabilidad única

- Un servicio = un motivo de cambio (`CLAUDE.md` §2.2). No mezclar lógica ML pesada
  dentro de `strategy-orchestrator` sin servicio `ml-inference` dedicado.
- FastAPI: lifespan para conexiones; cancelar background tasks en shutdown.

## execution-engine (Alpaca / CCXT)

- Interfaz canónica: `BrokerAdapter` en `app/brokers/base.py` (no duplicar ABC).
- `AlpacaAdapter`: llamadas SDK sync vía `asyncio.to_thread`; no bloquear el event loop.
- `OrderIntent` / `OrderResult` desde `quant_shared.schemas.orders` — Decimal, UUID v7.
- Symbol mapping: usar `_symbol_mapping` / `_symbol_mapping` modules; no strings
  Alpaca hardcodeados en `service.py`.
- Risk gate debe ejecutarse **antes** de `submit`; rechazos auditables.
- Reconciler: discrepancia broker vs interno → no silenciar; preparar freeze de nuevas órdenes.

## strategy-orchestrator

- Kill-switch: preferir persistencia (Redis/DB), no solo flag global en memoria.
- `_pnl_cache` / drawdown gate: si el PR toca gates, debe haber fuente de verdad
  de fills (o documentar limitación explícita).
- Señales Kafka: validar freshness de `ts` en feature vectors (>60s stale → skip).

## Redis / claves conocidas

- Flaggear desalineación de keys entre productores y consumidores
  (p. ej. whale sentiment: writer vs reader distintos).
- TTL documentado en writes `setex`.

## Frontend (`platform/frontend/`)

- No exponer secrets en bundle; env vars con prefijo `VITE_` solo para no-secretos.
- Tipos alineados con `shared` / API contracts.

## Tests

- Todo cambio en `app/*.py` de un servicio debe tocar `tests/` del mismo servicio.
- Integration tests con Postgres: marcar skip si docker no disponible, no fallar CI opaco.

## Docker / compose

- No commitear `.env` con valores reales; solo `.env.example`.
