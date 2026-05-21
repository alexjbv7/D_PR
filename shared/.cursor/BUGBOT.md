# Bugbot — shared (`quant_shared`)

## Schemas (Pydantic v2)

- Eventos Kafka canónicos: `quant_shared/schemas/events.py`.
- Órdenes dominio: `quant_shared/schemas/orders.py` — `Decimal`, enums, `_uuid7()`.
- No romper compatibilidad de campos sin bump semver de feature set o nota de migración.

## Cambios breaking

- Renombrar topic o campo en eventos → requiere actualizar shim
  `platform/libs/shared/events.py` y todos los productores/consumidores en el mismo PR.

## Features canónicos

- Definiciones en `quant_shared/features/definitions.py`; validadores en `validators.py`.
- Nuevas features: rango declarado, política NaN, online/offline flags.

## Model registry

- Metadata de modelos (PSR, DSR, ECE) no sustituye gates de promoción en código.

## Tests

- Roundtrip serialización JSON para schemas tocados.
- UUID v7: verificar orden temporal si se comparan IDs en tests.
