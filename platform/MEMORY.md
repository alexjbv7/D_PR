# Los Ojos — memoria de proyecto (integrado en el monorepo)

**Los Ojos** fue originalmente un repositorio separado (`C:\Users\alexj\OneDrive\Desktop\los_ojos\`),
hermano de `quant_bot`. Ambos fueron fusionados en el monorepo **PROJECT ML** (ver
`CLAUDE.md` raíz, cabecera "MONOREPO"): esta carpeta, **`platform/`**, es hoy
Los Ojos. Ya no existe como repo independiente — no busques rutas relativas a
`../quant_bot/` ni a un checkout separado de `los_ojos/`.

Mapa de las tres capas del monorepo (ver `CLAUDE.md` §0 / §15.1):

| Carpeta | Antes era | Contenido |
|---------|-----------|-----------|
| `research/` | `quant_bot/` | I+D, backtesting, entrenamiento ML |
| `platform/` (aquí) | `los_ojos/` | 8+ microservicios FastAPI, frontend React, Kafka, Redis |
| `shared/` | — (nuevo) | Librería `quant_shared`: features canónicos, schemas Kafka, model registry |

## Qué leer

| Archivo | Rol |
|---------|-----|
| `../CLAUDE.md` | Manual técnico interno completo — arquitectura, estrategias, riesgo, roadmap, reglas para agentes. **Fuente de verdad única** para todo el monorepo. |
| `../README.md` | Entrada del repo: setup rápido, visión general. |

Secciones relevantes de `CLAUDE.md` para esta carpeta:

| Sección | Contenido relevante para `platform/` (Los Ojos) |
|---------|-----------------------------------|
| §5 | Capa "Los Ojos": servicios, puertos, stack |
| §15.1 | Estructura del monorepo (`research/` / `platform/` / `shared/`) |
| §10 | Arquitectura de eventos Kafka (Los Ojos usa prefijo `los_ojos.*` en topics; convención target del monorepo es `<domain>.<entity>.<action>` — ver migración de schemas abajo) |
| §11–12 | Motor de ejecución y risk (`execution-engine` en esta carpeta ya implementa buena parte de esto; ver `services/execution-engine/`) |
| §18 | Roadmap (checkboxes Fase 2–3) |

## Núcleo ML (research)

Walk-forward, model zoo, calibración, meta-labeling, Bayesian sizing, etc.
viven en **`../research/`** (`models/`, `risk/`, `features/`). `platform/`
consume vectores de features y emite señales; la inferencia "institucional"
descrita en `CLAUDE.md` se integra vía artefactos de `research/` cargados por
servicios como `ml-feature-store` y `execution-engine`, según roadmap §18.3.

## Migración de schemas en curso

Los eventos Kafka se están moviendo de `platform/libs/shared/events.py` hacia
`shared/quant_shared/schemas/events.py` (fuente de verdad nueva del monorepo).
`platform/libs/shared/events.py` re-exporta desde ahí por compatibilidad
temporal. Los servicios deben migrar gradualmente sus imports:

```python
# antes
from libs.shared.events import MarketDataEvent
# después
from quant_shared.schemas.events import MarketDataEvent
```

## Servicios en esta carpeta (`platform/services/`)

Incluye los 7 servicios originales documentados en `CLAUDE.md` §5.0
(`market-intelligence`, `macroeconomic`, `onchain-analysis`, `context-engine`,
`realtime-signal`, `ml-feature-store`, `strategy-orchestrator`) más servicios
añadidos después de esa documentación: `execution-engine`, `sec-research`,
`openbb-adapter`. Si `CLAUDE.md` §5.0 no los lista, el código manda — actualizar
esa tabla o abrir un ADR si el gap es arquitectónicamente relevante.

## Convención para agentes

1. **`../README.md`** primero si necesitas contexto breve del monorepo.
2. **`../CLAUDE.md`** para arquitectura, estrategias, riesgo y roadmap en detalle.
3. Código y `infra/sql/` / `infra/kafka/topics.yml` en esta carpeta para el
   estado real del stack desplegado.
4. Si hay divergencia doc ↔ código, priorizar el código y anotar el gap en un
   PR o ADR en `docs/adr/`.

---

*Última actualización: 2026-07-08 — corregido para reflejar la fusión a
monorepo (ya no es un repo independiente; ver cabecera de `CLAUDE.md`).*
