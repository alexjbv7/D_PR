# Los Ojos — memoria de proyecto (pointer)

Este repositorio (**los_ojos**) es la **Fase 2** de PROJECT ML: microservicios de inteligencia financiera, Kafka, Redis, TimescaleDB y dashboard. La documentación de convenciones y el núcleo ML viven en el repo hermano **`quant_bot`** (guión bajo, sin espacio), carpeta típica:

`C:\Users\alexj\OneDrive\Desktop\quant_bot\`

Ahí tienes al menos:

| Archivo | Rol |
|---------|-----|
| **`CLAUDE.md`** | Manual técnico interno: arquitectura objetivo, estrategias, riesgo, roadmap, reglas para agentes — **fuente de verdad** para PROJECT ML. |
| **`README.md`** | Entrada del repo: setup rápido, visión general, enlaces útiles. |

**Enlaces locales** (desde `los_ojos`, hermanos en el mismo Desktop):

- [`../quant_bot/CLAUDE.md`](../quant_bot/CLAUDE.md)
- [`../quant_bot/README.md`](../quant_bot/README.md)

## Qué leer en `CLAUDE.md`

| Sección | Contenido relevante para Los Ojos |
|---------|-----------------------------------|
| §5 | Capa “Los Ojos”: servicios, puertos, stack |
| §15.1 | Estructura de **ambos** repos (`quant_bot/` vs `los_ojos/`) |
| §10 | Arquitectura de eventos Kafka (**convención target** `raw.*` / `features.*`; Los Ojos usa `los_ojos.*` en Makefile — alinear con código) |
| §11–12 | Motor de ejecución y risk (**target**; executor/risk-engine aún no equivalentes en este repo) |
| §18 | Roadmap (checkboxes Fase 2–3) |

## Núcleo ML (research)

Walk-forward, model zoo, calibración, meta-labeling, Bayesian sizing, etc. están en **quant_bot** (`models/`, `risk/`, `features/`). Los Ojos consume vectores de features y emite señales; la inferencia “institucional” descrita en `CLAUDE.md` debe **integrarse** (servicio `ml-inference` / artefactos desde quant_bot) según roadmap §18.3.

## Convención para agentes

1. **`quant_bot/README.md`** primero si necesitas contexto breve del monolito ML.  
2. **`quant_bot/CLAUDE.md`** para arquitectura, estrategias, riesgo y roadmap en detalle.  
3. Código y **`infra/sql/schema.sql`** en **los_ojos** para el estado real del stack desplegado aquí.  
4. Si hay divergencia doc ↔ código, priorizar el código y anotar el gap en un PR o ADR en **quant_bot**.

---

*Última actualización: alinear con `CLAUDE.md` del mismo día que editaste PROJECT ML.*
