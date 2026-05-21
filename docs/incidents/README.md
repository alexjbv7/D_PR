# Incident Logbook — paper trading run

Cada incident P0/P1 durante el run de 30 días se documenta como
`docs/incidents/YYYY-MM-DD-<short-slug>.md` siguiendo
`tools/incidents/template.md`.

## Cuándo crear incident

- ALERT-002 (Alpaca circuit open) cualquier duración.
- ALERT-004 (reconciler discrepancies P0).
- ALERT-001 (daily DD kill).
- Cualquier excepción no manejada que pause el sistema > 5 min.

## Cuándo NO crear incident

- Alerts P2 transitorios que se autorresuelven.
- Restart programado de servicios (documentar en CHANGELOG).

## Plantilla

Copiar `tools/incidents/template.md` o `docs/incidents/template.md` y rellenar
todas las secciones antes de cerrar el incident.
