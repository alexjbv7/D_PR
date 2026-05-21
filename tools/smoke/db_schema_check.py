"""Postgres schema/table existence checks."""
from __future__ import annotations

from typing import Any


def table_exists(dsn: str, schema: str, table: str) -> bool:
    """Return True when ``schema.table`` exists."""
    import psycopg2  # type: ignore[import-untyped]

    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = %s AND table_name = %s
                """,
                (schema, table),
            )
            return cur.fetchone() is not None


def latest_migration_prefix(migrations_dir: str) -> str | None:
    """Return numeric prefix of the highest migration file (e.g. ``009``)."""
    from pathlib import Path

    files = sorted(Path(migrations_dir).glob("*.sql"))
    if not files:
        return None
    return files[-1].stem.split("_")[0]


def migration_artifacts_present(dsn: str, migration_prefix: str) -> bool:
    """Verify tables introduced up to *migration_prefix* (e.g. ``009``) exist."""
    checks: dict[str, list[tuple[str, str]]] = {
        "004": [
            ("data", "universe_historical"),
            ("market", "corporate_actions"),
        ],
        "006": [("risk", "position_actions")],
        "008": [("drift", "psi_history"), ("audit", "predictions")],
        "009": [("risk", "allocator_state"), ("risk", "allocator_updates")],
    }
    prefix_int = int(migration_prefix)
    for mig_num, tables in checks.items():
        if int(mig_num) > prefix_int:
            continue
        for schema, table in tables:
            if not table_exists(dsn, schema, table):
                return False
    return True


def applied_migration_version(dsn: str) -> str | None:
    """Read MAX(version) from schema_migrations, or None if table missing."""
    import psycopg2

    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("SELECT MAX(version) FROM schema_migrations")
                row: tuple[Any, ...] | None = cur.fetchone()
            except Exception:
                return None
    if row is None or row[0] is None:
        return None
    return str(row[0])
