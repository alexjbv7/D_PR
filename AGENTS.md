# AGENTS.md

## Cursor Cloud specific instructions

### Repository Structure

This is a monorepo with three layers: `shared/`, `research/`, and `platform/`. See `CLAUDE.md` for full architecture documentation and `README.md` for quick-start commands.

### Running Services

| Service | How to start | Health check |
|---------|--------------|--------------|
| Platform infra (Kafka, Redis, Postgres, MongoDB) | `cd platform && sudo docker compose up -d zookeeper kafka redis postgres mongodb` | `sudo docker compose ps` |
| Platform microservices | `cd platform && sudo docker compose up -d <service>` (build first with `docker compose build <service>`) | `curl http://localhost:<port>/health` |
| Frontend (dev mode) | `cd platform/frontend && npx vite --host 0.0.0.0 --port 5173` | `curl http://localhost:5173/` |
| Research pipeline | `cd research && python3 examples/pipeline_ml_real_data.py` (needs `PYTHONPATH=/workspace`) | N/A |

Service ports: market-intelligence:8001, macroeconomic:8002, onchain-analysis:8003, context-engine:8004, realtime-signal:8005, ml-feature-store:8006, strategy-orchestrator:8007, sec-research:8008, openbb-adapter:8009, execution-engine:8010, frontend:5173 (dev) or 3000 (Docker).

### Key Gotchas

1. **Docker required for platform**: Docker must be running with `fuse-overlayfs` storage driver and `iptables-legacy`. Start dockerd with `sudo dockerd &` if not running.
2. **TimescaleDB auto-tuner may crash on low-memory VMs**: The postgres container's `timescaledb-tune` script panics when memory is very low. If postgres fails to start, remove the volume (`sudo docker compose rm -f postgres && sudo docker volume rm platform_postgres-data`) and re-start; the schema must then be applied manually: `sudo docker exec -i platform-postgres-1 psql -U trading -d trading_db < platform/infra/sql/schema.sql`.
3. **Platform .env file required**: Services reference `env_file: .env` in docker-compose.yml. Create `platform/.env` with at minimum `KAFKA_BOOTSTRAP_SERVERS`, `REDIS_URL`, `POSTGRES_DSN`. API keys (FRED, Alpaca, CCXT) are optional for development.
4. **`data.real_data` import path**: The research examples (`pipeline_ml_real_data.py`) import `from data.real_data import ...` but `/workspace/data/` lacks `__init__.py`. Running from the `research/` dir alone won't resolve this. Use `PYTHONPATH=/workspace` or generate synthetic data to test the ML pipeline.
5. **PATH for pip-installed tools**: User-installed binaries go to `~/.local/bin`. Export `PATH="$HOME/.local/bin:$PATH"` before using `pytest`, `ruff`, etc.

### Testing

- `make test-shared` — 80 tests (shared library parity)
- `make test-research` — 516+ tests (ML pipeline); exclude `test_alpaca_bars.py`, `test_alpaca_universe.py`, `test_real_data.py` which require network/data deps
- `make test-platform` — 288+ tests (per-service; `macroeconomic` has 7 pre-existing test failures)
- Frontend lint: `cd platform/frontend && npm run lint`

### Lint

- Python: `ruff check shared/ research/ platform/` (existing lint warnings in shared)
- Frontend: `cd platform/frontend && npm run lint`
