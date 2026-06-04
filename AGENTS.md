# PROJECT ML — Agent instructions

Canonical architecture and conventions: [`CLAUDE.md`](CLAUDE.md). Quick start: [`README.md`](README.md).

## Cursor Cloud specific instructions

### VM bootstrap (already applied once)

- **Python**: use `python3` (there is no `python` on PATH unless you `alias python=python3` in `~/.bashrc`).
- **User tools**: add `export PATH="$HOME/.local/bin:$PATH"` in `~/.bashrc` so `pytest`, `ruff`, and `uvicorn` resolve after `pip install --user`.
- **Docker**: Docker CE is installed; if you get permission errors on the socket, `sudo chmod 666 /var/run/docker.sock` or add the user to the `docker` group. Prefer `docker compose` (plugin) — root `platform/Makefile` still calls `docker-compose` in some targets; run equivalent `docker compose` commands from `platform/` if `make kafka-create-topics` fails.
- **Shared tests**: install `pytest-benchmark` or `shared` tests will error on `test_market_calendar.py::test_is_open_benchmark_warm_cache`.

### Dependency install (see also update script)

From repo root:

```bash
pip install -e shared/
cd research && pip install -e ".[dev,data]"
pip install pytest-benchmark ruff
```

Platform **unit tests** (no Docker): per service, `pip install -e ./shared/ -e ./platform/libs/shared/` then `pip install -r platform/services/<svc>/requirements.txt`, then `cd platform/services/<svc> && python3 -m pytest tests/ -q`.

Frontend: `cd platform/frontend && npm install` (no `package-lock.json` in tree — use `npm install`, not `npm ci`).

### Running the stack

Full commands: `platform/Makefile` and root `Makefile` (`make help`, `cd platform && make help`).

**Infra only** (from `platform/`):

```bash
docker compose up -d zookeeper kafka redis postgres mongodb
```

**Postgres on Cloud VMs**: `timescale/timescaledb:latest-pg16` init can fail when cgroup memory files are missing (`timescaledb-tune` panic). If `platform-postgres-1` exits during init, services that need Postgres will not start; Kafka/Redis-only services (`context-engine`, `onchain-analysis`, `realtime-signal` run locally) still work. On a normal Linux host with cgroups, `make up` + `make db-migrate` is the intended path.

**Minimal platform demo without Postgres**:

```bash
cd platform
cp /dev/null .env   # or copy from repo root .env.example for Alpaca keys
docker compose up -d zookeeper kafka redis mongodb
docker compose build context-engine onchain-analysis
docker compose up -d context-engine onchain-analysis
# realtime-signal (host process example):
cd services/realtime-signal
KAFKA_BOOTSTRAP_SERVERS=localhost:9092 REDIS_URL=redis://localhost:6379/0 \
  PYTHONPATH=$PWD:../../.. python3 -m uvicorn app.main:app --port 8005
```

**Frontend dev**: `cd platform/frontend && npm run dev -- --host 0.0.0.0 --port 5173` → http://localhost:5173

### Lint / test (CI parity)

| Layer | Lint | Test |
|-------|------|------|
| `shared/` | `ruff check shared/quant_shared` | `cd shared && python3 -m pytest tests/ -q` |
| `research/` | `ruff check research/models research/features research/risk` | `cd research && python3 -m pytest tests/ -q` |
| `platform/` | `cd platform && make lint` (needs `python` alias or edit to `python3`) | Per-service pytest under `platform/services/*/tests` |
| `frontend/` | `npm run lint` | `npm run typecheck` / `npm run build` |

Root `make test` / `make lint` call `python` and may fail until `python` is on PATH.

### Research “hello world” (no Docker)

```bash
cd research
python3 -c "from models.rl_agent import QLearningAgent, QLearningConfig; print('ok')"
# Train on synthetic data — see README § Arranque rápido
```

Verify shared contract: `python3 -c "from quant_shared.features import FEATURE_COUNT; assert FEATURE_COUNT == 19"`.

### Health checks (when services are up)

- Realtime API/WS: http://localhost:8005/health
- Context engine: http://localhost:8004/health
- Strategy orchestrator: http://localhost:8007/health (needs Postgres + `make up`)

Post-deploy smoke (stack + migrations): `cd platform && make ops-smoke` from repo root after `pip install -e tools/`.

### Secrets

`platform/.env` is required by Compose for app services (`env_file: .env`). Use repo root [`.env.example`](.env.example) for Alpaca/Discord placeholders. Do not commit `.env`.
