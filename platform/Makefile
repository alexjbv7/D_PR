# =============================================================================
# Los Ojos — Makefile
# =============================================================================

.PHONY: help up down build logs ps clean frontend-dev \
        infra services monitoring frontend \
        db-shell db-migrate redis-cli kafka-topics kafka-create-topics \
        env test lint k8s-apply k8s-diff sec-research

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Docker Compose ───────────────────────────────────────────────────────────

up:  ## Start all services
	docker-compose up -d

down:  ## Stop all services
	docker-compose down

build:  ## Build all images
	docker-compose build --parallel

restart:  ## Restart all services
	docker-compose down && docker-compose up -d

logs:  ## Tail logs from all services
	docker-compose logs -f

ps:  ## Show service status
	docker-compose ps

# ── Selective service starts ──────────────────────────────────────────────────

infra:  ## Start infrastructure only (kafka, redis, postgres, mongodb)
	docker-compose up -d zookeeper kafka redis postgres mongodb

services:  ## Start microservices (requires infra running)
	docker-compose up -d market-intelligence macroeconomic onchain-analysis context-engine \
	                      realtime-signal ml-feature-store strategy-orchestrator sec-research

monitoring:  ## Start monitoring (prometheus + grafana)
	docker-compose up -d prometheus grafana

frontend:  ## Start frontend
	docker-compose up -d frontend

# ── Development ──────────────────────────────────────────────────────────────

frontend-dev:  ## Start frontend in dev mode (hot reload)
	cd frontend && npm run dev

# ── Cleanup ──────────────────────────────────────────────────────────────────

clean:  ## Remove containers, volumes, images
	docker-compose down -v --rmi local

clean-volumes:  ## Remove data volumes only
	docker-compose down -v

# ── Database ─────────────────────────────────────────────────────────────────

db-shell:  ## Open psql shell
	docker-compose exec postgres psql -U trading -d trading_db

ops-test:  ## Run tools/ briefing + smoke unit tests (from repo root)
	cd .. && python -m pytest tools/briefing/tests tools/smoke/test_post_deploy_smoke.py -q

ops-smoke:  ## Post-deploy smoke (requires make up + db-migrate)
	cd .. && python -m pytest tools/smoke/test_post_deploy_smoke.py -xvs

db-migrate:  ## Apply SQL migrations (platform/infra/sql/migrations/*.sql in order)
	@echo "Applying migrations from infra/sql/migrations/ ..."
	@for f in $$(ls infra/sql/migrations/*.sql 2>/dev/null | sort); do \
	  echo "  -> $$f"; \
	  docker-compose exec -T postgres psql -U trading -d trading_db < $$f || exit 1; \
	done
	@echo "Migrations applied."

redis-cli:  ## Open redis-cli
	docker-compose exec redis redis-cli

# ── Kafka ────────────────────────────────────────────────────────────────────

kafka-topics:  ## List Kafka topics
	docker-compose exec kafka kafka-topics --list --bootstrap-server localhost:9092

kafka-create-topics:  ## Create all topics from infra/kafka/topics.yml
	@echo "Creating Kafka topics..."
	@for topic in los_ojos.market.data los_ojos.market.orderbook los_ojos.market.funding \
	              los_ojos.market.normalized \
	              los_ojos.macro.indicators los_ojos.macro.regime los_ojos.macro.recession_alert \
	              los_ojos.macro.signal \
	              los_ojos.onchain.whale_alert los_ojos.onchain.smart_money \
	              los_ojos.context.regime los_ojos.context.anomaly los_ojos.context.state \
	              los_ojos.ml.feature_vector los_ojos.features.vector \
	              los_ojos.signals.trading los_ojos.bot.kill_switch \
	              los_ojos.sec.signal los_ojos.dlq; do \
	  docker-compose exec kafka kafka-topics --create --if-not-exists \
	    --bootstrap-server localhost:9092 \
	    --partitions 4 --replication-factor 1 --topic $$topic; \
	done
	@echo "Done."

# ── Testing ──────────────────────────────────────────────────────────────────

test:  ## Run all Python tests (per-service isolation)
	@echo "Running tests for all services..."
	@for svc in market-intelligence macroeconomic onchain-analysis context-engine \
	            ml-feature-store realtime-signal strategy-orchestrator sec-research; do \
	  if [ -d "services/$$svc/tests" ] && [ -n "$$(ls services/$$svc/tests/test_*.py 2>/dev/null)" ]; then \
	    echo "\n[$$svc]"; \
	    cd services/$$svc && python -m pytest tests/ -q --tb=short || exit 1; cd ../..; \
	  fi; \
	done
	@echo "\nAll tests passed."

lint:  ## Lint all Python services with ruff
	@pip install -q ruff
	@for svc in market-intelligence macroeconomic onchain-analysis context-engine \
	            ml-feature-store realtime-signal strategy-orchestrator sec-research; do \
	  ruff check services/$$svc/app; \
	done
	@echo "All services linted."

# ── Kubernetes ───────────────────────────────────────────────────────────────

k8s-apply:  ## Apply k8s base manifests
	kubectl apply -k infra/k8s/base

k8s-apply-prod:  ## Apply k8s production overlay
	kubectl apply -k infra/k8s/overlays/production

k8s-diff:  ## Diff k8s base against cluster
	kubectl diff -k infra/k8s/base

k8s-delete:  ## Delete all k8s resources (DESTRUCTIVE)
	kubectl delete -k infra/k8s/base

# ── Env ──────────────────────────────────────────────────────────────────────

env:  ## Copy .env.example to .env
	@if [ ! -f .env ]; then cp .env.example .env && echo ".env created"; else echo ".env already exists"; fi
