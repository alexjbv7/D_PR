# =============================================================================
# quant_bot — Monorepo Makefile
# =============================================================================
# Capas:
#   shared/    → librería quant_shared (features canónicos, schemas, registry)
#   research/  → I+D, backtesting, entrenamiento ML
#   platform/  → 8 microservicios FastAPI + frontend React + Kafka + Redis
# =============================================================================

.PHONY: help install install-shared install-research \
        test test-shared test-research test-platform \
        lint lint-shared lint-research lint-platform \
        up down build infra services \
        clean docs

help:  ## Muestra este help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ── Instalación ──────────────────────────────────────────────────────────────

install-shared:  ## Instala quant_shared en modo editable
	pip install -e shared/

install-research: install-shared  ## Instala research/ (I+D) en modo editable
	pip install -e research/

install: install-research  ## Instala todo el stack Python (shared + research)
	@echo "Stack Python instalado."

# ── Tests ────────────────────────────────────────────────────────────────────

test-shared:  ## Tests de paridad en shared/ (36 tests, guardianes del monorepo)
	cd shared && python -m pytest tests/ -q --tb=short

test-research:  ## Tests de la capa de I+D en research/
	cd research && python -m pytest tests/ -q --tb=short

test-platform:  ## Tests de todos los microservicios en platform/ (196 tests)
	@echo "Corriendo tests de todos los servicios..."
	@for svc in market-intelligence macroeconomic onchain-analysis context-engine \
	            ml-feature-store realtime-signal strategy-orchestrator sec-research; do \
	  if [ -d "platform/services/$$svc/tests" ]; then \
	    echo "[$$svc]"; \
	    cd platform/services/$$svc && python -m pytest tests/ -q --tb=short || exit 1; \
	    cd ../../..; \
	  fi; \
	done

test:  ## Corre TODO: shared + research + platform (238+ tests)
	@$(MAKE) test-shared
	@$(MAKE) test-research
	@$(MAKE) test-platform
	@echo "\nAll tests passed."

# ── Lint ─────────────────────────────────────────────────────────────────────

lint-shared:  ## Ruff en shared/
	ruff check shared/quant_shared

lint-research:  ## Ruff en research/
	ruff check research/models research/features research/risk

lint-platform:  ## Ruff en todos los microservicios
	@for svc in market-intelligence macroeconomic onchain-analysis context-engine \
	            ml-feature-store realtime-signal strategy-orchestrator sec-research; do \
	  ruff check platform/services/$$svc/app; \
	done

lint: lint-shared lint-research lint-platform  ## Lint completo del monorepo

# ── Docker / Platform ────────────────────────────────────────────────────────

up:  ## Levanta el stack completo (docker-compose en platform/)
	cd platform && docker-compose up -d

down:  ## Para el stack
	cd platform && docker-compose down

build:  ## Build todas las imágenes Docker
	cd platform && docker-compose build --parallel

infra:  ## Solo infraestructura (kafka, redis, postgres, mongodb)
	cd platform && docker-compose up -d zookeeper kafka redis postgres mongodb

services:  ## Solo microservicios (requiere infra corriendo)
	cd platform && docker-compose up -d market-intelligence macroeconomic \
	  onchain-analysis context-engine realtime-signal ml-feature-store \
	  strategy-orchestrator sec-research

frontend-dev:  ## Frontend en modo dev (hot reload)
	cd platform/frontend && npm run dev

# ── Kubernetes ───────────────────────────────────────────────────────────────

k8s-apply:  ## Apply k8s base
	kubectl apply -k platform/infra/k8s/base

k8s-apply-prod:  ## Apply k8s producción
	kubectl apply -k platform/infra/k8s/overlays/production

# ── Limpieza ─────────────────────────────────────────────────────────────────

clean:  ## Limpia contenedores, volúmenes y __pycache__
	cd platform && docker-compose down -v --rmi local
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true

# ── Docs ─────────────────────────────────────────────────────────────────────

docs:  ## Abre CLAUDE.md (fuente de verdad del proyecto)
	@cat CLAUDE.md | head -60
