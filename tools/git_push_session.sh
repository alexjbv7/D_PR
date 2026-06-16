#!/usr/bin/env bash
# git_push_session.sh
# Commitea y pushea TODO el trabajo de la sesión 2026-06-03
# Corre desde la raíz del repo: bash tools/git_push_session.sh

set -e
cd "$(git rev-parse --show-toplevel)"

echo "=== Configurando line endings (.gitattributes) ==="
git add .gitattributes

echo "=== Cursor agents.md ==="
git add .cursor/agents.md

echo "=== P1-001: Circuit breaker ==="
git add platform/services/execution-engine/app/brokers/_alpaca/circuit_breaker.py
git add platform/services/execution-engine/app/brokers/alpaca.py
git add platform/services/execution-engine/tests/test_circuit_breaker.py

echo "=== P1-002: RiskGate kill switch ==="
git add platform/services/execution-engine/app/risk_gate.py
git add platform/services/execution-engine/app/main.py
git add platform/services/execution-engine/tests/test_risk_gate.py

echo "=== Alert rules Prometheus ==="
git add platform/monitoring/rules/
git add platform/monitoring/prometheus.yml

echo "=== S11: Nightly retrain DAG ==="
git add research/pipelines/
git add research/cli/run_nightly_retrain.py
git add research/tests/test_nightly_retrain.py
git add research/pyproject.toml

echo "=== S12: Docs (ADRs, runbooks, handoff) ==="
git add docs/adr/034-resmlp-replacement.md
git add docs/adr/035-alpaca-slo-reconciliation.md
git add docs/incidents/2026-05-24-drill-alpaca-outage.md
git add docs/incidents/2026-05-31-drill-reconciliation-drift.md
git add docs/incidents/2026-06-03-redrill-p1-fixes.md
git add docs/incidents/2026-06-03-s12-handoff.md
git add docs/runbooks/

echo "=== CLAUDE.md actualizado ==="
git add CLAUDE.md

echo "=== Staged files: ==="
git diff --cached --name-only

echo ""
echo "=== Commit ==="
git commit -m "feat: S10-S12 complete — circuit breaker, RiskGate kill switch, nightly retrain DAG, S12 hardening

P1-001: AlpacaCircuitBreaker CLOSED→OPEN→HALF_OPEN (circuit_breaker.py)
P1-002: RiskGate.trip_kill_switch() as step-0 check, propagated to REST + Kafka
DRILL-004: 21/21 checks PASS — both P1 gaps closed

S10: ALERT-004/005/006/007/008 in platform/monitoring/rules/alpaca.yml
     rule_files enabled in prometheus.yml

S11: NightlyRetrainDAG (research/pipelines/nightly_retrain.py)
     Gates: DSR≥0.4, ECE≤0.05, no collapse, DSR_new≥DSR_prod×0.95
     CLI: research/cli/run_nightly_retrain.py
     17/17 tests passing, dry-run exits 0, JSON run log in artifacts/runs/

S12: ADR-035 (SLO reconciliation), paper_trading_ops runbook,
     s12-handoff checklist, CLAUDE.md §18 updated, .gitattributes LF,
     .cursor/agents.md for Background Agents setup

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"

echo ""
echo "=== Push a GitHub ==="
git push origin main

echo ""
echo "✅ Done. Verifica en: https://github.com/alexjbv7/ML_PR"
