#!/usr/bin/env bash
# .cursor/setup.sh
# Development environment setup for Cursor Cloud agents.
# Runs automatically before any background agent starts working.
# ---------------------------------------------------------------

set -euo pipefail

echo "========================================"
echo " quant_bot — Cursor Cloud Environment Setup"
echo "========================================"

# ── 1. Python version check ──────────────────────────────────────
echo ""
echo "▶ Checking Python version..."
python3 --version
PYTHON_MIN="3.11"
PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if python3 -c "import sys; exit(0 if sys.version_info >= (3,11) else 1)"; then
    echo "  ✅ Python $PYTHON_VER OK"
else
    echo "  ❌ Python $PYTHON_VER — requires >= $PYTHON_MIN"
    exit 1
fi

# ── 2. Shared library (siempre primero — es dependencia de todo) ──
echo ""
echo "▶ Installing quant-shared..."
pip install -e shared/ --quiet
echo "  ✅ quant-shared installed"

# ── 3. Research package (ML, DRL, backtesting) ───────────────────
echo ""
echo "▶ Installing quant-research [dev]..."
cd research
pip install -e ".[dev]" --quiet
cd ..
echo "  ✅ quant-research installed"

# ── 4. Tools package (briefings, smoke tests) ────────────────────
echo ""
echo "▶ Installing tools..."
if [ -f "tools/pyproject.toml" ]; then
    pip install -e tools/ --quiet
    echo "  ✅ tools installed"
fi

# ── 5. DRL dependencies (gymnasium + stable-baselines3) ──────────
echo ""
echo "▶ Installing DRL dependencies..."
pip install gymnasium stable-baselines3 --quiet
echo "  ✅ DRL deps installed"

# ── 6. Verify critical imports ───────────────────────────────────
echo ""
echo "▶ Verifying imports..."

python3 -c "from quant_shared.schemas.orders import OrderIntent; print('  ✅ quant_shared')"
python3 -c "
import sys
sys.path.insert(0, 'research')
sys.path.insert(0, 'shared')
from models.walk_forward_runner import WalkForwardRunner
print('  ✅ WalkForwardRunner')
"
python3 -c "
import sys
sys.path.insert(0, 'research')
sys.path.insert(0, 'shared')
from pipelines.nightly_retrain import NightlyRetrainDAG
print('  ✅ NightlyRetrainDAG')
"

# DRL (solo si el agente ya mergeó PR #3)
python3 -c "
import sys
sys.path.insert(0, 'research')
sys.path.insert(0, 'shared')
try:
    from models.drl import TradingDQN, TradingResMLP
    print('  ✅ DRL backbone + DQN')
except ImportError:
    print('  ⚠️  DRL not yet merged — OK if working on PR #3')
" 2>/dev/null || true

python3 -c "
import sys
sys.path.insert(0, 'research')
sys.path.insert(0, 'shared')
try:
    from envs.trading_env import TradingEnvironment
    print('  ✅ TradingEnvironment')
except ImportError:
    print('  ⚠️  TradingEnvironment not yet merged — OK if working on PR #3')
" 2>/dev/null || true

# ── 7. Quick smoke test: dry-run del DAG nocturno ────────────────
echo ""
echo "▶ Smoke test: nightly retrain dry-run..."
cd research
python3 -m cli.run_nightly_retrain --dry-run 2>&1 | tail -3
EXIT_CODE=$?
cd ..
if [ $EXIT_CODE -eq 0 ]; then
    echo "  ✅ Dry-run OK (exit 0)"
else
    echo "  ❌ Dry-run failed (exit $EXIT_CODE)"
    exit 1
fi

# ── 8. Summary ───────────────────────────────────────────────────
echo ""
echo "========================================"
echo " Setup complete ✅"
echo " Python: $(python3 --version)"
echo " Packages: quant-shared, quant-research[dev], DRL deps"
echo " Smoke: nightly_retrain --dry-run OK"
echo ""
echo " Archivos clave:"
echo "   research/models/drl/      ← DRL backbone (PR #3)"
echo "   research/envs/            ← TradingEnvironment (PR #3)"
echo "   research/pipelines/       ← NightlyRetrainDAG (S11)"
echo "   platform/services/        ← Execution engine"
echo ""
echo " Reglas (CLAUDE.md §20):"
echo "   - Lee antes de escribir"
echo "   - Walk-forward o nada (no métricas IS)"
echo "   - UTC + Decimal + UUID v7"
echo "   - No commitear secrets"
echo "========================================"
