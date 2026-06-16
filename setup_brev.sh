#!/usr/bin/env bash
# Provision a Brev / NVIDIA GPU VM to train the full quant_bot stack.
# Idempotent: safe to re-run. Run from the repo root: bash setup_brev.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PY="${PYTHON:-python3}"

echo "== GPU =="
nvidia-smi || echo "WARN: no nvidia-smi (CPU-only box?)"
echo "== Python: $($PY --version) =="

# 1) Editable installs using the repo's declared dependencies.
#    shared first, then research with the [data,dev] extras
#    (pulls ccxt/yfinance/alpaca-py + pytest from research/pyproject.toml).
$PY -m pip install -U pip wheel
$PY -m pip install -e ./shared
$PY -m pip install -e "./research[data,dev]"

# 2) Torch: keep the image's CUDA build if present; install a matching one otherwise.
if $PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "== torch CUDA already available: $($PY -c 'import torch;print(torch.__version__, torch.version.cuda)') =="
elif command -v nvidia-smi >/dev/null 2>&1; then
  echo "== installing CUDA torch (cu124) =="
  $PY -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124
else
  echo "== no GPU detected: installing CPU torch =="
  $PY -m pip install --upgrade torch
fi

# 3) Verify torch + GPU visibility
$PY -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),
       'device',(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'))"

# 4) Alpaca keys check (crypto needs them; EUR/USD via yfinance does not)
if [[ -z "${ALPACA_API_KEY:-}" || -z "${ALPACA_API_SECRET:-}" ]]; then
  echo "WARN: ALPACA_API_KEY/ALPACA_API_SECRET not set."
  echo "      BTC/USD & ETH/USD will fail to load; EUR/USD (yfinance) still works."
  echo "      'cp .env.example .env' and fill them, or export them, before the full run."
fi

# 5) Import / wiring sanity (no training)
cd research && $PY -m pipelines.train_all --smoke --dry-run >/dev/null && echo "== import + dry-run OK =="

echo ""
echo "== setup complete =="
echo "Smoke (tiny, validates end-to-end training):"
echo "  cd research && python -m pipelines.train_all --smoke --device cpu"
echo "Full run (CPU-parallel across folds is fastest for the small DRL nets):"
echo "  cd research && python -m pipelines.train_all --device cpu --n-jobs \$(nproc)"
echo "Supervised LSTM/ResMLP benefit from GPU: add --stages supervised --device cuda"
echo "Reports land in research/artifacts/runs/train_all_*.json"
