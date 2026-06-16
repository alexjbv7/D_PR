# Runbook — Deploy & train everything on the NVIDIA / Brev VM

Trains the full stack (XGBoost + DQN + PPO + SAC + supervised ResMLP/LSTM +
stat-arb) on BTC/USD, ETH/USD and EUR/USD at 4H, on a Brev GPU VM.

Repo: `github.com/alexjbv7/D_PR` · branch `feat/drl-training-cloud`.

---

## 0. Push the code first (local machine)

The training harness lives in your working tree — the VM clones from GitHub, so
push it before deploying:

```bash
git add research/models/drl/multi_algo_gate.py research/pipelines/train_all.py \
        research/experiments/btc_eth_eur_4h research/tests/test_*.py \
        setup_brev.sh bootstrap_brev.sh Dockerfile.brev RUNBOOK_BREV.md
git commit -m "feat(train): full multi-algo training harness + Brev deploy

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push origin feat/drl-training-cloud
```

(If you'd rather not push, you can upload the folder with `brev cp` / `rsync`
instead — then skip the clone in step 2 and run `bash setup_brev.sh` directly.)

---

## 1. Launch the VM

`brev.nvidia.com/environment/new` → Mid-Range → **L40S** (44GB; best price/perf
for tabular DL + RL). Cheapest cloud/region, **spot** if offered, **≥16 vCPU**
(fold parallelism on CPU is the real speed lever for these small nets). L4 /
A10G also work. Open a terminal (`brev shell <name>`) or VS Code.

---

## 2. Get the code + secrets on the VM

Private repo → use a GitHub Personal Access Token in the clone URL. The
bootstrap does clone + provision in one step:

```bash
# fresh box (one command):
GIT_URL="https://<GITHUB_PAT>@github.com/alexjbv7/D_PR.git" bash <(curl -fsSL \
  "https://<GITHUB_PAT>@raw.githubusercontent.com/alexjbv7/D_PR/feat/drl-training-cloud/bootstrap_brev.sh")
```

Or the explicit two-liner (clearer, same result):

```bash
git clone -b feat/drl-training-cloud "https://<GITHUB_PAT>@github.com/alexjbv7/D_PR.git" ~/D_PR
cd ~/D_PR
```

Secrets (never commit them). Crypto bars need Alpaca keys; EUR/USD (yfinance)
does not:

```bash
cp .env.example .env        # then edit .env and fill ALPACA_API_KEY / ALPACA_API_SECRET
# or just:
export ALPACA_API_KEY=...    ALPACA_API_SECRET=...
```

---

## 3. Provision

```bash
cd ~/D_PR
bash setup_brev.sh
```

Installs `shared` + `research[data,dev]` (numpy/pandas/sklearn/xgboost/scipy/
statsmodels/optuna/gymnasium/torch + ccxt/yfinance/alpaca-py), keeps/installs a
CUDA torch, prints the detected GPU, and runs an import/dry-run check.

Docker alternative (NGC base, fully reproducible):

```bash
docker build -f Dockerfile.brev -t quant_bot:brev .
docker run --gpus all -e ALPACA_API_KEY -e ALPACA_API_SECRET quant_bot:brev
```

---

## 4. Smoke test (minutes)

```bash
cd ~/D_PR/research
python -m pipelines.train_all --smoke --device cpu
```

Tiny folds/episodes for one algo + one supervised model — confirms data,
training and the gate wire up before spending GPU-hours.

---

## 5. Full run (background, survives disconnect)

The bootstrap can do clone + setup + launch in one shot:

```bash
RUN=1 GIT_URL="https://<GITHUB_PAT>@github.com/alexjbv7/D_PR.git" bash bootstrap_brev.sh
```

Or launch manually:

```bash
cd ~/D_PR/research
nohup python -m pipelines.train_all --device cpu --n-jobs $(nproc) \
      > artifacts/runs/train_all.log 2>&1 &
tail -f artifacts/runs/train_all.log
```

- Per asset: DQN, PPO, SAC (each gated vs XGBoost + buy-and-hold), supervised
  ResMLP/LSTM, then stat-arb on BTC/USD–ETH/USD.
- `--device cpu --n-jobs $(nproc)` is usually fastest for the small DRL nets.
  For the supervised deep models, add a GPU pass:
  `python -m pipelines.train_all --stages supervised --device cuda`.

Useful subsets:

```bash
python -m pipelines.train_all --stages drl --algos ppo,sac --n-jobs $(nproc)
python -m pipelines.train_all --stages statarb
python -m pipelines.train_all --n-seeds 10 --episodes 300 --n-folds 5   # heavier
```

---

## 6. Results & shutdown

Output: `research/artifacts/runs/train_all_<timestamp>.json` + a console table.
Promotion targets: ADR-040 §3.2 (DSR > 0.4, beats XGBoost and buy-and-hold) and
ADR-039 (DSR(SAC) > DSR(PPO) > DSR(XGBoost)).

Pull the results back to your machine:

```bash
brev cp <vm-name>:~/D_PR/research/artifacts/runs/ ./runs/     # or: scp -r
```

**Stop the VM** as soon as the JSON lands — GPU runs are bursty and idle time is
the main cost.

---

## Caveats

- The new harness must be pushed (step 0) before the VM can clone it.
- EUR/USD 4H is limited to ~700 days (yfinance 1H cap) — shorter than the crypto
  history, so its DSR is noisier.
- **SAC** needs more env steps than DQN/PPO to converge — raise `--episodes` for
  a fair SAC comparison on the full run.
- Crypto bars require valid Alpaca keys; without them only EUR/USD loads.
- Never put the GitHub PAT or Alpaca keys in a committed file; pass them via the
  clone URL / `.env` / `export` only.
