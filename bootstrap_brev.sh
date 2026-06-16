#!/usr/bin/env bash
# One-command deploy on a fresh NVIDIA / Brev VM.
#
# Clones (or updates) the repo, provisions the environment via setup_brev.sh,
# and — with RUN=1 — launches the full training in the background.
#
# First time on a fresh box (private repo → use a GitHub PAT in the URL):
#   GIT_URL="https://<TOKEN>@github.com/alexjbv7/D_PR.git" RUN=1 bash bootstrap_brev.sh
#
# Re-deploy / update an existing checkout:
#   RUN=1 bash bootstrap_brev.sh
#
# Env vars:
#   GIT_URL  git remote (default: public HTTPS; add a PAT for private clone)
#   BRANCH   branch to deploy (default: feat/drl-training-cloud)
#   TARGET   checkout dir (default: $HOME/D_PR)
#   RUN      "1" to launch training in the background after setup (default: 0)
set -euo pipefail

GIT_URL="${GIT_URL:-https://github.com/alexjbv7/D_PR.git}"
BRANCH="${BRANCH:-feat/drl-training-cloud}"
TARGET="${TARGET:-$HOME/D_PR}"
RUN="${RUN:-0}"

# 1) Clone or update
if [[ -d "$TARGET/.git" ]]; then
  echo "== updating $TARGET ($BRANCH) =="
  git -C "$TARGET" fetch origin "$BRANCH"
  git -C "$TARGET" checkout "$BRANCH"
  git -C "$TARGET" pull --ff-only origin "$BRANCH"
else
  echo "== cloning $BRANCH -> $TARGET =="
  git clone -b "$BRANCH" "$GIT_URL" "$TARGET"
fi
cd "$TARGET"

# 2) Load secrets if present (never committed). ALPACA keys enable crypto bars.
if [[ -f .env ]]; then
  echo "== sourcing .env =="
  set -a; source .env; set +a
fi

# 3) Provision
bash setup_brev.sh

# 4) Optionally launch the full run in the background (survives SSH disconnect)
if [[ "$RUN" == "1" ]]; then
  JOBS="$(nproc)"
  mkdir -p research/artifacts/runs
  LOG="research/artifacts/runs/train_all.log"
  echo "== launching full training (n_jobs=$JOBS) -> $LOG =="
  ( cd research && nohup python -m pipelines.train_all --device cpu --n-jobs "$JOBS" \
      > "../$LOG" 2>&1 & echo "PID $!" )
  echo "Follow: tail -f $TARGET/$LOG"
  echo "Results: $TARGET/research/artifacts/runs/train_all_*.json"
else
  echo ""
  echo "Setup done. Start training with:"
  echo "  cd '$TARGET/research' && python -m pipelines.train_all --device cpu --n-jobs \$(nproc)"
fi
