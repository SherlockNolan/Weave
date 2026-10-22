#!/usr/bin/env bash
# The author's 8 x 16384 environment recipe; start fresh unless --resume is explicit.
set -eo pipefail
WEAVE_PAPER_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$WEAVE_PAPER_REPO"
export WEAVE_GPU_INDICES="${WEAVE_GPU_INDICES:-0,1,2,3,4,5,6,7}"
# Match the upstream install.sh runtime by default; the separate 6.2/Viser env remains installed.
export WEAVE_TRAIN_ACTIVATION="${WEAVE_TRAIN_ACTIVATION:-scripts/reproduce/activate.sh}"
export WEAVE_TRAIN_TASK=G1-Inspire-HOI-v0
export WEAVE_TRAIN_CONFIG=train_paper
export WEAVE_TRAIN_RUN_NAME="${WEAVE_TRAIN_RUN_NAME:-paper-issue2-full}"
export WEAVE_TRAIN_ITERATIONS=100000
export WEAVE_TRAIN_ENVS_PER_GPU=16384
exec bash scripts/reproduce/train_gpu01.sh --paper_recipe "$@"
