#!/usr/bin/env bash
# Resume the unified policy on GPUs 0 and 1 using the isolated Isaac Sim 6.2 environment.
set -euo pipefail
WEAVE_RESUME_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$WEAVE_RESUME_REPO"
# The later 64500/99999 checkpoints were trained with the audited resume/layout
# bugs. Default to the saved checkpoint before the first affected resume.
WEAVE_RESUME_CHECKPOINT="${1:-$WEAVE_RESUME_REPO/logs/rsl_rl/g1_inspire_hoi/2026-10-13_17-04-51_paper-reference-dual/model_300.pt}"
if [ ! -f "$WEAVE_RESUME_CHECKPOINT" ]; then
    echo "Checkpoint does not exist: $WEAVE_RESUME_CHECKPOINT" >&2
    exit 1
fi
WEAVE_RESUME_NAME="$(basename "$WEAVE_RESUME_CHECKPOINT")"
if [[ ! "$WEAVE_RESUME_NAME" =~ ^model_([0-9]+)\.pt$ ]]; then
    echo "Expected a model_ITERATION.pt checkpoint." >&2
    exit 1
fi
WEAVE_RESUME_ITERATION="${BASH_REMATCH[1]}"
WEAVE_RESUME_TARGET=100000
if [ "$WEAVE_RESUME_ITERATION" -ge "$WEAVE_RESUME_TARGET" ]; then
    echo "Checkpoint already reached the target iteration." >&2
    exit 1
fi
export WEAVE_TRAIN_ACTIVATION=scripts/reproduce/activate_isaac62.sh
export WEAVE_TRAIN_TASK=G1-Inspire-HOI-v0
export WEAVE_TRAIN_CONFIG=train_reference_dual
export WEAVE_TRAIN_RUN_NAME=reference-scaled-dual-isaac62-corrected
export WEAVE_TRAIN_ITERATIONS="$((WEAVE_RESUME_TARGET - WEAVE_RESUME_ITERATION))"
export WEAVE_TRAIN_ENVS_PER_GPU=16384
export WEAVE_JOB_PTR="$WEAVE_RESUME_REPO/logs/preparation/isaac62-resume-job.path"
exec bash scripts/reproduce/train_gpu01.sh --resume \
    --load_run "$(cd "$(dirname "$WEAVE_RESUME_CHECKPOINT")" && pwd)" \
    --checkpoint "$WEAVE_RESUME_NAME"
