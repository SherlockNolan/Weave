#!/usr/bin/env bash
# Run a Weave RL command only when physical GPU 0 has no existing workload.
set -eo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
if [ "$#" -eq 0 ]; then
    echo "Usage: bash scripts/reproduce/run_gpu0.sh scripts/rsl_rl/train.py [arguments...]" >&2
    exit 2
fi
case "$1" in
    scripts/rsl_rl/train.py|scripts/rsl_rl/eval.py|scripts/rsl_rl/play.py) ;;
    *) echo "Expected a Weave train.py, eval.py, or play.py command." >&2; exit 2 ;;
esac

# The lock coordinates commands using this launcher. The NVIDIA query also
# checks workloads started by other launchers; it does not reserve the GPU.
mkdir -p logs/preparation
exec 9>logs/preparation/gpu0.lock
flock -n 9 || { echo "Another Weave GPU 0 command is running." >&2; exit 1; }
WEAVE_GPU_UUID="$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)"
WEAVE_GPU_MEMORY="$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)"
WEAVE_GPU_UTIL="$(nvidia-smi -i 0 --query-gpu=utilization.gpu --format=csv,noheader,nounits)"
WEAVE_GPU_PROCESSES="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)"
if [ "$WEAVE_GPU_MEMORY" -gt 64 ] || [ "$WEAVE_GPU_UTIL" -ne 0 ] || \
    printf '%s\n' "$WEAVE_GPU_PROCESSES" | grep -Fxq "$WEAVE_GPU_UUID"; then
    echo "GPU 0 is busy; refusing to launch. Existing jobs were left running." >&2
    exit 1
fi

source scripts/reproduce/activate.sh
export CUDA_VISIBLE_DEVICES="$WEAVE_GPU_UUID"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec python -u "$@" --device cuda:0 \
    --kit_args="--/renderer/multiGpu/enabled=false --/renderer/multiGpu/maxGpuCount=1 --/plugins/carb.tasking.plugin/threadCount=4 --/plugins/omni.tbb.globalcontrol/maxThreadCount=4"
