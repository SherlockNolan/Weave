#!/usr/bin/env bash
# Two-rank training on idle physical GPUs 0 and 1 only.
set -eo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
mkdir -p logs/preparation logs/training
exec 9>logs/preparation/gpu0.lock
exec 8>logs/preparation/gpu1.lock
flock -n 9 || { echo "GPU 0 is reserved by another Weave launcher." >&2; exit 1; }
flock -n 8 || { echo "GPU 1 is reserved by another Weave launcher." >&2; exit 1; }

WEAVE_GPU_UUIDS=()
WEAVE_GPU_PROCESSES="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)"
for WEAVE_GPU_INDEX in 0 1; do
    WEAVE_GPU_UUID="$(nvidia-smi -i "$WEAVE_GPU_INDEX" --query-gpu=uuid --format=csv,noheader)"
    WEAVE_GPU_MEMORY="$(nvidia-smi -i "$WEAVE_GPU_INDEX" --query-gpu=memory.used --format=csv,noheader,nounits)"
    WEAVE_GPU_UTIL="$(nvidia-smi -i "$WEAVE_GPU_INDEX" --query-gpu=utilization.gpu --format=csv,noheader,nounits)"
    if [ "$WEAVE_GPU_MEMORY" -gt 64 ] || [ "$WEAVE_GPU_UTIL" -ne 0 ] || \
        printf '%s\n' "$WEAVE_GPU_PROCESSES" | grep -Fxq "$WEAVE_GPU_UUID"; then
        echo "GPU $WEAVE_GPU_INDEX is busy; refusing to start training." >&2
        exit 1
    fi
    WEAVE_GPU_UUIDS+=("$WEAVE_GPU_UUID")
done

source scripts/reproduce/activate.sh
export CUDA_VISIBLE_DEVICES="${WEAVE_GPU_UUIDS[0]},${WEAVE_GPU_UUIDS[1]}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TZ=Asia/Shanghai
export WEAVE_RUN_TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export TORCHINDUCTOR_COMPILE_THREADS=2
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WEAVE_TRAIN_CONFIG="${WEAVE_TRAIN_CONFIG:-train_reference_dual}"
export WEAVE_TRAIN_TASK="${WEAVE_TRAIN_TASK:-G1-Inspire-HOI-v0}"
export WEAVE_TRAIN_RUN_NAME="${WEAVE_TRAIN_RUN_NAME:-paper-reference-dual}"
export WEAVE_TRAIN_ITERATIONS="${WEAVE_TRAIN_ITERATIONS:-100000}"
export WEAVE_TRAIN_ENVS_PER_GPU="${WEAVE_TRAIN_ENVS_PER_GPU:-2048}"
WEAVE_JOB_DIR="$REPO_DIR/logs/training/${WEAVE_RUN_TIMESTAMP}_${WEAVE_TRAIN_RUN_NAME}"
export WEAVE_JOB_DIR
mkdir -p "$WEAVE_JOB_DIR"
ln -sfn "$(basename "$WEAVE_JOB_DIR")" logs/training/latest-dual
if [ "$WEAVE_TRAIN_TASK" = G1-Inspire-HOI-Pickup-v0 ]; then
    ln -sfn "$(basename "$WEAVE_JOB_DIR")" logs/training/latest-pickups
fi
if [ -n "${WEAVE_JOB_PTR:-}" ]; then
    printf '%s\n' "$WEAVE_JOB_DIR" > "$WEAVE_JOB_PTR"
fi
echo "STARTING" > "$WEAVE_JOB_DIR/status"
echo "$$" > "$WEAVE_JOB_DIR/launcher.pid"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > "$WEAVE_JOB_DIR/existing-jobs.csv"
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader > "$WEAVE_JOB_DIR/gpus-before.csv"

# Cap this job's CPU affinity and lower its CPU scheduling priority. These
# settings apply to this launcher and its children, not to existing jobs.
WEAVE_CPU_LIST="$(python -I -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[-16:])))')"
export WEAVE_CPU_LIST
python -I - <<'PY'
import json, os
from pathlib import Path
root = Path.cwd()
report = {
    "started_at": os.environ["WEAVE_RUN_TIMESTAMP"], "timezone": "Asia/Shanghai",
    "gpus": [0, 1], "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    "world_size": 2, "config": os.environ["WEAVE_TRAIN_CONFIG"],
    "task": os.environ["WEAVE_TRAIN_TASK"],
    "envs_per_gpu": int(os.environ["WEAVE_TRAIN_ENVS_PER_GPU"]),
    "total_envs": 2 * int(os.environ["WEAVE_TRAIN_ENVS_PER_GPU"]),
    "train_clips": 12 if os.environ["WEAVE_TRAIN_TASK"].endswith("Pickup-v0") else 7869,
    "objects": 3 if os.environ["WEAVE_TRAIN_TASK"].endswith("Pickup-v0") else 9,
    "max_iterations": int(os.environ["WEAVE_TRAIN_ITERATIONS"]),
    "save_interval": 100, "seed_rank0": 42,
    "cpu_affinity": os.environ["WEAVE_CPU_LIST"], "cpu_nice": 10,
    "checkpoint_dir": str(root / "logs/rsl_rl/g1_inspire_hoi" / (os.environ["WEAVE_RUN_TIMESTAMP"] + "_" + os.environ["WEAVE_TRAIN_RUN_NAME"])),
    "note": "Two-rank joint training. Per-rank environment counts are recorded above.",
}
Path(os.environ["WEAVE_JOB_DIR"], "launch.json").write_text(json.dumps(report, indent=2) + "\n")
PY
echo "RUNNING" > "$WEAVE_JOB_DIR/status"
set +e
taskset -c "$WEAVE_CPU_LIST" nice -n 10 python -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=2 --max_restarts=0 \
    --log-dir "$WEAVE_JOB_DIR/workers" --tee 3 \
    scripts/rsl_rl/train.py --task "$WEAVE_TRAIN_TASK" --headless --distributed \
    --config-dir ./configs/track --config-name "$WEAVE_TRAIN_CONFIG" \
    --run_name "$WEAVE_TRAIN_RUN_NAME" --logger tensorboard \
    --num_envs "$WEAVE_TRAIN_ENVS_PER_GPU" --max_iterations "$WEAVE_TRAIN_ITERATIONS" \
    --kit_args="--/renderer/multiGpu/enabled=false --/renderer/multiGpu/maxGpuCount=1 --/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8" \
    "$@" > "$WEAVE_JOB_DIR/train.log" 2>&1
WEAVE_EXIT_CODE=$?
if [ "$WEAVE_EXIT_CODE" -eq 0 ]; then
    echo "COMPLETED" > "$WEAVE_JOB_DIR/status"
else
    echo "FAILED exit=$WEAVE_EXIT_CODE" > "$WEAVE_JOB_DIR/status"
fi
exit "$WEAVE_EXIT_CODE"
