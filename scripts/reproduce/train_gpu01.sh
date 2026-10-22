#!/usr/bin/env bash
# Distributed launcher. Defaults to the reduced GPU 0/1 variant; train_paper.sh selects eight ranks.
set -eo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
export WEAVE_GPU_INDICES="${WEAVE_GPU_INDICES:-0,1}"
export WEAVE_TRAIN_CONFIG="${WEAVE_TRAIN_CONFIG:-train_reference_dual}"
export WEAVE_TRAIN_TASK="${WEAVE_TRAIN_TASK:-G1-Inspire-HOI-v0}"
export WEAVE_TRAIN_RUN_NAME="${WEAVE_TRAIN_RUN_NAME:-reference-scaled-dual}"
export WEAVE_TRAIN_ITERATIONS="${WEAVE_TRAIN_ITERATIONS:-100000}"
export WEAVE_TRAIN_ENVS_PER_GPU="${WEAVE_TRAIN_ENVS_PER_GPU:-16384}"
export WEAVE_ALLOW_GPU_SHARING="${WEAVE_ALLOW_GPU_SHARING:-0}"
export WEAVE_MIN_FREE_GPU_MIB="${WEAVE_MIN_FREE_GPU_MIB:-35000}"
if [[ ! "$WEAVE_ALLOW_GPU_SHARING" =~ ^[01]$ ]] || [[ ! "$WEAVE_MIN_FREE_GPU_MIB" =~ ^[1-9][0-9]*$ ]]; then
    echo "WEAVE_ALLOW_GPU_SHARING must be 0 or 1 and WEAVE_MIN_FREE_GPU_MIB must be positive." >&2
    exit 2
fi
if [[ ! "$WEAVE_GPU_INDICES" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]]; then
    echo "WEAVE_GPU_INDICES must be comma-separated numeric GPU indices (for example 0,1)." >&2
    exit 2
fi
IFS=, read -r -a WEAVE_SELECTED_GPUS <<< "$WEAVE_GPU_INDICES"
export WEAVE_WORLD_SIZE="${#WEAVE_SELECTED_GPUS[@]}"
declare -A WEAVE_SEEN_GPUS=()
for WEAVE_GPU_INDEX in "${WEAVE_SELECTED_GPUS[@]}"; do
    if [[ ! "$WEAVE_GPU_INDEX" =~ ^[0-9]+$ ]] || [[ -n "${WEAVE_SEEN_GPUS[$WEAVE_GPU_INDEX]:-}" ]]; then
        echo "WEAVE_GPU_INDICES must contain distinct numeric GPU indices." >&2
        exit 2
    fi
    WEAVE_SEEN_GPUS[$WEAVE_GPU_INDEX]=1
done
WEAVE_TRAIN_ARGS=()
WEAVE_DRY_RUN=false
for WEAVE_ARG in "$@"; do
    if [ "$WEAVE_ARG" = --dry-run ]; then
        WEAVE_DRY_RUN=true
    else
        WEAVE_TRAIN_ARGS+=("$WEAVE_ARG")
    fi
done
# Print/validate the declared plan before activating Isaac, locking GPUs, or creating job directories.
if [ "$WEAVE_DRY_RUN" = true ]; then
    exec python -I scripts/rsl_rl/training_recipe.py
fi
if [ "$WEAVE_TRAIN_CONFIG" = train_paper ]; then
    python -I scripts/rsl_rl/training_recipe.py > /dev/null
fi
mkdir -p logs/preparation logs/training
WEAVE_AVAILABLE_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)"
WEAVE_LOCK_FDS=()
for WEAVE_GPU_INDEX in "${WEAVE_SELECTED_GPUS[@]}"; do
    if ! printf '%s\n' "$WEAVE_AVAILABLE_GPUS" | rg -Fxq "$WEAVE_GPU_INDEX"; then
        echo "Requested GPU $WEAVE_GPU_INDEX is unavailable; this recipe requires $WEAVE_WORLD_SIZE GPUs. No training started." >&2
        exit 1
    fi
    exec {WEAVE_LOCK_FD}>"logs/preparation/gpu${WEAVE_GPU_INDEX}.lock"
    flock -n "$WEAVE_LOCK_FD" || { echo "GPU $WEAVE_GPU_INDEX is reserved by another Weave launcher." >&2; exit 1; }
    WEAVE_LOCK_FDS+=("$WEAVE_LOCK_FD")
done

WEAVE_GPU_UUIDS=()
WEAVE_GPU_PROCESSES="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)"
for WEAVE_GPU_INDEX in "${WEAVE_SELECTED_GPUS[@]}"; do
    WEAVE_GPU_UUID="$(nvidia-smi -i "$WEAVE_GPU_INDEX" --query-gpu=uuid --format=csv,noheader)"
    WEAVE_GPU_MEMORY="$(nvidia-smi -i "$WEAVE_GPU_INDEX" --query-gpu=memory.used --format=csv,noheader,nounits)"
    WEAVE_GPU_FREE="$(nvidia-smi -i "$WEAVE_GPU_INDEX" --query-gpu=memory.free --format=csv,noheader,nounits)"
    WEAVE_GPU_UTIL="$(nvidia-smi -i "$WEAVE_GPU_INDEX" --query-gpu=utilization.gpu --format=csv,noheader,nounits)"
    if [ "$WEAVE_ALLOW_GPU_SHARING" = 1 ]; then
        if [ "$WEAVE_GPU_FREE" -lt "$WEAVE_MIN_FREE_GPU_MIB" ]; then
            echo "GPU $WEAVE_GPU_INDEX has only $WEAVE_GPU_FREE MiB free; shared launch requires $WEAVE_MIN_FREE_GPU_MIB MiB." >&2
            exit 1
        fi
        echo "GPU $WEAVE_GPU_INDEX: explicit shared launch, $WEAVE_GPU_FREE MiB free; existing jobs keep running."
    elif [ "$WEAVE_GPU_MEMORY" -gt 64 ] || [ "$WEAVE_GPU_UTIL" -ne 0 ] || \
        printf '%s\n' "$WEAVE_GPU_PROCESSES" | rg -Fxq "$WEAVE_GPU_UUID"; then
        echo "GPU $WEAVE_GPU_INDEX is busy; refusing to start training." >&2
        exit 1
    fi
    WEAVE_GPU_UUIDS+=("$WEAVE_GPU_UUID")
done

source "${WEAVE_TRAIN_ACTIVATION:-scripts/reproduce/activate.sh}"
WEAVE_VISIBLE_UUIDS="$(printf '%s,' "${WEAVE_GPU_UUIDS[@]}")"
export CUDA_VISIBLE_DEVICES="${WEAVE_VISIBLE_UUIDS%,}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TZ=Asia/Shanghai
export WEAVE_RUN_TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
WEAVE_THREADS_PER_RANK="$((16 / WEAVE_WORLD_SIZE))"
if [ "$WEAVE_THREADS_PER_RANK" -lt 1 ]; then WEAVE_THREADS_PER_RANK=1; fi
export OMP_NUM_THREADS="$WEAVE_THREADS_PER_RANK"
export MKL_NUM_THREADS="$WEAVE_THREADS_PER_RANK"
export NUMEXPR_NUM_THREADS="$WEAVE_THREADS_PER_RANK"
export TORCHINDUCTOR_COMPILE_THREADS=2
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
WEAVE_JOB_DIR="$REPO_DIR/logs/training/${WEAVE_RUN_TIMESTAMP}_${WEAVE_TRAIN_RUN_NAME}"
export WEAVE_JOB_DIR
mkdir -p "$WEAVE_JOB_DIR"
ln -sfn "$(basename "$WEAVE_JOB_DIR")" logs/training/latest-distributed
if [ "$WEAVE_WORLD_SIZE" -eq 2 ]; then
    ln -sfn "$(basename "$WEAVE_JOB_DIR")" logs/training/latest-dual
fi
if [ "$WEAVE_TRAIN_CONFIG" = train_paper ]; then
    ln -sfn "$(basename "$WEAVE_JOB_DIR")" logs/training/latest-paper
fi
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
python -I - "${WEAVE_TRAIN_ARGS[@]}" <<'PY'
import json, os, sys
from pathlib import Path
root = Path.cwd()
report = {
    "started_at": os.environ["WEAVE_RUN_TIMESTAMP"], "timezone": "Asia/Shanghai",
    "gpus": [int(i) for i in os.environ["WEAVE_GPU_INDICES"].split(",")],
    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    "world_size": int(os.environ["WEAVE_WORLD_SIZE"]), "config": os.environ["WEAVE_TRAIN_CONFIG"],
    "activation": os.environ.get("WEAVE_TRAIN_ACTIVATION", "scripts/reproduce/activate.sh"),
    "allow_gpu_sharing": os.environ["WEAVE_ALLOW_GPU_SHARING"] == "1",
    "min_free_gpu_mib": int(os.environ["WEAVE_MIN_FREE_GPU_MIB"]),
    "extra_args": sys.argv[1:],
    "task": os.environ["WEAVE_TRAIN_TASK"],
    "envs_per_gpu": int(os.environ["WEAVE_TRAIN_ENVS_PER_GPU"]),
    "total_envs": int(os.environ["WEAVE_WORLD_SIZE"]) * int(os.environ["WEAVE_TRAIN_ENVS_PER_GPU"]),
    "paper_total_envs": 131072,
    "matches_author_parallelism": int(os.environ["WEAVE_WORLD_SIZE"]) * int(os.environ["WEAVE_TRAIN_ENVS_PER_GPU"]) == 131072,
    "recipe_source": "https://github.com/xiaohu-art/Weave/issues/2",
    "train_clips": 12 if os.environ["WEAVE_TRAIN_TASK"].endswith("Pickup-v0") else 7869,
    "objects": 3 if os.environ["WEAVE_TRAIN_TASK"].endswith("Pickup-v0") else 9,
    "max_iterations": int(os.environ["WEAVE_TRAIN_ITERATIONS"]),
    "save_interval": 100, "seed_rank0": 42,
    "cpu_affinity": os.environ["WEAVE_CPU_LIST"], "cpu_nice": 10,
    "checkpoint_dir": str(root / "logs/rsl_rl/g1_inspire_hoi" / (os.environ["WEAVE_RUN_TIMESTAMP"] + "_" + os.environ["WEAVE_TRAIN_RUN_NAME"])),
    "note": "Declared launch settings; resolved settings are saved as training_recipe.json in the checkpoint directory.",
}
Path(os.environ["WEAVE_JOB_DIR"], "launch.json").write_text(json.dumps(report, indent=2) + "\n")
PY
echo "RUNNING" > "$WEAVE_JOB_DIR/status"
set +e
taskset -c "$WEAVE_CPU_LIST" nice -n 10 python -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node="$WEAVE_WORLD_SIZE" --max_restarts=0 \
    --log-dir "$WEAVE_JOB_DIR/workers" --tee 3 \
    scripts/rsl_rl/train.py --task "$WEAVE_TRAIN_TASK" --headless --distributed \
    --config-dir ./configs/track --config-name "$WEAVE_TRAIN_CONFIG" \
    --run_name "$WEAVE_TRAIN_RUN_NAME" --logger tensorboard \
    --num_envs "$WEAVE_TRAIN_ENVS_PER_GPU" --max_iterations "$WEAVE_TRAIN_ITERATIONS" \
    --kit_args="--/renderer/multiGpu/enabled=false --/renderer/multiGpu/maxGpuCount=1 --/plugins/carb.tasking.plugin/threadCount=$WEAVE_THREADS_PER_RANK --/plugins/omni.tbb.globalcontrol/maxThreadCount=$WEAVE_THREADS_PER_RANK" \
    "${WEAVE_TRAIN_ARGS[@]}" > "$WEAVE_JOB_DIR/train.log" 2>&1
WEAVE_EXIT_CODE=$?
if [ "$WEAVE_EXIT_CODE" -eq 0 ] && rg -q 'torch\.(cuda\.)?OutOfMemoryError:|Error executing job with overrides:' "$WEAVE_JOB_DIR/train.log"; then
    # Isaac shutdown can mask a Python/Hydra failure with a zero process exit.
    echo "Training logged a fatal Python error despite exit=0; marking this run as failed." >&2
    WEAVE_EXIT_CODE=1
fi
if [ "$WEAVE_EXIT_CODE" -eq 0 ]; then
    echo "COMPLETED" > "$WEAVE_JOB_DIR/status"
else
    echo "FAILED exit=$WEAVE_EXIT_CODE" > "$WEAVE_JOB_DIR/status"
fi
exit "$WEAVE_EXIT_CODE"
