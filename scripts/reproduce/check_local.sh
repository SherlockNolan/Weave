#!/usr/bin/env bash
set -eo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
source scripts/reproduce/activate.sh
echo "Checking CUDA compiler discovery..."
python - <<'PY'
from torch.utils.cpp_extension import CUDA_HOME
from pathlib import Path
assert CUDA_HOME and (Path(CUDA_HOME) / "bin/nvcc").is_file(), "CUDA compiler not found"
print(f"CUDA_HOME={CUDA_HOME}")
PY
nvcc --version
echo "Checking the local training split with torch.compile enabled (256 envs, 2 iterations)..."
timeout 1200 python -u scripts/rsl_rl/train.py --task G1-Inspire-HOI-v0 \
    --headless --config-dir ./configs/track --config-name train_local \
    --max_iterations 2 --run_name preparation-local-compiled
