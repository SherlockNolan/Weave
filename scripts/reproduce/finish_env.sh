#!/usr/bin/env bash
set -eo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export WORKSPACE="${WORKSPACE:-$(dirname "$REPO_DIR")}"
export PATH="$HOME/.local/bin:$PATH"
VENV="$WORKSPACE/sim51"
cd "$REPO_DIR"
# The activation hook sets Isaac Sim's library and Python paths.
source "$REPO_DIR/scripts/reproduce/activate.sh"
uv pip freeze --python "$VENV/bin/python" > "$REPO_DIR/logs/preparation/environment.freeze.txt"
uv pip check --python "$VENV/bin/python"
python - <<'PY'
import json
import os
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
import torch
import numpy
report = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "cuda_home": os.environ.get("CUDA_HOME"),
    "runtime": {
        "numpy_version": numpy.__version__, "numpy_path": numpy.__file__,
        "torch_version": torch.__version__, "torch_path": torch.__file__,
    },
    "venv_packages": json.loads(subprocess.check_output([
        sys.executable, "-I", "-c",
        "import json; from importlib.metadata import version; "
        "print(json.dumps({p: version(p) for p in ('numpy', 'wheel', 'packaging')}))",
    ], text=True)),
    "packages": {name: version(name) for name in ("numpy", "isaaclab", "rsl-rl-lib", "g1_hoi_learning")},
    "weave_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "isaaclab_commit": subprocess.check_output(["git", "-C", "../IsaacLab", "rev-parse", "HEAD"], text=True).strip(),
}
Path("logs/preparation/environment.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY
timeout 300 python scripts/list_envs.py
