#!/usr/bin/env bash
# Finish an installed environment without rerunning IsaacLab's Torch installation.
set -eo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${WORKSPACE:-$(dirname "$REPO_DIR")}"
export PATH="$HOME/.local/bin:$PATH"
export UV_HTTP_TIMEOUT=120
uv pip install --python "$WORKSPACE/sim51/bin/python" \
    --index-url "${WEAVE_PYPI_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}" wheel==0.45.1
bash "$REPO_DIR/scripts/reproduce/finish_env.sh"
