#!/usr/bin/env bash
set -eo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export WORKSPACE="${WORKSPACE:-$(dirname "$REPO_DIR")}"
export PATH="$HOME/.local/bin:$PATH"
export UV_HTTP_TIMEOUT=300
export UV_HTTP_RETRIES=5
export UV_INDEX_URL="${WEAVE_PYPI_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_INDEX_URL="$UV_INDEX_URL"
export PIP_DISABLE_PIP_VERSION_CHECK=1
VENV="$WORKSPACE/sim51"
cd "$REPO_DIR"
if [ ! -x "$VENV/bin/python" ]; then
    uv venv --python 3.11 "$VENV"
fi
# IsaacLab otherwise attempts a sudo apt installation on this host.
uv pip install --python "$VENV/bin/python" cmake pip setuptools wheel==0.45.1
export PATH="$VENV/bin:$PATH"
bash "$REPO_DIR/install.sh"
bash "$REPO_DIR/scripts/reproduce/finish_env.sh"
