#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
cd "$REPO_DIR"
git lfs pull
git lfs fsck
python3 - <<'PY'
import subprocess
from pathlib import Path
for name in subprocess.check_output(["git", "lfs", "ls-files", "--name-only"], text=True).splitlines():
    with Path(name).open("rb") as stream:
        if stream.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise SystemExit(f"Unresolved LFS pointer: {name}")
print("All Git LFS assets are hydrated.")
PY
