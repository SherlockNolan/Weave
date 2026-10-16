#!/usr/bin/env bash
set -euo pipefail
WEAVE_UPGRADE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WEAVE_UPGRADE_LAB="$(dirname "$WEAVE_UPGRADE_REPO")/IsaacLab-3.0-latest"
WEAVE_UPGRADE_REV="$(cat "$WEAVE_UPGRADE_REPO/environments/isaac62/isaaclab-revision")"
if [ ! -d "$WEAVE_UPGRADE_LAB" ]; then
    git clone --filter=blob:none --no-checkout --branch release/3.0.0 \
        https://github.com/isaac-sim/IsaacLab.git "$WEAVE_UPGRADE_LAB"
    git -C "$WEAVE_UPGRADE_LAB" fetch --depth 1 origin "$WEAVE_UPGRADE_REV"
    git -C "$WEAVE_UPGRADE_LAB" checkout --detach "$WEAVE_UPGRADE_REV"
fi
WEAVE_UPGRADE_ACTUAL="$(git -c safe.directory="$WEAVE_UPGRADE_LAB" -C "$WEAVE_UPGRADE_LAB" rev-parse HEAD)"
if [ "$WEAVE_UPGRADE_ACTUAL" != "$WEAVE_UPGRADE_REV" ]; then
    echo "Isaac Lab revision differs from the pinned version; existing checkout left intact: $WEAVE_UPGRADE_LAB" >&2
    exit 1
fi
WEAVE_UPGRADE_UV="$(command -v uv || true)"
if [ -z "$WEAVE_UPGRADE_UV" ]; then
    WEAVE_UPGRADE_UV=/mnt/data/ruofei/zhangyu/cache/uv/bin/uv
fi
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-8}"
WEAVE_UPGRADE_ENV="$WEAVE_UPGRADE_REPO/environments/isaac62/.venv"
# Put the many small runtime files on local disk. The project, wheel cache,
# reference data and checkpoints remain on the shared filesystem.
if [ -L "$WEAVE_UPGRADE_ENV" ]; then
    WEAVE_UPGRADE_LOCAL="$(readlink -m "$WEAVE_UPGRADE_ENV")"
elif [ ! -e "$WEAVE_UPGRADE_ENV" ]; then
    WEAVE_UPGRADE_HASH="$(printf '%s' "$WEAVE_UPGRADE_REPO" | sha256sum | cut -c1-12)"
    WEAVE_UPGRADE_LOCAL="${WEAVE_RUNTIME_ROOT:-/var/tmp}/weave-isaac62-$WEAVE_UPGRADE_HASH/venv"
else
    WEAVE_UPGRADE_LOCAL="$WEAVE_UPGRADE_ENV"
fi
if [ ! -x "$WEAVE_UPGRADE_LOCAL/bin/python" ]; then
    "$WEAVE_UPGRADE_UV" venv --project "$WEAVE_UPGRADE_REPO/environments/isaac62" \
        --python 3.12 "$WEAVE_UPGRADE_LOCAL"
fi
if [ ! -e "$WEAVE_UPGRADE_ENV" ] && [ ! -L "$WEAVE_UPGRADE_ENV" ]; then
    ln -s "$WEAVE_UPGRADE_LOCAL" "$WEAVE_UPGRADE_ENV"
fi
"$WEAVE_UPGRADE_LOCAL/bin/python" "$WEAVE_UPGRADE_REPO/scripts/reproduce/prepare_isaac62_wheels.py"
"$WEAVE_UPGRADE_LOCAL/bin/python" "$WEAVE_UPGRADE_REPO/scripts/reproduce/prepare_isaac62_robot.py"
"$WEAVE_UPGRADE_UV" sync --project "$WEAVE_UPGRADE_REPO/environments/isaac62" --python 3.12 --locked
echo "Activate: source $WEAVE_UPGRADE_REPO/scripts/reproduce/activate_isaac62.sh"
