#!/usr/bin/env bash
# Run locally inside tmux. Reconnect with the authorized SSH key after a server restart.
set -euo pipefail
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*' no_proxy='*' TZ=Asia/Shanghai
export PATH="$HOME/.local/bin:$PATH"
WEAVE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL_WORKSPACE="$(dirname "$WEAVE_REPO")"
PAI_HOST=root@8.130.141.87
PAI_PORT=1023
PAI_WORKSPACE=/mnt/data/ruofei/zhangyu/code
PAI_SOCKET="$HOME/.ssh/weave-pai-1023.sock"
TRANSFER_LOGS="$LOCAL_WORKSPACE/.weave-pai-transfer"
mkdir -p "$TRANSFER_LOGS"
printf 'CONNECTING\n' > "$TRANSFER_LOGS/sync.status"
trap 'result=$?; if [ "$result" -ne 0 ]; then printf "FAILED exit=%s\n" "$result" > "$TRANSFER_LOGS/sync.status"; fi' EXIT
if ! ssh -S "$PAI_SOCKET" -O check -p "$PAI_PORT" "$PAI_HOST" >/dev/null 2>&1; then
    echo "Establishing a direct SSH connection with the authorized key..."
    ssh -M -N -f -S "$PAI_SOCKET" -p "$PAI_PORT" \
        -o BatchMode=yes -o ConnectTimeout=20 -o ControlPersist=8h \
        -o ProxyCommand=none -o ProxyJump=none -o StrictHostKeyChecking=accept-new \
        -o ServerAliveInterval=30 -o ServerAliveCountMax=3 "$PAI_HOST"
fi
SSH_ARGS=(-S "$PAI_SOCKET" -p "$PAI_PORT" -o BatchMode=yes -o ConnectTimeout=20 \
    -o ProxyCommand=none -o ProxyJump=none -o ServerAliveInterval=30 -o ServerAliveCountMax=3)
ssh "${SSH_ARGS[@]}" "$PAI_HOST" 'bash -s' < "$WEAVE_REPO/scripts/reproduce/probe_pai.sh" \
    | tee "$TRANSFER_LOGS/probe.log"
ssh "${SSH_ARGS[@]}" "$PAI_HOST" "mkdir -p '$PAI_WORKSPACE'"
RSYNC_SSH="ssh -S $PAI_SOCKET -p $PAI_PORT -o BatchMode=yes -o ProxyCommand=none -o ProxyJump=none -o ServerAliveInterval=30 -o ServerAliveCountMax=3"
BACKUP_DIR="$PAI_WORKSPACE/.weave-transfer-backups/$(date +%Y%m%d-%H%M%S)"
printf 'SYNCING_REPOSITORY_AND_DATA\n' > "$TRANSFER_LOGS/sync.status"
rsync -a --partial --partial-dir=.rsync-partial --info=progress2 --human-readable --stats \
    --backup --backup-dir="$BACKUP_DIR" \
    --exclude='__pycache__/' --exclude='.pytest_cache/' --exclude='.rsync-partial/' \
    -e "$RSYNC_SSH" "$WEAVE_REPO/" "$PAI_HOST:$PAI_WORKSPACE/Weave/"
printf 'REPOSITORY_AND_DATA_SYNCED\n' > "$TRANSFER_LOGS/sync.status"
# Reuse the downloaded simulator and pinned IsaacLab source on an x86_64 server.
PAI_ARCH="$(ssh "${SSH_ARGS[@]}" "$PAI_HOST" uname -m)"
if [ "$PAI_ARCH" = x86_64 ]; then
    printf 'SYNCING_INSTALLATION_ASSETS\n' > "$TRANSFER_LOGS/sync.status"
    rsync -a --partial --partial-dir=.rsync-partial --info=progress2 --human-readable --stats \
        -e "$RSYNC_SSH" "$LOCAL_WORKSPACE/isaac-sim-standalone-5.1.0-linux-x86_64.zip" \
        "$PAI_HOST:$PAI_WORKSPACE/"
    rsync -a --partial --partial-dir=.rsync-partial --info=progress2 --human-readable --stats \
        --exclude='_isaac_sim' --exclude='__pycache__/' --exclude='.pytest_cache/' \
        --backup --backup-dir="$BACKUP_DIR/IsaacLab" \
        -e "$RSYNC_SSH" "$LOCAL_WORKSPACE/IsaacLab/" "$PAI_HOST:$PAI_WORKSPACE/IsaacLab/"
    ssh "${SSH_ARGS[@]}" "$PAI_HOST" "mkdir -p '$PAI_WORKSPACE/.tools'"
    rsync -a -e "$RSYNC_SSH" "$(command -v uv)" "$PAI_HOST:$PAI_WORKSPACE/.tools/uv"
else
    echo "PAI architecture is $PAI_ARCH; x86_64 installation assets were skipped."
fi
printf 'SUCCESS\n' > "$TRANSFER_LOGS/sync.status"
echo "Transfer complete. Remote GPU and driver details are in $TRANSFER_LOGS/probe.log."
echo "The remote Python environment must be created for the PAI host; the local venv was not copied."
