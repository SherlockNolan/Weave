#!/usr/bin/env bash
# Read-only probe to run on PAI before choosing its CUDA/PyTorch environment.
set -uo pipefail
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*' no_proxy='*'
printf 'Host: '; hostname
id
uname -m
cat /etc/os-release
for cmd in uv rsync tmux curl git g++ unzip nvcc; do
    printf '%-10s ' "$cmd"
    command -v "$cmd" || true
done
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; fi
if command -v nvcc >/dev/null 2>&1; then nvcc --version; fi
df -h /mnt/data
free -h
for path in /mnt/data/ruofei/zhangyu/code /mnt/data/ruofei/zhangyu/code/Weave \
            /mnt/data/ruofei/zhangyu/code/IsaacLab /mnt/data/ruofei/zhangyu/code/isaacsim \
            /mnt/data/ruofei/zhangyu/code/sim51; do
    if [ -e "$path" ]; then ls -ld "$path"; fi
done
for path in /AGENTS.md /mnt/AGENTS.md /mnt/data/AGENTS.md /mnt/data/ruofei/AGENTS.md \
            /mnt/data/ruofei/zhangyu/AGENTS.md /mnt/data/ruofei/zhangyu/code/AGENTS.md \
            /mnt/data/ruofei/zhangyu/code/Weave/AGENTS.md; do
    if [ -f "$path" ]; then printf '\nInstructions: %s\n' "$path"; cat "$path"; fi
done
