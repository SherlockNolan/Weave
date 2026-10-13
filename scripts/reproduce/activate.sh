#!/usr/bin/env bash
# Source this file to activate Isaac Sim and expose the installed CUDA compiler.
WEAVE_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WEAVE_WORKSPACE="${WORKSPACE:-$(dirname "$WEAVE_REPO_DIR")}"
if [ -x "$WEAVE_REPO_DIR/.venv/bin/python" ]; then
    source "$WEAVE_REPO_DIR/.venv/bin/activate"
    export ISAACLAB_PATH="${ISAACLAB_PATH:-$WEAVE_WORKSPACE/IsaacLab}"
    export RESOURCE_NAME="IsaacSim"
    if [ "$(id -u)" -eq 0 ]; then
        export OMNI_KIT_ALLOW_ROOT=1
    fi
    if [ -f "$ISAACLAB_PATH/_isaac_sim/setup_conda_env.sh" ]; then
        source "$ISAACLAB_PATH/_isaac_sim/setup_conda_env.sh"
    fi
else
    source "$WEAVE_WORKSPACE/sim51/bin/activate"
fi
# Use NVIDIA's EGL Vulkan entry point on this headless container. This changes
# only applications launched from this shell, not system drivers or other jobs.
if [ -f "$WEAVE_REPO_DIR/scripts/reproduce/vulkan/nvidia_icd.json" ]; then
    export VK_DRIVER_FILES="$WEAVE_REPO_DIR/scripts/reproduce/vulkan/nvidia_icd.json"
    export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
fi
WEAVE_SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
for WEAVE_CUDA_CANDIDATE in "$WEAVE_SITE_PACKAGES/nvidia/cu13" "$WEAVE_WORKSPACE/cuda-12.8" /usr/local/cuda; do
    if [ -x "$WEAVE_CUDA_CANDIDATE/bin/nvcc" ]; then
        export CUDA_HOME="$WEAVE_CUDA_CANDIDATE"
        export PATH="$CUDA_HOME/bin:$PATH"
        break
    fi
done
unset WEAVE_SITE_PACKAGES WEAVE_CUDA_CANDIDATE WEAVE_WORKSPACE WEAVE_REPO_DIR
