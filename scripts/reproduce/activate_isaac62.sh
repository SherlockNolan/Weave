#!/usr/bin/env bash
# Source this script from any directory. The Isaac Sim 5.1 environment stays intact.
WEAVE_ISAAC62_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WEAVE_ISAAC62_ENV="$WEAVE_ISAAC62_REPO/environments/isaac62/.venv"
if [ ! -x "$WEAVE_ISAAC62_ENV/bin/python" ]; then
    echo "Run bash $WEAVE_ISAAC62_REPO/scripts/reproduce/setup_isaac62.sh first." >&2
    return 1
fi
# Clear paths injected by an earlier Isaac Sim 5.1 activation in this shell.
WEAVE_ISAAC62_FILTERED_LD=""
IFS=: read -r -a WEAVE_ISAAC62_LD_PARTS <<< "${LD_LIBRARY_PATH:-}"
for WEAVE_ISAAC62_LD_PART in "${WEAVE_ISAAC62_LD_PARTS[@]}"; do
    case "$WEAVE_ISAAC62_LD_PART" in
        "$WEAVE_ISAAC62_REPO/.venv/"*|"$(dirname "$WEAVE_ISAAC62_REPO")/isaacsim/"*|"$(dirname "$WEAVE_ISAAC62_REPO")/IsaacLab/_isaac_sim/"*) continue ;;
    esac
    if [ -n "$WEAVE_ISAAC62_LD_PART" ]; then
        WEAVE_ISAAC62_FILTERED_LD="${WEAVE_ISAAC62_FILTERED_LD:+$WEAVE_ISAAC62_FILTERED_LD:}$WEAVE_ISAAC62_LD_PART"
    fi
done
export LD_LIBRARY_PATH="$WEAVE_ISAAC62_FILTERED_LD"
unset PYTHONPATH ISAAC_PATH CARB_APP_PATH EXP_PATH
source "$WEAVE_ISAAC62_ENV/bin/activate"
export ISAACLAB_PATH="$(dirname "$WEAVE_ISAAC62_REPO")/IsaacLab-3.0-latest"
export OMNI_KIT_ACCEPT_EULA=YES
export OMNI_KIT_ALLOW_ROOT=1
export ACCEPT_EULA=Y
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VK_DRIVER_FILES="$WEAVE_ISAAC62_REPO/scripts/reproduce/vulkan/nvidia_icd.json"
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
# Some CUDA containers expose compatibility libraries newer than the host's
# kernel driver. Prefer matching libraries for this process without altering
# the host driver or the system symlinks.
WEAVE_ISAAC62_DRIVER="$(cat /sys/module/nvidia/version 2>/dev/null)"
WEAVE_ISAAC62_LIBDIR="$WEAVE_ISAAC62_REPO/environments/isaac62/driver-libs"
mkdir -p "$WEAVE_ISAAC62_LIBDIR"
for WEAVE_ISAAC62_LIB in libcuda.so libnvidia-ptxjitcompiler.so libnvidia-nvvm.so; do
    WEAVE_ISAAC62_REAL="/usr/lib/x86_64-linux-gnu/$WEAVE_ISAAC62_LIB.$WEAVE_ISAAC62_DRIVER"
    if [ -f "$WEAVE_ISAAC62_REAL" ]; then
        case "$WEAVE_ISAAC62_LIB" in
            libnvidia-nvvm.so) WEAVE_ISAAC62_SONAME=libnvidia-nvvm.so.4 ;;
            *) WEAVE_ISAAC62_SONAME="$WEAVE_ISAAC62_LIB.1" ;;
        esac
        ln -sfn "$WEAVE_ISAAC62_REAL" "$WEAVE_ISAAC62_LIBDIR/$WEAVE_ISAAC62_SONAME"
        ln -sfn "$WEAVE_ISAAC62_REAL" "$WEAVE_ISAAC62_LIBDIR/$WEAVE_ISAAC62_LIB"
    fi
done
export LD_LIBRARY_PATH="$WEAVE_ISAAC62_LIBDIR:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset WEAVE_ISAAC62_REPO WEAVE_ISAAC62_ENV WEAVE_ISAAC62_DRIVER WEAVE_ISAAC62_LIBDIR
unset WEAVE_ISAAC62_LIB WEAVE_ISAAC62_REAL WEAVE_ISAAC62_SONAME
unset WEAVE_ISAAC62_FILTERED_LD WEAVE_ISAAC62_LD_PARTS WEAVE_ISAAC62_LD_PART
