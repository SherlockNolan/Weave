#!/usr/bin/env bash
# Run a preparation command with persistent logs and an explicit completion status.
set -uo pipefail
export TZ=Asia/Shanghai
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOB="$1"
shift
LOG_DIR="$REPO_DIR/logs/preparation"
mkdir -p "$LOG_DIR"
cd "$REPO_DIR"
printf 'RUNNING\n' > "$LOG_DIR/$JOB.status"
printf 'Started: %s\nCommand:' "$(date --iso-8601=seconds)" | tee -a "$LOG_DIR/$JOB.log"
printf ' %q' "$@" | tee -a "$LOG_DIR/$JOB.log"
printf '\n' | tee -a "$LOG_DIR/$JOB.log"
"$@" 2>&1 | tee -a "$LOG_DIR/$JOB.log"
RESULT=${PIPESTATUS[0]}
if [ "$RESULT" -eq 0 ]; then
    printf 'SUCCESS\n' > "$LOG_DIR/$JOB.status"
else
    printf 'FAILED exit=%s\n' "$RESULT" > "$LOG_DIR/$JOB.status"
fi
printf '\nFinished: %s; exit=%s\n' "$(date --iso-8601=seconds)" "$RESULT" | tee -a "$LOG_DIR/$JOB.log"
exit "$RESULT"
