#!/usr/bin/env bash
set -eo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
echo "Waiting for env, data, and assets jobs to succeed..."
while true; do
    READY=true
    for job in env data assets; do
        STATUS="$(cat "logs/preparation/$job.status" 2>/dev/null || true)"
        case "$STATUS" in
            SUCCESS) ;;
            FAILED*) echo "$job: $STATUS; fix and rerun this validation job."; exit 1 ;;
            *) READY=false ;;
        esac
    done
    if [ "$READY" = true ]; then break; fi
    sleep 10
done
source scripts/reproduce/activate.sh
python scripts/reproduce/verify_data.py
echo "Running a 32-environment, two-iteration smoke check..."
timeout 1200 python scripts/rsl_rl/train.py --task G1-Inspire-HOI-v0 \
    --headless --config-dir ./configs/track --config-name smoke
