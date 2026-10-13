#!/usr/bin/env bash
set -eo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
source scripts/reproduce/activate.sh
CHECKPOINT="$(python - <<'PY'
from pathlib import Path
paths = list(Path('logs/rsl_rl/g1_inspire_hoi').glob('*_preparation-smoke/model_1.pt'))
if not paths:
    raise SystemExit('Run the preparation smoke check first.')
print(max(paths, key=lambda p: p.stat().st_mtime).resolve())
PY
)"
echo "Checking checkpoint reload and clip-wise evaluation using $CHECKPOINT"
timeout 1200 python -u scripts/rsl_rl/eval.py --task G1-Inspire-HOI-v0 \
    --headless --config-dir ./configs/track --config-name eval_reference \
    --checkpoint "$CHECKPOINT" --output_dir logs/preparation/eval-smoke \
    'env.commands.motion.motion_files=[./data/test/smallbox_24clip_var5.npz]'
python - <<'PY'
import json
from pathlib import Path
path = Path('logs/preparation/eval-smoke/metrics.json')
metrics = json.loads(path.read_text())
assert metrics['num_clips'] == 120 and len(metrics['clips']) == 120, 'Incomplete clip set'
assert all(clip['steps'] > 0 and clip['cause'] for clip in metrics['clips']), 'Some clips never terminated'
print(f'Evaluation report written: {path.resolve()}')
print(f'Report fields: {list(metrics)}')
PY
