#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
for job in env data assets validate local eval; do
    STATUS="$(cat "logs/preparation/$job.status" 2>/dev/null || echo NOT_STARTED)"
    printf '%-10s %s\n' "$job" "$STATUS"
done
python3 - <<'PY'
import json
from pathlib import Path
p = Path('logs/preparation/dataset.remote.json')
if p.exists():
    manifest = json.loads(p.read_text())
    total = sum(item['size'] for item in manifest['files'])
    transferred = 0
    complete = 0
    for item in manifest['files']:
        target = Path('data') / item['path'].removeprefix('reference/')
        part = target.with_name(target.name + '.part')
        if target.exists():
            transferred += target.stat().st_size
            complete += 1
        elif part.exists():
            transferred += part.stat().st_size
    print(f'Data: {transferred/1e9:.2f}/{total/1e9:.2f} GB; {complete}/{len(manifest["files"])} completed files')
for p in (Path('../isaac-sim-standalone-5.1.0-linux-x86_64.zip.part'), Path('../isaac-sim-standalone-5.1.0-linux-x86_64.zip')):
    if p.exists():
        print(f'Isaac Sim archive: {p.stat().st_size/1e9:.2f}/8.77 GB ({p.name})')
PY
