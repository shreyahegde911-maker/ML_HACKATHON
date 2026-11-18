#!/usr/bin/env bash
# macOS/Linux: bash start.sh [--api-only | --ui-only] [--no-browser]
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

for candidate in .venv/bin/python python3.12 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && \
       "$candidate" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))' >/dev/null 2>&1; then
        exec "$candidate" scripts/launch.py "$@"
    fi
done

echo "PathSnap needs Python 3.12. Install it, then run: bash start.sh" >&2
exit 1
