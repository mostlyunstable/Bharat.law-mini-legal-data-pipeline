#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PY="${PYTHON:-python3}"

if [[ ! -d ".venv" ]]; then
  "$PY" -m venv .venv
fi

. .venv/bin/activate
python -m pip install -q -r requirements.txt

python -m scripts.run_pipeline "$@"
