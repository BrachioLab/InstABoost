#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

RECREATE=${1:-}

if [ "$RECREATE" = "--recreate" ] && [ -d ".venv" ]; then
    backup=".venv.broken.$(date +%Y%m%d%H%M%S)"
    mv .venv "$backup"
    echo "Moved existing .venv to $backup"
fi

if [ ! -d ".venv" ]; then
    python3 -m venv --system-site-packages .venv
fi

. "$REPO_ROOT/.venv/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

python - <<'PY'
import importlib

mods = [
    "torch",
    "transformers",
    "transformer_lens",
    "datasets",
    "sklearn",
    "pandas",
    "unillm",
    "googleapiclient",
]

for mod_name in mods:
    importlib.import_module(mod_name)

print("Local environment is ready.")
PY
