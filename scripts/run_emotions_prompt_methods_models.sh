#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

EMOTION=${1:-joy}
MAX_TOKENS_GENERATED=${2:-${MAX_TOKENS_GENERATED:-}}

if [ "$EMOTION" = "all" ]; then
    SELECTOR="emotions"
else
    SELECTOR="emotions:${EMOTION}"
fi

"$SCRIPT_DIR/run_prompt_methods_models.sh" "$SELECTOR" "$MAX_TOKENS_GENERATED"
