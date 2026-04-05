#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

EMOTION=${1:-joy}
MAX_TOKENS_GENERATED=${2:-${MAX_TOKENS_GENERATED:-}}

"$SCRIPT_DIR/run_emotions_prompt_methods.sh" "$EMOTION" "Qwen/Qwen3-14B" "" "$MAX_TOKENS_GENERATED"
"$SCRIPT_DIR/run_emotions_prompt_methods.sh" "$EMOTION" "openai/gpt-oss-20b" "" "$MAX_TOKENS_GENERATED"
