#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

SELECTOR=${1:-all}
MAX_TOKENS_GENERATED=${2:-${MAX_TOKENS_GENERATED:-}}

"$SCRIPT_DIR/run_prompt_methods.sh" "$SELECTOR" "Qwen/Qwen3-14B" "" "$MAX_TOKENS_GENERATED"
"$SCRIPT_DIR/run_prompt_methods.sh" "$SELECTOR" "openai/gpt-oss-20b" "" "$MAX_TOKENS_GENERATED"
