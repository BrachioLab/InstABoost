#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

SELECTOR=${1:-all}
MAX_TOKENS_GENERATED=${2:-${MAX_TOKENS_GENERATED:-}}
METHODS_SELECTOR=${3:-${METHODS:-}}
FAILED_MODELS=

if ! "$SCRIPT_DIR/run_prompt_methods.sh" "$SELECTOR" "Qwen/Qwen3-14B" "" "$MAX_TOKENS_GENERATED" "" "" "$METHODS_SELECTOR"; then
    FAILED_MODELS="Qwen/Qwen3-14B"
fi

if ! "$SCRIPT_DIR/run_prompt_methods.sh" "$SELECTOR" "openai/gpt-oss-20b" "" "$MAX_TOKENS_GENERATED" "" "" "$METHODS_SELECTOR"; then
    if [ -n "$FAILED_MODELS" ]; then
        FAILED_MODELS="${FAILED_MODELS}
openai/gpt-oss-20b"
    else
        FAILED_MODELS="openai/gpt-oss-20b"
    fi
fi

if [ -n "$FAILED_MODELS" ]; then
    echo "One or more model runs failed:"
    printf '%s\n' "$FAILED_MODELS"
    exit 1
fi
