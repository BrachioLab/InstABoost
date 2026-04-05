#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

EMOTION_INPUT=${1:-joy}
MODEL=${2:-${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}}
BATCH_SIZE=${3:-${BATCH_SIZE:-}}
MAX_TOKENS_GENERATED=${4:-${MAX_TOKENS_GENERATED:-}}
REASONING_EFFORT=${5:-${REASONING_EFFORT:-}}
THINKING_MODE=${6:-${THINKING_MODE:-}}

if [ "$EMOTION_INPUT" = "all" ]; then
    SELECTOR="emotions"
else
    SELECTOR="emotions:${EMOTION_INPUT}"
fi

"$SCRIPT_DIR/run_prompt_methods.sh" \
    "$SELECTOR" \
    "$MODEL" \
    "$BATCH_SIZE" \
    "$MAX_TOKENS_GENERATED" \
    "$REASONING_EFFORT" \
    "$THINKING_MODE"
