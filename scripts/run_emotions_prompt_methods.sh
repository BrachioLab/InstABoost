#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

if [ ! -d ".venv" ]; then
    echo "Missing local virtual environment at .venv"
    echo "Create it with: scripts/setup_local_env.sh"
    exit 1
fi

. "$REPO_ROOT/.venv/bin/activate"

python - <<'PY' >/dev/null
import importlib
for module in ("torch", "transformers", "transformer_lens", "datasets", "pandas", "sklearn", "unillm"):
    importlib.import_module(module)
PY

if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    . "$REPO_ROOT/.env"
    set +a
fi

EMOTION_INPUT=${1:-joy}
MODEL=${2:-${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}}
MODEL_SHORT=${MODEL##*/}
BATCH_SIZE=${3:-${BATCH_SIZE:-}}
MAX_TOKENS_GENERATED=${4:-${MAX_TOKENS_GENERATED:-}}
REASONING_EFFORT=${5:-${REASONING_EFFORT:-}}
THINKING_MODE=${6:-${THINKING_MODE:-}}
EMOTIONS="anger disgust fear joy sadness surprise"
MAX_TOKENS_ARGS=
REASONING_ARGS=
THINKING_ARGS=

if [ -z "$BATCH_SIZE" ]; then
    case "$MODEL" in
        *gpt-oss-20b*)
            BATCH_SIZE=2
            ;;
        *Qwen3-14B*|*14B*)
            BATCH_SIZE=4
            ;;
        *)
            BATCH_SIZE=15
            ;;
    esac
fi

if [ -n "${MAX_TOKENS_GENERATED:-}" ]; then
    MAX_TOKENS_ARGS="--max_tokens_generated $MAX_TOKENS_GENERATED"
fi

if [ -z "${REASONING_EFFORT:-}" ]; then
    case "$MODEL" in
        *gpt-oss-20b*)
            REASONING_EFFORT=low
            ;;
    esac
fi

if [ -n "${REASONING_EFFORT:-}" ]; then
    case "$REASONING_EFFORT" in
        low|medium|high) REASONING_ARGS="--reasoning_effort $REASONING_EFFORT" ;;
        *)
            echo "Unsupported reasoning effort: $REASONING_EFFORT"
            echo "Supported values: low medium high"
            exit 1
            ;;
    esac
fi

if [ -z "${THINKING_MODE:-}" ]; then
    case "$MODEL" in
        *Qwen3-*)
            THINKING_MODE=off
            ;;
    esac
fi

case "${THINKING_MODE:-}" in
    ""|default) ;;
    on) THINKING_ARGS="--enable_thinking" ;;
    off) THINKING_ARGS="--disable_thinking" ;;
    *)
        echo "Unsupported thinking mode: $THINKING_MODE"
        echo "Supported values: on off default"
        exit 1
        ;;
esac

if [ "$EMOTION_INPUT" = "all" ]; then
    EMOTIONS_TO_RUN=$EMOTIONS
else
    EMOTIONS_TO_RUN=$EMOTION_INPUT
fi

for EMOTION in $EMOTIONS_TO_RUN; do
    case " $EMOTIONS " in
        *" $EMOTION "*) ;;
        *)
            echo "Unsupported emotion: $EMOTION"
            echo "Supported emotions: $EMOTIONS"
            exit 1
            ;;
    esac

    OUTPUT_DIR="results/emotions/${EMOTION}"
    LOG_DIR="${OUTPUT_DIR}/${MODEL_SHORT}"
    LOG_FILE="${LOG_DIR}/out_${EMOTION}.log"
    RESULTS_FILE="${LOG_DIR}/all_methods_results.json"

    mkdir -p "$LOG_DIR"

    if [ -f "$RESULTS_FILE" ]; then
        echo "Skipping ${EMOTION} for ${MODEL}: found existing results at ${RESULTS_FILE}"
        continue
    fi

    python src/steering.py \
        --dataset emotions-qa \
        --dataset_dir "data/emotions_train_${EMOTION}" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size "$BATCH_SIZE" \
        --model_path "$MODEL" \
        --grid_layers \
        --use_fluency \
        --normalize_dir \
        --trust_remote_code \
        --methods prompt prompt-attention \
        $MAX_TOKENS_ARGS \
        $REASONING_ARGS \
        $THINKING_ARGS \
        >"$LOG_FILE" 2>&1

    echo "Finished emotions experiment for ${EMOTION}"
    echo "Log: $LOG_FILE"
    echo "Model: $MODEL"
    echo "Batch size: $BATCH_SIZE"
    if [ -n "${MAX_TOKENS_GENERATED:-}" ]; then
        echo "Max tokens generated: $MAX_TOKENS_GENERATED"
    fi
    if [ -n "${REASONING_EFFORT:-}" ]; then
        echo "Reasoning effort: $REASONING_EFFORT"
    fi
    if [ -n "${THINKING_MODE:-}" ]; then
        echo "Thinking mode: $THINKING_MODE"
    fi
done
