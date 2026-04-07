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

SELECTOR=${1:-all}
MODEL=${2:-${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}}
MODEL_SHORT=${MODEL##*/}
BATCH_SIZE=${3:-${BATCH_SIZE:-}}
MAX_TOKENS_GENERATED=${4:-${MAX_TOKENS_GENERATED:-}}
REASONING_EFFORT=${5:-${REASONING_EFFORT:-}}
THINKING_MODE=${6:-${THINKING_MODE:-}}

EMOTIONS="anger disgust fear joy sadness surprise"
AI_RISK_BEHAVIORS="power wealth"
DATASET_SELECTORS="all emotions ai-risk-mcq ai-risk-qa triviaqa truthfulqa safety jbb toxicity"
MAX_TOKENS_ARGS=
REASONING_ARGS=
THINKING_ARGS=
FAILED_CASES=

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

run_case() {
    CASE_LABEL=$1
    DATASET=$2
    DATASET_DIR=$3
    OUTPUT_DIR=$4
    LOG_FILE=$5

    LOG_DIR=$(dirname "$LOG_FILE")
    RESULTS_FILE="${OUTPUT_DIR}/${MODEL_SHORT}/all_methods_results.json"

    mkdir -p "$LOG_DIR"

    if [ -f "$RESULTS_FILE" ]; then
        echo "Skipping ${CASE_LABEL} for ${MODEL}: found existing results at ${RESULTS_FILE}"
        return
    fi

    if python src/steering.py \
        --dataset "$DATASET" \
        --dataset_dir "$DATASET_DIR" \
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
    then
        echo "Finished ${CASE_LABEL}"
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
    else
        echo "Failed ${CASE_LABEL} for ${MODEL}"
        echo "Log: $LOG_FILE"
        if [ -n "$FAILED_CASES" ]; then
            FAILED_CASES="${FAILED_CASES}
${CASE_LABEL}: ${LOG_FILE}"
        else
            FAILED_CASES="${CASE_LABEL}: ${LOG_FILE}"
        fi
    fi
}

run_emotions() {
    EMOTION_SELECTOR=${1:-all}

    if [ "$EMOTION_SELECTOR" = "all" ]; then
        EMOTIONS_TO_RUN=$EMOTIONS
    else
        EMOTIONS_TO_RUN=$EMOTION_SELECTOR
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
        LOG_FILE="${OUTPUT_DIR}/${MODEL_SHORT}/out_${EMOTION}.log"
        run_case "emotions:${EMOTION}" "emotions-qa" "data/emotions_train_${EMOTION}" "$OUTPUT_DIR" "$LOG_FILE"
    done
}

run_ai_risk() {
    DATASET=$1
    BEHAVIOR_SELECTOR=${2:-all}

    if [ "$BEHAVIOR_SELECTOR" = "all" ]; then
        BEHAVIORS_TO_RUN=$AI_RISK_BEHAVIORS
    else
        BEHAVIORS_TO_RUN=$BEHAVIOR_SELECTOR
    fi

    for BEHAVIOR in $BEHAVIORS_TO_RUN; do
        case " $AI_RISK_BEHAVIORS " in
            *" $BEHAVIOR "*) ;;
            *)
                echo "Unsupported ai-risk behavior: $BEHAVIOR"
                echo "Supported behaviors: $AI_RISK_BEHAVIORS"
                exit 1
                ;;
        esac

        case "$DATASET" in
            ai-risk-mcq)
                DATASET_DIR="data/ai-risk_mcq_${BEHAVIOR}-seeking-inclination"
                OUTPUT_DIR="results/ai-risk-mcq/${BEHAVIOR}"
                LOG_FILE="${OUTPUT_DIR}/${MODEL_SHORT}/out_${BEHAVIOR}.log"
                ;;
            ai-risk-qa)
                DATASET_DIR="data/ai-risk_qa_${BEHAVIOR}-seeking-inclination"
                OUTPUT_DIR="results/ai-risk-qa/${BEHAVIOR}"
                LOG_FILE="${OUTPUT_DIR}/${MODEL_SHORT}/out_${BEHAVIOR}.log"
                ;;
            *)
                echo "Unsupported ai-risk dataset: $DATASET"
                exit 1
                ;;
        esac

        run_case "${DATASET}:${BEHAVIOR}" "$DATASET" "$DATASET_DIR" "$OUTPUT_DIR" "$LOG_FILE"
    done
}

run_single_dataset() {
    DATASET=$1

    case "$DATASET" in
        triviaqa|truthfulqa|safety|jbb|toxicity) ;;
        *)
            echo "Unsupported dataset: $DATASET"
            echo "Supported selectors: $DATASET_SELECTORS, emotions:<emotion>, ai-risk-mcq:<behavior>, ai-risk-qa:<behavior>"
            exit 1
            ;;
    esac

    OUTPUT_DIR="results/${DATASET}"
    LOG_FILE="${OUTPUT_DIR}/${MODEL_SHORT}/out.log"
    run_case "$DATASET" "$DATASET" "data/${DATASET}" "$OUTPUT_DIR" "$LOG_FILE"
}

case "$SELECTOR" in
    all)
        run_emotions all
        run_ai_risk ai-risk-mcq all
        run_ai_risk ai-risk-qa all
        for DATASET in triviaqa truthfulqa safety jbb toxicity; do
            run_single_dataset "$DATASET"
        done
        ;;
    emotions)
        run_emotions all
        ;;
    emotions:*)
        run_emotions "${SELECTOR#emotions:}"
        ;;
    ai-risk-mcq)
        run_ai_risk ai-risk-mcq all
        ;;
    ai-risk-mcq:*)
        run_ai_risk ai-risk-mcq "${SELECTOR#ai-risk-mcq:}"
        ;;
    ai-risk-qa)
        run_ai_risk ai-risk-qa all
        ;;
    ai-risk-qa:*)
        run_ai_risk ai-risk-qa "${SELECTOR#ai-risk-qa:}"
        ;;
    anger|disgust|fear|joy|sadness|surprise)
        run_emotions "$SELECTOR"
        ;;
    power|wealth)
        echo "Ambiguous selector: $SELECTOR"
        echo "Use ai-risk-mcq:${SELECTOR} or ai-risk-qa:${SELECTOR}"
        exit 1
        ;;
    triviaqa|truthfulqa|safety|jbb|toxicity)
        run_single_dataset "$SELECTOR"
        ;;
    *)
        echo "Unsupported selector: $SELECTOR"
        echo "Supported selectors: $DATASET_SELECTORS"
        echo "Also supported: emotions:<emotion>, ai-risk-mcq:<behavior>, ai-risk-qa:<behavior>"
        exit 1
        ;;
esac

if [ -n "$FAILED_CASES" ]; then
    echo "One or more runs failed:"
    printf '%s\n' "$FAILED_CASES"
    exit 1
fi
