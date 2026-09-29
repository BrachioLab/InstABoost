#!/usr/bin/env bash
# Run the IFEval experiment (prompt / InstABoost / Spotlight / PASTA) for one model
# with the settings used for the reported results.
#
# Usage:
#   bash scripts/run_ifeval.sh <llama3|gemma|qwen3|gpt-oss> [extra run_ifeval_experiment.py args...]
#
# Examples:
#   bash scripts/run_ifeval.sh llama3
#   bash scripts/run_ifeval.sh qwen3 --n_val 10 --n_test 20      # quick check
#   METHODS="prompt instaboost" bash scripts/run_ifeval.sh gemma  # subset of methods
#
# Environment overrides: OUTPUT_DIR, N_VAL, N_TEST, BATCH_SIZE, MAX_TOKENS_GENERATED,
# METHODS, PASTA_HEAD_CONFIG, SEED. Set DRY_RUN=1 to print the command without running.
#
# Outputs go to ${OUTPUT_DIR}/<model_short>/<method>/ and can be summarized with
# src/build_ifeval_summary.py, src/ifeval_paired_stats.py and
# src/build_ifeval_per_instruction_summary.py.

set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <llama3|gemma|qwen3|gpt-oss> [extra args...]" >&2
  exit 1
fi
MODEL_KEY="$1"
shift
EXTRA_ARGS=("$@")

# Shared settings
OUTPUT_DIR="${OUTPUT_DIR:-results/ifeval}"
N_VAL="${N_VAL:-100}"
SEED="${SEED:-42}"
PASTA_ALPHA="${PASTA_ALPHA:-0.01}"
MODEL_FLAGS=()

# Per-model settings (batch size only affects memory, not results)
case "$MODEL_KEY" in
  llama3)
    MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
    N_TEST="${N_TEST:-400}"
    BATCH_SIZE="${BATCH_SIZE:-10}"
    MAX_TOKENS_GENERATED="${MAX_TOKENS_GENERATED:-4000}"
    METHODS="${METHODS:-prompt instaboost spotlight pasta}"
    PASTA_HEAD_CONFIG="${PASTA_HEAD_CONFIG:-configs/pasta_ifeval/Meta-Llama-3-8B-Instruct.json}"
    ;;
  gemma)
    MODEL="google/gemma-7b-it"
    # 500 requested test samples = all 441 prompts left after the validation split.
    N_TEST="${N_TEST:-500}"
    BATCH_SIZE="${BATCH_SIZE:-10}"
    MAX_TOKENS_GENERATED="${MAX_TOKENS_GENERATED:-4000}"
    METHODS="${METHODS:-prompt instaboost spotlight pasta}"
    PASTA_HEAD_CONFIG="${PASTA_HEAD_CONFIG:-configs/pasta_ifeval/gemma-7b-it.json}"
    ;;
  qwen3)
    MODEL="Qwen/Qwen3-14B"
    N_TEST="${N_TEST:-400}"
    BATCH_SIZE="${BATCH_SIZE:-5}"
    MAX_TOKENS_GENERATED="${MAX_TOKENS_GENERATED:-4000}"
    METHODS="${METHODS:-prompt instaboost spotlight pasta}"
    PASTA_HEAD_CONFIG="${PASTA_HEAD_CONFIG:-configs/pasta_ifeval/Qwen3-14B.json}"
    MODEL_FLAGS=(--disable_thinking --trust_remote_code --instaboost_multipliers 3 5 7 9)
    ;;
  gpt-oss)
    MODEL="openai/gpt-oss-20b"
    N_TEST="${N_TEST:-400}"
    BATCH_SIZE="${BATCH_SIZE:-8}"
    MAX_TOKENS_GENERATED="${MAX_TOKENS_GENERATED:-1524}"
    METHODS="${METHODS:-prompt instaboost spotlight}"
    PASTA_HEAD_CONFIG="${PASTA_HEAD_CONFIG:-}"   # no PASTA head config for GPT-OSS
    MODEL_FLAGS=(--reasoning_effort low --trust_remote_code
                 --instaboost_hook_mode mass_preserving --instaboost_kv_cache on
                 --instaboost_multipliers 3 5 7 11)
    ;;
  *)
    echo "Unknown model key: $MODEL_KEY (expected llama3, gemma, qwen3 or gpt-oss)" >&2
    exit 1
    ;;
esac

# shellcheck disable=SC2206
METHOD_LIST=($METHODS)
PASTA_ARGS=(--skip_pasta)
if [ -n "$PASTA_HEAD_CONFIG" ]; then
  PASTA_ARGS=(--pasta_head_config "$PASTA_HEAD_CONFIG" --pasta_alpha "$PASTA_ALPHA")
fi

CMD=(python src/run_ifeval_experiment.py
  --model_path "$MODEL"
  --output_dir "$OUTPUT_DIR"
  --n_val "$N_VAL"
  --n_test "$N_TEST"
  --batch_size "$BATCH_SIZE"
  --max_tokens_generated "$MAX_TOKENS_GENERATED"
  --seed "$SEED"
  --methods "${METHOD_LIST[@]}"
  "${PASTA_ARGS[@]}"
  "${MODEL_FLAGS[@]}"
  "${EXTRA_ARGS[@]}")

echo "=========================================="
echo "IFEval experiment"
echo "Model:       ${MODEL}"
echo "Methods:     ${METHOD_LIST[*]}"
echo "n_val/test:  ${N_VAL} / ${N_TEST}"
echo "batch_size:  ${BATCH_SIZE}"
echo "max_tokens:  ${MAX_TOKENS_GENERATED}"
echo "output_dir:  ${OUTPUT_DIR}"
echo "Started at:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

if [ "${DRY_RUN:-0}" = "1" ]; then
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi
"${CMD[@]}"

echo ""
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
