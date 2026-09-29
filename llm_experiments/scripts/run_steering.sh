#!/usr/bin/env bash
# Run the steering experiments (src/steering.py) on one or more tasks.
#
# Usage:
#   bash scripts/run_steering.sh <selector> [model] [extra steering.py args...]
#
# Selectors:
#   all                       every task below
#   emotions[:<emotion>]      anger disgust fear joy sadness surprise
#   ai-risk-mcq[:<behavior>]  power wealth
#   ai-risk-qa[:<behavior>]   power wealth
#   triviaqa | truthfulqa | safety | jbb | toxicity
#
# Examples:
#   bash scripts/run_steering.sh all                                   # Llama-3, all methods
#   bash scripts/run_steering.sh emotions:joy
#   METHODS="prompt instaboost spotlight" bash scripts/run_steering.sh all Qwen/Qwen3-14B
#   METHODS="prompt instaboost spotlight" bash scripts/run_steering.sh all openai/gpt-oss-20b
#
# Environment overrides:
#   METHODS               space-separated subset of steering.py --methods (default: all methods)
#   BATCH_SIZE            default 15 (Llama-3 / Gemma), 4 (Qwen3-14B), 2 (gpt-oss-20b)
#   MAX_TOKENS_GENERATED  override the dataset default
#   REASONING_EFFORT      low|medium|high (default: low for gpt-oss-20b)
#   THINKING_MODE         on|off (default: off for Qwen3)
#   DRY_RUN=1             print the commands without running them
#
# Each run writes to results/<task>/<model_short>/ and logs to out.log in that folder.
# Runs whose all_methods_results.json already exists are skipped.

set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

SELECTOR="${1:-all}"
[ "$#" -gt 0 ] && shift
MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
if [ "$#" -gt 0 ] && [ "${1#--}" = "$1" ]; then   # second positional (not a --flag) is the model
  MODEL="$1"; shift
fi
EXTRA_ARGS=("$@")
MODEL_SHORT="${MODEL##*/}"

EMOTIONS=(anger disgust fear joy sadness surprise)
BEHAVIORS=(power wealth)
SINGLE_TASKS=(triviaqa truthfulqa safety jbb toxicity)

if [ -z "${BATCH_SIZE:-}" ]; then
  case "$MODEL" in
    *gpt-oss*)  BATCH_SIZE=2 ;;
    *Qwen3*)    BATCH_SIZE=4 ;;
    *)          BATCH_SIZE=15 ;;
  esac
fi
if [ -z "${REASONING_EFFORT:-}" ]; then
  case "$MODEL" in *gpt-oss*) REASONING_EFFORT=low ;; esac
fi
if [ -z "${THINKING_MODE:-}" ]; then
  case "$MODEL" in *Qwen3*) THINKING_MODE=off ;; esac
fi

COMMON_ARGS=(--model_path "$MODEL" --batch_size "$BATCH_SIZE"
             --grid_layers --use_fluency --normalize_dir --trust_remote_code)
if [ -n "${METHODS:-}" ]; then
  # shellcheck disable=SC2206
  COMMON_ARGS+=(--methods $METHODS)
fi
if [ -n "${MAX_TOKENS_GENERATED:-}" ]; then
  COMMON_ARGS+=(--max_tokens_generated "$MAX_TOKENS_GENERATED")
fi
if [ -n "${REASONING_EFFORT:-}" ]; then
  COMMON_ARGS+=(--reasoning_effort "$REASONING_EFFORT")
fi
case "${THINKING_MODE:-}" in
  on)  COMMON_ARGS+=(--enable_thinking) ;;
  off) COMMON_ARGS+=(--disable_thinking) ;;
esac

FAILED=()

run_case() {
  local label="$1" dataset="$2" dataset_dir="$3" output_dir="$4"
  local run_dir="${output_dir}/${MODEL_SHORT}"
  local log="${run_dir}/out.log"
  if [ -f "${run_dir}/all_methods_results.json" ]; then
    echo "Skipping ${label} (${MODEL_SHORT}): results already exist in ${run_dir}"
    return
  fi
  mkdir -p "$run_dir"
  local cmd=(python src/steering.py --dataset "$dataset" --dataset_dir "$dataset_dir"
             --output_dir "$output_dir" "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}")
  echo "Running ${label} (${MODEL_SHORT}) -> ${run_dir}"
  if [ "${DRY_RUN:-0}" = "1" ]; then
    printf '  %q' "${cmd[@]}"; echo
    return
  fi
  if "${cmd[@]}" >"$log" 2>&1; then
    echo "Finished ${label}"
  else
    echo "Failed ${label}; see ${log}"
    FAILED+=("${label}: ${log}")
  fi
}

run_emotions() {
  local wanted=("$@"); [ "${#wanted[@]}" -eq 0 ] && wanted=("${EMOTIONS[@]}")
  for e in "${wanted[@]}"; do
    run_case "emotions:${e}" emotions-qa "data/emotions_train_${e}" "results/emotions/${e}"
  done
}

run_ai_risk() {
  local kind="$1"; shift
  local wanted=("$@"); [ "${#wanted[@]}" -eq 0 ] && wanted=("${BEHAVIORS[@]}")
  for b in "${wanted[@]}"; do
    run_case "ai-risk-${kind}:${b}" "ai-risk-${kind}" "data/ai-risk_${kind}_${b}-seeking-inclination" "results/ai-risk-${kind}/${b}"
  done
}

run_single() {
  run_case "$1" "$1" "data/$1" "results/$1"
}

case "$SELECTOR" in
  all)
    run_emotions
    run_ai_risk mcq
    run_ai_risk qa
    for t in "${SINGLE_TASKS[@]}"; do run_single "$t"; done
    ;;
  emotions)        run_emotions ;;
  emotions:*)      run_emotions "${SELECTOR#emotions:}" ;;
  ai-risk-mcq)     run_ai_risk mcq ;;
  ai-risk-mcq:*)   run_ai_risk mcq "${SELECTOR#ai-risk-mcq:}" ;;
  ai-risk-qa)      run_ai_risk qa ;;
  ai-risk-qa:*)    run_ai_risk qa "${SELECTOR#ai-risk-qa:}" ;;
  triviaqa|truthfulqa|safety|jbb|toxicity) run_single "$SELECTOR" ;;
  *)
    echo "Unknown selector: $SELECTOR" >&2
    echo "Expected: all | emotions[:<emotion>] | ai-risk-mcq[:<behavior>] | ai-risk-qa[:<behavior>] | triviaqa | truthfulqa | safety | jbb | toxicity" >&2
    exit 1
    ;;
esac

if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "One or more runs failed:"
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
