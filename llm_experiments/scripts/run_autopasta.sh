#!/usr/bin/env bash
# AutoPASTA head selection (coarse-to-fine layer-then-head sweep) for one or
# more tasks. Produces the {layer: [heads]} configs consumed by --pasta_head_config.
#
# Usage:
#   bash scripts/run_autopasta.sh <task...> [extra python args...]
#
# Tasks:
#   ifeval                    (src/autopasta_head_selection_ifeval.py)
#   emotions[:<emotion>]      anger disgust fear joy sadness surprise
#   ai-risk-mcq[:<behavior>]  power wealth
#   ai-risk-qa[:<behavior>]   power wealth
#   safety | jbb              (load Llama-Guard-3-8B for evaluation: needs HF auth)
#   all                       every task above except ifeval
#
# Examples:
#   bash scripts/run_autopasta.sh ifeval
#   MODEL=google/gemma-7b-it bash scripts/run_autopasta.sh ifeval
#   bash scripts/run_autopasta.sh emotions:anger ai-risk-mcq:power
#   bash scripts/run_autopasta.sh safety --phase 2               # resume at phase 2
#
# Environment overrides: MODEL (default Qwen/Qwen3-14B), OUTPUT_DIR, BATCH_SIZE,
# TOP_L, TOP_K, ALPHA, SEED. Set DRY_RUN=1 to print the commands without running.
#
# Outputs: ${OUTPUT_DIR}/<model_short>/<task>/autopasta/{layer_selection,head_selection,
# head_config_top<k>}.json (IFEval: results_pasta/ifeval_autopasta/<model_short>/autopasta/).

set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <task...> [extra args...]" >&2
  exit 1
fi

MODEL="${MODEL:-Qwen/Qwen3-14B}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ALPHA="${ALPHA:-0.01}"
SEED="${SEED:-42}"
case "$MODEL" in
  *Qwen3*) TOP_L="${TOP_L:-8}"; TOP_K="${TOP_K:-100}" ;;
  *)       TOP_L="${TOP_L:-6}"; TOP_K="${TOP_K:-50}" ;;
esac

EMOTIONS=(anger disgust fear joy sadness surprise)
BEHAVIORS=(power wealth)
TASKS=()
EXTRA_ARGS=()
for arg in "$@"; do
  case "$arg" in
    ifeval|all|emotions|emotions:*|ai-risk-mcq|ai-risk-mcq:*|ai-risk-qa|ai-risk-qa:*|safety|jbb)
      TASKS+=("$arg") ;;
    *) EXTRA_ARGS+=("$arg") ;;
  esac
done
if [ "${#TASKS[@]}" -eq 0 ]; then
  echo "No task given. Expected: ifeval | all | emotions[:<emotion>] | ai-risk-mcq[:<behavior>] | ai-risk-qa[:<behavior>] | safety | jbb" >&2
  exit 1
fi

run_cmd() {
  echo "------------------------------------------"
  echo ">> $1"
  shift
  if [ "${DRY_RUN:-0}" = "1" ]; then
    printf '  %q' "$@"; echo
  else
    "$@"
  fi
}

run_standard() {  # label dataset [dataset_dir]
  local label="$1" dataset="$2" dataset_dir="${3:-}"
  local args=(python -u src/autopasta_head_selection.py
              --model_path "$MODEL" --output_dir "${OUTPUT_DIR:-results_pasta/autopasta}"
              --dataset "$dataset" --task_label "$label"
              --batch_size "$BATCH_SIZE" --top_l "$TOP_L" --top_k "$TOP_K"
              --alpha "$ALPHA" --seed "$SEED")
  [ -n "$dataset_dir" ] && args+=(--dataset_dir "$dataset_dir")
  run_cmd "$label" "${args[@]}" "${EXTRA_ARGS[@]}"
}

run_emotions() {
  local wanted=("$@"); [ "${#wanted[@]}" -eq 0 ] && wanted=("${EMOTIONS[@]}")
  for e in "${wanted[@]}"; do run_standard "emotions-${e}" emotions-qa "emotions_train_${e}"; done
}

run_ai_risk() {
  local kind="$1"; shift
  local wanted=("$@"); [ "${#wanted[@]}" -eq 0 ] && wanted=("${BEHAVIORS[@]}")
  for b in "${wanted[@]}"; do
    run_standard "ai-risk-${kind}_${b}-seeking-inclination" "ai-risk-${kind}" "ai-risk-${kind}_${b}-seeking-inclination"
  done
}

if [ -z "${HF_TOKEN:-}" ] && [ ! -f "${HOME}/.cache/huggingface/token" ]; then
  case " ${TASKS[*]} " in
    *" safety "*|*" jbb "*|*" all "*)
      echo "WARNING: safety/jbb evaluation loads the gated meta-llama/Llama-Guard-3-8B; no HF auth detected." >&2 ;;
  esac
fi

echo "=========================================="
echo "AutoPASTA head selection"
echo "Model:    ${MODEL}"
echo "Tasks:    ${TASKS[*]}"
echo "Settings: batch=${BATCH_SIZE} top_l=${TOP_L} top_k=${TOP_K} alpha=${ALPHA} seed=${SEED}"
echo "Started:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

for task in "${TASKS[@]}"; do
  case "$task" in
    ifeval)
      run_cmd ifeval python -u src/autopasta_head_selection_ifeval.py \
        --model_path "$MODEL" --output_dir "${OUTPUT_DIR:-results_pasta/ifeval_autopasta}" \
        --batch_size "$BATCH_SIZE" --top_l "$TOP_L" --top_k "$TOP_K" \
        --alpha "$ALPHA" --seed "$SEED" "${EXTRA_ARGS[@]}" ;;
    all)            run_emotions; run_ai_risk mcq; run_ai_risk qa; run_standard safety safety; run_standard jbb jbb ;;
    emotions)       run_emotions ;;
    emotions:*)     run_emotions "${task#emotions:}" ;;
    ai-risk-mcq)    run_ai_risk mcq ;;
    ai-risk-mcq:*)  run_ai_risk mcq "${task#ai-risk-mcq:}" ;;
    ai-risk-qa)     run_ai_risk qa ;;
    ai-risk-qa:*)   run_ai_risk qa "${task#ai-risk-qa:}" ;;
    safety|jbb)     run_standard "$task" "$task" ;;
  esac
done

echo ""
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
