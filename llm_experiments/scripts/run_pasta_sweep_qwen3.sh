#!/usr/bin/env bash
# Run the PASTA per-(layer, head) scoring sweep for Qwen3 on a single task.
#
# Usage:
#   bash scripts/run_pasta_sweep_qwen3.sh <task>
#   bash scripts/run_pasta_sweep_qwen3.sh <task> --layer_start 0 --layer_end 20
#
# <task> must be one of DATASET_CHOICES in src/data_utils.py:
#   triviaqa | truthfulqa | safety | jbb | toxicity | ai-risk-mcq | ai-risk-qa | emotions-qa
#
# To shard across machines, give each instance a non-overlapping
# [layer_start, layer_end) range — output goes to a shard-suffixed JSON so
# they don't race on the same file. Merge before running pasta_head_selection.py.

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <task> [extra args]" >&2
  exit 1
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

TASK="$1"
shift

MODEL="Qwen/Qwen3-14B"
OUTPUT_DIR="results_pasta"
BATCH_SIZE=8
ALPHA=0.01

EXTRA_ARGS=("$@")

echo "=========================================="
echo "PASTA head selection (Qwen3)"
echo "Model:     ${MODEL}"
echo "Task:      ${TASK}"
echo "Output:    ${OUTPUT_DIR}/${MODEL##*/}/pasta/"
echo "Batch:     ${BATCH_SIZE}"
echo "Alpha:     ${ALPHA}"
echo "Started:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

python -u src/pasta_head_selection_qwen.py \
  --model_path "${MODEL}" \
  --output_dir "${OUTPUT_DIR}" \
  --dataset "${TASK}" \
  --batch_size "${BATCH_SIZE}" \
  --alpha "${ALPHA}" \
  "${EXTRA_ARGS[@]}"

echo ""
echo "Finished:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
