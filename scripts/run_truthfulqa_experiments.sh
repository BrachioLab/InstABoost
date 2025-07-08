#!/usr/bin/env bash
model='meta-llama/Meta-Llama-3-8B-Instruct'
model_short="Meta-Llama-3-8B-Instruct"

out="results/truthfulqa/${model_short}/out.log"
if [ ! -d "results/truthfulqa/${model_short}" ]; then
    mkdir -p "results/truthfulqa/${model_short}"
fi
python src/steering.py \
    --dataset truthfulqa \
    --dataset_dir truthfulqa \
    --model_path "${model}" \
    --output_dir "results/truthfulqa" \
    --batch_size 10 \
    --grid_layers \
    --use_fluency \
    --normalize_dir \
    >"$out" 2>&1

echo "TruthfulQA experiments complete."