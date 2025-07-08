#!/usr/bin/env bash
model='meta-llama/Meta-Llama-3-8B-Instruct'
model_short="Meta-Llama-3-8B-Instruct"

out="results/toxicity/${model_short}/out.log"
if [ ! -d "results/toxicity/${model_short}" ]; then
    mkdir -p "results/toxicity/${model_short}"
fi
python src/steering.py \
    --dataset toxicity \
    --dataset_dir "data/toxicity" \
    --model_path "${model}" \
    --output_dir "results/toxicity" \
    --batch_size 10 \
    --grid_layers \
    --use_fluency \
    --normalize_dir \
    >"$out" 2>&1

echo "Toxicity experiments complete."