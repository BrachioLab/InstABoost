#!/usr/bin/env bash
model='meta-llama/Meta-Llama-3-8B-Instruct'
model_short="Meta-Llama-3-8B-Instruct"

out="results/triviaqa/${model_short}/out.log"
if [ ! -d "results/triviaqa/${model_short}" ]; then
    mkdir -p "results/triviaqa/${model_short}"
fi
python src/steering.py \
    --dataset triviaqa \
    --dataset_dir triviaqa \
    --model_path "${model}" \
    --output_dir "results/triviaqa" \
    --batch_size 10 \
    --grid_layers \
    --use_fluency \
    --normalize_dir \
    >"$out" 2>&1

echo "TriviaQA experiments complete."