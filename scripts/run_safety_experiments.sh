#!/bin/bash
model='meta-llama/Meta-Llama-3-8B-Instruct'
model_short="Meta-Llama-3-8B-Instruct"
dataset="safety"

out="results/${dataset}/${model_short}/out.log"
if [ ! -d "results/${dataset}/${model_short}" ]; then
	mkdir -p "results/${dataset}/${model_short}"
fi
python src/steering.py \
	--dataset ${dataset} \
	--dataset_dir "data/${dataset}" \
	--model_path "${model}" \
	--output_dir "results/${dataset}" \
	--batch_size 10 \
	--grid_layers \
	--use_fluency \
	--normalize_dir \
	>"$out" 2>&1

echo "Safety experiments complete."
