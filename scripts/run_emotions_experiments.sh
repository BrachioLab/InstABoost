#!/bin/bash
model='meta-llama/Meta-Llama-3-8B-Instruct'
model_short="Meta-Llama-3-8B-Instruct"
emotions=(anger disgust fear joy sadness surprise)

for emotion in "${emotions[@]}"; do
    if [ ! -d "results/emotions/${emotion}/${model_short}" ]; then
        mkdir -p "results/emotions/${emotion}/${model_short}"
    fi
    out="results/emotions/${emotion}/${model_short}/out_${emotion}.log"
    echo "Running $emotion..."
    python src/steering.py --dataset emotions-qa \
        --dataset_dir "data/emotions_train_${emotion}" \
        --output_dir "results/emotions/${emotion}" \
        --batch_size 15 \
        --model_path "${model}" \
        --grid_layers \
        --use_fluency \
        --normalize_dir \
        > "$out" 2>&1
    echo "Finished $emotion"
done

echo "All emotions experiments complete."
