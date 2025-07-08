#!/bin/bash
model='meta-llama/Meta-Llama-3-8B-Instruct'
model_short="Meta-Llama-3-8B-Instruct"
behaviors=(power wealth)
qtypes=(mcq qa)

for qtype in "${qtypes[@]}"; do
    for be in "${behaviors[@]}"; do
        if [ ! -d "results/ai-risk-${qtype}/${be}/${model_short}" ]; then
            mkdir -p "results/ai-risk-${qtype}/${be}/${model_short}"
        fi
        out="results/ai-risk-${qtype}/${be}/${model_short}/out_${be}.log"
        echo "Running ${qtype} - ${be}..."
        python src/steering.py --dataset ai-risk-${qtype} \
            --dataset_dir "data/ai-risk_${qtype}_${be}-seeking-inclination" \
            --output_dir "results/ai-risk-${qtype}/${be}" \
            --batch_size 10 \
            --model_path "${model}" \
            --grid_layers \
            --use_fluency \
            --normalize_dir \
            > "$out" 2>&1
        echo "Finished ${qtype} - ${be}"
    done
done

echo "All ai-risk experiments complete."
