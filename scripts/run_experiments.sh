#!/bin/bash
model='meta-llama/Meta-Llama-3-8B-Instruct'
model_short="Meta-Llama-3-8B-Instruct"
base_dir="results"

echo "================================================"
echo "Running experiments..."
echo "================================================"

# Emotions
echo "Running emotions experiments..."
emotions=(anger disgust fear joy sadness surprise)
for emotion in "${emotions[@]}"; do
    if [ ! -d "${base_dir}/emotions/${emotion}/${model_short}" ]; then
        mkdir -p "${base_dir}/emotions/${emotion}/${model_short}"
    fi
    out="${base_dir}/emotions/${emotion}/${model_short}/out_${emotion}.log"
    echo "Running $emotion..."
    python src/steering.py --dataset emotions-qa \
        --dataset_dir "data/emotions_train_${emotion}" \
        --output_dir "${base_dir}/emotions/${emotion}" \
        --batch_size 15 \
        --model_path "${model}" \
        --grid_layers \
        --use_fluency \
        --normalize_dir \
        > "$out" 2>&1
    echo "Finished $emotion"
done
echo "All emotions experiments complete."
echo "------------------------------------------------"

# AI Persona
echo "Running ai-persona experiments..."
behaviors=(power wealth)
qtypes=(mcq qa)
for qtype in "${qtypes[@]}"; do
    for be in "${behaviors[@]}"; do
        if [ ! -d "${base_dir}/ai-risk-${qtype}/${be}/${model_short}" ]; then
            mkdir -p "${base_dir}/ai-risk-${qtype}/${be}/${model_short}"
        fi
        out="${base_dir}/ai-risk-${qtype}/${be}/${model_short}/out_${be}.log"
        echo "Running ${qtype} - ${be}..."
        python src/steering.py --dataset ai-risk-${qtype} \
            --dataset_dir "data/ai-risk_${qtype}_${be}-seeking-inclination" \
        --output_dir "${base_dir}/ai-risk-${qtype}/${be}" \
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
echo "------------------------------------------------"

# Others
others=(triviaqa truthfulqa safety jbb toxicity)
for other in "${others[@]}"; do
    echo "Running ${other} experiments..."
    out="${base_dir}/${other}/${model_short}/out.log"
    if [ ! -d "${base_dir}/${other}/${model_short}" ]; then
        mkdir -p "${base_dir}/${other}/${model_short}"
    fi
    python src/steering.py \
        --dataset ${other} \
        --dataset_dir "data/${other}" \
        --model_path "${model}" \
        --output_dir "${base_dir}/${other}" \
        --batch_size 15 \
        --grid_layers \
        --use_fluency \
        --normalize_dir \
        >"$out" 2>&1
    echo "${other} experiments complete."
    echo "------------------------------------------------"
done
echo "All other experiments complete."
echo "================================================"
echo "All experiments complete."
echo "================================================"
