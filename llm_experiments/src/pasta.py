import torch
import functools
import pandas as pd
import textwrap
import gc
import csv
import time
from datetime import datetime

from pastalib.pasta import PASTA 

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer, utils
from colorama import Fore
import os
import json
from argparse import ArgumentParser
import numpy as np
import random

from data_utils import (
    get_dataset,
    DATASET_CHOICES,
    DATASET_CONFIGS
)
from model_interaction_utils import (
    tokenize_instructions,
    get_generations,
    get_hiddens
)

from evaluation_utils import (
    evaluate_generations, 
    load_existing_results, 
    save_all_methods_results, 
    run_evaluation_pipeline, 
    print_evaluation_results
)


if __name__ == "__main__":
    start_time = time.time()
    print(f"Starting script at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--output_dir", type=str, default="qa_results_for_steering", help="Output directory for steering results")
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="triviaqa", choices=DATASET_CHOICES, help="Dataset to use for evaluation")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Directory containing the dataset results")
    parser.add_argument("--baseline_generations_dir", type=str, default=None, help="Directory containing baseline generations (should contain results.csv)")
    parser.add_argument("--answer_file_name", type=str, default="new_answer.txt", help="Name of the file containing the answers")
    # Steering arguments
    parser.add_argument("--steering_type", type=str, default="refusal", choices=["refusal", "random", "mean", "prompt", "pca", "repe", "linear", "instaboost"], help="Type of steering to perform")
    parser.add_argument("--steer_to_pt", action="store_true", help="Steer to PT (defaults to CT)")
    parser.add_argument("--cache_harmless", action="store_true", help="Cache harmless instructions. By default, only harmful instructions are cached.")
    # Misc arguments
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--use_fluency", action="store_true", help="Whether to calculate and use fluency in steering")
    parser.add_argument("--use_relevance", action="store_true", help="Whether to calculate and use relevance in steering")
    # PASTA arguments
    parser.add_argument("--alpha", type=float, default=0.01, help="Alpha value for PASTA steering")
    args = parser.parse_args()

    dataset = args.dataset
    MODEL_PATH = args.model_path
    output_dir = args.output_dir
    DEVICE = args.device
    batch_size = args.batch_size
    max_tokens_generated = DATASET_CONFIGS[dataset]["max_tokens_generated"]
    N_INST_TRAIN = DATASET_CONFIGS[dataset]["n_inst_train"]
    N_INST_TEST = DATASET_CONFIGS[dataset]["n_inst_test"]
    N_INST_VAL = DATASET_CONFIGS[dataset]["n_inst_val"]
    ALPHA = args.alpha

    assert dataset in DATASET_CONFIGS, f"Dataset {dataset} not found in DATASET_CONFIGS, please update the DATASET_CONFIGS dictionary in data_utils.py"

    task_type = DATASET_CONFIGS[dataset]["task_type"]
    eval_type = DATASET_CONFIGS[dataset]["eval_type"]
    apply_chat_template = DATASET_CONFIGS[dataset].get("apply_chat_template", True)

    output_dir = os.path.join(output_dir, f"{MODEL_PATH.split('/')[1]}")
    if args.baseline_generations_dir is None:
        baseline_generations_dir = output_dir
    else:
        baseline_generations_dir = args.baseline_generations_dir

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Ensure padding is on the left side for causal models
    model.tokenizer = tokenizer

    tokenize_instructions_fn = functools.partial(tokenize_instructions, tokenizer=tokenizer, model_name=MODEL_PATH, apply_chat_template=apply_chat_template)
    
    # Get total number of layers
    # total_layers = model.cfg.n_layers
    total_layers = model.config.num_hidden_layers
    # Get the number of heads per layer
    # n_heads = model.cfg.n_heads
    n_heads = model.config.num_attention_heads

    print(f"Number of layers: {total_layers}, Number of heads per layer: {n_heads}")

    # 1. Load dataset
    print(50*'=' + '\n')
    print(f"### Loading dataset: {dataset} ###")
    (harmful_inst_train, harmless_inst_train, steer_dataset, steer_val_dataset, steer_questions, steer_answers, steer_alt_answers,
     steer_val_questions, steer_val_answers, steer_val_alt_answers, eval_args, steer_prompt) = get_dataset(
        dataset=dataset,
        dataset_dir=args.dataset_dir,
        answer_file_name=args.answer_file_name,
        n_inst_train=N_INST_TRAIN,
        n_inst_test=N_INST_TEST,
        n_inst_val=N_INST_VAL,
        steer_to_pt=args.steer_to_pt,
        tokenizer=tokenizer,
        model=model,
        baseline_generations_dir=baseline_generations_dir,
        cache_harmless=args.cache_harmless,
        tokenize_instructions_fn=tokenize_instructions_fn,
        max_tokens_generated=max_tokens_generated,
        batch_size=batch_size
    )
    print('Done loading dataset. \n')
    print(50*'=' + '\n')

    print(f"N_INST_TRAIN: {N_INST_TRAIN}, N_INST_TEST: {N_INST_TEST}, N_INST_VAL: {N_INST_VAL}")
    print(f"Len of harmful_inst_train: {len(harmful_inst_train)}, Len of harmless_inst_train: {len(harmless_inst_train)}")
    print(f"Len of steer_dataset: {len(steer_dataset)}, Len of steer_val_dataset: {len(steer_val_dataset)}")
    print(f"Len of steer_questions: {len(steer_questions)}, Len of steer_answers: {len(steer_answers)}, Len of steer_alt_answers: {len(steer_alt_answers)}")
    print(f"Len of steer_val_questions: {len(steer_val_questions)}, Len of steer_val_answers: {len(steer_val_answers)}, Len of steer_val_alt_answers: {len(steer_val_alt_answers)}")
    print(f"Eval args: {eval_args}")

    print('\n' + 50*'=')
    print(f'### Running PASTA Head Selection ###')
    print(50*'=' + '\n')

    pasta_start = time.time()
    pasta_output_dir = os.path.join(args.output_dir, f"{MODEL_PATH.split('/')[1]}", "pasta")
    os.makedirs(pasta_output_dir, exist_ok=True)

    # Check if a json file with head_selection_results exists
    pasta_layer_head_selection_results_path = os.path.join(pasta_output_dir, "layer_head_selection_results.json")
    pasta_layer_head_selection_results = {}
    if os.path.exists(pasta_layer_head_selection_results_path):
        print(f"Found existing head selection results at {pasta_layer_head_selection_results_path}. Skipping re-evaluation for heads / layers already processed.")
        with open(pasta_layer_head_selection_results_path, 'r') as f:
            pasta_layer_head_selection_results = json.load(f)

    for layer in range(total_layers):
        for head in range(n_heads):
            print(f"\nProcessing layer {layer}, head {head} for PASTA head selection")
            if str(layer) in pasta_layer_head_selection_results and str(head) in pasta_layer_head_selection_results[str(layer)]:
                print(f"Skipping layer {layer}, head {head} - already processed.")
                continue

            head_time_start = time.time()
            
            # PASTA head config
            head_config = {
                f"{layer}": [head]
            }

            pasta = PASTA(
                model=model,
                tokenizer=tokenizer,
                head_config=head_config,
                alpha=ALPHA,
                scale_position="exclude"
            )

            # Run PASTA
            emphasized_texts = [steer_prompt]
            texts = [steer_prompt + sample for sample in steer_val_dataset]
            
            # Convert each instruction into a chat message format
            messages = [{"role": "user", "content": text} for text in texts]
            
            # Apply the chat template using the tokenizer's built-in method
            if apply_chat_template and hasattr(tokenizer, "apply_chat_template"):
                print(f"Using tokenizer's apply_chat_template method to format instructions.")
                texts = [tokenizer.apply_chat_template([msg], tokenize=False, add_generation_prompt=True) for msg in messages]

            val_steer_generations = []

            for i in range(0, len(steer_val_dataset), batch_size):
                print(f"Processing batch {i // batch_size + 1} of {len(steer_val_dataset) // batch_size + 1}")
                emphasized_texts = [steer_prompt] * len(texts[i:i + batch_size])

                inputs, offset_mapping = pasta.inputs_from_batch(texts[i:i + batch_size])
                inputs = inputs.to(DEVICE)

                with pasta.apply_steering(
                    model=model,
                    strings=texts[i:i + batch_size],
                    substrings=emphasized_texts,
                    model_input=inputs,
                    offsets_mapping=offset_mapping,
                ) as steered_model:
                    val_steer_generations_batch = steered_model.generate(**inputs, max_new_tokens=max_tokens_generated)
                # Decode the generated tokens
                val_steer_generations_batch = val_steer_generations_batch[:, inputs.input_ids.shape[1]:]  # Remove input tokens
                val_steer_generations.extend(
                    tokenizer.batch_decode(val_steer_generations_batch, skip_special_tokens=True)
                )

            print(val_steer_generations[:5])  # Print first 5 generations for debugging

            val_results = evaluate_generations(
                    generations=val_steer_generations,
                    dataset=dataset,
                    eval_type=eval_type,
                    eval_args=eval_args,
                    questions=steer_val_questions,
                    answers=steer_val_answers,
                    alt_answers=steer_val_alt_answers,
                    is_validation=True,
                    use_fluency=args.use_fluency,
                    use_relevance=args.use_relevance,
                    steer_dataset=steer_val_dataset
                )
            
            head_time_end = time.time()
            
            score = val_results["score"]
            print(f"Validation score for layer {layer}, head {head}: {score:.3f}")
            pasta_layer_head_selection_results.setdefault(layer, {})[head] = {
                "score": score,
            }

            if args.use_fluency:
                print(f"Fluency for layer {layer}, head {head}: {val_results['fluency']:.3f}")
                pasta_layer_head_selection_results[layer][head]["fluency"] = val_results["fluency"]

            pasta_layer_head_selection_results[layer][head]["time"] = head_time_end - head_time_start

            # Save the results after each head
            with open(pasta_layer_head_selection_results_path, 'w') as f:
                json.dump(pasta_layer_head_selection_results, f, indent=2)

