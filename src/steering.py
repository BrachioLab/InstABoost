import torch
import functools
import einops
import pandas as pd
import textwrap
import gc
from sklearn.decomposition import PCA
import csv
from sklearn.linear_model import LogisticRegression
import time
from datetime import datetime

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch import Tensor
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from jaxtyping import Float, Int
from colorama import Fore
import os
import pickle
import json
from argparse import ArgumentParser
import numpy as np
import random
import math

from data_utils import (
    get_eval_dataset, 
    get_qa_data, 
    get_harmful_instructions, 
    get_harmless_instructions,
    get_harmful_jbb_instructions,
    get_harmless_jbb_instructions,
    get_dataset,
    DATASET_CHOICES,
    DATASET_CONFIGS
)
from prompts import PROMPTS, EVAL_STRATEGY
from model_interaction_utils import (
    tokenize_instructions,
    get_generations,
    get_hiddens
)
from evaluation_utils import evaluate_generations, load_existing_results, save_all_methods_results, run_evaluation_pipeline, print_evaluation_results

STEERING_FACTORS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def direction_ablation_hook(
    activation: Float[Tensor, "... d_act"],
    hook: HookPoint,
    direction: Float[Tensor, "d_act"]
):
    proj = einops.einsum(activation, direction.view(-1, 1).to(activation.device), '... d_act, d_act single -> ... single') * direction.to(activation.device)
    return activation - proj

def in_direction_ablation_hook(
    activation: Float[Tensor, "... d_act"],
    hook: HookPoint,
    direction: Float[Tensor, "d_act"],
    multiplier: float = 1.0
):
    direction = direction.to(activation.device)
    return activation + multiplier * direction.unsqueeze(0)

def prompt_attention_ablation_hook(
    attn_scores: Float[Tensor, "... d_act"],
    hook: HookPoint,
    prompt_len: int,
    multiplier: float = 3.0
):
    attn_scores[:, :, :, :prompt_len] *= multiplier
    attn_scores = torch.nn.functional.normalize(attn_scores, p=1, dim=-1)
    return attn_scores

def get_steering_direction(harmful_hiddens, harmless_hiddens, steering_type="mean", normalize: bool = True):
    dir = None
    if steering_type == "mean":
        dir =  harmful_hiddens.mean(dim=0) - harmless_hiddens.mean(dim=0)
    elif steering_type == "random":
        dir = torch.randn_like(harmful_hiddens.mean(dim=0) - harmless_hiddens.mean(dim=0))
    elif steering_type == "repe":
        diffs = (harmful_hiddens - harmless_hiddens).to(torch.float32)
        diffs /= diffs.norm(dim=-1, keepdim=True)
        diffs_float32 = diffs.to(torch.float32)
        dir = PCA(n_components=1).fit_transform(diffs_float32.cpu().T)
        dir = torch.from_numpy(dir).to(harmful_hiddens.device).T
        dir = dir.to(torch.bfloat16)
    elif steering_type == "pca":
        harmful_hiddens_norm = (harmful_hiddens / harmful_hiddens.norm(dim=-1, keepdim=True)).to(torch.float32).cpu().T
        dir = PCA(n_components=1).fit_transform(harmful_hiddens_norm)
        dir = torch.from_numpy(dir).to(harmful_hiddens.device).T
        dir = dir.to(torch.bfloat16)
    elif steering_type == "linear":
        harmful_np = harmful_hiddens.to(torch.float32).cpu().numpy()
        harmless_np = harmless_hiddens.to(torch.float32).cpu().numpy()
        X = np.vstack([harmful_np, harmless_np])
        y = np.concatenate([np.zeros(len(harmful_np)), np.ones(len(harmless_np))])
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        dir = torch.from_numpy(clf.coef_[0]).to(harmful_hiddens.device)
        dir = dir.to(torch.bfloat16)
    elif steering_type == "refusal":
        dir = harmful_hiddens.mean(dim=0) - harmless_hiddens.mean(dim=0)
        dir = dir / dir.norm()
    elif steering_type == "prompt":
        dir = None
    elif steering_type == "prompt-attention":
        dir = None
    else:
        raise ValueError(f"Unknown steering type: {steering_type}")

    if not dir is None and normalize:
        dir = dir / dir.norm()

    return dir

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
    parser.add_argument("--steering_type", type=str, default="refusal", choices=["refusal", "random", "mean", "prompt", "pca", "repe", "linear", "prompt-attention"], help="Type of steering to perform")
    parser.add_argument("--steer_to_pt", action="store_true", help="Steer to PT (defaults to CT)")
    parser.add_argument("--cache_harmless", action="store_true", help="Cache harmless instructions. By default, only harmful instructions are cached.")
    # Misc arguments
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--use_fluency", action="store_true", help="Whether to calculate and use fluency in steering")
    parser.add_argument("--grid_layers", action="store_true", help="Whether to grid search over layers")
    parser.add_argument("--normalize_dir", action="store_true", help="Whether to normalize the steering direction")
    args = parser.parse_args()

    dataset = args.dataset
    MODEL_PATH = args.model_path
    output_dir = args.output_dir
    DEVICE = args.device
    batch_size = args.batch_size
    normalize_dir = args.normalize_dir
    max_tokens_generated = DATASET_CONFIGS[dataset]["max_tokens_generated"]
    N_INST_TRAIN = DATASET_CONFIGS[dataset]["n_inst_train"]
    N_INST_TEST = DATASET_CONFIGS[dataset]["n_inst_test"]
    N_INST_VAL = DATASET_CONFIGS[dataset]["n_inst_val"]

    assert dataset in DATASET_CONFIGS, f"Dataset {dataset} not found in DATASET_CONFIGS, please update the DATASET_CONFIGS dictionary in data_utils.py"

    task_type = DATASET_CONFIGS[dataset]["task_type"]
    eval_type = DATASET_CONFIGS[dataset]["eval_type"]
    apply_chat_template = DATASET_CONFIGS[dataset].get("apply_chat_template", True)

    output_dir = os.path.join(output_dir, f"{MODEL_PATH.split('/')[1]}")
    if args.baseline_generations_dir is None:
        baseline_generations_dir = output_dir
    else:
        baseline_generations_dir = args.baseline_generations_dir

    model = HookedTransformer.from_pretrained_no_processing(
        MODEL_PATH,
        device=DEVICE,
        dtype=torch.bfloat16,
        default_padding_side='left',
    )

    model.tokenizer.pad_token = model.tokenizer.eos_token

    tokenize_instructions_fn = functools.partial(tokenize_instructions, tokenizer=model.tokenizer, model_name=MODEL_PATH, apply_chat_template=apply_chat_template)

    # Get total number of layers
    if args.grid_layers:
        total_layers = model.cfg.n_layers
        middle_percentage = 0.2  # 20% of layers in the middle
        n_middle_layers = int(total_layers * middle_percentage)
        start_layer = (total_layers - n_middle_layers) // 2
        layers_to_grid = list(range(start_layer, start_layer + n_middle_layers))
    else:
        layers_to_grid = [DATASET_CONFIGS[dataset]["layer"]]

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
        tokenizer=model.tokenizer,
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
    print(f'### Running Evaluations ###')
    print(50*'=' + '\n')

    output_json_data = {}
    method_times = {}
    all_methods_results = {}
    
    # 2. Run baseline evaluation
    baseline_output_dir = os.path.join(args.output_dir, f"{MODEL_PATH.split('/')[1]}", "baseline")
    os.makedirs(baseline_output_dir, exist_ok=True)
    
    baseline_start = time.time()
    print("\nRunning baseline evaluation")
    # Check if baseline results exist
    baseline_results, baseline_generations = load_existing_results(baseline_output_dir, "results")
    if baseline_results is None:
        print("\nRunning baseline evaluation")
        baseline_results, baseline_generations = run_evaluation_pipeline(
            model=model,
            dataset=dataset,
            eval_type=eval_type,
            eval_args=eval_args,
            steer_dataset=steer_dataset,
            steer_questions=steer_questions,
            steer_answers=steer_answers,
            steer_alt_answers=steer_alt_answers,
            tokenize_instructions_fn=tokenize_instructions_fn,
            fwd_hooks=[],
            max_tokens_generated=max_tokens_generated,
            batch_size=batch_size,
            use_fluency=args.use_fluency,
            output_dir=baseline_output_dir
        )
    baseline_time = time.time() - baseline_start
    method_times["baseline"] = baseline_time
    print_evaluation_results(baseline_results, prefix="Baseline", use_fluency=args.use_fluency)
    print(f"Baseline evaluation took {baseline_time:.2f} seconds")
    print(50*'-', '\n\n')
    
    # Add baseline results
    all_methods_results["baseline"] = baseline_results

    # 3. Run prompt method
    prompt_start = time.time()
    print("\nRunning prompt method")
    print(50*'-')
    prompt_output_dir = os.path.join(args.output_dir, f"{MODEL_PATH.split('/')[1]}", "prompt")
    os.makedirs(prompt_output_dir, exist_ok=True)

    # Check if prompt results exist
    prompt_results, prompt_generations = load_existing_results(prompt_output_dir, "results")
    if prompt_results is None:
        print("Running prompt evaluation")
        fwd_hooks = []
        prompt_results, prompt_generations = run_evaluation_pipeline(
            model=model,
            dataset=dataset,
            eval_type=eval_type,
            eval_args=eval_args,
            steer_dataset=steer_dataset,
            steer_questions=steer_questions,
            steer_answers=steer_answers,
            steer_alt_answers=steer_alt_answers,
            tokenize_instructions_fn=tokenize_instructions_fn,
            fwd_hooks=fwd_hooks,
            steer_prompt=steer_prompt,
            max_tokens_generated=max_tokens_generated,
            batch_size=batch_size,
            use_fluency=args.use_fluency,
            output_dir=prompt_output_dir,
            baseline_results=baseline_results,
            baseline_generations=baseline_generations,
            best_factor=None  # Prompt method doesn't use a steering factor
        )
    prompt_time = time.time() - prompt_start
    method_times["prompt"] = prompt_time
    print_evaluation_results(prompt_results, prefix="prompt", use_fluency=args.use_fluency)
    print(f"Prompt method took {prompt_time:.2f} seconds")
    print(50*'-', '\n\n')
    all_methods_results["prompt"] = prompt_results

    # 4. Run prompt-attention method
    prompt_attention_start = time.time()
    print("\nRunning prompt-attention method")
    print(50*'-')
    prompt_attention_output_dir = os.path.join(args.output_dir, f"{MODEL_PATH.split('/')[1]}", "prompt-attention")
    os.makedirs(prompt_attention_output_dir, exist_ok=True)

    # Check if prompt-attention results exist
    prompt_attention_results, prompt_attention_generations = load_existing_results(prompt_attention_output_dir, "results")
    if prompt_attention_results is None:
        print("Running prompt-attention evaluation")
        prompt_len = tokenize_instructions_fn(instructions=[steer_prompt], add_generation_prompt=True)[0].shape[0] - 5
        print(f"Prompt length: {prompt_len}")
        
        # Check if best steering factor exists
        best_factor_path = os.path.join(prompt_attention_output_dir, "best_steering_factor.txt")
        if os.path.exists(best_factor_path):
            with open(best_factor_path, 'r') as f:
                best_steering_factor = float(f.read())
        else:
            steering_success_rates = []
            fluencies = []
            checked_factors = []
            validation_examples = {}
            for multiplier in range(1, 20, 2):
                print(f"Testing multiplier: {multiplier}")
                checked_factors.append(multiplier)
                hook_fn = functools.partial(prompt_attention_ablation_hook, prompt_len=prompt_len, multiplier=multiplier)
                fwd_hooks = [(utils.get_act_name('pattern', l), hook_fn) for l in range(model.cfg.n_layers)]

                val_steer_generations = get_generations(
                    model,
                    [steer_prompt + sample for sample in steer_val_dataset],
                    tokenize_instructions_fn,
                    fwd_hooks=fwd_hooks,
                    max_tokens_generated=max_tokens_generated,
                    batch_size=1
                )

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
                    steer_dataset=steer_val_dataset
                )

                steering_success_rates.append(val_results["score"])
                if args.use_fluency:
                    fluencies.append(val_results["fluency"])
                    print(f"Fluency: {val_results['fluency']}")
                    if val_results["fluency"] < 1:
                        print(f"Fluency: {val_results['fluency']} < 1, ending search")
                        break
                
                validation_examples[multiplier] = []
                for i in range(len(val_steer_generations)):
                    example = {'sample': steer_val_dataset[i], 
                    'generation': val_steer_generations[i], 
                    'score': val_results['per_sample'][i]['score'], 
                    'fluency': val_results['per_sample'][i]['fluency'] if args.use_fluency else None
                    }
                    validation_examples[multiplier].append(example)

                print(f"Score: {val_results['score']}")
                print(50*'-')

            # Save validation examples
            with open(os.path.join(prompt_attention_output_dir, f"validation_examples.json"), 'w') as f:
                json.dump(validation_examples, f, indent=2)

            # Select best steering factor
            steering_success_rates = np.array(steering_success_rates)
            fluencies = np.array(fluencies) if args.use_fluency else None
            with open(os.path.join(prompt_attention_output_dir, f"acc_fluency.txt"), 'w') as f:
                for i in range(len(steering_success_rates)):
                    if args.use_fluency:
                        f.write(f"{checked_factors[i]}, {steering_success_rates[i]}, {fluencies[i]}\n")
                    else:
                        f.write(f"{checked_factors[i]}, {steering_success_rates[i]}\n")
            if args.use_fluency:
                steering_success_rates[fluencies < 1] = -1
            best_steering_factor = checked_factors[np.argmax(steering_success_rates)]
            print(f"Best steering factor: {best_steering_factor}")
            with open(best_factor_path, 'w') as f:
                f.write(f"{best_steering_factor}")

        hook_fn = functools.partial(prompt_attention_ablation_hook, prompt_len=prompt_len, multiplier=best_steering_factor)
        fwd_hooks = [(utils.get_act_name('pattern', l), hook_fn) for l in range(model.cfg.n_layers)]

        prompt_attention_results, prompt_attention_generations = run_evaluation_pipeline(
            model=model,
            dataset=dataset,
            eval_type=eval_type,
            eval_args=eval_args,
            steer_dataset=steer_dataset,
            steer_questions=steer_questions,
            steer_answers=steer_answers,
            steer_alt_answers=steer_alt_answers,
            tokenize_instructions_fn=tokenize_instructions_fn,
            fwd_hooks=fwd_hooks,
            steer_prompt=steer_prompt,
            max_tokens_generated=max_tokens_generated,
            batch_size=1,
            use_fluency=args.use_fluency,
            output_dir=prompt_attention_output_dir,
            baseline_results=baseline_results,
            baseline_generations=baseline_generations,
            best_factor=best_steering_factor
        )
    prompt_attention_time = time.time() - prompt_attention_start
    method_times["prompt-attention"] = prompt_attention_time
    print_evaluation_results(prompt_attention_results, prefix="prompt-attention", use_fluency=args.use_fluency)
    print(f"Prompt-attention method took {prompt_attention_time:.2f} seconds")
    print(50*'-', '\n\n')
    all_methods_results["prompt-attention"] = prompt_attention_results

    # 5. Run other steering methods
    other_methods = ["refusal", "random", "mean", "pca", "repe", "linear"]
    
    # Dictionary to store validation results for each method
    validation_results = {}
    
    # For each method, do grid search over layers and steering factors
    for steering_type in other_methods:
        gc.collect()
        torch.cuda.empty_cache()
        method_start = time.time()
        print(f"\nRunning {steering_type} method")
        print(50*'-')
        
        method_output_dir = os.path.join(args.output_dir, f"{MODEL_PATH.split('/')[1]}", steering_type)
        os.makedirs(method_output_dir, exist_ok=True)
        
        # Attempt to load existing results to avoid redundant computation
        existing_results, _ = load_existing_results(method_output_dir, "results")
        if existing_results is not None:
            print(f"Found cached results for {steering_type}. Skipping re-evaluation.")
            print_evaluation_results(existing_results, prefix=f"{steering_type} (cached)", use_fluency=args.use_fluency)
            method_times[steering_type] = time.time() - method_start
            all_methods_results[steering_type] = existing_results
            continue
        
        # Check if best parameters exist (to avoid repeating grid search)
        best_params_path = os.path.join(method_output_dir, "best_params.json")
        if os.path.exists(best_params_path):
            print(f"Loading best parameters from {best_params_path}")
            with open(best_params_path, 'r') as f:
                best_params = json.load(f)
            best_layer = best_params["layer"]
            best_factor = best_params["factor"]
            best_score = best_params["score"]
            
            print(f"Best parameters for {steering_type}:")
            print(f"Layer: {best_layer}, Factor: {best_factor}")
            print(f"Validation score: {best_score:.3f}")
            
            # Load steering direction
            layer_output_dir = os.path.join(method_output_dir, f"layer_{best_layer}")
            steering_direction = torch.load(os.path.join(layer_output_dir, "intervention_dir.pt"))
            best_direction = steering_direction
            
        else:
            # Store validation results for this method
            validation_results[steering_type] = {}
            best_score = -1
            best_layer = None
            best_factor = None
            best_direction = None
            
            # Grid search over layers and steering factors
            for layer in layers_to_grid:
                print(f"\nLayer {layer}")
                print(50*'-')
                pos = DATASET_CONFIGS[dataset]["pos"]
                layer_output_dir = os.path.join(method_output_dir, f"layer_{layer}")
                os.makedirs(layer_output_dir, exist_ok=True)
                
                # Compute hiddens for this layer
                hiddens_start = time.time()
                print(f"Computing hiddens for layer {layer}")
                resid_pres_harmful, resid_pres_harmless = get_hiddens(
                    model=model,
                    harmful_inst=harmful_inst_train,
                    harmless_inst=harmless_inst_train,
                    layer=layer,
                    pos=pos,
                    tokenize_instructions_fn=tokenize_instructions_fn,
                    batch_size=batch_size
                )
                hiddens_time = time.time() - hiddens_start
                print(f"Computing hiddens took {hiddens_time:.2f} seconds")
                
                # Get steering direction for this method
                steering_direction = get_steering_direction(resid_pres_harmful, resid_pres_harmless, steering_type, 
                                                            normalize=normalize_dir)
                
                # Store validation results for this layer
                validation_results[steering_type][layer] = {}
                
                # Grid search over steering factors
                if steering_type == "refusal":
                    hook_fn = functools.partial(direction_ablation_hook,direction=steering_direction)
                    fwd_hooks = [(utils.get_act_name('resid_post', l), hook_fn) for l in range(model.cfg.n_layers)]

                    val_steer_generations = get_generations(
                        model,
                        [sample for sample in steer_val_dataset],
                        tokenize_instructions_fn,
                        fwd_hooks=fwd_hooks,
                        max_tokens_generated=max_tokens_generated,
                        batch_size=batch_size
                    )

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
                        steer_dataset=steer_val_dataset
                    )

                    validation_results[steering_type][layer] = {
                        "score": val_results["score"],
                        "fluency": val_results.get("fluency", None)
                    }

                    if val_results["score"] > best_score and (not args.use_fluency or val_results.get("fluency", 0) >= 1):
                        best_score = val_results["score"]
                        best_layer = layer
                        best_factor = None
                        best_direction = steering_direction
                else:
                    for steering_factor in STEERING_FACTORS:
                        print(f"Testing layer {layer}, factor {steering_factor}")
                        
                        hook_fn = functools.partial(in_direction_ablation_hook, direction=steering_direction, multiplier=steering_factor)
                        fwd_hooks = [(utils.get_act_name('resid_post', l), hook_fn) for l in range(model.cfg.n_layers)]
                        
                        # Run validation
                        val_steer_generations = get_generations(
                            model,
                            [sample for sample in steer_val_dataset],
                            tokenize_instructions_fn,
                            fwd_hooks=fwd_hooks,
                            max_tokens_generated=max_tokens_generated,
                            batch_size=batch_size
                        )
                        
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
                            steer_dataset=steer_val_dataset
                        )
                        
                        # Store validation results
                        validation_results[steering_type][layer][steering_factor] = {
                            "score": val_results["score"],
                            "fluency": val_results.get("fluency", None)
                        }
                        
                        # Update best combination if this is better
                        if val_results["score"] > best_score and (not args.use_fluency or val_results.get("fluency", 0) >= 1):
                            best_score = val_results["score"]
                            best_layer = layer
                            best_factor = steering_factor
                            best_direction = steering_direction

                        if args.use_fluency and val_results["fluency"] < 1:
                            print(f"Fluency: {val_results['fluency']} < 1, ending search at layer {layer}.")
                            break
                        
                        print(f"Validation score: {val_results['score']:.3f}")
                        if args.use_fluency:
                            print(f"Validation fluency: {val_results.get('fluency', 0):.3f}")
                        print(50*'-')
                
                # Clean up memory
                del resid_pres_harmful, resid_pres_harmless
                gc.collect()
                torch.cuda.empty_cache()
            
            # Save validation results
            with open(os.path.join(method_output_dir, "validation_results.json"), 'w') as f:
                json.dump(validation_results[steering_type], f, indent=2)
            
            # Save best parameters
            best_params = {
                "layer": best_layer,
                "factor": best_factor,
                "score": best_score
            }
            with open(best_params_path, 'w') as f:
                json.dump(best_params, f, indent=2)
            
            # Save best steering direction
            layer_output_dir = os.path.join(method_output_dir, f"layer_{best_layer}")
            os.makedirs(layer_output_dir, exist_ok=True)
            torch.save(best_direction, os.path.join(layer_output_dir, "intervention_dir.pt"))
            
            print(f"\nBest combination for {steering_type}:")
            print(f"Layer: {best_layer}, Factor: {best_factor}")
            print(f"Validation score: {best_score:.3f}")
        
        # Run test set with best combination
        if best_layer is not None:
            print(f"\nRunning test set with best combination")
            layer = best_layer
            layer_output_dir = os.path.join(method_output_dir, f"layer_{layer}")
            
            print(f"Best direction type: {type(best_direction)}")

            # Run test set
            if steering_type == "refusal":
                hook_fn = functools.partial(direction_ablation_hook, direction=best_direction)
                fwd_hooks = [(utils.get_act_name('resid_post', l), hook_fn) for l in range(model.cfg.n_layers)]
            else:
                hook_fn = functools.partial(in_direction_ablation_hook, direction=best_direction, multiplier=best_factor)
                fwd_hooks = [(utils.get_act_name('resid_post', l), hook_fn) for l in range(model.cfg.n_layers)]
            
            intervention_results, intervention_generations = run_evaluation_pipeline(
                model=model,
                dataset=dataset,
                eval_type=eval_type,
                eval_args=eval_args,
                steer_dataset=steer_dataset,
                steer_questions=steer_questions,
                steer_answers=steer_answers,
                steer_alt_answers=steer_alt_answers,
                tokenize_instructions_fn=tokenize_instructions_fn,
                fwd_hooks=fwd_hooks,
                steer_prompt='',
                max_tokens_generated=max_tokens_generated,
                batch_size=batch_size,
                use_fluency=args.use_fluency,
                output_dir=method_output_dir,  # Save in method directory instead of layer directory
                layer=layer,
                best_factor=best_factor,
                baseline_results=baseline_results,
                baseline_generations=baseline_generations
            )
            
            method_time = time.time() - method_start
            method_times[steering_type] = method_time
            
            print_evaluation_results(intervention_results, prefix=f"{steering_type} (layer {layer})", use_fluency=args.use_fluency)
            all_methods_results[steering_type] = intervention_results
        
        print(f"{steering_type} method took {method_time:.2f} seconds")

    # 6. Save all results
    save_all_methods_results(
        output_dir=args.output_dir,
        model_name=MODEL_PATH.split('/')[1],
        dataset=dataset,
        methods_results=all_methods_results,
        method_times=method_times
    )

    total_time = time.time() - start_time
    print(f'\n### Done steering ###')
    print(f'Total script runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)')
    print(f'Timing breakdown:')
    for method, time in method_times.items():
        print(f'- {method}: {time:.2f} seconds')
    print(50*'=' + '\n')