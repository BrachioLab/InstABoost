import json
import os
import numpy as np
from prompts import EVAL_STRATEGY
from typing import List, Dict, Tuple, Optional, Any
import random
from scipy import stats
from model_interaction_utils import get_generations

def load_existing_results(output_dir, results_name):
    """
    Check if results exist and load them if they do.
    
    Args:
        output_dir (str): Directory to look for results
        results_name (str): Name of the results file (without extension)
        
    Returns:
        tuple: (results dict, generations list) if files exist, (None, None) otherwise
    """
    results_path = os.path.join(output_dir, f"{results_name}.json")
    generations_path = os.path.join(output_dir, f"{results_name}_by_sample.json")
    
    if os.path.exists(results_path) and os.path.exists(generations_path):
        print(f"Loading {results_name} results from cache")
        with open(results_path, 'r') as f:
            results = json.load(f)
        with open(generations_path, 'r') as f:
            generations_data = json.load(f)
            
        # Extract per_sample data from generations_data
        per_sample = []
        for idx in range(len(generations_data)):
            sample_data = generations_data[str(idx)]
            evaluation = sample_data.get("baseline_evaluation") or sample_data.get("intervention_evaluation")
            score = sample_data.get("baseline_score") or sample_data.get("intervention_score")
            fluency = sample_data.get("baseline_fluency") or sample_data.get("intervention_fluency")
            
            per_sample.append({
                "evaluation": evaluation,
                "score": score,
                "fluency": fluency
            })
            
        # Handle both baseline and intervention results
        if "intervention" in results:
            results_data = results["intervention"]
        else:
            results_data = results
            
        results_data["per_sample"] = per_sample
        generations = [data["intervention"] for data in generations_data.values()]

        return results_data, generations
    return None, None

def bootstrap_confidence_interval(
    data: List[float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for a list of values using scipy.stats.
    
    Args:
        data: List of values to bootstrap
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for the interval
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    # Convert to numpy array
    data_array = np.array(data)
    
    # Calculate confidence interval using scipy
    bootstrap_result = stats.bootstrap(
        (data_array,),
        np.mean,
        n_resamples=n_bootstrap,
        confidence_level=confidence_level,
        method='percentile'
    )
    
    return bootstrap_result.confidence_interval.low, bootstrap_result.confidence_interval.high

def evaluate_generations(
    generations,
    dataset,
    eval_type,
    eval_args,
    questions=None,
    answers=None,
    alt_answers=None,
    is_validation=False,
    use_fluency=False,
    steer_dataset=None,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
):
    """
    Evaluate generations for different datasets and cases.
    
    Args:
        generations (list): List of generated responses
        dataset (str): Dataset name
        eval_type (str): Evaluation type from EVAL_STRATEGY
        eval_args (dict): Additional arguments for evaluation
        questions (list, optional): List of questions
        answers (list, optional): List of correct answers
        alt_answers (list, optional): List of alternative answers
        is_validation (bool): Whether this is validation set evaluation
        use_fluency (bool): Whether to evaluate fluency
        steer_dataset (list, optional): Original dataset for safety/jbb evaluation
        n_bootstrap (int): Number of bootstrap samples for confidence intervals
        confidence_level (float): Confidence level for intervals (0-1)
        
    Returns:
        dict: Dictionary containing evaluation metrics and per-sample results
    """
    n_true = 0
    n_false = 0
    n_unsure = 0
    eval_scores = []
    fluency_scores = []
    per_sample_results = []
    
    for i in range(len(generations)):
        if dataset in ["triviaqa", "truthfulqa", "toxicity", "emotions-qa", "ai-risk-mcq", "ai-risk-qa"]:
            eval_args["answer"] = answers[i]
            eval_args["alt_answer"] = alt_answers[i]
            eval_args["question"] = questions[i]
            eval_result = EVAL_STRATEGY[eval_type](generations[i], eval_args)
            
            if use_fluency:
                fluency = EVAL_STRATEGY["fluency"](generations[i], None)
                fluency_scores.append(fluency)
            
            if eval_result not in [-1, 0, 1]:
                eval_result = 1 if eval_result >= 0.5 else 0 if eval_result != -1 and eval_result < 0.5 else -1

            eval_scores.append(eval_result)
            
            n_true += eval_result >= 0.5
            n_false += eval_result < 0.5 and eval_result >= 0
            n_unsure += eval_result == -1
            
            per_sample_results.append({
                "evaluation": eval_result,
                "score": eval_scores[-1],
                "fluency": fluency_scores[-1] if use_fluency else None
            })
            
        elif dataset in ["safety", "jbb"]:
            eval_result = EVAL_STRATEGY[eval_type](steer_dataset[i], generations[i])
            
            if use_fluency:
                fluency = EVAL_STRATEGY["fluency"](generations[i], None)
                fluency_scores.append(fluency)
            
            eval_scores.append(eval_result)
            n_true += eval_result < 0.5 and eval_result >= 0
            n_false += eval_result >= 0.5
            n_unsure += eval_result == -1
            
            per_sample_results.append({
                "evaluation": eval_result,
                "score": eval_scores[-1],
                "fluency": fluency_scores[-1] if use_fluency else None
            })
    
    # Calculate bootstrap confidence intervals
    score_ci = bootstrap_confidence_interval(eval_scores, n_bootstrap, confidence_level)
    fluency_ci = bootstrap_confidence_interval(fluency_scores, n_bootstrap, confidence_level) if use_fluency else None
    
    results = {
        "true": n_true,
        "false": n_false,
        "unsure": n_unsure,
        "total": n_true + n_false + n_unsure,
        "score": np.mean(eval_scores) if eval_scores else 0,
        "score_ci": {
            "lower": score_ci[0],
            "upper": score_ci[1],
            "confidence_level": confidence_level
        },
        "per_sample": per_sample_results
    }
    
    if use_fluency:
        results["fluency"] = np.mean(fluency_scores) if fluency_scores else 0
        results["fluency_ci"] = {
            "lower": fluency_ci[0],
            "upper": fluency_ci[1],
            "confidence_level": confidence_level
        }
    
    return results

def save_results(
    layer_output_dir,
    layer,
    intervention_results,
    intervention_generations,
    steer_questions,
    steer_dataset,
    steer_answers=None,
    steer_alt_answers=None,
    use_fluency=False,
    best_factor=None,
    baseline_results=None,
    baseline_generations=None
):
    """
    Save evaluation results to JSON files.
    
    Args:
        layer_output_dir (str): Directory to save results
        layer (int): Layer number
        intervention_results (dict): Results from intervention evaluation
        intervention_generations (list): List of intervention generations
        steer_questions (list): List of questions
        steer_answers (list, optional): List of answers
        steer_alt_answers (list, optional): List of alternative answers
        use_fluency (bool): Whether fluency was evaluated
        best_factor (float, optional): Best steering factor found
        baseline_results (dict, optional): Results from baseline evaluation
        baseline_generations (list, optional): List of baseline generations
    """
    # Create results with intervention data
    results = {
        "true": intervention_results["true"],
        "false": intervention_results["false"],
        "unsure": intervention_results["unsure"],
        "total": intervention_results["total"],
        "score": intervention_results["score"],
        "score_ci": intervention_results["score_ci"],
        "fluency": intervention_results["fluency"] if use_fluency else None,
        "fluency_ci": intervention_results["fluency_ci"] if use_fluency else None,
        "best_layer": layer,
        "best_factor": best_factor
    }

    # Create per-sample results
    layer_data = {
        idx: {
            "instruction": steer_questions[idx],
            "intervention": intervention_generations[idx],
            "intervention_evaluation": intervention_results["per_sample"][idx]["evaluation"],
            "intervention_score": intervention_results["per_sample"][idx]["score"],
        }
        for idx in range(len(intervention_generations))
    }

    # Add baseline data to per-sample results if available
    if baseline_results is not None and baseline_generations is not None:
        for idx in range(len(intervention_generations)):
            layer_data[idx].update({
                "baseline": baseline_generations[idx],
                "baseline_evaluation": baseline_results["per_sample"][idx]["evaluation"],
                "baseline_score": baseline_results["per_sample"][idx]["score"],
            })
            if use_fluency:
                layer_data[idx]["baseline_fluency"] = baseline_results["per_sample"][idx]["fluency"]

    if steer_answers is not None:
        for idx in range(len(intervention_generations)):
            layer_data[idx]["answer"] = steer_answers[idx]
            layer_data[idx]["alt_answer"] = steer_alt_answers[idx]

    if use_fluency:
        for idx in range(len(intervention_generations)):
            layer_data[idx]["intervention_fluency"] = intervention_results["per_sample"][idx]["fluency"]

    # Save results
    with open(os.path.join(layer_output_dir, f"results_by_sample.json"), 'w') as f:
        json.dump(layer_data, f, indent=2)
    
    with open(os.path.join(layer_output_dir, f"results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    return results

def save_all_methods_results(output_dir, model_name, dataset, methods_results, method_times=None):
    """
    Save aggregated results from all steering methods to a single file.
    
    Args:
        output_dir (str): Base output directory
        model_name (str): Name of the model
        dataset (str): Name of the dataset
        methods_results (dict): Dictionary containing results for each method
            Format: {
                "method_name": {
                    "true": int,
                    "false": int,
                    "unsure": int,
                    "total": int,
                    "score": float,
                    "score_ci": dict,
                    "fluency": float,
                    "fluency_ci": dict
                }
            }
        method_times (dict): Dictionary containing timing information for each method
            Format: {
                "method_name": time_in_seconds
            }
    """
    # Create the aggregated results structure, excluding per_sample data
    aggregated_results = {
        "model": model_name,
        "dataset": dataset,
        "methods": {
            method: {
                k: v for k, v in results.items() 
                if k != "per_sample"  # Exclude per_sample data
            }
            for method, results in methods_results.items()
        },
        "method_times": method_times if method_times is not None else {}
    }
    
    # Save to a single JSON file
    output_path = os.path.join(output_dir, f"{model_name}", "all_methods_results.json")
    with open(output_path, 'w') as f:
        json.dump(aggregated_results, f, indent=2)
    
    return output_path

def print_evaluation_results(results: Dict[str, Any], prefix: str = "", use_fluency: bool = False):
    """
    Print evaluation results in a formatted way.
    
    Args:
        results: Dictionary containing evaluation results
        prefix: Prefix to add before the output (e.g., "Baseline", "Intervention")
        use_fluency: Whether to print fluency metrics
    """
    print(f"{prefix}:")
    print(f"True: {results['true']}, False: {results['false']}, Unsure: {results['unsure']}, Total: {results['total']}")
    print(f"Score: {results['score']:.3f} (95% CI: [{results['score_ci']['lower']:.3f}, {results['score_ci']['upper']:.3f}])")
    if use_fluency and 'fluency' in results:
        print(f"Fluency: {results['fluency']:.3f} (95% CI: [{results['fluency_ci']['lower']:.3f}, {results['fluency_ci']['upper']:.3f}])")

def run_evaluation_pipeline(
    model,
    dataset: str,
    eval_type: str,
    eval_args: Dict[str, Any],
    steer_dataset: List[str],
    steer_questions: List[str],
    steer_answers: Optional[List[str]],
    steer_alt_answers: Optional[List[str]],
    tokenize_instructions_fn,
    fwd_hooks: List[Tuple[str, Any]],
    steer_prompt: str = "",
    max_tokens_generated: int = 50,
    batch_size: int = 4,
    use_fluency: bool = False,
    cache_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    layer: Optional[int] = None,
    best_factor: Optional[float] = None,
    baseline_results: Optional[Dict[str, Any]] = None,
    baseline_generations: Optional[List[str]] = None
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Run the complete evaluation pipeline including generation, evaluation, and saving results.
    
    Args:
        model: The model to use for generation
        dataset: Dataset name
        eval_type: Evaluation type
        eval_args: Evaluation arguments
        steer_dataset: Dataset for steering
        steer_questions: Questions for evaluation
        steer_answers: Correct answers (if applicable)
        steer_alt_answers: Alternative answers (if applicable)
        tokenize_instructions_fn: Function to tokenize instructions
        fwd_hooks: Forward hooks for the model
        steer_prompt: Prompt to prepend (if any)
        max_tokens_generated: Maximum tokens to generate
        batch_size: Batch size for generation
        use_fluency: Whether to evaluate fluency
        cache_dir: Directory for caching results
        output_dir: Directory to save results
        layer: Layer number (if applicable)
        best_factor: Best steering factor found
        baseline_results: Baseline results (if available)
        baseline_generations: Baseline generations (if available)
        
    Returns:
        Tuple of (evaluation results, generations)
    """
    # Generate completions
    generations = get_generations(
        model,
        [steer_prompt + sample for sample in steer_dataset],
        tokenize_instructions_fn,
        fwd_hooks=fwd_hooks,
        max_tokens_generated=max_tokens_generated,
        batch_size=batch_size,
        cache_dir=cache_dir
    )
    
    # Evaluate generations
    results = evaluate_generations(
        generations=generations,
        dataset=dataset,
        eval_type=eval_type,
        eval_args=eval_args,
        questions=steer_questions,
        answers=steer_answers,
        alt_answers=steer_alt_answers,
        is_validation=False,
        use_fluency=use_fluency,
        steer_dataset=steer_dataset
    )
    
    # Save results if output directory is provided
    if output_dir is not None:
        save_results(
            layer_output_dir=output_dir,
            layer=layer,
            intervention_results=results,
            intervention_generations=generations,
            steer_dataset=steer_dataset,
            steer_questions=steer_questions,
            steer_answers=steer_answers if dataset in ["triviaqa", "truthfulqa", "toxicity", "emotions-qa", "ai-risk-mcq", "ai-risk-qa"] else None,
            steer_alt_answers=steer_alt_answers if dataset in ["triviaqa", "truthfulqa", "toxicity", "emotions-qa", "ai-risk-mcq", "ai-risk-qa"] else None,
            use_fluency=use_fluency,
            best_factor=best_factor,
            baseline_results=baseline_results,
            baseline_generations=baseline_generations
        )
    
    return results, generations 