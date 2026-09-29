"""
Selects the top k layer-heads for each task based on their scores.
The intersection across tasks is stored in {output_dir}/pasta_layer_head_selection_results.json.
The format is {"layer": [head_0, head_1, ...]} and can be directly used with PASTA's API.
"""
from argparse import ArgumentParser
import json
import os
from collections import defaultdict

TASK_DIRS = {
    "safety": "results_pasta/safety/",
    "jbb": "results_pasta/jbb/",
    "triviaqa": "results_pasta/triviaqa/",
    "truthfulqa": "results_pasta/truthfulqa/",
    "toxicity": "results_pasta/toxicity/",
    "ai-risk-qa-power": "results_pasta/ai-risk-qa/power/",
    "ai-risk-qa-wealth": "results_pasta/ai-risk-qa/wealth/",
    "ai-risk-mcq-wealth": "results_pasta/ai-risk-mcq/wealth/",
    "ai-risk-mcq-power": "results_pasta/ai-risk-mcq/power",
    "emotions-anger": "results_pasta/emotions/anger",
    "emotions-joy": "results_pasta/emotions/joy",
    "emotions-sadness": "results_pasta/emotions/sadness",
    "emotions-disgust": "results_pasta/emotions/disgust",
    "emotions-fear": "results_pasta/emotions/fear",
    "emotions-surprise": "results_pasta/emotions/surprise"
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--k", type=int, default=400, help="Number of top layer-heads to select per task")
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--output_dir", type=str, default="qa_results_for_steering", help="Output directory for steering results")
    args = parser.parse_args()

    k = args.k
    model_path = args.model_path.split("/")[1]
    output_dir = args.output_dir

    # Get the top-k layer-heads per task (top layer-head combos)
    top_k_layer_heads = defaultdict(list)
    for task, task_dir in TASK_DIRS.items():
        json_path = os.path.join(task_dir, model_path, "pasta/layer_head_selection_results.json")
        if not os.path.exists(json_path):
            print(f"JSON file not found for task {task}: {json_path}")
            continue
        with open(json_path, "r") as f:
            results = json.load(f)
            # First flatten, then get top-k
            flat_results = [(layer, head, data["score"]) for layer, heads in results.items() for head, data in heads.items()]
            top_heads_all = sorted(flat_results, key=lambda x: x[2], reverse=True)
            # Don't break ties (just take the top k scores)
            top_heads = []
            for i, (layer, head, score) in enumerate(top_heads_all):
                if i < k or score == top_heads[-1][2]:
                    top_heads.append((layer, head, score))
            # top_heads = sorted(flat_results, key=lambda x: x[2], reverse=True)[:k]
            top_k_layer_heads[task].extend([(layer, head) for layer, head, score in top_heads])
        print(f"Top {k} layer-heads for task {task}: {len(top_k_layer_heads[task])} found.")
        print(f"Top 10 layer-heads for task {task}: {top_k_layer_heads[task][:10]}")

    tasks = list(top_k_layer_heads.keys())
    # Get the intersection of all top-k layer-heads across tasks
    common_layer_heads = set(top_k_layer_heads[tasks[0]])
    for task in tasks[1:]:
        common_layer_heads.intersection_update(top_k_layer_heads[task])

    # Create the final result dictionary ({"layer": [heads]})
    final_results = defaultdict(list)
    for layer, head in common_layer_heads:
        final_results[layer].append(int(head))
    print(f"Final results: {final_results}")
    print(f"Number of unique layer-heads: {sum(len(heads) for heads in final_results.values())}")

    # Save the final results to the output directory
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"pasta_layer_head_selection_results_k-{k}.json")
    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"Final results saved to: {output_path}")
