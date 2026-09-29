# InstABoost: Instruction Following by Principled Attention Boosting

InstABoost steers a language model by boosting the attention paid to the instruction tokens during generation. This repository contains the steering experiments (emotions, AI-risk personas, TriviaQA, TruthfulQA, safety, JailbreakBench, toxicity), the IFEval benchmark, the baselines (prompting, latent-steering methods, Spotlight, PASTA/AutoPASTA), and the scripts that aggregate the results.

## Repository layout

```
configs/pasta_ifeval/         PASTA head configs used for the reported IFEval numbers
scripts/
  setup_local_env.sh          create .venv and install the requirements
  run_steering.sh             steering experiments on any task / model
  run_ifeval.sh               IFEval experiment for llama3 | gemma | qwen3 | gpt-oss
  run_autopasta.sh            AutoPASTA head selection (coarse-to-fine)
  run_pasta_sweep_qwen3.sh    full per-(layer, head) PASTA sweep for Qwen3
src/
  steering.py                 steering experiments and baselines
  run_ifeval_experiment.py    IFEval experiment
  hooks.py                    attention-boosting hooks
  model_interaction_utils.py  generation with hooks
  autopasta_head_selection*.py, pasta.py, pasta_head_selection*.py   head selection
  build_ifeval_summary.py, ifeval_paired_stats.py,
  build_ifeval_per_instruction_summary.py, score_ifeval_judges.py    IFEval reporting
  SteeringResults-Data.ipynb, SteeringResults-Vis.ipynb              steering tables and figures
  ifeval_scripts/             IFEval instruction checkers (Google Research, Apache-2.0)
```

All datasets are downloaded automatically on first use. Outputs are written to `results/` (steering and IFEval) and `results_pasta/` (head selection); both are git-ignored.

## Setup

The experiments were run inside the `pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel` Docker image, which provides the CUDA-enabled PyTorch build (`requirements.txt` therefore does not pin `torch`). Install `git` in the container, then:

```
bash scripts/setup_local_env.sh          # creates .venv and installs requirements.txt
pip install -r requirements-ifeval.txt   # extra dependencies for IFEval and PASTA
```

Set the following variables, either exported in the shell or placed in a `.env` file at the repository root (loaded automatically):

```
HF_TOKEN=...              # Hugging Face token (gated models such as Llama-3 and Llama-Guard)
GOOGLE_API_KEY=...        # Gemini key for the LLM judges (fluency / relevance / AI-risk)
PERSPECTIVE_API_KEY=...   # Perspective API key for toxicity scoring
JUDGE_MODEL=...           # optional judge model, default gemini-2.5-flash
```

The reported results used `gemini-2.0-flash` as the judge, which Google has since retired from its API.

## Steering experiments

`scripts/run_steering.sh` runs `src/steering.py` on one or more tasks. Without a method list it runs the baseline and every method: prompting, InstABoost, Spotlight, PASTA, and the latent-steering baselines.

```
bash scripts/run_steering.sh all                    # Llama-3-8B-Instruct, all tasks and methods
bash scripts/run_steering.sh emotions:joy           # one emotion
bash scripts/run_steering.sh ai-risk-mcq:power      # one behavior
bash scripts/run_steering.sh safety                 # triviaqa | truthfulqa | safety | jbb | toxicity
```

The additional models run the prompt-based methods only:

```
METHODS="prompt instaboost spotlight" bash scripts/run_steering.sh all Qwen/Qwen3-14B
METHODS="prompt instaboost spotlight" bash scripts/run_steering.sh all openai/gpt-oss-20b
```

Per-model defaults (batch size, thinking disabled for Qwen3, low reasoning effort for GPT-OSS) are set in the script and can be overridden with the environment variables documented at the top of the file. Extra arguments are forwarded to `src/steering.py`. Each run writes `results/<task>/<model>/<method>/` and an `all_methods_results.json` summary. GPT-OSS-20B is loaded from its cached safetensors shards with the expert weights dequantized on CPU (`src/gpt_oss_loading.py`).

Aggregate the steering results with `src/SteeringResults-Data.ipynb` and produce the figures with `src/SteeringResults-Vis.ipynb`.

## IFEval

`scripts/run_ifeval.sh` runs `src/run_ifeval_experiment.py` with the settings behind the reported numbers. The InstABoost multiplier and the Spotlight target attention are selected on the validation split and evaluated on the test split.

| Key | Model | Methods | n_val / n_test | Notes |
|---|---|---|---|---|
| `llama3` | meta-llama/Meta-Llama-3-8B-Instruct | prompt, instaboost, spotlight, pasta | 100 / 400 | |
| `gemma` | google/gemma-7b-it | prompt, instaboost, spotlight, pasta | 100 / 441 | all prompts left after validation |
| `qwen3` | Qwen/Qwen3-14B | prompt, instaboost, spotlight, pasta | 100 / 400 | thinking disabled, multipliers {3, 5, 7, 9} |
| `gpt-oss` | openai/gpt-oss-20b | prompt, instaboost, spotlight | 100 / 400 | low reasoning effort, multipliers {3, 5, 7, 11} |

```
bash scripts/run_ifeval.sh llama3
bash scripts/run_ifeval.sh gemma
bash scripts/run_ifeval.sh qwen3
bash scripts/run_ifeval.sh gpt-oss
bash scripts/run_ifeval.sh llama3 --n_val 10 --n_test 20   # quick check; extra arguments are forwarded
```

Outputs go to `results/ifeval/<model>/<method>/` as `results.json`, `results_by_sample.json`, and `best_params.json`. Summaries and statistics:

```
python src/build_ifeval_summary.py                  # results/ifeval_summary.{md,tex,png,pdf}
python src/ifeval_paired_stats.py                   # paired bootstrap, McNemar and Wilcoxon tests
python src/build_ifeval_per_instruction_summary.py  # per-category and per-instruction-type accuracy
python src/score_ifeval_judges.py                   # optional fluency / relevance judge scores
```

PASTA on IFEval uses the AutoPASTA head configs in `configs/pasta_ifeval/` (top-50 heads for Llama-3 and Gemma, top-100 for Qwen3). To regenerate them:

```
bash scripts/run_autopasta.sh ifeval                                   # Qwen3-14B
MODEL=meta-llama/Meta-Llama-3-8B-Instruct bash scripts/run_autopasta.sh ifeval
MODEL=google/gemma-7b-it bash scripts/run_autopasta.sh ifeval
```

## PASTA head selection

AutoPASTA scores whole layers first, keeps the top `top_l`, then scores each head within those layers and writes `head_config_top<k>.json`. Both phases are resumable.

```
bash scripts/run_autopasta.sh all                          # emotions, ai-risk, safety, jbb on Qwen3-14B
bash scripts/run_autopasta.sh emotions:anger ai-risk-qa:wealth
bash scripts/run_autopasta.sh safety --phase 2
```

The full per-(layer, head) sweep used for the Llama-3 steering experiments is `src/pasta.py` (architectures supported by pastalib) and `scripts/run_pasta_sweep_qwen3.sh` (Qwen3). Per-task sweep scores are combined by `src/pasta_head_selection.py` into `pasta_layer_head_selection_results_k-<k>.json`, which `src/steering.py` reads via `--pasta_k`. Sharded sweeps are merged with `src/merge_pasta_shards.py`.
