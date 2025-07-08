# InstABoost: Instruction Following by Boosting Attention of Large Language Models

[<a href="https://arxiv.org/abs/2506.13734">Paper</a>] [<a href="https://debugml.github.io/instaboost/">Blog post</a>] 

Official implementation for "Instruction Following by Boosting Attention of Large Language Models".

Authors: Vitoria Guardieiro\*, Adam Stein\*, Avishree Khare\*, Eric Wong

## Steps to reproduce experiments

### Step 1: Setup

The experiments are designed to be run within a Docker container. We use the `pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel` image. Install `git` in your container.

Install python dependencies with:
```
pip install -r requirements.txt
```

Then, you need to set environment variables with your Huggingface token (for running experiments with gated models), your Google API key (for llm-judge), and Perspective API key (to evaluate generation toxicity).

Replace `hf_XXXX` with your Huggingface token, `g_XXXX` with your Google API key, and `p_XXXX` with you Perspective API key below.

```
export HF_TOKEN=hf_XXXX
export GOOGLE_API_KEY=g_XXXX
export PERSPECTIVE_API_KEY=p_XXXX
```

### Step 2: Running experiments

The experiments are executed with the python script `src/steering.py`.

You can run all experiments at once with:
```
bash scripts/run_experiments.sh
```

Otherwise, you can run individual experiments by directly calling `src/steering.py` with the hyperparameters of dataset, model, and other hyperparameters. For example, you can run the experiment for the emotion joy with the following:

```
python src/steering.py \
        --dataset emotions-qa \
        --dataset_dir data/emotions_train_joy \
        --model_path meta-llama/Meta-Llama-3-8B-Instruct \
        --output_dir results/emotions/joy/ \
        --grid_layers \
        --use_fluency \
        --normalize_dir \
        --batch_size 1
```

We also provide scripts to run all methods on each dataset on the folder `scripts`. 

### Step 3: Aggregate results and generate figures

To aggregate all results, use the notebook `src/SteeringResults-Data.ipynb`. 

Then, you can generate the figures with the notebook `src/SteeringResults-Vis.ipynb`.

## Citation

If you find our work helpful, please cite:

```bibtex
@article{guardieiro2025instruction,
  title={Instruction Following by Boosting Attention of Large Language Models},
  author={Guardieiro, Vitoria and Stein, Adam and Khare, Avishree and Wong, Eric},
  journal={arXiv preprint arXiv:2506.13734},
  year={2025},
  url={https://arxiv.org/abs/2506.13734}
}
```