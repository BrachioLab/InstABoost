# instaboost_theory_exp

InstaBoost experiments for theory-style attacks on learned reasoners in `tf_logic`.

This folder implements **InstaBoost**: a defense that boosts attention to rule tokens by adding a
fixed positive bias `B` to the rulebook rows' attention logits (pre-softmax) at every reasoning
step. It then measures how this changes the success of the theory attacks as the adversarial
suffix is repeated, what the boost costs in a controlled two-rule setting, and how the cost of
subverting the boosted model grows with `B`.

## Important dependency

This code is **not standalone**. To run `instaboost_theory_exp`, clone the full `tf_logic`
repository:

- https://github.com/AntonXue/tf_logic

The InstaBoost scripts import core models, datasets, and checkpoint-loading utilities from
that repo:

- `models/*`
- `my_datasets/*`
- `_dump/autoreg_experiments/*` (learned model checkpoints)

## Setup

Clone the `tf_logic` repository and place this directory at `tf_logic/instaboost_theory_exp`:

```bash
git clone https://github.com/AntonXue/tf_logic.git
cd tf_logic
```

Create the environment from the `tf_logic` root using either option:

```bash
# Option A: conda
conda env create -f environment.yml
conda activate tfl
```

```bash
# Option B: pip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then install this directory's own requirement on top:

```bash
pip install -r instaboost_theory_exp/requirements.txt
```

This directory needs `transformers>=4.52`. The parent `tf_logic` pin of `4.40.2` cannot run
these runners: GPT-2 only accepts a 4-D attention mask (which `RuleBiasAutoregWrapper` needs to
bias rulebook rows) from 4.52 onward, and earlier versions fail with a tensor-size mismatch in
the attention mask.

## What it does

- Applies the InstaBoost rule-attention bias defense to learned autoregressive reasoners.
- Runs bias sweeps for the `fact_amnesia` and `suppress_rule` attacks, in two conventions:
  - **replace** mode: the adversarial suffix replaces rulebook rows, so the suffix is capped at
    the rulebook size;
  - **append** mode: the suffix is appended and the rulebook kept whole, so the sweep can run
    well past that cap.
- Runs the coerce-state variance analysis (fixed attack suffix `Δ`, many random `X0`).
- Runs the **boost trade-off** experiment: one boosted rule against one exchangeable unboosted
  rule as added context `K` and bias `B` vary, measuring what the bias buys and what it costs.
- Measures the **subversion cost**: the number of suffix repetitions needed to reach a fixed
  failure level, as a function of `B`.
- Checks attention to applicable vs non-applicable rules on the clean, unboosted reasoner.

## Structure

- `models/biased_learned_reasoner.py`
  - Wrapper model (`RuleBiasAutoregWrapper`) that injects rulebook-key attention bias.
- `datasets/append_attack_datasets.py`
  - Append-mode variants of the attack datasets.
- `datasets/boost_tradeoff_dataset.py`
  - `BoostTradeoffDataset`: one boosted rule and one exchangeable unboosted rule, plus `K`
    further rules appended as context. `K = 0` is literally two rules and the initial state.
- `runners/run_learned_bias_theory_attacks.py`
  - Main attack runner. Writes `learned_<attack>_rule_attention_bias.csv`, with an `_append`
    suffix when the config sets `append_attack_tokens`.
- `runners/run_learned_bias_coerce_state_variance.py`
  - Coerce-state variance runner: sample many `Δ`, evaluate each on many `X0`.
  - Writes `learned_coerce_state_variance_rule_attention_bias.csv`.
- `runners/run_boost_tradeoff.py`
  - Firing rate of each rule as `B` and `K` vary. Writes `learned_boost_tradeoff.csv`.
- `runners/run_applicable_attention.py`
  - Per-rule-row attention (post-softmax weight and centred pre-softmax logit) on the clean
    model, split by applicable vs non-applicable, plus per-group attention mass.
- `plotting/plot_full_defense_summary.py`
  - Three-panel defense summary (fact amnesia, rule suppression, state-coercion variance)
    from the replace-mode sweeps.
- `plotting/plot_full_defense_summary_append.py`
  - The same summary from the append-mode sweeps.
- `plotting/plot_repeats_to_threshold_vs_bias.py`
  - Subversion cost: suffix repetitions to a fixed failure level against `B`, and against the
    attention multiplier `M = exp(B)`.
- `plotting/plot_boost_tradeoff.py`
  - Boost trade-off figures: both firing-rate curves plus the tolerance trade-off.
- `plotting/plot_applicable_attention.py`
  - Attention distributions by rule status (post-softmax weights and pre-softmax logits).
- `configs/*.json`
  - Experiment configs (bias values, repeats, samples, seeds, etc.).
- `results/`
  - Attack sweep CSVs and the replace-mode defense summary.
- `results/subversion_lower_bounds/`
  - Append-mode defense summary, subversion-cost figures, and the repetitions-to-threshold table.
- `results/boost_tradeoff/`
  - Boost trade-off sweep CSV and figures.
- `results/lambda_separation/`
  - Applicable vs non-applicable attention: per-row CSV, summaries, and figures.

## Typical usage

Run from the `tf_logic` root.

```bash
# Attack sweeps, replace mode
python instaboost_theory_exp/runners/run_learned_bias_theory_attacks.py \
  --config_file instaboost_theory_exp/configs/learned_suppress_rule_bias.json \
  --output_dir instaboost_theory_exp/results --device cpu

python instaboost_theory_exp/runners/run_learned_bias_theory_attacks.py \
  --config_file instaboost_theory_exp/configs/learned_fact_amnesia_bias.json \
  --output_dir instaboost_theory_exp/results --device cpu

# Attack sweeps, append mode (suffix appended, rulebook kept whole)
python instaboost_theory_exp/runners/run_learned_bias_theory_attacks.py \
  --config_file instaboost_theory_exp/configs/learned_suppress_rule_bias_append.json \
  --output_dir instaboost_theory_exp/results --device cpu

python instaboost_theory_exp/runners/run_learned_bias_theory_attacks.py \
  --config_file instaboost_theory_exp/configs/learned_fact_amnesia_bias_append.json \
  --output_dir instaboost_theory_exp/results --device cpu
```

```bash
# Coerce-state variance analysis (fixed Delta, many X0)
python instaboost_theory_exp/runners/run_learned_bias_coerce_state_variance.py \
  --config_file instaboost_theory_exp/configs/learned_coerce_state_bias.json \
  --output_dir instaboost_theory_exp/results --device cpu

# Optional: extend the B = 2.5 sweep past the rulebook cap (reruns skip rows already present)
python instaboost_theory_exp/runners/run_learned_bias_coerce_state_variance.py \
  --config_file instaboost_theory_exp/configs/learned_coerce_state_variance_extend.json \
  --output_dir instaboost_theory_exp/results --device cpu
```

```bash
# Defense summaries
python instaboost_theory_exp/plotting/plot_full_defense_summary.py
python instaboost_theory_exp/plotting/plot_full_defense_summary_append.py

# Subversion cost vs bias (reads the append-mode sweeps and the variance CSV)
python instaboost_theory_exp/plotting/plot_repeats_to_threshold_vs_bias.py
```

```bash
# Boost trade-off: one boosted rule vs one exchangeable unboosted rule, K rules added as context
python instaboost_theory_exp/runners/run_boost_tradeoff.py \
  --config_file instaboost_theory_exp/configs/boost_tradeoff.json \
  --output_dir instaboost_theory_exp/results/boost_tradeoff --device cpu

python instaboost_theory_exp/plotting/plot_boost_tradeoff.py
```

```bash
# Attention to applicable vs non-applicable rules (clean model, no bias, no attack)
python instaboost_theory_exp/runners/run_applicable_attention.py \
  --config_file instaboost_theory_exp/configs/applicable_attention.json \
  --output_dir instaboost_theory_exp/results/lambda_separation --device cpu

python instaboost_theory_exp/plotting/plot_applicable_attention.py
```

## Checkpoints

The upstream repo may not include all learned checkpoints by default.
This project expects local checkpoints under:

- `_dump/autoreg_experiments/SynSAR_gpt2_d{embed_dim}_L1_H1__DMY_nv{num_vars}_nr32_exph3.000__ntr262144_ntt65536_bsz512_steps8192_lr0.00050_seed601/checkpoint-*/model.safetensors`

The runs here use `(num_vars, embed_dim) = (64, 128)` with `seed=601`.

If missing, regenerate with `experiments/autoreg_experiments_base.py`:

```bash
python experiments/autoreg_experiments_base.py \
  --model_name gpt2 \
  --embed_dim 128 \
  --num_layers 1 \
  --num_heads 1 \
  --dataset_name autoreg_custom \
  --num_vars 64 \
  --num_steps 3 \
  --num_rules 32 \
  --exp_hots 3.0 \
  --train_len 262144 \
  --eval_len 65536 \
  --batch_size 512 \
  --num_epochs 16 \
  --learning_rate 0.0005 \
  --seed 601
```

Note: this training recipe is reconstructed from the run-name conventions in the upstream repo.

## Notes

- These scripts are currently focused on `reasoner_types = ["learned"]`.
- `RuleBiasAutoregWrapper` needs a transformers version that passes a 4-D additive attention mask
  through GPT-2. 4.57 works; 4.40 flattens the mask and fails. The wrapper builds the causal mask
  itself when `_prepare_4d_causal_attention_mask_with_cache_position` is absent, which is
  equivalent, not an approximation.
- Runners skip `(seed, n, d, bias, k)` rows already present in their output CSV, so a preempted
  sweep can simply be resubmitted.

## Metrics

The result CSVs contain multiple metrics. The key ones used in the figures are:

- Accuracy - `raw_state_acc`
  - Clean (no-attack) exact state accuracy across all predicted steps.
- Failure Induction Rate - `adv_ns3_state_acc`
  - Attacked-input exact state accuracy across all 3 predicted steps (`ns3` = steps 1..3).
  - This is the y-axis of the fact-amnesia and rule-suppression panels.
- Variance - `variance` (from `learned_coerce_state_variance_rule_attention_bias.csv`)
  - Coerce-state concentration metric: output variance across many random `X0` for a fixed
    attack suffix `Δ`.
  - Lower variance means stronger concentration/coercion effect; higher variance means weaker
    coercion.

Additional attack metrics in the attack CSVs:

- `simple_asr`
  - Untargeted success proxy: attack counts as successful if any bit/step differs from clean
    labels.
- `simple_ssr`
  - Targeted success proxy (attack-specific):
    - fact amnesia: target fact suppressed
    - suppress rule: target consequence suppressed
