from __future__ import annotations

import json
import os
import sys
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

os.environ.setdefault("USE_TF", "0")
from transformers import HfArgumentParser

PROJ_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, PROJ_ROOT)

from instaboost_theory_exp.models.biased_learned_reasoner import RuleBiasAutoregWrapper
from instaboost_theory_exp.runners.run_learned_bias_theory_attacks import load_model_and_dataset_local
from my_datasets.attack_datasets import CoerceStateDataset


@dataclass
class BiasCoerceVarianceArguments:
    config_file: str = field(metadata={"help": "Path to JSON config."})
    output_dir: str = field(default=str(Path(PROJ_ROOT, "instaboost_theory_exp", "results")))
    device: str = field(default="cuda")


def _build_adv_suffix_for_delta(
    item: dict,
    k: int,
    num_vars: int,
    rulebook_plus_init_len: int,
    hot_prob: float,
    device: torch.device,
) -> torch.Tensor:
    """Build one fixed coerce-state suffix (Delta) to reuse across many X0."""
    n = num_vars
    L = rulebook_plus_init_len

    hints = item["hints"].to(device).float()  # (3, n)
    tgt1, tgt2, tgt3 = hints.chunk(3, dim=0)  # each (1, n)
    tgt0 = (torch.rand(1, n, device=device) < hot_prob).float()

    # Mirror theory coerce_state_variance scaling (2 * L).
    atk_rule1 = torch.cat([tgt0, 2 * L * (tgt1 - 0.5)], dim=-1)
    atk_rule2 = torch.cat([tgt1, 2 * L * (tgt2 - 0.5)], dim=-1)
    atk_rule3 = torch.cat([tgt2, 2 * L * (tgt3 - 0.5)], dim=-1)
    atk_rules = torch.cat([atk_rule1, atk_rule2, atk_rule3], dim=0).repeat(k, 1)
    init_token = torch.cat([torch.zeros_like(tgt0), 2 * L * (tgt0 - 0.5)], dim=-1)
    adv_suffix = torch.cat([atk_rules, init_token], dim=0)  # (3k+1, 2n)
    return adv_suffix


@torch.no_grad()
def run(config: Namespace, output_dir: Path) -> None:
    assert config.reasoner_types == ["learned"], "This runner supports learned models only."
    assert config.attack_name == "coerce_state", "Use attack_name='coerce_state' in config."

    num_deltas = int(getattr(config, "num_deltas", 1024))
    num_x0_per_delta = int(getattr(config, "num_x0_per_delta", 512))
    eval_batch_size = int(getattr(config, "batch_size", 512))

    device = torch.device(config.device)
    out_file = output_dir / "learned_coerce_state_variance_rule_attention_bias.csv"

    cols = [
        "reasoner_type",
        "train_seed",
        "num_vars",
        "embed_dim",
        "attn_bias",
        "num_attack_tokens",
        "num_deltas",
        "num_x0_per_delta",
        "variance",
        "variance_std",
    ]
    if out_file.exists():
        df = pd.read_csv(out_file)
        for c in cols:
            if c not in df.columns:
                df[c] = pd.NA
        df = df[cols]
    else:
        df = pd.DataFrame(columns=cols)

    completed_keys = set()
    for _, row in df.iterrows():
        completed_keys.add(
            (
                int(row["train_seed"]),
                int(row["num_vars"]),
                int(row["embed_dim"]),
                float(row["attn_bias"]),
                int(row["num_attack_tokens"]),
            )
        )

    for train_seed in config.train_seeds:
        for (n, embed_dim) in config.nd_pairs:
            base_model, base_dataset = load_model_and_dataset_local(
                embed_dim=embed_dim,
                num_vars=n,
                seed=train_seed,
            )
            base_model.num_steps = config.num_reasoner_steps
            base_model.eval().to(device)

            # Match paper's coerce-state variance setup: fixed-size sampled items
            # that define Delta, then evaluate each Delta over many random X0.
            delta_dataset = CoerceStateDataset(base_dataset, num_attack_tokens=4, dataset_len=num_deltas)
            hot_prob = float(delta_dataset.hot_prob)
            L = int(delta_dataset[0]["tokens"].size(0))

            for attn_bias in config.attn_bias_values:
                model = RuleBiasAutoregWrapper(base_model, attn_bias=attn_bias).eval().to(device)

                for k in config.num_attack_tokens:
                    run_key = (int(train_seed), int(n), int(embed_dim), float(attn_bias), int(k))
                    if run_key in completed_keys:
                        continue

                    variances = torch.tensor([], device=device)
                    pbar = tqdm(range(num_deltas))

                    for idx in pbar:
                        item = delta_dataset[idx]
                        adv_suffix = _build_adv_suffix_for_delta(
                            item=item,
                            k=int(k),
                            num_vars=n,
                            rulebook_plus_init_len=L,
                            hot_prob=hot_prob,
                            device=device,
                        )

                        preds = torch.tensor([], device=device)
                        done = 0
                        while done < num_x0_per_delta:
                            bsz = min(eval_batch_size, num_x0_per_delta - done)
                            other_tokens = (torch.rand(bsz, L, 2 * n, device=device) < hot_prob).float()
                            adv_tokens = torch.cat(
                                [other_tokens, adv_suffix.view(1, -1, 2 * n).repeat(bsz, 1, 1)],
                                dim=1,
                            )
                            adv_out = model(tokens=adv_tokens, rulebook_len=L - 1)
                            adv_pred = (adv_out.logits > 0).float()
                            preds = torch.cat([preds, adv_pred], dim=0)
                            done += bsz

                        # Variance of the same Delta evaluated on many X0.
                        delta_var = preds.var(dim=0).mean().view(1)
                        variances = torch.cat([variances, delta_var], dim=0)

                        pbar.set_description(
                            f"learned (n={n},d={embed_dim},b={attn_bias},k={k}) "
                            f"var={variances.mean().item():.3e}"
                        )

                    row = {
                        "reasoner_type": "learned",
                        "train_seed": train_seed,
                        "num_vars": n,
                        "embed_dim": embed_dim,
                        "attn_bias": float(attn_bias),
                        "num_attack_tokens": int(k),
                        "num_deltas": num_deltas,
                        "num_x0_per_delta": num_x0_per_delta,
                        "variance": variances.mean().item(),
                        "variance_std": variances.var().sqrt().item(),
                    }
                    df = pd.concat([df, pd.DataFrame(row, index=[0])], ignore_index=True)
                    df.to_csv(out_file, index=False)
                    completed_keys.add(run_key)

    print(f"Saved results to: {out_file}")


if __name__ == "__main__":
    parser = HfArgumentParser(BiasCoerceVarianceArguments)
    args = parser.parse_args_into_dataclasses()[0]

    config = Namespace(**json.load(open(args.config_file)))
    config.device = args.device

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run(config, output_dir)
