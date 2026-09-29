"""One boosted rule against one unboosted rule, as added context K and bias B vary.

Every sample is two rules and an initial state; `K` further rules are appended as context.  Both
rules fire at the first step, so the measurement is simply whether each one's consequence is on.
No eligibility filter is needed -- the ground truth is the same for every sample -- and a rule
cannot fire "too early" because the tested step is the first one.

At `B = 0` the two rules are exchangeable, so `symmetry_gap` must come out at zero; that is the
only calibration this design needs.

Writes `learned_<experiment_name>.csv`; reruns skip rows already present.
"""

from __future__ import annotations

import json
import os
import sys
from argparse import Namespace
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ.setdefault("USE_TF", "0")
from transformers import HfArgumentParser

PROJ_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, PROJ_ROOT)

from instaboost_theory_exp.datasets.boost_tradeoff_dataset import BoostTradeoffDataset
from instaboost_theory_exp.models.biased_learned_reasoner import RuleBiasAutoregWrapper
from instaboost_theory_exp.runners.run_learned_bias_theory_attacks import load_model_and_dataset_local


def _require_eager_attention(base_model) -> None:
    """Ask GPT-2 for eager attention so `output_attentions=True` actually returns weights.

    Recent transformers default to SDPA and no longer silently fall back to eager when attention
    weights are requested -- they just come back as `None`.  Eager and SDPA are the same function
    up to floating point, and this experiment reads the attention map, so pin it.
    """
    for cfg in (base_model.seqcls_model.gpt2s.config,
                base_model.seqcls_model.gpt2s.transformer.config):
        cfg._attn_implementation = "eager"


@dataclass
class BoostTradeoffArguments:
    config_file: str = field(metadata={"help": "Path to JSON config."})
    output_dir: str = field(
        default=str(Path(PROJ_ROOT, "instaboost_theory_exp", "results", "boost_tradeoff"))
    )
    device: str = field(default="cuda")


KEY_COLS = ["train_seed", "num_vars", "embed_dim", "num_init_facts", "applicable_fraction",
            "num_competing", "attn_bias"]
COLS = ["num_samples", "reasoner_type", *KEY_COLS, "seq_len", "mean_applicable",
        "boosted_firing_rate", "unboosted_firing_rate", "joint_rate", "symmetry_gap",
        "boosted_logit_mean", "unboosted_logit_mean",
        "attn_x", "attn_y", "attn_competing"]


@torch.no_grad()
def run(config: Namespace, output_dir: Path) -> None:
    assert config.num_samples % config.batch_size == 0
    device = torch.device(config.device)
    out_file = output_dir / f"learned_{config.experiment_name}.csv"

    if out_file.exists():
        df = pd.read_csv(out_file)
        for c in COLS:
            if c not in df.columns:
                df[c] = pd.NA
        df = df[COLS]
    else:
        df = pd.DataFrame(columns=COLS)
    done = {(int(r.train_seed), int(r.num_vars), int(r.embed_dim), int(r.num_init_facts),
             float(r.applicable_fraction), int(r.num_competing), float(r.attn_bias))
            for r in df.itertuples()}

    for train_seed in config.train_seeds:
        for (n, embed_dim) in config.nd_pairs:
            base_model, base_dataset = load_model_and_dataset_local(
                embed_dim=embed_dim, num_vars=n, seed=train_seed
            )
            base_model.num_steps = config.num_reasoner_steps
            _require_eager_attention(base_model)
            base_model.eval().to(device)

            grid = product(config.attn_bias_values,
                           getattr(config, "num_init_facts", [3]),
                           getattr(config, "applicable_fractions", [0.0]),
                           config.num_competing)
            for attn_bias, j, frac, K in grid:
                key = (int(train_seed), int(n), int(embed_dim), int(j), float(frac),
                       int(K), float(attn_bias))
                if key in done:
                    continue
                model = RuleBiasAutoregWrapper(base_model, attn_bias=attn_bias).eval().to(device)
                dataset = BoostTradeoffDataset(base_dataset, num_competing=int(K),
                                         dataset_len=config.num_samples,
                                         applicable_fraction=float(frac),
                                         num_init_facts=int(j),
                                         seed=int(getattr(config, "seed", 0)),
                                         pool_size=int(getattr(config, "pool_size", 256)))
                loader = DataLoader(dataset, batch_size=config.batch_size)

                fx, fy, joint, napp = [], [], [], []
                lx, ly, ax, ay, ac = [], [], [], [], []
                pbar = tqdm(loader)
                for batch in pbar:
                    tokens = batch["tokens"].to(device)
                    xi = batch["x_index"].to(device).view(-1, 1)
                    yi = batch["y_index"].to(device).view(-1, 1)
                    N = tokens.size(0)
                    # The boosted rule is row 0; the initial-state token sits at row 1.
                    xpos = torch.zeros(N, 1, dtype=torch.long, device=device)
                    out = model(tokens=tokens, output_attentions=True, rulebook_len=1,
                                bias_positions=xpos)
                    step = out.logits[:, 0]
                    pred = (step > 0).long()
                    on_x = (pred.gather(1, xi) == 1).squeeze(1)
                    on_y = (pred.gather(1, yi) == 1).squeeze(1)
                    fx.append(on_x); fy.append(on_y); joint.append(on_x & on_y)
                    lx.append(step.gather(1, xi).squeeze(1))
                    ly.append(step.gather(1, yi).squeeze(1))
                    napp.append(batch["num_applicable"].to(device).float())

                    attn = out.all_seqcls_outputs[0].attentions[0][:, 0][:, -1]
                    ypos = 1 + batch["y_position"].to(device).view(-1, 1)
                    y_share = attn.gather(1, ypos).squeeze(1)
                    ax.append(attn[:, 0])
                    ay.append(y_share)
                    ac.append(attn[:, 1:1 + 1 + K].sum(-1) - y_share)
                    pbar.set_description(
                        f"B={attn_bias} K={K} f={frac} j={j} "
                        f"x={torch.cat(fx).float().mean():.3f} y={torch.cat(fy).float().mean():.3f}")

                mean = lambda p: torch.cat(p).float().mean().item()
                xr, yr = mean(fx), mean(fy)
                row = {"num_samples": config.num_samples, "reasoner_type": "learned",
                       "train_seed": train_seed, "num_vars": n, "embed_dim": embed_dim,
                       "num_init_facts": int(j), "applicable_fraction": float(frac),
                       "num_competing": int(K), "attn_bias": attn_bias,
                       "seq_len": 3 + int(K), "mean_applicable": mean(napp),
                       "boosted_firing_rate": xr, "unboosted_firing_rate": yr,
                       "joint_rate": mean(joint), "symmetry_gap": xr - yr,
                       "boosted_logit_mean": mean(lx), "unboosted_logit_mean": mean(ly),
                       "attn_x": mean(ax), "attn_y": mean(ay), "attn_competing": mean(ac)}
                df = pd.concat([df, pd.DataFrame(row, index=[0])], ignore_index=True)
                df.to_csv(out_file, index=False)
                done.add(key)

    print(f"Saved results to: {out_file}")


if __name__ == "__main__":
    args = HfArgumentParser(BoostTradeoffArguments).parse_args_into_dataclasses()[0]
    config = Namespace(**json.load(open(args.config_file)))
    config.device = args.device
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    run(config, out)
