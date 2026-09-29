"""Attention to applicable vs non-applicable rule rows, per reasoning step.

At each step we
split the rule rows into applicable and non-applicable (from the ground-truth proof state
s_{t-1}) and record the attention each row receives, so the two distributions can be compared
directly.

Two quantities per row:
  * `attn_weight`   -- post-softmax attention weight (the attention actually given to the row).
  * `logit_raw` -- the pre-softmax attention logit as-is.
  * `logit_centred` -- the same, minus the mean logit over the rule rows of that (example,
    step). Softmax is shift-invariant, so logits are in principle only defined up to a
    per-example offset. Empirically this model barely uses that freedom (offset std 0.35), and
    centring in fact *widens* the applicable groups, so the figures use `logit_raw`.

Everything is measured on the CLEAN, UNBOOSTED model on the clean rulebook. No InstaBoost /
attention bias, no adversarial suffix.
"""

from __future__ import annotations

import json
import math
import os
import sys
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ.setdefault("USE_TF", "0")
from transformers import HfArgumentParser

PROJ_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, PROJ_ROOT)

from instaboost_theory_exp.runners.run_learned_bias_theory_attacks import (
    hot,
    load_model_and_dataset_local,
)
from my_datasets.attack_datasets import SuppressRuleDataset

QUANTILES = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]


class PreSoftmaxLogitTap:
    """Capture pre-softmax attention logits of the single-layer / single-head GPT-2 block.

    `output_attentions=True` only exposes post-softmax weights, so we tap the fused QKV
    projection and recompute z = q_last . k^T / sqrt(head_dim) ourselves. This is exactly the
    quantity `eager_attention_forward` feeds to the softmax (before the causal mask). One entry
    is recorded per seqcls forward, i.e. per autoregressive reasoning step.
    """

    def __init__(self, model: torch.nn.Module):
        gpt2 = model.seqcls_model.gpt2s.transformer
        assert len(gpt2.h) == 1, "This tap assumes a 1-layer model."
        self.attn = gpt2.h[0].attn
        assert self.attn.num_heads == 1, "This tap assumes a 1-head model."
        self.per_step_logits: list[torch.Tensor] = []
        self._handle = self.attn.c_attn.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        q, k, _ = output.split(self.attn.embed_dim, dim=-1)
        z = torch.einsum("ne,nle->nl", q[:, -1, :], k)
        if self.attn.scale_attn_weights:
            z = z / math.sqrt(self.attn.head_dim)
        if self.attn.scale_attn_by_inverse_layer_idx:
            z = z / float(self.attn.layer_idx + 1)
        self.per_step_logits.append(z)

    def reset(self):
        self.per_step_logits = []

    def remove(self):
        self._handle.remove()


def ground_truth_states(infos: torch.Tensor, n: int) -> torch.Tensor:
    """Proof states (s_0..s_3) of the `make_common_stuff` rulebook: {a} -> {abcd} -> {abcdef} -> {abcdefg}."""
    a, b, c, d, e, f, g, h = infos.chunk(infos.size(-1), dim=-1)
    s0 = hot(a, n)
    s1 = s0 + hot(b, n) + hot(c, n) + hot(d, n)
    s2 = s1 + hot(e, n) + hot(f, n)
    s3 = s2 + hot(g, n)
    return torch.cat([s0, s1, s2, s3], dim=1).clamp(max=1).float()


@dataclass
class ApplicableAttentionArguments:
    config_file: str = field(metadata={"help": "Path to JSON config."})
    output_dir: str = field(
        default=str(Path(PROJ_ROOT, "instaboost_theory_exp", "results", "lambda_separation"))
    )
    device: str = field(default="cpu")


@torch.no_grad()
def run(config: Namespace, output_dir: Path) -> tuple[Path, Path]:
    assert config.reasoner_types == ["learned"], "This runner supports learned models only."
    assert config.num_samples % config.batch_size == 0

    # The eval datasets sample a fresh rulebook on every access, so seed for reproducibility.
    torch.manual_seed(int(getattr(config, "eval_seed", 0)))
    device = torch.device(config.device)
    rows: list[dict] = []
    mass: list[dict] = []

    for train_seed in config.train_seeds:
        for (n, embed_dim) in config.nd_pairs:
            model, base_dataset = load_model_and_dataset_local(
                embed_dim=embed_dim, num_vars=n, seed=train_seed
            )
            model.num_steps = config.num_reasoner_steps
            model.eval().to(device)
            tap = PreSoftmaxLogitTap(model)

            dataset = SuppressRuleDataset(base_dataset, num_attack_tokens=0, dataset_len=config.num_samples)
            dataloader = DataLoader(dataset, batch_size=config.batch_size)

            for batch in tqdm(dataloader, desc=f"attn (n={n},d={embed_dim},seed={train_seed})"):
                tokens = batch["tokens"].to(device).float()
                infos = batch["infos"].to(device)
                L = tokens.size(1)
                num_rules = L - 1  # last row is the initial-state token

                tap.reset()
                out = model(tokens=tokens, output_attentions=True)

                states = ground_truth_states(infos, n)
                antecedents = tokens[:, :num_rules, :n]
                consequents = tokens[:, :num_rules, n:]
                nonempty = antecedents.sum(dim=-1) > 0

                for t in range(1, config.num_reasoner_steps + 1):
                    z = tap.per_step_logits[t - 1]
                    p = out.all_seqcls_outputs[t - 1].attentions[0][:, 0, -1, :]
                    assert z.size(-1) == num_rules + t, (z.shape, num_rules + t)
                    max_dev = (z.softmax(dim=-1) - p).abs().max().item()
                    assert max_dev < 1e-4, f"pre-softmax recovery mismatch: {max_dev}"

                    z_rules, p_rules = z[:, :num_rules], p[:, :num_rules]
                    z_centred = z_rules - z_rules.mean(dim=-1, keepdim=True)

                    state = states[:, t - 1].unsqueeze(1)  # (N,1,n)
                    app = ((antecedents * (1.0 - state)).sum(dim=-1) == 0) & nonempty
                    is_new = app & ((consequents * (1.0 - state)).sum(dim=-1) > 0)

                    # The step at which a rule FIRST becomes applicable -- a fixed property of
                    # the rule, identical across steps. 0 means it never applies in 3 steps.
                    first_app = torch.zeros_like(app, dtype=torch.long)
                    for k in range(config.num_reasoner_steps, 0, -1):
                        applies_then = (
                            (antecedents * (1.0 - states[:, k - 1].unsqueeze(1))).sum(dim=-1) == 0
                        ) & nonempty
                        first_app = torch.where(applies_then, torch.full_like(first_app, k), first_app)

                    # Per-row observations, long format.
                    ex_idx, row_idx = torch.nonzero(nonempty, as_tuple=True)
                    applicable = app[ex_idx, row_idx]
                    newly = is_new[ex_idx, row_idx]
                    rows.extend(
                        {
                            "num_vars": int(n),
                            "embed_dim": int(embed_dim),
                            "step": t,
                            "num_rules": int(num_rules),
                            "group": (
                                ("applicable_new" if bool(nw) else "applicable_fired")
                                if bool(ap) else "non_applicable"
                            ),
                            "applicable": bool(ap),
                            "first_applicable_step": int(fa),
                            "attn_weight": float(w),
                            "logit_raw": float(zr),
                            "logit_centred": float(zc),
                        }
                        for ap, nw, fa, w, zr, zc in zip(
                            applicable.tolist(),
                            newly.tolist(),
                            first_app[ex_idx, row_idx].tolist(),
                            p_rules[ex_idx, row_idx].tolist(),
                            z_rules[ex_idx, row_idx].tolist(),
                            z_centred[ex_idx, row_idx].tolist(),
                        )
                    )

                    # Per-example attention mass by group (rule rows + the state tokens).
                    m_app = p_rules.masked_fill(~app, 0.0).sum(dim=-1)
                    m_non = p_rules.masked_fill(app | ~nonempty, 0.0).sum(dim=-1)
                    m_state = p[:, num_rules:].sum(dim=-1)
                    mass.extend(
                        {
                            "num_vars": int(n),
                            "embed_dim": int(embed_dim),
                            "step": t,
                            "num_rules": int(num_rules),
                            "num_applicable": int(app[i].sum().item()),
                            "mass_applicable": float(m_app[i].item()),
                            "mass_non_applicable": float(m_non[i].item()),
                            "mass_state_tokens": float(m_state[i].item()),
                        }
                        for i in range(tokens.size(0))
                    )

            tap.remove()

    rows_file = output_dir / "applicable_attention_rows.csv"
    pd.DataFrame.from_records(rows).to_csv(rows_file, index=False, float_format="%.6g")
    print(f"Saved per-row attention to: {rows_file}  ({len(rows)} rows)")
    # The per-example mass table is only ever aggregated, so it stays in memory.
    return rows_file, pd.DataFrame.from_records(mass)


def write_summary(rows_file: Path, mass: pd.DataFrame) -> None:
    rows = pd.read_csv(rows_file)

    recs = []
    for metric in ["attn_weight", "logit_raw", "logit_centred"]:
        for scope, sub_all in [("all_steps", rows)] + [
            (f"step_{int(t)}", g) for t, g in rows.groupby("step")
        ]:
            for group in ["applicable", "applicable_new", "applicable_fired", "non_applicable"]:
                sub = sub_all[sub_all["applicable"]] if group == "applicable" else sub_all[sub_all["group"] == group]
                if sub.empty:
                    continue
                q = sub[metric].quantile(QUANTILES)
                recs.append(
                    {
                        "metric": metric,
                        "scope": scope,
                        "group": group,
                        "n_rows": int(len(sub)),
                        "mean": float(sub[metric].mean()),
                        "median": float(sub[metric].median()),
                        **{f"q{int(round(k * 100)):02d}": float(v) for k, v in q.items()},
                    }
                )
    summary_file = rows_file.parent / "applicable_attention_summary.csv"
    pd.DataFrame(recs).to_csv(summary_file, index=False)

    mass_cols = ["mass_applicable", "mass_non_applicable", "mass_state_tokens"]
    mass_summary = mass.groupby("step").agg(
        num_examples=("num_applicable", "size"),
        num_applicable=("num_applicable", "mean"),
        **{
            f"{c}_{stat}": (c, stat)
            for c in mass_cols
            for stat in ("mean", "std")
        },
    )
    mass_file_out = rows_file.parent / "applicable_attention_mass_summary.csv"
    mass_summary.to_csv(mass_file_out)

    print(f"Saved summaries to: {summary_file} and {mass_file_out}")
    print("\nMean attention weight per rule row:")
    print(
        rows.pivot_table(index="step", columns="group", values="attn_weight", aggfunc="mean")
        .to_string(float_format=lambda x: f"{x:.4f}")
    )
    print("\nAttention mass by group (mean, std over examples):")
    print(mass_summary.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    parser = HfArgumentParser(ApplicableAttentionArguments)
    args = parser.parse_args_into_dataclasses()[0]

    config = Namespace(**json.load(open(args.config_file)))
    config.device = args.device

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_summary(*run(config, out_dir))
