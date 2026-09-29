from __future__ import annotations

import json
import os
import sys
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ.setdefault("USE_TF", "0")
from transformers import HfArgumentParser

PROJ_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, PROJ_ROOT)

from instaboost_theory_exp.models.biased_learned_reasoner import RuleBiasAutoregWrapper
from models.seqcls_models.gpt2 import MyGPT2Config, MyGPT2SeqClsModel
from models.task_models import AutoregKStepsTaskModel
from my_datasets.attack_datasets import CoerceStateDataset, FactAmnesiaDataset, SuppressRuleDataset
from instaboost_theory_exp.datasets.append_attack_datasets import APPEND_DATASETS
from my_datasets.task_datasets import AutoregDiamondTokensDataset


def hot(i: torch.Tensor, n: int) -> torch.Tensor:
    return F.one_hot(i, n)


@dataclass
class BiasAttackArguments:
    config_file: str = field(metadata={"help": "Path to JSON config."})
    output_dir: str = field(default=str(Path(PROJ_ROOT, "instaboost_theory_exp", "results")))
    device: str = field(default="cuda")


def load_model_and_dataset_local(
    embed_dim: int,
    num_vars: int,
    seed: int,
    num_steps: int = 3,
    num_rules: int = 32,
    exph: float = 3.0,
    train_len: int = 262144,
    eval_len: int = 65536,
    num_train_steps: int = 8192,
    learning_rate: float = 5e-4,
    batch_size: int = 512,
    max_test_seq_len: int = 1024,
):
    seqcls_model = MyGPT2SeqClsModel(
        MyGPT2Config(
            input_dim=2 * num_vars,
            num_vars=num_vars,
            embed_dim=embed_dim,
            num_heads=1,
            num_layers=1,
            use_positional_embedding=False,
            max_seq_len=max_test_seq_len,
        )
    )
    model = AutoregKStepsTaskModel(seqcls_model=seqcls_model, num_steps=num_steps, train_supervision_mode="all")

    run_name = (
        f"SynSAR_gpt2_d{embed_dim}_L1_H1__DMY_nv{num_vars}_nr{num_rules}_exph{exph:.3f}__"
        f"ntr{train_len}_ntt{eval_len}_bsz{batch_size}_steps{num_train_steps}_lr{learning_rate:.5f}_seed{seed}"
    )
    ckpt_root = Path(PROJ_ROOT, "_dump", "autoreg_experiments", run_name)
    if not ckpt_root.exists():
        raise FileNotFoundError(f"Local checkpoint dir not found: {ckpt_root}")

    checkpoints = sorted(
        [int(p.name.split("-")[-1]) for p in ckpt_root.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under: {ckpt_root}")
    ckpt_file = ckpt_root / f"checkpoint-{checkpoints[-1]}" / "model.safetensors"

    with safe_open(str(ckpt_file), framework="pt", device="cpu") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}

    if "seqcls_model.gpt2s.transformer.wpe.weight" in tensors:
        del tensors["seqcls_model.gpt2s.transformer.wpe.weight"]
    model.load_state_dict(tensors, strict=False)

    model.seqcls_model.gpt2s.transformer.wpe = torch.nn.Embedding(max_test_seq_len, embed_dim)
    model.seqcls_model.gpt2s.transformer.wpe.requires_grad_(False)
    model.seqcls_model.gpt2s.transformer.wpe.weight.fill_(0)

    dataset = AutoregDiamondTokensDataset(
        num_vars=num_vars,
        num_rules=num_rules,
        exp_hots=exph,
        dataset_len=train_len,
    )
    return model, dataset


def _build_dataset(config: Namespace, base_dataset, k: int):
    # Append mode swaps in local subclasses that keep the base rulebook whole; the upstream
    # my_datasets classes are used unchanged for the default replace-mode behaviour.
    append = bool(getattr(config, "append_attack_tokens", False))
    n_tokens = 3 * k + 1 if config.attack_name == "coerce_state" else k
    if append:
        cls = APPEND_DATASETS.get(config.attack_name)
        if cls is None:
            raise ValueError(f"Unknown attack_name {config.attack_name}")
        return cls(base_dataset, num_attack_tokens=n_tokens, dataset_len=config.num_samples)
    if config.attack_name == "suppress_rule":
        return SuppressRuleDataset(base_dataset, num_attack_tokens=k, dataset_len=config.num_samples)
    if config.attack_name == "fact_amnesia":
        return FactAmnesiaDataset(base_dataset, num_attack_tokens=k, dataset_len=config.num_samples)
    if config.attack_name == "coerce_state":
        return CoerceStateDataset(base_dataset, num_attack_tokens=n_tokens,
                                  dataset_len=config.num_samples)
    raise ValueError(f"Unknown attack_name {config.attack_name}")


def _build_adv_suffix(config: Namespace, batch: dict, k: int, n: int, L: int, device: torch.device):
    N = batch["tokens"].size(0)
    infos = batch["infos"].to(device)
    a, b, c, d, e, f, g, h = infos.chunk(infos.size(-1), dim=-1)
    init_token = torch.cat([torch.zeros(N, 1, n, device=device), hot(a, n)], dim=-1)

    if config.attack_name == "suppress_rule":
        atk_rule = torch.cat([hot(c, n) + hot(d, n), -1 * hot(f, n)], dim=-1)
        adv_suffix = torch.cat([atk_rule.view(N, 1, 2 * n).repeat(1, k - 1, 1), init_token.view(N, 1, -1)], dim=1)

    elif config.attack_name == "fact_amnesia":
        atk_rule = torch.cat([torch.zeros(N, 1, n, device=device), -L * hot(a, n)], dim=-1)
        adv_suffix = torch.cat([atk_rule.view(N, 1, -1).repeat(1, k - 1, 1), init_token.view(N, 1, -1)], dim=1)

    elif config.attack_name == "coerce_state":
        hints = batch["hints"].to(device)
        tgt1, tgt2, tgt3 = hints.chunk(3, dim=1)
        tgt0 = hot(a, n).view(-1, 1, n)

        atk_rule1 = torch.cat([tgt0, L * (tgt1 - 0.5)], dim=-1)
        atk_rule2 = torch.cat([tgt1, L * (tgt2 - 0.5)], dim=-1)
        atk_rule3 = torch.cat([tgt2, L * (tgt3 - 0.5)], dim=-1)
        one_copy = torch.cat([atk_rule1, atk_rule2, atk_rule3], dim=1)

        adv_suffix = torch.cat([one_copy.repeat(1, k, 1), init_token.view(N, 1, -1)], dim=1)

    else:
        raise ValueError(f"Unknown attack_name {config.attack_name}")

    return adv_suffix


@torch.no_grad()
def run(config: Namespace, output_dir: Path):
    assert config.reasoner_types == ["learned"], "This runner supports learned models only."
    assert config.num_samples % config.batch_size == 0

    device = torch.device(config.device)
    # A run tag keeps incompatible conventions in separate files: the resume logic keys on
    # (seed, n, d, bias, k) only, so append-mode rows would silently shadow replace-mode ones.
    run_tag = str(getattr(config, "run_tag", "") or "")
    suffix = f"_{run_tag}" if run_tag else ""
    out_file = output_dir / f"learned_{config.attack_name}_rule_attention_bias{suffix}.csv"

    cols = [
        "num_samples",
        "reasoner_type",
        "train_seed",
        "num_vars",
        "embed_dim",
        "num_attack_tokens",
        "attn_bias",
        "raw_state_acc",
        "simple_ssr",
        "simple_asr",
        "adv_ns1_state_acc",
        "adv_ns2_state_acc",
        "adv_ns3_state_acc",
        "adv_ns1_atk_wts",
        "adv_ns2_atk_wts",
        "adv_ns3_atk_wts",
        "adv_ns1_atk_wts_std",
        "adv_ns2_atk_wts_std",
        "adv_ns3_atk_wts_std",
    ]
    if out_file.exists():
        df = pd.read_csv(out_file)
        # Keep backward compatibility if older CSV is missing columns.
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

    # Optional early-stop logic for a specific bias value (useful for large-k extension runs).
    early_stop_bias = getattr(config, "early_stop_bias_value", None)
    early_stop_asr = float(getattr(config, "early_stop_asr_threshold", 1.0))

    for train_seed in config.train_seeds:
        for (n, embed_dim) in config.nd_pairs:
            base_model, base_dataset = load_model_and_dataset_local(
                embed_dim=embed_dim,
                num_vars=n,
                seed=train_seed,
            )
            base_model.num_steps = config.num_reasoner_steps
            base_model.eval().to(device)

            # attack_datasets.make_common_stuff always injects 6 special rules.
            # So we must keep at least 6 non-attack rules in the base rulebook.
            append = bool(getattr(config, "append_attack_tokens", False))
            if append:
                # Nothing is taken from the rulebook, so k is bounded only by the model's
                # context. Sequence = base rules + init token + the attack suffix.
                max_seq = int(getattr(config, "max_test_seq_len", 1024))
                suffix_len = (lambda kk: 3 * kk + 2 if config.attack_name == "coerce_state" else kk)
                longest = int(base_dataset.num_rules) + 1 + suffix_len(max(config.num_attack_tokens))
                if longest > max_seq:
                    raise ValueError(
                        f"append_attack_tokens: longest sequence {longest} exceeds max_test_seq_len "
                        f"{max_seq}. Reduce the largest num_attack_tokens."
                    )
            elif config.attack_name in {"suppress_rule", "fact_amnesia"}:
                max_k = int(base_dataset.num_rules) - 6
                bad_ks = [int(k) for k in config.num_attack_tokens if int(k) > max_k]
                if bad_ks:
                    raise ValueError(
                        f"Invalid num_attack_tokens for {config.attack_name}: {bad_ks}. "
                        f"With base num_rules={base_dataset.num_rules}, maximum supported k is {max_k}. "
                        "Reduce k or use a dataset construction that does not subtract k from the base rule count."
                    )
            elif config.attack_name == "coerce_state":
                # In _build_dataset, coerce_state consumes total_attack_tokens = 3*k + 1.
                # CoerceStateDataset then sets num_rules = base_num_rules - total_attack_tokens.
                # We need num_rules >= 6 for make_common_stuff.
                base_num_rules = int(base_dataset.num_rules)
                max_k = (base_num_rules - 7) // 3
                bad_ks = [int(k) for k in config.num_attack_tokens if int(k) > max_k]
                if bad_ks:
                    raise ValueError(
                        f"Invalid num_attack_tokens for coerce_state: {bad_ks}. "
                        f"With base num_rules={base_num_rules} and total_attack_tokens=3*k+1, "
                        f"maximum supported k is {max_k}. "
                        "Reduce k or use a dataset construction that does not subtract 3*k+1 from the base rule count."
                    )

            for attn_bias in config.attn_bias_values:
                model = RuleBiasAutoregWrapper(base_model, attn_bias=attn_bias).eval().to(device)
                stop_this_bias = False

                for k in config.num_attack_tokens:
                    if stop_this_bias:
                        break

                    run_key = (int(train_seed), int(n), int(embed_dim), float(attn_bias), int(k))
                    if run_key in completed_keys:
                        continue

                    atk_dataset = _build_dataset(config, base_dataset, k)
                    dataloader = DataLoader(atk_dataset, batch_size=config.batch_size)

                    raw_state_hits = torch.tensor([], device=device)
                    adv_ns1_state_hits = torch.tensor([], device=device)
                    adv_ns2_state_hits = torch.tensor([], device=device)
                    adv_ns3_state_hits = torch.tensor([], device=device)

                    adv_ns1_atk_wts = torch.tensor([], device=device)
                    adv_ns2_atk_wts = torch.tensor([], device=device)
                    adv_ns3_atk_wts = torch.tensor([], device=device)

                    simple_ssr_oks = torch.tensor([], device=device)
                    simple_asr_oks = torch.tensor([], device=device)

                    pbar = tqdm(dataloader)
                    for batch in pbar:
                        raw_tokens = batch["tokens"].to(device)
                        N, L = raw_tokens.size(0), raw_tokens.size(1)

                        raw_out = model(tokens=raw_tokens, output_attentions=True, rulebook_len=L - 1)
                        raw_pred = (raw_out.logits > 0).long()

                        infos = batch["infos"].to(device)
                        a, b, c, d, e, f, g, h = infos.chunk(infos.size(-1), dim=-1)
                        raw_labels = torch.cat(
                            [
                                hot(a, n) + hot(b, n) + hot(c, n) + hot(d, n),
                                hot(a, n) + hot(b, n) + hot(c, n) + hot(d, n) + hot(e, n) + hot(f, n),
                                hot(a, n) + hot(b, n) + hot(c, n) + hot(d, n) + hot(e, n) + hot(f, n) + hot(g, n),
                            ],
                            dim=1,
                        )

                        adv_labels = batch["labels"].to(device)
                        if adv_labels.size(1) == 4:
                            adv_labels = adv_labels[:, 1:]
                        assert adv_labels.shape == (N, 3, n)

                        adv_suffix = _build_adv_suffix(config, batch, k, n, L, device)
                        adv_tokens = torch.cat([raw_tokens, adv_suffix], dim=1)
                        adv_out = model(tokens=adv_tokens, output_attentions=True, rulebook_len=L - 1)
                        adv_pred = (adv_out.logits > 0).long()

                        all_raw_hits = raw_pred == raw_labels
                        raw_state_hits = torch.cat([raw_state_hits, all_raw_hits.all(dim=-1).all(dim=-1)])

                        adv_hits = adv_pred == adv_labels
                        adv_ns1_state_hits = torch.cat([adv_ns1_state_hits, adv_hits[:, 0:1].all(dim=-1).all(dim=-1)])
                        adv_ns2_state_hits = torch.cat([adv_ns2_state_hits, adv_hits[:, 0:2].all(dim=-1).all(dim=-1)])
                        adv_ns3_state_hits = torch.cat([adv_ns3_state_hits, adv_hits[:, 0:3].all(dim=-1).all(dim=-1)])

                        adv_out1, adv_out2, adv_out3 = adv_out.all_seqcls_outputs
                        adv_attn1 = adv_out1.attentions[0][:, 0]
                        adv_attn2 = adv_out2.attentions[0][:, 0]
                        adv_attn3 = adv_out3.attentions[0][:, 0]

                        atk_len = adv_suffix.size(1)
                        adv_ns1_atk_wts = torch.cat([adv_ns1_atk_wts, adv_attn1[:, -1, L : L + atk_len]])
                        adv_ns2_atk_wts = torch.cat([adv_ns2_atk_wts, adv_attn2[:, -1, L : L + atk_len]])
                        adv_ns3_atk_wts = torch.cat([adv_ns3_atk_wts, adv_attn3[:, -1, L : L + atk_len]])

                        if config.attack_name == "fact_amnesia":
                            simple_ssr_oks = torch.cat([simple_ssr_oks, (adv_pred * hot(a, n)).sum(dim=(-1, -2)) == 0])
                        elif config.attack_name == "suppress_rule":
                            simple_ssr_oks = torch.cat([simple_ssr_oks, (adv_pred * hot(f, n)).sum(dim=(-1, -2)) == 0])
                        elif config.attack_name == "coerce_state":
                            simple_ssr_oks = torch.cat([simple_ssr_oks, torch.ones(N, device=device).bool()])

                        simple_asr_oks = torch.cat([simple_asr_oks, (adv_pred != raw_labels).any(dim=-1).any(dim=-1)])

                        pbar.set_description(
                            f"learned (n={n},d={embed_dim},b={attn_bias},k={k}) "
                            f"raw={raw_state_hits.float().mean():.2f} "
                            f"asr={simple_asr_oks.float().mean():.2f}"
                        )

                    row = {
                        "num_samples": config.num_samples,
                        "reasoner_type": "learned",
                        "train_seed": train_seed,
                        "num_vars": n,
                        "embed_dim": embed_dim,
                        "num_attack_tokens": k,
                        "attn_bias": attn_bias,
                        "raw_state_acc": raw_state_hits.float().mean().item(),
                        "simple_ssr": simple_ssr_oks.float().mean().item(),
                        "simple_asr": simple_asr_oks.float().mean().item(),
                        "adv_ns1_state_acc": adv_ns1_state_hits.float().mean().item(),
                        "adv_ns2_state_acc": adv_ns2_state_hits.float().mean().item(),
                        "adv_ns3_state_acc": adv_ns3_state_hits.float().mean().item(),
                        "adv_ns1_atk_wts": adv_ns1_atk_wts.mean().item(),
                        "adv_ns2_atk_wts": adv_ns2_atk_wts.mean().item(),
                        "adv_ns3_atk_wts": adv_ns3_atk_wts.mean().item(),
                        "adv_ns1_atk_wts_std": adv_ns1_atk_wts.var().sqrt().item(),
                        "adv_ns2_atk_wts_std": adv_ns2_atk_wts.var().sqrt().item(),
                        "adv_ns3_atk_wts_std": adv_ns3_atk_wts.var().sqrt().item(),
                    }
                    df = pd.concat([df, pd.DataFrame(row, index=[0])], ignore_index=True)
                    df.to_csv(out_file, index=False)
                    completed_keys.add(run_key)

                    if (
                        config.attack_name == "suppress_rule"
                        and early_stop_bias is not None
                        and abs(float(attn_bias) - float(early_stop_bias)) < 1e-12
                        and float(row["simple_asr"]) >= early_stop_asr
                    ):
                        stop_this_bias = True

    print(f"Saved results to: {out_file}")


if __name__ == "__main__":
    parser = HfArgumentParser(BiasAttackArguments)
    args = parser.parse_args_into_dataclasses()[0]

    config = Namespace(**json.load(open(args.config_file)))
    config.device = args.device

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run(config, out_dir)
