import gc
import json
from pathlib import Path

import einops
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.integrations.mxfp4 import convert_moe_packed_tensors

from transformer_lens import HookedTransformer
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


_OPEN_FILES = {}


def get_gpt_oss_model_path(model_name: str = "openai/gpt-oss-20b") -> Path:
    org, repo = model_name.split("/", 1)
    cache_path = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{org}--{repo}"
    snapshots = cache_path / "snapshots"

    if snapshots.exists():
        snapshot_dirs = sorted(snapshots.iterdir())
        if snapshot_dirs:
            return snapshot_dirs[-1]

    return Path(snapshot_download(model_name))


def create_gpt_oss_config(device: str = "cpu", n_layers: int = 24) -> HookedTransformerConfig:
    return HookedTransformerConfig(
        n_layers=n_layers,
        d_model=2880,
        d_head=64,
        n_heads=64,
        d_mlp=2880,
        n_ctx=4096,
        d_vocab=201088,
        act_fn="silu",
        normalization_type="RMS",
        positional_embedding_type="rotary",
        rotary_base=150000,
        eps=1e-5,
        n_key_value_heads=8,
        gated_mlp=True,
        use_local_attn=False,
        rotary_dim=64,
        num_experts=32,
        experts_per_token=4,
        dtype=torch.bfloat16,
        device=device,
        original_architecture="GptOssForCausalLM",
        model_name="openai/gpt-oss-20b",
    )


def _get_tensor(hf_name: str, weight_map: dict[str, str], model_path: Path) -> torch.Tensor:
    st_file = weight_map[hf_name]
    filepath = str(model_path / st_file)
    if filepath not in _OPEN_FILES:
        _OPEN_FILES[filepath] = safe_open(filepath, framework="pt", device="cpu")
    return _OPEN_FILES[filepath].get_tensor(hf_name)


def load_gpt_oss_layer_weights(layer_idx: int, cfg: HookedTransformerConfig, index: dict, model_path: Path) -> dict[str, torch.Tensor]:
    state_dict = {}
    weight_map = index["weight_map"]
    prefix = f"model.layers.{layer_idx}"

    def gt(name: str) -> torch.Tensor:
        return _get_tensor(name, weight_map, model_path)

    state_dict[f"blocks.{layer_idx}.ln1.w"] = gt(f"{prefix}.input_layernorm.weight")
    state_dict[f"blocks.{layer_idx}.ln2.w"] = gt(f"{prefix}.post_attention_layernorm.weight")

    q_w = gt(f"{prefix}.self_attn.q_proj.weight")
    k_w = gt(f"{prefix}.self_attn.k_proj.weight")
    v_w = gt(f"{prefix}.self_attn.v_proj.weight")
    o_w = gt(f"{prefix}.self_attn.o_proj.weight")

    state_dict[f"blocks.{layer_idx}.attn.W_Q"] = einops.rearrange(q_w, "(n h) m -> n m h", n=cfg.n_heads)
    state_dict[f"blocks.{layer_idx}.attn._W_K"] = einops.rearrange(k_w, "(n h) m -> n m h", n=cfg.n_key_value_heads)
    state_dict[f"blocks.{layer_idx}.attn._W_V"] = einops.rearrange(v_w, "(n h) m -> n m h", n=cfg.n_key_value_heads)
    state_dict[f"blocks.{layer_idx}.attn.W_O"] = einops.rearrange(o_w, "m (n h) -> n h m", n=cfg.n_heads)
    del q_w, k_w, v_w, o_w

    q_bias_key = f"{prefix}.self_attn.q_proj.bias"
    if q_bias_key in weight_map:
        state_dict[f"blocks.{layer_idx}.attn.b_Q"] = einops.rearrange(gt(q_bias_key), "(n h) -> n h", n=cfg.n_heads)
        state_dict[f"blocks.{layer_idx}.attn._b_K"] = einops.rearrange(
            gt(f"{prefix}.self_attn.k_proj.bias"), "(n h) -> n h", n=cfg.n_key_value_heads
        )
        state_dict[f"blocks.{layer_idx}.attn._b_V"] = einops.rearrange(
            gt(f"{prefix}.self_attn.v_proj.bias"), "(n h) -> n h", n=cfg.n_key_value_heads
        )
    else:
        state_dict[f"blocks.{layer_idx}.attn.b_Q"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)
        state_dict[f"blocks.{layer_idx}.attn._b_K"] = torch.zeros(cfg.n_key_value_heads, cfg.d_head, dtype=cfg.dtype)
        state_dict[f"blocks.{layer_idx}.attn._b_V"] = torch.zeros(cfg.n_key_value_heads, cfg.d_head, dtype=cfg.dtype)

    o_bias_key = f"{prefix}.self_attn.o_proj.bias"
    if o_bias_key in weight_map:
        state_dict[f"blocks.{layer_idx}.attn.b_O"] = gt(o_bias_key)
    else:
        state_dict[f"blocks.{layer_idx}.attn.b_O"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

    state_dict[f"blocks.{layer_idx}.mlp.W_gate.weight"] = gt(f"{prefix}.mlp.router.weight")
    state_dict[f"blocks.{layer_idx}.mlp.W_gate.bias"] = gt(f"{prefix}.mlp.router.bias")

    gate_up_blocks = gt(f"{prefix}.mlp.experts.gate_up_proj_blocks")
    gate_up_scales = gt(f"{prefix}.mlp.experts.gate_up_proj_scales")
    gate_up_bias = gt(f"{prefix}.mlp.experts.gate_up_proj_bias")

    print(f"  Dequantizing layer {layer_idx} gate_up_proj on CPU...", end="", flush=True)
    gate_up_proj = convert_moe_packed_tensors(gate_up_blocks, gate_up_scales)
    del gate_up_blocks, gate_up_scales
    print(" done")

    down_blocks = gt(f"{prefix}.mlp.experts.down_proj_blocks")
    down_scales = gt(f"{prefix}.mlp.experts.down_proj_scales")
    down_bias = gt(f"{prefix}.mlp.experts.down_proj_bias")

    print(f"  Dequantizing layer {layer_idx} down_proj on CPU...", end="", flush=True)
    down_proj = convert_moe_packed_tensors(down_blocks, down_scales)
    del down_blocks, down_scales
    print(" done")

    for expert_idx in range(cfg.num_experts):
        state_dict[f"blocks.{layer_idx}.mlp.experts.{expert_idx}.W_gate.weight"] = gate_up_proj[expert_idx, :, ::2].T.contiguous()
        state_dict[f"blocks.{layer_idx}.mlp.experts.{expert_idx}.W_gate.bias"] = gate_up_bias[expert_idx, ::2].contiguous()
        state_dict[f"blocks.{layer_idx}.mlp.experts.{expert_idx}.W_in.weight"] = gate_up_proj[expert_idx, :, 1::2].T.contiguous()
        state_dict[f"blocks.{layer_idx}.mlp.experts.{expert_idx}.W_in.bias"] = gate_up_bias[expert_idx, 1::2].contiguous()
        state_dict[f"blocks.{layer_idx}.mlp.experts.{expert_idx}.W_out.weight"] = down_proj[expert_idx].T.contiguous()
        state_dict[f"blocks.{layer_idx}.mlp.experts.{expert_idx}.W_out.bias"] = down_bias[expert_idx].contiguous()

    del gate_up_proj, gate_up_bias, down_proj, down_bias
    return state_dict


def load_gpt_oss_model(device: str = "cuda", n_layers: int = 24) -> HookedTransformer:
    model_path = get_gpt_oss_model_path()
    print(f"Loading GPT-OSS directly from safetensors at {model_path}")

    with open(model_path / "model.safetensors.index.json") as f:
        index = json.load(f)
    with open(model_path / "generation_config.json") as f:
        generation_config = json.load(f)

    cfg = create_gpt_oss_config(device="cpu", n_layers=n_layers)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = HookedTransformer(cfg, tokenizer, move_to_device=False)

    weight_map = index["weight_map"]
    model.load_state_dict({"embed.W_E": _get_tensor("model.embed_tokens.weight", weight_map, model_path)}, strict=False)
    gc.collect()

    for layer_idx in range(n_layers):
        print(f"Loading layer {layer_idx}/{n_layers - 1}...")
        layer_dict = load_gpt_oss_layer_weights(layer_idx, cfg, index, model_path)
        for key in list(layer_dict.keys()):
            model.load_state_dict({key: layer_dict[key]}, strict=False)
            del layer_dict[key]
        del layer_dict
        gc.collect()

    model.load_state_dict({"ln_final.w": _get_tensor("model.norm.weight", weight_map, model_path)}, strict=False)
    model.load_state_dict({"unembed.W_U": _get_tensor("lm_head.weight", weight_map, model_path).T}, strict=False)
    model.load_state_dict({"unembed.b_U": torch.zeros(cfg.d_vocab, dtype=cfg.dtype)}, strict=False)
    gc.collect()

    if device != "cpu":
        print(f"Moving GPT-OSS model to {device}...")
        model.to(device)
        torch.cuda.empty_cache()
    model.cfg.device = device
    model.generation_stop_token_ids = generation_config.get("eos_token_id", [model.tokenizer.eos_token_id])
    return model
