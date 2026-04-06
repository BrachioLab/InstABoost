import gc
import json
from pathlib import Path

import einops
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers.integrations.mxfp4 import convert_moe_packed_tensors

from transformer_lens import HookedTransformer
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

from transformer_lens_compat import patch_gpt_oss_attention_sinks


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


def get_gpt_oss_dequant_cache_dir(
    model_path: Path,
    model_name: str = "openai/gpt-oss-20b",
) -> Path:
    cache_root = Path.home() / ".cache" / "instaboost" / "gpt_oss_dequantized"
    org, repo = model_name.split("/", 1)
    snapshot_id = model_path.name
    return cache_root / f"{org}--{repo}" / snapshot_id


def get_gpt_oss_layer_cache_path(cache_dir: Path, layer_idx: int) -> Path:
    return cache_dir / f"layer_{layer_idx:02d}.safetensors"


def load_cached_gpt_oss_layer(cache_path: Path) -> dict[str, torch.Tensor] | None:
    if not cache_path.exists():
        return None
    return load_file(str(cache_path), device="cpu")


def save_cached_gpt_oss_layer(cache_path: Path, state_dict: dict[str, torch.Tensor]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    serializable_state = {
        key: tensor.detach().cpu().contiguous()
        for key, tensor in state_dict.items()
    }
    save_file(serializable_state, str(tmp_path))
    tmp_path.replace(cache_path)


def create_gpt_oss_config(
    hf_config,
    device: str = "cpu",
    n_layers: int | None = None,
) -> HookedTransformerConfig:
    total_layers = hf_config.num_hidden_layers if n_layers is None else n_layers
    layer_types = getattr(hf_config, "layer_types", None)
    if layer_types is not None:
        layer_types = layer_types[:total_layers]
        attn_types = [
            "local" if layer_type == "sliding_attention" else "global"
            for layer_type in layer_types
        ]
        use_local_attn = any(attn_type == "local" for attn_type in attn_types)
        window_size = getattr(hf_config, "sliding_window", None)
    else:
        attn_types = None
        use_local_attn = False
        window_size = None
    return HookedTransformerConfig(
        n_layers=total_layers,
        d_model=hf_config.hidden_size,
        d_head=hf_config.head_dim,
        n_heads=hf_config.num_attention_heads,
        d_mlp=hf_config.intermediate_size,
        n_ctx=getattr(hf_config, "initial_context_length", 4096),
        d_vocab=hf_config.vocab_size,
        act_fn=hf_config.hidden_act,
        normalization_type="RMS",
        positional_embedding_type="rotary",
        rotary_base=hf_config.rope_theta,
        eps=hf_config.rms_norm_eps,
        n_key_value_heads=hf_config.num_key_value_heads,
        gated_mlp=True,
        final_rms=True,
        use_local_attn=use_local_attn,
        attn_types=attn_types,
        window_size=window_size,
        rotary_dim=hf_config.head_dim,
        num_experts=hf_config.num_local_experts,
        experts_per_token=hf_config.num_experts_per_tok,
        dtype=torch.bfloat16,
        device=device,
        init_weights=False,
        default_prepend_bos=False,
        original_architecture="GptOssForCausalLM",
        model_name="openai/gpt-oss-20b",
    )


def _get_tensor(hf_name: str, weight_map: dict[str, str], model_path: Path) -> torch.Tensor:
    st_file = weight_map[hf_name]
    filepath = str(model_path / st_file)
    if filepath not in _OPEN_FILES:
        _OPEN_FILES[filepath] = safe_open(filepath, framework="pt", device="cpu")
    return _OPEN_FILES[filepath].get_tensor(hf_name)


def load_gpt_oss_layer_weights(
    layer_idx: int,
    cfg: HookedTransformerConfig,
    index: dict,
    model_path: Path,
    cache_dir: Path | None = None,
) -> tuple[dict[str, torch.Tensor], str]:
    if cache_dir is not None:
        cache_path = get_gpt_oss_layer_cache_path(cache_dir, layer_idx)
        cached_state = load_cached_gpt_oss_layer(cache_path)
        sink_key = f"blocks.{layer_idx}.attn.sinks"
        if cached_state is not None and sink_key in cached_state:
            return cached_state, "cached"

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
    state_dict[f"blocks.{layer_idx}.attn.sinks"] = gt(f"{prefix}.self_attn.sinks")
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
    if cache_dir is not None:
        save_cached_gpt_oss_layer(cache_path, state_dict)
    return state_dict, "dequantized"


def materialize_gpt_oss_meta_buffers(model: HookedTransformer) -> None:
    for block in model.blocks:
        attn = block.attn
        if getattr(attn.mask, "is_meta", False):
            causal_mask = torch.tril(torch.ones((model.cfg.n_ctx, model.cfg.n_ctx), dtype=torch.bool))
            if attn.attn_type == "global":
                attn.mask = causal_mask
            elif attn.attn_type == "local":
                if not isinstance(model.cfg.window_size, int):
                    raise ValueError("Window size must be an integer for local attention")
                attn.mask = torch.triu(causal_mask, 1 - model.cfg.window_size)
            else:
                raise ValueError(f"Invalid attention type: {attn.attn_type}")
        if getattr(attn.IGNORE, "is_meta", False):
            attn.IGNORE = torch.tensor(-torch.inf)
        if hasattr(attn, "rotary_sin") and getattr(attn.rotary_sin, "is_meta", False):
            if model.cfg.rotary_dim is None:
                raise ValueError("Rotary dim must be provided for rotary positional embeddings")
            if model.cfg.rotary_base_local is not None and attn.attn_type == "local":
                rope_base = model.cfg.rotary_base_local
            else:
                rope_base = model.cfg.rotary_base
            sin, cos = attn.calculate_sin_cos_rotary(
                model.cfg.rotary_dim,
                model.cfg.n_ctx,
                base=rope_base,
                dtype=model.cfg.dtype,
            )
            attn.rotary_sin = sin
            attn.rotary_cos = cos


def load_gpt_oss_model(device: str = "cuda", n_layers: int = 24) -> HookedTransformer:
    model_path = get_gpt_oss_model_path()
    print(f"Loading GPT-OSS directly from safetensors at {model_path}")
    patch_gpt_oss_attention_sinks()

    hf_config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    with open(model_path / "model.safetensors.index.json") as f:
        index = json.load(f)
    with open(model_path / "generation_config.json") as f:
        generation_config = json.load(f)

    cache_dir = get_gpt_oss_dequant_cache_dir(model_path)
    cfg = create_gpt_oss_config(hf_config=hf_config, device="cpu", n_layers=n_layers)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    tokenizer.padding_side = "left"
    with torch.device("meta"):
        model = HookedTransformer(
            cfg,
            tokenizer,
            move_to_device=False,
            default_padding_side="left",
        )

    weight_map = index["weight_map"]
    model.load_state_dict(
        {"embed.W_E": _get_tensor("model.embed_tokens.weight", weight_map, model_path)},
        strict=False,
        assign=True,
    )
    gc.collect()

    layer_progress = tqdm(range(n_layers), desc="Loading GPT-OSS layers", unit="layer")
    for layer_idx in layer_progress:
        layer_progress.set_postfix_str(f"layer={layer_idx}")
        layer_dict, layer_source = load_gpt_oss_layer_weights(
            layer_idx,
            cfg,
            index,
            model_path,
            cache_dir=cache_dir,
        )
        layer_progress.set_postfix_str(f"layer={layer_idx} source={layer_source}")
        model.load_state_dict(layer_dict, strict=False, assign=True)
        del layer_dict
        gc.collect()

    model.load_state_dict(
        {"ln_final.w": _get_tensor("model.norm.weight", weight_map, model_path)},
        strict=False,
        assign=True,
    )
    model.load_state_dict(
        {"unembed.W_U": _get_tensor("lm_head.weight", weight_map, model_path).T},
        strict=False,
        assign=True,
    )
    model.load_state_dict(
        {"unembed.b_U": torch.zeros(cfg.d_vocab, dtype=cfg.dtype)},
        strict=False,
        assign=True,
    )
    materialize_gpt_oss_meta_buffers(model)
    gc.collect()

    if device != "cpu":
        print(f"Moving GPT-OSS model to {device}...")
        model.to(device)
        torch.cuda.empty_cache()
    model.cfg.device = device
    model.tokenizer.padding_side = "left"
    model.generation_stop_token_ids = generation_config.get("eos_token_id", [model.tokenizer.eos_token_id])
    model.generation_config_overrides = {
        "do_sample": False,
        "temperature": float(generation_config.get("temperature", 1.0)),
        "top_k": int(generation_config.get("top_k", 0) or 0),
        "top_p": float(generation_config.get("top_p", 1.0)),
    }
    return model
