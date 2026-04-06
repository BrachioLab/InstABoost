import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_lens.components.abstract_attention import AbstractAttention
from transformer_lens.components.grouped_query_attention import GroupedQueryAttention
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


_PATCHED = False
_ORIGINAL_GQA_INIT = GroupedQueryAttention.__init__
_ORIGINAL_ABSTRACT_ATTENTION_FORWARD = AbstractAttention.forward


def _is_gpt_oss_attention(module: AbstractAttention) -> bool:
    return (
        getattr(module.cfg, "original_architecture", None) == "GptOssForCausalLM"
        and hasattr(module, "sinks")
    )


def _patched_grouped_query_attention_init(self, cfg, attn_type="global", layer_id=None):
    _ORIGINAL_GQA_INIT(self, cfg, attn_type, layer_id)
    cfg = HookedTransformerConfig.unwrap(cfg)
    if cfg.original_architecture == "GptOssForCausalLM":
        self.sinks = nn.Parameter(torch.zeros(cfg.n_heads, dtype=cfg.dtype))


def _patched_abstract_attention_forward(
    self,
    query_input,
    key_input,
    value_input,
    past_kv_cache_entry=None,
    additive_attention_mask=None,
    attention_mask=None,
    position_bias=None,
):
    if not _is_gpt_oss_attention(self):
        return _ORIGINAL_ABSTRACT_ATTENTION_FORWARD(
            self,
            query_input,
            key_input,
            value_input,
            past_kv_cache_entry=past_kv_cache_entry,
            additive_attention_mask=additive_attention_mask,
            attention_mask=attention_mask,
            position_bias=position_bias,
        )

    q, k, v = self.calculate_qkv_matrices(query_input, key_input, value_input)

    if past_kv_cache_entry is not None:
        kv_cache_pos_offset = past_kv_cache_entry.past_keys.size(1)
        k, v = past_kv_cache_entry.append(k, v)
    else:
        kv_cache_pos_offset = 0

    if self.cfg.positional_embedding_type == "rotary":
        q = self.hook_rot_q(self.apply_rotary(q, kv_cache_pos_offset, attention_mask))
        k = self.hook_rot_k(self.apply_rotary(k, 0, attention_mask))

    if self.cfg.dtype not in [torch.float32, torch.float64]:
        q = q.to(torch.float32)
        k = k.to(torch.float32)

    attn_scores = self.calculate_attention_scores(q, k)

    if self.cfg.positional_embedding_type == "alibi":
        query_ctx = attn_scores.size(-2)
        key_ctx = attn_scores.size(-1)
        if self.alibi is None or key_ctx > self.alibi.size(-1):
            self.alibi = AbstractAttention.create_alibi_bias(
                self.cfg.n_heads, key_ctx, self.cfg.device
            )
        attn_scores += self.alibi[:, -query_ctx:, :key_ctx]
    elif self.cfg.positional_embedding_type == "relative_positional_bias":
        if position_bias is None:
            if self.has_relative_attention_bias:
                raise ValueError("Positional bias is required for relative_positional_bias")
            position_bias = torch.zeros(
                1,
                self.cfg.n_heads,
                attn_scores.shape[2],
                attn_scores.shape[3],
                device=attn_scores.device,
            )
        attn_scores += position_bias

    if self.cfg.attention_dir == "causal":
        attn_scores = self.apply_causal_mask(attn_scores, kv_cache_pos_offset, attention_mask)
    if additive_attention_mask is not None:
        attn_scores += additive_attention_mask

    attn_scores = self.hook_attn_scores(attn_scores)

    sink_logits = self.sinks.to(device=attn_scores.device, dtype=attn_scores.dtype)
    sink_logits = sink_logits.reshape(1, -1, 1, 1).expand(
        attn_scores.shape[0], -1, attn_scores.shape[2], 1
    )
    combined_logits = torch.cat([attn_scores, sink_logits], dim=-1)
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    pattern = F.softmax(combined_logits, dim=-1)[..., :-1]
    pattern = torch.where(torch.isnan(pattern), torch.zeros_like(pattern), pattern)
    pattern = self.hook_pattern(pattern)
    pattern = pattern.to(self.cfg.dtype)
    pattern = pattern.to(v.device)
    z = self.calculate_z_scores(v, pattern)

    if not self.cfg.use_attn_result:
        w = einops.rearrange(
            self.W_O, "head_index d_head d_model -> d_model (head_index d_head)"
        )
        if self.b_O.device != w.device:
            w = w.to(self.b_O.device)
        if self.b_O.device != z.device:
            z = z.to(self.b_O.device)
        z = z.reshape(z.shape[0], z.shape[1], self.cfg.d_head * self.cfg.n_heads)
        if z.device.type == "mps":
            out = torch.matmul(z, w.T) + self.b_O
        else:
            out = F.linear(z, w, self.b_O)
    else:
        w = einops.rearrange(
            self.W_O,
            "head_index d_head d_model -> 1 1 head_index d_head d_model",
        )
        z = einops.rearrange(
            z, "batch pos head_index d_head -> batch pos head_index d_head 1"
        )
        unhooked_result = (z * w).sum(-2)
        result = self.hook_result(unhooked_result)
        out = (
            einops.reduce(result, "batch position index model->batch position model", "sum")
            + self.b_O
        )
    return out


def patch_gpt_oss_attention_sinks() -> None:
    global _PATCHED
    if _PATCHED:
        return
    GroupedQueryAttention.__init__ = _patched_grouped_query_attention_init
    AbstractAttention.forward = _patched_abstract_attention_forward
    _PATCHED = True
