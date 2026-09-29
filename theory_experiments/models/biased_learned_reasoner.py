from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from models.task_models import AutoregKStepsTaskModel, AutoregKStepsTaskOutput


class RuleBiasAutoregWrapper(nn.Module):
    """Add a fixed positive bias to rule-token attention logits in a learned autoregressive reasoner."""

    def __init__(
        self,
        base_model: AutoregKStepsTaskModel,
        attn_bias: float,
        rulebook_len: Optional[int] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.attn_bias = float(attn_bias)
        self.rulebook_len = rulebook_len

    @property
    def num_steps(self) -> int:
        return self.base_model.num_steps

    @property
    def num_labels(self) -> int:
        return self.base_model.num_labels

    @property
    def input_dim(self) -> int:
        return self.base_model.input_dim

    @property
    def embed_dim(self) -> int:
        return self.base_model.embed_dim

    @property
    def model_name(self) -> str:
        return f"{self.base_model.model_name}_rulebias"

    def _make_biased_causal_mask(
        self,
        batch_size: int,
        seq_len: int,
        rulebook_len: int,
        device: torch.device,
        dtype: torch.dtype,
        bias_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        gpt2 = self.base_model.seqcls_model.gpt2s.transformer
        cache_position = torch.arange(seq_len, device=device)

        # `_prepare_4d_causal_attention_mask_with_cache_position` only exists on GPT2Model in a
        # window of transformers releases (it arrived with the cache_position refactor and later
        # moved to `masking_utils`).  With `attention_mask=None`, a full `cache_position` and
        # `target_length == sequence_length` it reduces exactly to the strict upper-triangular
        # additive mask built below, so the fallback is equivalent, not an approximation.
        if hasattr(gpt2, "_prepare_4d_causal_attention_mask_with_cache_position"):
            causal_mask = gpt2._prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask=None,
                sequence_length=seq_len,
                target_length=seq_len,
                dtype=dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), min_dtype, dtype=dtype, device=device), diagonal=1
            )
            causal_mask = causal_mask[None, None].expand(batch_size, 1, -1, -1)

        if self.attn_bias == 0:
            return causal_mask

        # Bias only valid key positions; masked future positions remain masked.
        valid = (causal_mask == 0).to(dtype)
        bias_mask = torch.zeros_like(causal_mask)
        if bias_positions is not None:
            # An explicit per-sample set of boosted key positions, for boosting individual rule
            # rows rather than a contiguous instruction block.
            idx = bias_positions.to(device).view(bias_mask.size(0), 1, 1, -1)
            bias_mask.scatter_(3, idx.expand(-1, 1, seq_len, -1), self.attn_bias)
        elif rulebook_len > 0:
            bias_mask[:, :, :, :rulebook_len] = self.attn_bias
        else:
            return causal_mask
        return causal_mask + bias_mask * valid

    def forward(
        self,
        tokens: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        rulebook_len: Optional[int] = None,
        bias_positions: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """`bias_positions` (N, m) boosts exactly those key positions per sample; when it is
        given `rulebook_len` no longer selects what is boosted, only the prefix default is
        overridden.  Leaving it None keeps the original behaviour: boost `[0, rulebook_len)`."""
        all_succ_logits = ()
        all_seqcls_outs = ()
        all_tokens = tokens

        if rulebook_len is None:
            rulebook_len = self.rulebook_len
        if rulebook_len is None:
            rulebook_len = tokens.size(1) - 1

        for _ in range(self.base_model.num_steps):
            n_batch, seq_len, _ = all_tokens.shape
            attn_mask = self._make_biased_causal_mask(
                batch_size=n_batch,
                seq_len=seq_len,
                rulebook_len=min(rulebook_len, seq_len),
                device=all_tokens.device,
                dtype=all_tokens.dtype,
                bias_positions=bias_positions,
            )

            seqcls_out = self.base_model.seqcls_model(
                all_tokens.float(),
                labels=None,
                attention_mask=attn_mask,
                output_attentions=output_attentions,
                **kwargs,
            )
            all_seqcls_outs += (seqcls_out,)

            succ_logits = seqcls_out.logits
            all_succ_logits += (succ_logits,)

            succ = (succ_logits > 0)
            succ_token = self.base_model.state_to_token(succ.unsqueeze(1).float())
            all_tokens = torch.cat([all_tokens, succ_token], dim=1).float()

        all_succ_logits = torch.stack(all_succ_logits, dim=1)

        loss = None
        if labels is not None:
            if self.base_model.train_supervision_mode == "all":
                loss = self.base_model.loss_fn(all_succ_logits, labels.float()).to(tokens.device)
            elif self.base_model.train_supervision_mode == "final":
                loss = self.base_model.loss_fn(all_succ_logits[:, -1], labels[:, -1].float()).to(tokens.device)
            else:
                raise ValueError(f"Unknown train_supervision_mode: {self.base_model.train_supervision_mode}")

        return AutoregKStepsTaskOutput(
            loss=loss,
            logits=all_succ_logits,
            all_seqcls_outputs=all_seqcls_outs,
        )
