import torch
from jaxtyping import Float
from transformer_lens.hook_points import HookPoint
import einops
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

def direction_ablation_hook(
    activation: Float[torch.Tensor, "... d_act"],
    hook: HookPoint,
    direction: Float[torch.Tensor, "d_act"]
):
    proj = einops.einsum(activation, direction.view(-1, 1).to(activation.device), '... d_act, d_act single -> ... single') * direction.to(activation.device)
    return activation - proj

def in_direction_ablation_hook(
    activation: Float[torch.Tensor, "... d_act"],
    hook: HookPoint,
    direction: Float[torch.Tensor, "d_act"],
    multiplier: float = 1.0
):
    direction = direction.to(activation.device)
    return activation + multiplier * direction.unsqueeze(0)

def instaboost_hook(
    pattern: torch.Tensor,
    hook: HookPoint,
    instruction_len: int,
    multiplier: float
):
    """
    Instaboost: boosts the attention probabilities on the instruction tokens by a fixed multiplier.

    Args:
        pattern (torch.Tensor): Softmax-normalized attention probabilities 
            (shape: [batch, heads, query_len, key_len]).
        hook (HookPoint): TransformerLens hook point (not used directly here).
        instruction_len (int): Number of tokens at the beginning of the input 
            corresponding to the instruction span (these will be boosted).
        multiplier (float): Multiplicative factor to boost the instruction tokens' attention.

    Returns:
        torch.Tensor: Modified attention pattern with the first `instruction_len` tokens
        scaled by `multiplier` and renormalized along the key dimension.
    """
    pattern[:, :, :, :instruction_len] *= multiplier
    return torch.nn.functional.normalize(pattern, p=1, dim=-1)

def spotlight_hook(
    attn_scores: torch.Tensor,
    hook: HookPoint,
    instruction_start: int,
    instruction_end: int,
    target_attention: float
):
    """
    Spotlight: adds a log-bias to the instruction tokens' attention logits
    to ensure their attention proportion reaches at least `target_attention`.

    Args:
        attn_scores (torch.Tensor): Attention logits before softmax 
            (shape: [batch, heads, query_len, key_len]).
        hook (HookPoint): TransformerLens hook point (not used directly here).
        instruction_start (int): Index of the first instruction token in the input sequence.
        instruction_end (int): Index after the last instruction token in the input sequence.
        target_attention (float): Minimum desired proportion of attention mass 
            on the instruction tokens (0 < target_attention < 1).

    Returns:
        torch.Tensor: Modified attention logits where a uniform log-bias has been added 
        to the instruction tokens for query/head positions whose attention proportion 
        was below the `target_attention`.
    """
    # Softmax over logits to compute current attention distribution
    attn_probs = attn_scores.softmax(dim=-1)  # Shape [b, h, q, k]

    # Current proportion of attention on instruction tokens
    current_attn = attn_probs[..., instruction_start:instruction_end].sum(dim=-1) # [b, h, q]
    total_attn = attn_probs.sum(dim=-1)
    proportion = current_attn / (total_attn + 1e-8) # [b, h, q]

    # Compute bias (log-ratio) only if proportion < target
    boost_needed = proportion < target_attention # [b, h, q]
    bias = torch.zeros_like(proportion)
    bias[boost_needed] = torch.log(target_attention / (proportion[boost_needed] + 1e-8))

    # Expand bias and apply to instruction token logits
    bias = bias.unsqueeze(-1)  # [b, h, q, 1]
    attn_scores[..., instruction_start:instruction_end] += bias

    return attn_scores

def get_steering_direction(harmful_hiddens, 
                           harmless_hiddens,
                           steering_type="mean",
                           normalize: bool = True
                           ):
    """
    Get the steering direction for the latent steering methods.
    """
    dir = None
    if steering_type == "mean":
        dir =  harmful_hiddens.mean(dim=0) - harmless_hiddens.mean(dim=0)
    elif steering_type == "random":
        dir = torch.randn_like(harmful_hiddens.mean(dim=0) - harmless_hiddens.mean(dim=0))
    elif steering_type == "repe":
        diffs = (harmful_hiddens - harmless_hiddens).to(torch.float32)
        diffs /= diffs.norm(dim=-1, keepdim=True)
        diffs_float32 = diffs.to(torch.float32)
        dir = PCA(n_components=1).fit_transform(diffs_float32.cpu().T)
        dir = torch.from_numpy(dir).to(harmful_hiddens.device).T
        dir = dir.to(torch.bfloat16)
    elif steering_type == "pca":
        harmful_hiddens_norm = (harmful_hiddens / harmful_hiddens.norm(dim=-1, keepdim=True)).to(torch.float32).cpu().T
        dir = PCA(n_components=1).fit_transform(harmful_hiddens_norm)
        dir = torch.from_numpy(dir).to(harmful_hiddens.device).T
        dir = dir.to(torch.bfloat16)
    elif steering_type == "linear":
        harmful_np = harmful_hiddens.to(torch.float32).cpu().numpy()
        harmless_np = harmless_hiddens.to(torch.float32).cpu().numpy()
        X = np.vstack([harmful_np, harmless_np])
        y = np.concatenate([np.zeros(len(harmful_np)), np.ones(len(harmless_np))])
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        dir = torch.from_numpy(clf.coef_[0]).to(harmful_hiddens.device)
        dir = dir.to(torch.bfloat16)
    elif steering_type == "refusal":
        dir = harmful_hiddens.mean(dim=0) - harmless_hiddens.mean(dim=0)
        dir = dir / dir.norm()
    elif steering_type == "prompt":
        dir = None
    elif steering_type == "prompt-attention":
        dir = None
    else:
        raise ValueError(f"Unknown steering type: {steering_type}")

    if not dir is None and normalize:
        dir = dir / dir.norm()

    return dir