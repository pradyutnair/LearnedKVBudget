"""Phase 1 scaffolding for per-head feature extraction.

This module defines the feature schema and helper functions for:
1) attention entropy
2) top-k attention mass
3) key norm variance
4) Ada-KV analytical score (provided by caller)

PRD reference:
- Section 4.2 (Per-Head Feature Vector)
- Section 5.3 "Features" phase

Documentation links:
- PyTorch tensor ops: https://pytorch.org/docs/stable/torch.html
- PyTorch topk: https://pytorch.org/docs/stable/generated/torch.topk.html
- HF model outputs (attentions): https://huggingface.co/docs/transformers/main_classes/output
"""

from dataclasses import dataclass

import torch


@dataclass
class FeatureConfig:
    """Configuration for feature computation in Phase 1.

    TODO(phase1): confirm `topk_fraction` and `eps` with your experiments.
    """

    topk_fraction: float = 0.1
    eps: float = 1e-8


def validate_attention_shape(attentions: torch.Tensor) -> None:
    """Validate attention tensor shape.

    Expected: (batch, num_heads, query_len, key_len)
    """
    if attentions.ndim != 4:
        raise ValueError(
            f"Expected attentions with 4 dims (B, H, Q, K), got shape={tuple(attentions.shape)}"
        )


def compute_attention_entropy(attentions: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute per-head attention entropy averaged over queries.

    Returns:
        Tensor with shape (batch, num_heads)

    TODO(phase1): verify whether to compute this on full prefill window or a fixed trailing window.
    TODO(phase1): confirm whether to aggregate over batch before or after logging.
    """
    validate_attention_shape(attentions)
    # Clamp probabilities to avoid log(0)
    probs = attentions.clamp_min(eps)
    # Compute entropy per query, then average over queries
    entropy_per_query = -(probs * probs.log()).sum(dim=-1)  # (B, H, Q)
    return entropy_per_query.mean(dim=-1)  # (B, H)


def compute_topk_attention_mass(
    attentions: torch.Tensor, topk_fraction: float = 0.1
) -> torch.Tensor:
    """Compute per-head top-k attention mass averaged over queries.

    Returns:
        Tensor with shape (batch, num_heads)

    TODO(phase1): align top-k definition with PRD language.
    - Option A: fixed ratio of key_len (used here as scaffold)
    - Option B: derive k from current per-head budget candidate
    """
    validate_attention_shape(attentions)
    # Compute top-k values, then sum over keys and average over queries
    key_len = attentions.shape[-1]
    k = max(1, int(key_len * topk_fraction))
    topk_vals, _ = torch.topk(attentions, k=k, dim=-1)
    return topk_vals.sum(dim=-1).mean(dim=-1)  # (B, H)


def compute_key_norm_variance(keys: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute variance of key-vector norms per head.

    Expected keys shape: (batch, num_kv_heads, key_len, head_dim)
    Returns shape: (batch, num_kv_heads)

    TODO(phase1): decide whether population variance (`unbiased=False`) is preferred.
    """
    if keys.ndim != 4:
        raise ValueError(
            f"Expected keys with 4 dims (B, H_kv, K, D), got shape={tuple(keys.shape)}"
        )
    norms = torch.linalg.vector_norm(keys, ord=2, dim=-1)  # (B, H_kv, K)
    return norms.var(dim=-1, unbiased=False).clamp_min(eps)  # (B, H_kv)


def build_feature_tensor(
    attention_entropy: torch.Tensor,
    topk_attention_mass: torch.Tensor,
    key_norm_variance: torch.Tensor,
    adakv_l1_score: torch.Tensor,
) -> torch.Tensor:
    """Stack per-head scalar features into f_{l,h} in R^4.

    Input tensors should each be shape (batch, num_heads).
    Returns:
        Tensor with shape (batch, num_heads, 4)

    TODO(phase1): if model uses GQA and `num_heads != num_kv_heads`, define a canonical
    alignment strategy before training.
    """
    parts = [
        attention_entropy.unsqueeze(-1), # (B, H, 1)
        topk_attention_mass.unsqueeze(-1), # (B, H, 1)
        key_norm_variance.unsqueeze(-1), # (B, H, 1)
        adakv_l1_score.unsqueeze(-1), # (B, H, 1)
    ]
    return torch.cat(parts, dim=-1) # (B, H, 4)

