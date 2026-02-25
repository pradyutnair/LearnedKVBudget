"""Reward computation for GRPO training of the budget allocator.

The reward signal drives what the allocator learns. PRD Section 4.5 defines:
    r = accuracy(y_pred, y_ref) + λ * budget_efficiency

Two components:
    1. Task accuracy: binary correctness for the downstream task.
       - RULER: string match (does prediction contain the reference answer?)
       - MATH-500: extract \\boxed{answer} and compare
       This reuses the existing kvpress benchmark metric functions.

    2. Budget efficiency: encourages sparse, decisive allocations.
       Measured as negative entropy of the budget fraction vector.
       Without this, the allocator may collapse to uniform allocation
       (which has maximum entropy and is the "safe" default).
       λ controls the trade-off (start with 0.1).

PRD references:
    - Section 4.5 (Reward Function)
    - Section 7 Risk: "Allocator collapses to uniform"

Documentation links:
    - kvpress RULER metrics: evaluation/benchmarks/ruler/calculate_metrics.py
    - kvpress MATH500 metrics: evaluation/benchmarks/math500/calculate_metrics.py
    - Entropy: https://en.wikipedia.org/wiki/Entropy_(information_theory)
"""

import torch


def compute_task_reward_ruler(prediction: str, references: list[str]) -> float:
    """Binary reward for RULER-style tasks: 1.0 if any reference is found in prediction.

    This mirrors ruler/calculate_metrics.py's string_match_part logic
    but for a single (prediction, references) pair.

    Args:
        prediction: Model-generated text.
        references: List of acceptable reference answers.

    Returns:
        1.0 if any reference string appears in prediction (case-insensitive), else 0.0.

    TODO(phase3): Implement:
        1. Lowercase prediction
        2. Check if any ref.lower() is a substring of prediction.lower()
        3. Return 1.0 or 0.0
    """
    raise NotImplementedError


def compute_task_reward_math(prediction: str, reference: str) -> float:
    """Binary reward for MATH-style tasks: extract \\boxed{} answer and compare.

    This mirrors math500/calculate_metrics.py's score_aime logic.

    Args:
        prediction: Model-generated text (should contain \\boxed{answer}).
        reference: Ground truth answer string.

    Returns:
        1.0 if extracted boxed answer matches reference, else 0.0.

    TODO(phase3): Implement:
        1. Try to extract content between "boxed{" and "}"
        2. Compare with str(reference)
        3. Return 1.0 or 0.0
    """
    raise NotImplementedError


def compute_budget_efficiency(budget_fractions: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Negative entropy of budget fractions — rewards sparse allocations.

    Entropy of a distribution p over H heads:
        H(p) = -Σ p_h log(p_h)
    Maximum entropy = log(H) (uniform). Minimum = 0 (all budget to one head).

    We return -H(p) so that sparser allocations get higher reward.
    Optionally normalize by log(H) to get values in [-1, 0].

    Args:
        budget_fractions: shape (B, H) or (H,), each row sums to 1.

    Returns:
        Negative entropy per batch item, shape (B,) or scalar.

    TODO(phase3): Implement:
        1. Clamp fractions to min eps
        2. entropy = -(fractions * fractions.log()).sum(dim=-1)
        3. Return -entropy  (so lower entropy → higher reward)
    """
    raise NotImplementedError


def compute_composite_reward(
    task_reward: float,
    budget_fractions: torch.Tensor,
    lambda_efficiency: float = 0.1,
) -> float:
    """Composite reward combining task accuracy and budget efficiency.

    r = task_reward + λ * budget_efficiency

    Args:
        task_reward: Binary accuracy (0.0 or 1.0) from task-specific metric.
        budget_fractions: shape (H,), the allocation used for this rollout.
        lambda_efficiency: Weight for the efficiency bonus. Start with 0.1.
            Set to 0.0 for accuracy-only ablation.

    Returns:
        Scalar composite reward.

    TODO(phase3): Implement:
        1. efficiency = compute_budget_efficiency(budget_fractions).item()
        2. return task_reward + lambda_efficiency * efficiency
    """
    raise NotImplementedError


def compute_group_advantages(
    rewards: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute group-relative advantages for GRPO.

    GRPO's core idea: instead of a learned value baseline, use the group mean
    as the baseline. For each prompt's G rollouts:
        A_i = (r_i - mean(r)) / (std(r) + eps)

    This normalizes rewards within each prompt's group, making the gradient
    signal independent of the absolute reward scale.

    Args:
        rewards: shape (B, G) — rewards for each rollout in each prompt's group.

    Returns:
        Advantages: shape (B, G), zero-mean within each group.

    TODO(phase3): Implement:
        1. group_mean = rewards.mean(dim=-1, keepdim=True)    # (B, 1)
        2. group_std = rewards.std(dim=-1, keepdim=True)      # (B, 1)
        3. advantages = (rewards - group_mean) / (group_std + eps)
        4. Return advantages

    Note: if all G rewards are identical (std=0), advantages are all 0,
    which is correct — no gradient signal from that prompt.
    """
    raise NotImplementedError