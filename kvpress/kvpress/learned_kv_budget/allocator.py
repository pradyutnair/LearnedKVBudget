"""Budget Allocator MLP: maps per-head features to a budget vector.

This module implements the tiny MLP from PRD Section 4.3 that replaces
Ada-KV's analytical L1 budget allocation with a learned, input-dependent
allocation policy.

Architecture (for H KV heads, F=4 features):
    Input:  f_l ∈ R^{HxF}  (flattened to R^{FxH})
    Hidden: Linear(FxH, 128) → ReLU → Linear(128, 64) → ReLU
    Output: Linear(64, H) → softmax(logits / τ) x total_budget
    Result: b_l ∈ R^H  where sum(b_l) = total_budget

Key design decisions to understand:
    1. Softmax constrains output to the probability simplex, so budgets
       automatically sum to total_budget. No clamping or renormalization needed.
    2. Temperature τ is a learnable scalar — low τ → sharp (peaky) allocations,
       high τ → flat (uniform-like) allocations. Initialized to tau_init.
    3. For GQA models like Llama-3.1-8B, H = num_kv_heads (8), NOT num_query_heads (32).
       The allocator decides budget per KV head group.

PRD references:
    - Section 4.3 (Budget Allocator Network)
    - Section 4.4 (GRPO Training Loop — the allocator is πθ)

Documentation links:
    - nn.Module: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    - nn.Linear: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    - F.softmax:  https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
    - nn.Parameter: https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html

Sizing reference (Llama-3.1-8B):
    H=8 KV heads, F=4 features → input_dim=FxH=32
    Linear(32,128) + Linear(128,64) + Linear(64,8) ≈ 1.5K params per layer
    32 layers x 1.5K = ~48K total params (negligible vs 8B LLM)
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BudgetAllocatorConfig:
    """Configuration for the per-layer budget allocator MLP.

    Attributes:
        num_kv_heads: Number of KV heads in the target model.
            Llama-3.1-8B = 8, Qwen-2.5-7B = 7 (check model config).
            Read from model.config.num_key_value_heads.
        num_features: Number of scalar features per head (default 4).
            Matches the feature vector from learned_budget_features.py.
        hidden_dims: Sizes of hidden layers. PRD suggests [128, 64].
        tau_init: Initial value for the learnable softmax temperature.
            Lower → sharper initial allocations (less uniform).
            PRD suggests starting at 1.0; risk mitigation suggests 0.5.
    """

    num_kv_heads: int = 8
    num_features: int = 4
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64])
    tau_init: float = 1.0


class BudgetAllocator(nn.Module):
    """Lightweight MLP that outputs a per-head budget vector.

    Given per-head feature summaries from the prefill pass, produces a
    continuous budget allocation over H KV heads that sums to total_budget.

    Usage during inference:
        features = extract_features(...)   # (B, H, 4)
        budget = allocator(features, total_budget=n_kept)  # (B, H)
        # budget[b, h] = number of tokens head h retains for batch item b

    Usage during GRPO training:
        # 1. Get concentration params for Dirichlet sampling
        fractions = allocator.get_budget_fractions(features)  # (B, H), sums to 1
        # 2. Sample G allocations via Dirichlet (see dirichlet.py)
        # 3. Evaluate each, compute advantages, update allocator weights
    """

    def __init__(self, config: BudgetAllocatorConfig):
        super().__init__()
        self.config = config

        # TODO(phase2): Build the MLP layers.
        #
        # The network flattens (B, H, F) → (B, H*F), passes through hidden layers
        # with ReLU activations, and outputs H logits.
        #
        # Steps:
        #   1. Compute input_dim = num_kv_heads * num_features
        #   2. Build nn.Sequential with:
        #      - Linear(input_dim, hidden_dims[0]), ReLU
        #      - Linear(hidden_dims[0], hidden_dims[1]), ReLU
        #      - Linear(hidden_dims[1], num_kv_heads)
        #   3. Store as self.mlp
        #
        # Hint: use a loop over hidden_dims to generalize to any depth.
        input_dim = self.config.num_features * self.config.num_kv_heads
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, self.config.hidden_dims[0]), # FxH -> 128
            nn.ReLU(),
            nn.Linear(self.config.hidden_dims[0], self.config.hidden_dims[1]), # 128 -> 64
            nn.ReLU(),
            nn.Linear(self.config.hidden_dims[1], self.config.num_kv_heads), # 64 -> H
        )

        # TODO(phase2): Create learnable temperature parameter.
        #
        self.log_tau = nn.Parameter(torch.log(torch.tensor(self.config.tau_init)))
        #
        # Why log_tau instead of tau directly?
        #   - tau must be positive. Storing log(tau) and exponentiating ensures
        #     positivity without constrained optimization.
        #   - Initialize: log_tau = log(tau_init)
        #
        # Then in forward: tau = self.log_tau.exp()

    @property
    def tau(self) -> torch.Tensor:
        """Current softmax temperature (always positive via exp)."""
        # TODO(phase2): return self.log_tau.exp()
        return self.log_tau.exp()

    def _compute_logits(self, features: torch.Tensor) -> torch.Tensor:
        """Run features through the MLP to get raw logits.

        Args:
            features: Per-head feature tensor, shape (B, H, F).
                B = batch size
                H = num_kv_heads (from config)
                F = num_features (from config, default 4)

        Returns:
            Raw logits, shape (B, H). NOT yet softmax'd.

        TODO(phase2): Implement:
            1. Flatten features from (B, H, F) → (B, H*F)
            2. Pass through self.mlp
            3. Return raw logits (B, H)
        """
        features_flat = features.flatten(start_dim=1, end_dim=2) # (B, H, F) -> (B, HxF)
        logits = self.mlp(features_flat) # (B, HxF) -> (B, H)
        return logits

    def get_budget_fractions(self, features: torch.Tensor) -> torch.Tensor:
        """Compute budget fractions (probability simplex, sums to 1).

        This is the "policy output" used for:
        - Deterministic inference (multiply by total_budget)
        - Dirichlet concentration params during GRPO training

        Args:
            features: shape (B, H, F)

        Returns:
            Budget fractions, shape (B, H), each row sums to 1.0.

        TODO(phase2): Implement:
            1. Get logits via self._compute_logits(features)
            2. Apply softmax(logits / tau, dim=-1)
            3. Return the result
        """
        logits = self._compute_logits(features)
        budget_fractions = F.softmax(logits / self.tau, dim=-1)
        return budget_fractions

    def forward(
        self,
        features: torch.Tensor,
        total_budget: int,
    ) -> torch.Tensor:
        """Compute per-head budget allocation (number of tokens per head).

        Args:
            features: Per-head features, shape (B, H, F).
            total_budget: Total number of tokens to retain across all heads.
                Typically: n_kept = int(seq_len * (1 - compression_ratio)).

        Returns:
            Per-head budgets, shape (B, H).
            budget[b, h] = number of tokens head h should retain.
            Sum across heads equals total_budget (approximately, due to rounding).

        TODO(phase2): Implement:
            1. fractions = self.get_budget_fractions(features)    # (B, H)
            2. budgets = fractions * total_budget                 # (B, H)
            3. (Optional) Round to integers while preserving the sum.
               Simplest: use floor + distribute remainder to heads with
               largest fractional parts. Or just return floats and let
               the press handle rounding.
            4. Return budgets
        """
        fractions = self.get_budget_fractions(features)
        budgets = fractions * total_budget
        budgets = budgets.round().long()
        return budgets


def create_allocators_for_model(
    num_layers: int,
    config: BudgetAllocatorConfig,
) -> nn.ModuleList:
    """Create one BudgetAllocator per layer (PRD Section 4.2 recommends per-layer).

    Args:
        num_layers: Number of transformer layers.
            Llama-3.1-8B: 32 layers.
            Read from model.config.num_hidden_layers.
        config: Shared config for all allocators.

    Returns:
        nn.ModuleList of BudgetAllocator instances, indexed by layer.

    TODO(phase2): Implement — just wrap BudgetAllocator(config) in a loop.
    Consider: should all layers share weights? PRD says no (independent per layer).
    """
    return nn.ModuleList([BudgetAllocator(config) for _ in range(num_layers)])