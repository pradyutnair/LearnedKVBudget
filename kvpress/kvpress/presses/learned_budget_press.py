"""LearnedBudgetPress: drop-in replacement for AdaKVPress using a learned allocator.

This press wraps any ScorerPress (e.g. SnapKVPress) and replaces Ada-KV's
analytical L1 budget computation with the learned BudgetAllocator MLP.

The integration is surgical (PRD Section 5.2):
    Before (AdaKVPress): budget comes from global top-k across flattened heads
    After  (this press): budget comes from allocator MLP, eviction is per-head

Architecture comparison:
    AdaKVPress.compress():
        1. score() → (B, H_kv, K)
        2. Safeguard top n_safe per head
        3. Global bottom-k across ALL heads (flattened) → masked indices
        → Result: variable tokens per head, but decided by global ranking

    LearnedBudgetPress.compress():
        1. score() → (B, H_kv, K)
        2. Extract features from attentions + keys
        3. allocator(features) → per-head budget vector (B, H_kv)
        4. Per-head top-k eviction using each head's individual budget
        → Result: variable tokens per head, decided by the learned policy

Critical implementation detail — kvpress's masking mechanism:
    AdaKVPress does NOT physically remove tokens. Instead, it stores
    `module.masked_key_indices` — a tuple of (batch, head, seq) indices
    that get masked to -inf during attention computation.
    See: kvpress/presses/attention_patch.py

    Your press MUST use the same mechanism. Study how AdaKVPress builds
    the masked_key_indices tuple.

PRD references:
    - Section 4.1 (Architecture Overview — component 3)
    - Section 5.2 (Key Implementation Detail)

Documentation links:
    - torch.topk: https://pytorch.org/docs/stable/generated/torch.topk.html
    - torch.scatter_: https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html
    - BasePress: see base_press.py in this directory
    - AdaKVPress: see adakv_press.py in this directory (your primary reference)
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress

from kvpress.learned_kv_budget.allocator import BudgetAllocatorConfig, create_allocators_for_model
from kvpress.learned_kv_budget.learned_budget_features import (
    compute_attention_entropy,
    compute_topk_attention_mass,
    compute_key_norm_variance,
    build_feature_tensor,
)
from kvpress.learned_kv_budget.learned_budget_feature_collection import (
    compute_adakv_target_proxy_from_attentions,
)


@dataclass
class LearnedBudgetPress(BasePress):
    """Press that uses a learned MLP to allocate per-head KV budgets.

    This is the core deliverable of the LearnedKVBudget project. It wraps
    a ScorerPress (which provides per-token importance scores) and adds
    a learned budget allocation layer on top.

    Parameters
    ----------
    press : ScorerPress
        The underlying scorer (e.g., SnapKVPress, ExpectedAttentionPress).
        Provides token-level importance scores per head.
    allocators : nn.ModuleList or None
        One BudgetAllocator per layer. If None, must be set before use
        via post_init_from_model() or direct assignment.
    alpha_safeguard : float
        Minimum fraction of budget each head must retain (same as AdaKVPress).
        Prevents any head from being completely emptied.
    topk_fraction : float
        Fraction used for top-k attention mass feature (passed to feature extractor).

    Example
    -------
    >>> from kvpress.presses.snapkv_press import SnapKVPress
    >>> scorer = SnapKVPress(compression_ratio=0.5)
    >>> press = LearnedBudgetPress(press=scorer)
    >>> press.load_allocators("path/to/checkpoint.pt")
    >>> with press(model):
    ...     outputs = model(input_ids, past_key_values=cache)
    """

    press: ScorerPress = field(default=None)
    allocators: object = field(default=None, repr=False)  # nn.ModuleList at runtime
    alpha_safeguard: float = 0.20
    topk_fraction: float = 0.1

    def __post_init__(self):
        """Validate configuration.

        TODO(phase2): Implement validation:
            1. Assert press is a ScorerPress instance
            2. Assert alpha_safeguard is in [0, 1]
            3. If allocators is not None, assert it's an nn.ModuleList
        """
        assert isinstance(self.press, ScorerPress), "LearnedBudgetPress requires a ScorerPress as input"
        assert 0 <= self.alpha_safeguard <= 1, "alpha_safeguard should be in [0, 1]"
        if self.allocators is not None:
            assert isinstance(self.allocators, nn.ModuleList), "allocators should be an nn.ModuleList"

    def post_init_from_model(self, model):
        """Initialize from model config if allocators not yet created.

        This is called automatically by BasePress.__call__() before hooks
        are registered.

        TODO(phase2): Implement:
            1. Call self.press.post_init_from_model(model)
            2. If self.allocators is None, create them:
               - Read num_layers from model.config.num_hidden_layers
               - Read num_kv_heads from model.config.num_key_value_heads
               - Create BudgetAllocatorConfig with those values
               - Call create_allocators_for_model()
               - Store as self.allocators
        """
        self.press.post_init_from_model(model)
        if self.allocators is None:
            num_layers = model.config.num_hidden_layers
            num_kv_heads = model.config.num_key_value_heads
            config = BudgetAllocatorConfig(num_kv_heads=num_kv_heads)
            self.allocators = create_allocators_for_model(num_layers, config)

    @property
    def compression_ratio(self):
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value):
        self.press.compression_ratio = value

    def _reduce_query_to_kv_heads(self, tensor: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
        """Reduce the number of query heads to the number of KV heads.

        Args:
            tensor: Tensor to reduce, shape (B, H_q, ...)
            num_kv_heads: Number of KV heads
        """
        bsz, num_q_heads = tensor.shape
        if num_q_heads == num_kv_heads:
            return tensor
        group_size = num_q_heads // num_kv_heads
        return tensor.reshape(bsz, num_kv_heads, group_size).mean(dim=-1)
    def _extract_features(
        self,
        module: torch.nn.Module,
        keys: torch.Tensor,
        attentions: torch.Tensor,
    ) -> torch.Tensor:
        """Extract per-head features for the allocator from this layer's data.

        This reuses the feature functions from Phase 1, called inside the
        compress() hook where we have direct access to keys and attentions.

        Args:
            module: The attention layer (has module.layer_idx, module.config, etc.)
            keys: Key tensor, shape (B, H_kv, K, D)
            attentions: Attention weights, shape (B, H_q, Q, K) or None

        Returns:
            Feature tensor, shape (B, H_kv, 4)

        TODO(phase2): Implement:
            1. Compute attention_entropy from attentions         → (B, H_q)
            2. Compute topk_attention_mass from attentions       → (B, H_q)
            3. Compute key_norm_variance from keys               → (B, H_kv)
            4. Compute adakv_proxy from attentions               → (B, H_q)
            5. Handle GQA: if H_q != H_kv, reduce H_q → H_kv
               For features derived from attentions (H_q), you need to
               aggregate over the query heads that share each KV head.
               Options: mean over grouped query heads, or just use the
               KV-head-aligned version. The align_kv_to_attention_heads
               function goes the OTHER direction (H_kv → H_q). For this
               direction, you need a reduce: reshape (B, H_q) → (B, H_kv, G)
               then mean over G.
            6. Stack into (B, H_kv, 4) via build_feature_tensor()
        """
        # Compute attention entropy
        attention_entropy = compute_attention_entropy(attentions) # (B, H_q)
        attention_entropy = self._reduce_query_to_kv_heads(attention_entropy, module.config.num_key_value_heads) # (B, H_kv)

        # Compute topk attention mass
        topk_attention_mass = compute_topk_attention_mass(attentions, topk_fraction=self.topk_fraction) # (B, H_q)
        topk_attention_mass = self._reduce_query_to_kv_heads(topk_attention_mass, module.config.num_key_value_heads) # (B, H_kv)

        # Compute Ada-KV L1 score
        adakv_proxy = compute_adakv_target_proxy_from_attentions(attentions, eps=1e-8) # (B, H_q)
        adakv_proxy = self._reduce_query_to_kv_heads(adakv_proxy, module.config.num_key_value_heads) # (B, H_kv)

        # Compute key norm variance
        key_norm_variance = compute_key_norm_variance(keys) # (B, H_kv)

        # Stack into (B, H_kv, 4)
        feature_tensor = build_feature_tensor(attention_entropy, topk_attention_mass, key_norm_variance, adakv_proxy) # (B, H_kv, 4)
        return feature_tensor
        
    def compress(
        self,
        module: torch.nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress KV cache using learned per-head budget allocation.

        This is the main method. It replaces AdaKVPress.compress().

        Compare with AdaKVPress.compress() (adakv_press.py lines 53-78):
            AdaKV: global top-k across all heads (implicit budget allocation)
            Ours:  explicit per-head budgets from the allocator MLP

        Args:
            module: Attention layer (module.layer_idx gives the layer index)
            hidden_states: (B, seq_len, hidden_dim)
            keys:   (B, H_kv, K, D)
            values: (B, H_kv, K, D)
            attentions: (B, H_q, Q, K) — attention weights from this layer
            kwargs: Forward pass kwargs (contains cache, position info)

        Returns:
            (keys, values) — unchanged tensors, but module.masked_key_indices
            is set so the attention patch masks out evicted tokens.

        TODO(phase2): Implement step by step:

            Step 1: Early exit
                if self.compression_ratio == 0: return keys, values

            Step 2: Compute token-level scores using the wrapped scorer
                scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)
                # scores shape: (B, H_kv, K)
                # Higher score = more important token

            Step 3: Compute total budget
                bsz, num_kv_heads, k_len = scores.shape
                n_kept_total = int(k_len * (1 - self.compression_ratio))
                # This is the total number of tokens to retain across ALL heads

            Step 4: Extract features and get per-head budgets
                features = self._extract_features(module, keys, attentions)
                layer_idx = module.layer_idx
                allocator = self.allocators[layer_idx]
                budgets = allocator(features, total_budget=n_kept_total)
                # budgets shape: (B, H_kv) — tokens to keep per head

            Step 5: Apply safeguard
                # Each head must keep at least alpha_safeguard * its_budget tokens
                min_per_head = max(1, int(n_kept_total / num_kv_heads * self.alpha_safeguard))
                budgets = budgets.clamp(min=min_per_head)
                # Re-normalize so budgets still sum to n_kept_total
                budgets = budgets / budgets.sum(dim=-1, keepdim=True) * n_kept_total
                budgets = budgets.round().long()

            Step 6: Per-head eviction → build masked indices
                # This is the KEY difference from AdaKVPress.
                # AdaKV does global top-k across all heads.
                # We do per-head top-k with each head's own budget.
                #
                # For each head h:
                #   n_evict_h = k_len - budgets[b, h]
                #   Find the n_evict_h lowest-scoring tokens in that head
                #   Add those (batch, head, seq) indices to the mask
                #
                # Efficient implementation hint:
                #   - For each head, get bottom-k indices from scores[:, h, :]
                #   - Concatenate all evicted indices across heads
                #   - Store as module.masked_key_indices = (batch_idx, head_idx, seq_idx)
                #
                # Study AdaKVPress lines 70-77 for the exact format expected
                # by the attention patch.

            Step 7: Return unchanged keys, values
                return keys, values
        """
        if self.compression_ratio == 0:
            return keys, values

        scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs) # (B, H_kv, K)

        # Total budget (number of tokens to retain across all heads)
        bsz, num_kv_heads, k_len = scores.shape # (B, H_kv, K)
        n_kept_total = int(k_len * (1 - self.compression_ratio))

        # Extract features and get per-head budgets
        features = self._extract_features(module, keys, attentions)
        layer_idx = module.layer_idx
        allocator = self.allocators[layer_idx]
        budgets = allocator(features, total_budget=n_kept_total) # (B, H_kv) - number of tokens to keep per head

        # Apply safeguard
        min_per_head = max(1, int(n_kept_total / num_kv_heads * self.alpha_safeguard))
        # Re-normalize so budgets still sum to n_kept_total
        budgets = budgets.clamp(min=min_per_head) 
        budgets = budgets / budgets.sum(dim=-1, keepdim=True) * n_kept_total
        budgets = budgets.round().long()

        # Per-head eviction
        all_batch = []
        all_head = []
        all_seq = []

        # Iterate over each batch and key-value head
        for b in range(bsz):
            for h in range(num_kv_heads):
                # Calculate the number of tokens to evict for this head
                n_evict = k_len - budgets[b, h].item()
                if n_evict <= 0:
                    continue
                # Find the indices of the lowest-scoring tokens to evict
                bottom_k_indices = torch.topk(-scores[b, h, :], n_evict, dim=-1).indices # (n_evict,)
                # Collect the indices into lists corresponding to batch, head, and sequence positions
                all_batch.append(torch.full((n_evict,), b, dtype=torch.long, device=scores.device))
                all_head.append(torch.full((n_evict,), h, dtype=torch.long, device=scores.device))
                all_seq.append(bottom_k_indices.flatten())
        
        # If any tokens are marked for eviction, store their indices in module.masked_key_indices
        if all_batch:
            module.masked_key_indices = (
                torch.cat(all_batch),
                torch.cat(all_head),
                torch.cat(all_seq),
            )
        return keys, values

    def load_allocators(self, checkpoint_path: str, device: str = "cuda"):
        """Load trained allocator weights from a checkpoint file.

        TODO(phase2): Implement:
            1. state_dict = torch.load(checkpoint_path, map_location=device)
            2. self.allocators.load_state_dict(state_dict)
        """
        state_dict = torch.load(checkpoint_path, map_location=torch.device(device))
        self.allocators.load_state_dict(state_dict)

    def save_allocators(self, checkpoint_path: str):
        """Save allocator weights for later inference or transfer.

        TODO(phase2): Implement:
            1. torch.save(self.allocators.state_dict(), checkpoint_path)
        """
        torch.save(self.allocators.state_dict(), checkpoint_path)