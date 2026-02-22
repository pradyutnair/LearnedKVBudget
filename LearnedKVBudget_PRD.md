# Technical PRD: LearnedKVBudget

**RL-Based Per-Head KV Cache Budget Allocation via GRPO**

| | |
|---|---|
| **Author** | Pradyut Nair |
| **Affiliation** | Prosus / University of Amsterdam |
| **Date** | February 2026 |
| **Timeline** | 14 days (2 weeks) |
| **Compute** | Snellius (NVIDIA A100 cluster) |
| **Status** | Planning |

---

## 1. Executive Summary

KV cache compression is critical for efficient long-context LLM inference. Existing methods allocate compression budgets either uniformly across all attention heads or via analytical heuristics (Ada-KV's L1 bound, HeadKV's retrieval head scores). No existing work learns an input-dependent, continuous budget allocation policy across attention heads using reinforcement learning.

This project trains a lightweight budget allocation network that, given per-head summary statistics from the prefill phase, outputs a continuous budget vector determining how many KV pairs each head retains. The allocator is trained via Group Relative Policy Optimization (GRPO) with downstream task accuracy as the reward signal. At inference time, the learned allocator replaces the analytical budget computation in Ada-KV, while the underlying eviction policy (SnapKV, H2O, etc.) remains unchanged.

**One-line novelty claim:** First work to train an input-dependent, continuous budget allocation policy across attention heads via RL, as opposed to learning eviction within fixed-budget heads (KV Policy, ForesightKV) or learning static head masks (RLKV).

---

## 2. Problem Statement

### 2.1 Background

Long-context LLM serving is bottlenecked by KV cache memory. For example, serving Llama 3.1-70B at 1M tokens in fp16 requires up to 330GB of KV cache memory. Eviction-based methods reduce this by discarding low-importance KV pairs at inference time.

A critical design choice in any eviction method is budget allocation: given a total cache budget B, how should it be distributed across the H attention heads per layer? Most methods apply a uniform budget B/H to every head. Ada-KV showed that non-uniform, head-wise allocation based on an analytical L1 loss bound significantly improves quality at the same compression ratio.

### 2.2 Gap in the Literature

The analytical approach has clear limitations: the L1 bound is a local, single-step approximation that does not account for how budget decisions propagate through autoregressive generation. An RL-trained allocator can optimize directly for end-task quality over full generation trajectories.

Three concurrent RL-based KV cache works exist, but each solves a different sub-problem:

- **RLKV** (arXiv 2510.08525): Learns a static binary mask over heads (full vs. compressed). The mask is input-independent—same allocation for all inputs. Does not learn continuous budgets.
- **KV Policy** (arXiv 2602.10238): Trains per-head RL agents that rank tokens within each head. Every head gets the same fixed budget—the inter-head allocation is not learned.
- **ForesightKV** (arXiv 2602.03203): Trains a scoring model to predict per-token long-term importance via GRPO. Operates under a fixed, global budget. Does not allocate different budgets to different heads.

None of these learn which heads should get more or fewer tokens as a function of the input.

---

## 3. Competitive Landscape

The following table decomposes the design space into four dimensions and shows where each method sits. Our contribution occupies the only unoccupied cell: learned, input-dependent, continuous inter-head budget allocation.

| Dimension | RLKV | KV Policy | ForesightKV | **Ours** |
|---|---|---|---|---|
| What is learned | Static binary head mask | Per-token ranking within head | Per-token importance scores | **Per-head budget fractions** |
| Action space | Binary gate per head | Token ranking per head | Token scores (global) | **Continuous vector b ∈ ℝᴴ** |
| Budget type | Fixed (full vs. local) | Uniform across heads | Fixed global budget | **Input-dependent per head** |
| Input-dependent | No (same mask for all) | Yes (token features) | Yes (token features) | **Yes (head statistics)** |
| RL algorithm | PPO + L1 penalty | Custom RL reward | GRPO (MDP) | **GRPO** |
| Complementary? | Orthogonal | Stackable | Stackable | **—** |

**Key insight:** Our method is complementary to all three. RLKV decides which heads matter at all; KV Policy and ForesightKV decide which tokens matter within a head; we decide how much budget each head gets. These can be stacked: our allocator outputs budget fractions, then KV Policy's agents rank tokens within each head's allocation.

---

## 4. Technical Approach

### 4.1 Architecture Overview

The system has three components that execute sequentially during inference:

1. **Feature extractor:** After prefill, extract per-head summary statistics from the KV cache and attention patterns. These are cheap to compute and already available in the attention forward pass.

2. **Budget allocator network πθ:** A small MLP (2–3 layers, ~50K parameters total) that takes the per-head features and outputs a continuous budget vector b ∈ ℝᴴ via softmax, where H is the number of KV heads. The softmax ensures budgets sum to the total budget B.

3. **Eviction policy:** Any existing scorer press (SnapKV, H2O, ExpectedAttention) from KVPress. Each head evicts tokens to meet its allocated budget bₕ. The eviction policy is frozen and not trained.

### 4.2 Per-Head Feature Vector

For each KV head h in each layer l, we extract the following features from the prefill pass. All features are scalars, yielding a feature vector f_{l,h} ∈ ℝ⁴ per head:

- **Attention entropy:** H(aₕ) = −Σ aᵢ log(aᵢ) averaged over queries in the observation window. High entropy indicates diffuse attention (head needs more tokens); low entropy indicates focused attention (head can be compressed aggressively).

- **Top-k attention mass:** Fraction of total attention captured by the top-k tokens (k = current budget candidate). Directly measures how much information is retained under compression.

- **Key norm variance:** σ²(‖kᵢ‖) across tokens in the cache. Heads with high key norm variance have more differentiated tokens, suggesting importance varies more and budget allocation matters more.

- **Ada-KV L1 score:** The analytical L1 loss upper bound from Ada-KV (Feng et al., 2024). Included as a feature so the network can learn to improve upon it rather than ignoring it.

The full input to the allocator for one layer is f_l ∈ ℝ^{H×4}. For a model with L layers, we train L independent allocators (one per layer) or a single allocator with layer index as an additional feature. The per-layer approach is preferred for simplicity.

### 4.3 Budget Allocator Network

Architecture per layer:

- **Input:** f_l ∈ ℝ^{H×4} (flattened to ℝ^{4H})
- **Hidden:** Linear(4H, 128) → ReLU → Linear(128, 64) → ReLU → Linear(64, H)
- **Output:** softmax(logits / τ) × B_l, yielding budget vector b_l ∈ ℝᴴ that sums to layer budget B_l
- **Temperature τ:** learnable scalar initialized to 1.0, allowing the network to sharpen or flatten the allocation

For Llama-3.1-8B (8 KV heads, 32 layers): 32 allocators × ~1.5K params each = ~48K total parameters. This is negligible compared to the LLM.

**Exploration:** During GRPO training, we sample from a Dirichlet distribution parameterized by the softmax outputs (treating them as concentration parameters scaled by a factor α). This provides smooth exploration over the simplex. At inference, we use the argmax (deterministic) softmax output.

### 4.4 GRPO Training Loop

We adapt the GRPO algorithm (Guo et al., 2025) for our setting. The key differences from standard GRPO for LLM training are: (a) the policy is a tiny MLP, not the LLM itself; (b) the action is a continuous budget vector, not a discrete token; (c) the LLM is frozen and used only for evaluation rollouts.

**Training procedure for each iteration:**

1. Sample a batch of prompts from the training set (RULER synthetic tasks or math problems).
2. Run prefill on each prompt through the frozen LLM to obtain per-head features.
3. For each prompt, sample G budget allocations from the policy (G = group size, typically 8–16).
4. For each sampled allocation, apply the eviction policy (e.g., SnapKV) with those budgets and run generation to completion.
5. Compute reward for each rollout: r = task_accuracy + λ × compression_bonus (see Section 4.5).
6. Compute group-relative advantages: Aᵢ = (rᵢ − mean(r)) / std(r) within each prompt's group.
7. Update πθ to maximize E[Aᵢ × log πθ(bᵢ | f)] with a KL penalty to a reference policy (uniform allocation).

### 4.5 Reward Function

The reward signal is the primary design choice. We propose a composite reward:

> *r = accuracy(y, y\*) + λ × budget_efficiency*

Where:

- **accuracy(y, y\*):** Binary correctness for math tasks (AIME, MATH-500) or exact match for RULER synthetic tasks. This is the primary signal.

- **budget_efficiency:** Bonus for allocations that concentrate budget in fewer heads (measured by negative entropy of the budget vector). This encourages the allocator to learn sparse, decisive allocations rather than collapsing to uniform. λ = 0.1 as starting point.

**Alternative reward (for ablation):** Use negative perplexity increase relative to the uncompressed model, similar to ForesightKV's LM loss signal. This is cheaper (no generation needed) but less directly aligned with task performance.

---

## 5. Implementation Plan

### 5.1 Codebase and Dependencies

- **Base framework:** Fork NVIDIA KVPress (`github.com/NVIDIA/kvpress`). Provides all eviction methods, AdaKVPress wrapper for per-head budgets, benchmarking infrastructure, and HuggingFace Transformers integration.

- **Variable-length KV kernels:** AdaKV's CUDA kernels (`github.com/FFY0/AdaKV`) using `flash_attn_varlen_func` for efficient inference with non-uniform head budgets.

- **RL training:** Custom GRPO loop in PyTorch (~200–300 lines). The policy is a tiny MLP, not the LLM, so standard RL infra (veRL, OpenRLHF) is overkill. We only need: sample from Dirichlet, compute log-prob, compute advantage, gradient step.

- **Target models:** Llama-3.1-8B-Instruct (8 KV heads, 32 layers, GQA), Qwen-2.5-7B-Instruct. Both supported by KVPress.

- **Evaluation benchmarks:** RULER (synthetic long-context, 13 subtasks), LongBench (16 real-world tasks), AIME 2024/2025 and MATH-500 (reasoning under compression).

### 5.2 Key Implementation Detail: Integrating with AdaKVPress

The integration point is surgical. AdaKVPress is a wrapper press in KVPress that takes any ScorerPress, computes per-head scores, and then redistributes the eviction budget across heads using an analytical formula. We replace exactly one function:

```
Before: budget_per_head = ada_kv_l1_allocation(attention_scores, total_budget)

After:  features = extract_head_features(kv_cache, attention_scores)
        budget_per_head = learned_allocator(features) * total_budget
```

Everything downstream (token scoring, eviction, variable-length KV cache handling) remains unchanged. This is the primary engineering advantage of building on KVPress.

### 5.3 Timeline (14 Days)

| Phase | Days | Deliverables |
|---|---|---|
| **Setup** | 1–2 | Fork KVPress, verify Ada-KV baseline runs on Snellius. Reproduce Ada-SnapKV numbers on RULER at 20%/40%/60% compression. Set up evaluation pipeline. |
| **Features** | 3–4 | Implement per-head feature extraction (attention entropy, top-k mass, key norm variance, L1 score). Verify features are informative via correlation analysis with Ada-KV's analytical budgets. |
| **Allocator** | 4–5 | Implement budget allocator MLP. Wire it into AdaKVPress replacing the analytical computation. Verify forward pass produces valid budget vectors that sum correctly. |
| **GRPO Loop** | 5–7 | Implement GRPO training: Dirichlet sampling, reward computation (RULER accuracy), advantage normalization, policy gradient update. Debug on small-scale (1 layer, Qwen-3-1.7B) before scaling. |
| **Training** | 7–9 | Full training on Llama-3.1-8B with RULER training split. Hyperparameter sweep: group size G ∈ {4, 8, 16}, learning rate, temperature init, λ for budget efficiency bonus. Save checkpoints. |
| **Evaluation** | 9–11 | Benchmark learned allocator vs. baselines (uniform, Ada-KV analytical, HeadKV heuristic, PyramidKV) across all compression ratios on RULER, LongBench, MATH-500. Generate accuracy-vs-compression curves. |
| **Ablations** | 11–12 | Ablate feature importance (drop each feature), reward function (accuracy-only vs. composite), allocator capacity (1-layer vs. 3-layer MLP), and cross-model transfer (train on Llama, eval on Qwen). |
| **Writeup** | 13–14 | Technical report with motivation, method, results, and analysis. Prepare figures: accuracy-vs-compression curves, learned budget heatmaps (layer × head), comparison table. |

---

## 6. Evaluation Plan

### 6.1 Baselines

1. **Uniform allocation:** Each head gets B/H tokens. Standard default for SnapKV, H2O, etc.
2. **Ada-KV (analytical):** L1 loss bound from Feng et al. (2024). State-of-the-art training-free head-wise allocation. This is the primary baseline to beat.
3. **HeadKV (heuristic):** Retrieval head importance scores for budget allocation (Fu et al., 2024, ICLR 2025).
4. **PyramidKV (fixed heuristic):** Layer-wise pyramid allocation. Not head-wise, but widely cited.
5. **LAVa (analytical):** Unified dynamic head + layer budget allocation via information loss minimization (Shen et al., 2025, EMNLP 2025).

### 6.2 Metrics

- **Primary:** Task accuracy at fixed compression ratios (20%, 40%, 60%, 80% cache retention). Reported as average score across RULER subtasks and individual LongBench categories.
- **Secondary:** Perplexity increase relative to uncompressed model (measures LM quality degradation).
- **Efficiency:** Allocator inference overhead in milliseconds (should be <1ms on GPU, negligible vs. prefill).

### 6.3 Key Experiments

1. **Main result:** Accuracy-vs-compression curves for all methods on RULER and LongBench. Target: beat Ada-KV by 2–5 points at 20–40% retention where the gap matters most.

2. **Reasoning under compression:** MATH-500 and AIME 2024 with Qwen-2.5-7B reasoning traces. Show that learned allocation better preserves reasoning chains than analytical methods.

3. **Budget heatmap analysis:** Visualize learned allocations (layer × head matrix) for different input types. Show the allocator learns interpretable patterns (e.g., retrieval heads get more budget on needle-in-haystack tasks).

4. **Ablations:** Feature dropout (which features matter?), reward function variants, allocator capacity, cross-model transfer.

---

## 7. Risks and Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| **GRPO training instability** | High | The continuous action space (Dirichlet) may have high variance gradients. Mitigation: start with large group size G=16, use baseline subtraction, clip advantages. Fallback: switch to PPO with a tiny value network. |
| **Marginal gains over Ada-KV** | Medium | The analytical L1 bound may be near-optimal, leaving little room for improvement. Mitigation: focus evaluation on low compression ratios (20–40%) and reasoning tasks where analytical bounds are known to be loose. If gains are <1 point, pivot to demonstrating the allocator learns interpretable patterns that explain why Ada-KV works. |
| **Training cost** | Medium | Each GRPO iteration requires G full generation rollouts per prompt. For reasoning tasks with long outputs, this is expensive. Mitigation: train on RULER (short outputs) first, then fine-tune on reasoning. Use perplexity reward as cheaper proxy during early training. |
| **Allocator collapses to uniform** | Low | Softmax with insufficient training converges to uniform. Mitigation: initialize temperature τ low (0.5) to encourage peaky initial allocations. Add budget efficiency bonus in reward to penalize entropy. |
| **Overfitting to training tasks** | Medium | Allocator may memorize RULER patterns. Mitigation: train on diverse RULER subtasks, evaluate on held-out LongBench tasks. Report zero-shot transfer results explicitly. |

---

## 8. Success Criteria

### 8.1 Minimum Viable Result (Must-Have)

- Learned allocator matches or exceeds Ada-KV on RULER at 40% compression for Llama-3.1-8B.
- Allocator overhead is <5ms per prefill (negligible vs. attention computation).
- Clear budget heatmap showing non-trivial, input-dependent allocation patterns.

### 8.2 Target Result (Nice-to-Have)

- 2–5 point improvement over Ada-KV at 20% compression on RULER.
- Demonstrated gains on reasoning benchmarks (MATH-500) under compression.
- Successful zero-shot transfer: train on Llama, evaluate on Qwen without retraining.
- Stackability demonstration: show gains from combining learned budgets + KV Policy's learned eviction.

---

## 9. Key References

1. Ada-KV: Feng et al. (2024). Optimizing KV Cache Eviction by Adaptive Budget Allocation. NeurIPS 2025. arXiv:2407.11550
2. RLKV: (2025). Which Heads Matter for Reasoning? RL-Guided KV Cache Compression. arXiv:2510.08525
3. KV Policy (KVP): Moschella et al. (2026). Learning to Evict from Key-Value Cache. arXiv:2602.10238
4. ForesightKV: (2026). Optimizing KV Cache Eviction for Reasoning Models by Learning Long-Term Contribution. arXiv:2602.03203 (ICLR 2026 submission)
5. LAVa: Shen et al. (2025). Layer-wise KV Cache Eviction with Dynamic Budget Allocation. EMNLP 2025 Findings. arXiv:2509.09754
6. HeadKV: Fu et al. (2024). Not All Heads Matter: A Head-Level KV Cache Compression Method. ICLR 2025. arXiv:2410.19258
7. GRPO: Guo et al. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.
8. NVIDIA KVPress: Devoto et al. (2025). Expected Attention: KV Cache Compression. github.com/NVIDIA/kvpress
9. R-KV: Cai et al. (2025). Redundancy-aware KV Cache Compression for Reasoning Models. NeurIPS 2025. arXiv:2505.24133
10. SCBench: Yue et al. (2025). Can't See the Forest for the Trees: KV Cache-Centric Benchmark. ICLR 2025.
