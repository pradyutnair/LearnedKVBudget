"""Dirichlet sampling and log-probability for GRPO exploration.

During GRPO training, the allocator's softmax output (budget fractions on the
simplex) is used to parameterize a Dirichlet distribution. Sampling from this
distribution provides smooth exploration around the policy's preferred allocation.

Why Dirichlet?
    The budget fractions must sum to 1 (they live on the simplex). The Dirichlet
    is the natural distribution over the simplex, just as the Gaussian is for R^n.
    - High concentration (alpha_scale large) → samples close to the mean (exploit)
    - Low concentration (alpha_scale small) → samples spread across simplex (explore)

Connection to the allocator:
    1. allocator.get_budget_fractions(features) → π ∈ R^H, sum = 1
    2. concentration = π * alpha_scale  (alpha_scale controls exploration)
    3. Sample G fraction vectors from Dir(concentration)
    4. Each sample is a candidate budget allocation to evaluate

PRD references:
    - Section 4.3 "Exploration" paragraph
    - Section 4.4 step 3 (sample G budget allocations)

Documentation links:
    - torch.distributions.Dirichlet:
      https://pytorch.org/docs/stable/distributions.html#dirichlet
    - Dirichlet distribution (Wikipedia):
      https://en.wikipedia.org/wiki/Dirichlet_distribution
    - Log-probability derivation:
      log p(x|alpha) = Σ[(alpha - 1) log(x)] + log Γ(Σalpha) - Σ log Γ(alpha)

Numerical stability notes:
    - Concentration params must be > 0. Clamp to min eps before constructing.
    - Sampled fractions can be very close to 0; clamp before computing log_prob.
    - torch.distributions.Dirichlet handles most of this internally.
"""

import torch
from torch.distributions import Dirichlet


def compute_concentration(
    budget_fractions: torch.Tensor,
    alpha_scale: float = 10.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Convert allocator's softmax output to Dirichlet concentration parameters.

    Args:
        budget_fractions: Policy output from allocator.get_budget_fractions(),
            shape (B, H), each row sums to 1.
        alpha_scale: Scaling factor controlling exploration variance.
            Higher → tighter around policy mean (more exploitation).
            Lower  → broader exploration across the simplex.
            PRD suggests sweeping {5, 10, 20}.
        eps: Floor for concentration to ensure valid Dirichlet params.

    Returns:
        Concentration parameters, shape (B, H), all entries > 0.

    TODO(phase3): Implement:
        1. concentration = budget_fractions * alpha_scale
        2. Clamp to min eps to avoid zero concentrations
        3. Return result
    """
    concentration = budget_fractions * alpha_scale
    concentration = concentration.clamp(min=eps)
    return concentration


def sample_dirichlet(
    concentration: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    """Sample G budget fraction vectors from Dirichlet(concentration).

    Each sample is a valid budget allocation (sums to 1, all positive).

    Args:
        concentration: Dirichlet params, shape (B, H).
        num_samples: G = group size for GRPO. Typically 4, 8, or 16.

    Returns:
        Sampled fractions, shape (B, G, H).
        samples[b, g, :] is one candidate budget allocation for batch item b.
        Each samples[b, g, :] sums to 1.0.

    TODO(phase3): Implement:
        1. Create Dirichlet distribution: dist = Dirichlet(concentration)
           Note: concentration is (B, H), Dirichlet treats last dim as event dim.
        2. Sample G times: samples = dist.sample((num_samples,))
           This gives shape (G, B, H)
        3. Permute to (B, G, H) for easier downstream use
        4. Return samples

    Hint: dist.sample((G,)) returns (G, B, H). Use .permute(1, 0, 2) → (B, G, H).
    """
    dist = Dirichlet(concentration)
    # Shape: (G, B, H)
    samples = dist.sample((num_samples,))
    # Shape: (B, G, H)
    samples = samples.permute(1, 0, 2)
    return samples


def dirichlet_log_prob(
    samples: torch.Tensor,
    concentration: torch.Tensor,
) -> torch.Tensor:
    """Compute log-probability of sampled fractions under the Dirichlet.

    This is needed for the GRPO policy gradient:
        loss = -E[advantage * log_prob(sampled_fractions | policy)]

    Args:
        samples: Sampled budget fractions, shape (B, G, H).
        concentration: Dirichlet params from compute_concentration(),
            shape (B, H).

    Returns:
        Log-probabilities, shape (B, G).
        log_prob[b, g] = log Dir(samples[b,g,:] | concentration[b,:])

    TODO(phase3): Implement:
        1. Create dist = Dirichlet(concentration)
        2. For each group sample, compute dist.log_prob(samples[:, g, :])
        3. Stack across G to get (B, G)

    Hint: You can loop over G (it's small, 4-16), or reshape cleverly:
        - Expand concentration from (B, H) to (B, G, H)
        - Create Dirichlet with (B, G, H) concentration
        - Call log_prob on samples directly
        Watch out: Dirichlet.log_prob expects the event dim to be last.
    """
    B, G, H = samples.shape
    concentration = concentration.unsqueeze(1).expand(B, G, H) # (B, H) -> (B, 1, H) -> (B, G, H)
    dist = Dirichlet(concentration)
    log_probs = dist.log_prob(samples) # (B, G)
    return log_probs


def sample_and_log_prob(
    budget_fractions: torch.Tensor,
    alpha_scale: float,
    num_samples: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience: sample G allocations and compute their log-probs in one call.

    This is the main entry point used by the GRPO trainer.

    Args:
        budget_fractions: From allocator.get_budget_fractions(), shape (B, H).
        alpha_scale: Dirichlet concentration scaling.
        num_samples: Group size G.
        eps: Numerical floor for concentrations.

    Returns:
        samples: (B, G, H) — sampled budget fraction vectors
        log_probs: (B, G) — log-probability of each sample under current policy

    TODO(phase3): Implement by composing the functions above:
        1. concentration = compute_concentration(budget_fractions, alpha_scale, eps)
        2. samples = sample_dirichlet(concentration, num_samples)
        3. log_probs = dirichlet_log_prob(samples, concentration)
        4. Return (samples, log_probs)
    """
    concentration = compute_concentration(budget_fractions, alpha_scale, eps) # (B, H)
    samples = sample_dirichlet(concentration, num_samples) # (B, G, H)
    log_probs = dirichlet_log_prob(samples, concentration) # (B, G)
    return samples, log_probs