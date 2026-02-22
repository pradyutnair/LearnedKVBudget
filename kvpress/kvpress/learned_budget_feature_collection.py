"""Phase 1 scaffolding to collect per-head feature datasets.

This script is intentionally scaffolded and does not implement full integration.
It defines a clean entry point and TODO-marked steps needed to:
1) run prefill with attention outputs
2) compute per-layer/per-head features
3) log Ada-KV analytical allocation targets
4) save a training-ready tabular artifact

Documentation links:
- Transformers generation + outputs:
  https://huggingface.co/docs/transformers/main_classes/text_generation
- Datasets loading:
  https://huggingface.co/docs/datasets/loading
- Torch inference mode:
  https://pytorch.org/docs/stable/generated/torch.inference_mode.html
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from kvpress.learned_budget_features import (
    FeatureConfig,
    build_feature_tensor,
    compute_attention_entropy,
    compute_key_norm_variance,
    compute_topk_attention_mass,
)


@dataclass
class Phase1CollectionConfig:
    """Configuration for Phase 1 feature extraction.

    TODO(phase1): tune fields based on GPU memory budget and target dataset split.
    """

    dataset_name: str = "ruler"
    dataset_config: str = "4096"
    split: str = "train"
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    output_csv: str = "kvpress/evaluation/results/phase1_features.csv"
    max_examples: int = 128
    topk_fraction: float = 0.1
    device: str = "cuda:0"


def compute_placeholder_adakv_score(
    batch_size: int, num_heads: int, device: torch.device
) -> torch.Tensor:
    """Temporary scaffold for Ada-KV analytical score tensor.

    TODO(phase1): replace this with the exact Ada-KV L1 score extraction.
    Suggested integration point: reuse the same tensors used in `AdaKVPress.compress`.
    """
    return torch.zeros((batch_size, num_heads), device=device)


def collect_phase1_features(config: Phase1CollectionConfig) -> Path:
    """Collect features and emit a CSV artifact for correlation analysis.

    Returns:
        Path to written CSV file.
    """
    out_path = Path(config.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    feature_cfg = FeatureConfig(topk_fraction=config.topk_fraction)
    records: list[dict] = []

    # TODO(phase1): load dataset examples.
    # - Use existing KVPress evaluation dataset loaders if possible.
    # - Keep prompt IDs stable so you can join with metrics later.
    # - Reference: kvpress/evaluation/benchmarks/ruler/create_huggingface_dataset.py
    # --------------------------------------------------------------------------

    # TODO(phase1): load tokenizer/model exactly as your baseline pipeline does.
    # - Enable outputs needed to recover attentions from prefill.
    # - Confirm how to capture per-layer keys in the same forward pass.
    # --------------------------------------------------------------------------

    # TODO(phase1): iterate over examples and run prefill in inference mode.
    # - For each layer:
    #   1) extract attentions and keys
    #   2) compute 4 feature channels
    #   3) store one row per (example_id, layer_id, head_id)
    # - Also store Ada-KV analytical allocation target for correlation checks.
    # --------------------------------------------------------------------------
    _ = feature_cfg  # remove once collection loop is implemented

    if not records:
        # Keep the scaffold runnable while logic is unimplemented.
        empty_df = pd.DataFrame(
            columns=[
                "example_id",
                "layer_id",
                "head_id",
                "attention_entropy",
                "topk_attention_mass",
                "key_norm_variance",
                "adakv_l1_score",
            ]
        )
        empty_df.to_csv(out_path, index=False)
        return out_path

    df = pd.DataFrame.from_records(records)
    df.to_csv(out_path, index=False)
    return out_path


def dryrun_feature_ops() -> None:
    """Small local tensor check to validate feature tensor plumbing.

    TODO(phase1): expand with shape assertions from real model tensors.
    """
    bsz, heads, q_len, k_len, d = 2, 8, 16, 16, 128
    attentions = torch.softmax(torch.randn(bsz, heads, q_len, k_len), dim=-1)
    keys = torch.randn(bsz, heads, k_len, d)
    adakv = compute_placeholder_adakv_score(bsz, heads, attentions.device)

    ent = compute_attention_entropy(attentions)
    topk = compute_topk_attention_mass(attentions, topk_fraction=0.1)
    var = compute_key_norm_variance(keys)
    feat = build_feature_tensor(ent, topk, var, adakv)

    assert feat.shape == (bsz, heads, 4), f"Unexpected feature shape: {tuple(feat.shape)}"


if __name__ == "__main__":
    cfg = Phase1CollectionConfig()
    output = collect_phase1_features(cfg)
    print(f"[phase1] wrote feature scaffold output to: {output}")

