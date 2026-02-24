"""Phase 1 feature collection for learned budget allocation.

This module extracts per-layer/per-head features from a frozen causal LM prefill:
1) attention entropy
2) top-k attention mass
3) key norm variance
4) Ada-KV target proxy score (placeholder approximation)

Documentation links:
- Transformers model outputs:
  https://huggingface.co/docs/transformers/main_classes/output
- Datasets loading:
  https://huggingface.co/docs/datasets/loading
- Torch inference mode:
  https://pytorch.org/docs/stable/generated/torch.inference_mode.html
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    split: str = "test"
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    output_csv: str = "kvpress/evaluation/results/phase1_features.csv"
    max_examples: int = 128
    topk_fraction: float = 0.1
    max_context_tokens: int = 4096
    device: str = "cuda:0"
    trust_remote_code: bool = True


DATASET_REGISTRY = {
    "ruler": "simonjegou/ruler",
    "longbench": "Xnhyacinth/LongBench",
    "math500": "alessiodevoto/math500",
}


def resolve_hf_dataset_name(dataset_name: str) -> str:
    if dataset_name in DATASET_REGISTRY:
        return DATASET_REGISTRY[dataset_name]
    return dataset_name


def build_prompt_from_example(example: dict) -> str:
    """Build a prefill prompt from a benchmark row."""
    if "context" in example and "question" in example:
        return f"{example['context']}\n\n{example['question']}"
    if "prompt" in example:
        return example["prompt"]
    if "input" in example:
        return example["input"]
    if "text" in example:
        return example["text"]
    raise ValueError("Could not infer prompt field from dataset example")


def get_example_id(example: dict, fallback_index: int) -> str:
    if "id" in example:
        return str(example["id"])
    if "example_id" in example:
        return str(example["example_id"])
    return str(fallback_index)


def align_kv_to_attention_heads(kv_tensor: torch.Tensor, num_attention_heads: int) -> torch.Tensor:
    """Align `(B, H_kv)` tensor to `(B, H)` for GQA models.

    If H is a multiple of H_kv, each KV-head value is repeated across the grouped query heads.
    """
    bsz, num_kv_heads = kv_tensor.shape
    if num_kv_heads == num_attention_heads:
        return kv_tensor
    if num_attention_heads % num_kv_heads != 0:
        raise ValueError(
            f"Cannot align KV heads to attention heads: H={num_attention_heads}, H_kv={num_kv_heads}"
        )
    repeat_factor = num_attention_heads // num_kv_heads
    return kv_tensor.repeat_interleave(repeat_factor, dim=1).reshape(bsz, num_attention_heads)


def compute_adakv_target_proxy_from_attentions(
    attentions: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute a stable target proxy from attentions with shape `(B, H, Q, K)`.

    This is not the full Ada-KV analytical bound. It is a monotonic proxy:
    mean over query positions of max attention over key positions.
    Higher values indicate heads that concentrate strongly on salient keys.
    """
    # (B, H, Q, K) -> (B, H, Q) -> (B, H)
    return attentions.clamp_min(eps).amax(dim=-1).mean(dim=-1)


def load_prefill_model_and_tokenizer(config: Phase1CollectionConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=config.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": config.trust_remote_code,
    }
    target_device = config.device
    if target_device != "auto" and target_device.startswith("cuda") and not torch.cuda.is_available():
        target_device = "cpu"

    if target_device == "auto":
        model = AutoModelForCausalLM.from_pretrained(config.model_name, device_map="auto", **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs).to(target_device)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def collect_phase1_features(config: Phase1CollectionConfig) -> Path:
    """Collect features and emit a CSV artifact for correlation analysis.

    Returns:
        Path to written CSV file.
    """
    out_path = Path(config.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    feature_cfg = FeatureConfig(topk_fraction=config.topk_fraction)
    records: list[dict] = []

    dataset = load_dataset(
        resolve_hf_dataset_name(config.dataset_name),
        config.dataset_config,
        split=config.split,
    )
    model, tokenizer = load_prefill_model_and_tokenizer(config)

    n_examples = min(config.max_examples, len(dataset))
    for idx in range(n_examples):
        example = dataset[idx]
        prompt = build_prompt_from_example(example)
        example_id = get_example_id(example, idx)

        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=config.max_context_tokens,
        )
        if config.device != "auto":
            encoded = {k: v.to(config.device) for k, v in encoded.items()}

        outputs = model(
            **encoded,
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )
        attentions_per_layer = outputs.attentions
        past_key_values = outputs.past_key_values
        if attentions_per_layer is None or past_key_values is None:
            raise ValueError("Model outputs missing attentions or past_key_values")

        for layer_id, (layer_attn, layer_kv) in enumerate(zip(attentions_per_layer, past_key_values)):
            layer_keys = layer_kv[0]
            attention_entropy = compute_attention_entropy(layer_attn, eps=feature_cfg.eps)
            topk_attention_mass = compute_topk_attention_mass(layer_attn, topk_fraction=feature_cfg.topk_fraction)
            key_norm_variance_kv = compute_key_norm_variance(layer_keys, eps=feature_cfg.eps)
            key_norm_variance = align_kv_to_attention_heads(key_norm_variance_kv, attention_entropy.shape[1])
            adakv_proxy = compute_adakv_target_proxy_from_attentions(layer_attn, eps=feature_cfg.eps)

            feature_tensor = build_feature_tensor(
                attention_entropy=attention_entropy,
                topk_attention_mass=topk_attention_mass,
                key_norm_variance=key_norm_variance,
                adakv_l1_score=adakv_proxy,
            )

            # Store one row per (example_id, layer_id, head_id).
            # We use the first batch item because collection runs one prompt at a time.
            for head_id in range(feature_tensor.shape[1]):
                records.append(
                    {
                        "example_id": example_id,
                        "layer_id": int(layer_id),
                        "head_id": int(head_id),
                        "attention_entropy": float(feature_tensor[0, head_id, 0].item()),
                        "topk_attention_mass": float(feature_tensor[0, head_id, 1].item()),
                        "key_norm_variance": float(feature_tensor[0, head_id, 2].item()),
                        "adakv_l1_score": float(feature_tensor[0, head_id, 3].item()),
                    }
                )

    if not records:
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
    keys = torch.randn(bsz, 4, k_len, d)
    var = compute_key_norm_variance(keys)
    var = align_kv_to_attention_heads(var, heads)
    adakv = compute_adakv_target_proxy_from_attentions(attentions, eps=1e-8)

    ent = compute_attention_entropy(attentions)
    topk = compute_topk_attention_mass(attentions, topk_fraction=0.1)
    feat = build_feature_tensor(ent, topk, var, adakv)

    assert feat.shape == (bsz, heads, 4), f"Unexpected feature shape: {tuple(feat.shape)}"


def parse_args() -> Phase1CollectionConfig:
    parser = argparse.ArgumentParser(description="Collect Phase 1 per-head feature dataset.")
    parser.add_argument("--dataset_name", type=str, default="ruler")
    parser.add_argument("--dataset_config", type=str, default="4096")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_csv", type=str, default="kvpress/evaluation/results/phase1_features.csv")
    parser.add_argument("--max_examples", type=int, default=128)
    parser.add_argument("--topk_fraction", type=float, default=0.1)
    parser.add_argument("--max_context_tokens", type=int, default=4096)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no-trust_remote_code", dest="trust_remote_code", action="store_false")
    args = parser.parse_args()
    return Phase1CollectionConfig(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        model_name=args.model_name,
        output_csv=args.output_csv,
        max_examples=args.max_examples,
        topk_fraction=args.topk_fraction,
        max_context_tokens=args.max_context_tokens,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    cfg = parse_args()
    output = collect_phase1_features(cfg)
    print(f"[phase1] wrote feature output to: {output}")
