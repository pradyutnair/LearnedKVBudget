"""Phase 1 scaffolding: feature informativeness checks.

Goal from PRD:
- Verify extracted features are informative by correlating them with Ada-KV
  analytical budgets/scores before moving to allocator training.

Documentation links:
- pandas groupby: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html
- scipy spearmanr: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


@dataclass
class Phase1AnalysisConfig:
    """Config for Phase 1 correlation analysis."""

    feature_csv: str = "kvpress/evaluation/results/phase1_features.csv"
    output_csv: str = "kvpress/evaluation/results/phase1_feature_correlations.csv"


FEATURE_COLUMNS = [
    "attention_entropy",
    "topk_attention_mass",
    "key_norm_variance",
]

TARGET_COLUMN = "adakv_l1_score"


def compute_layerwise_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-layer Spearman correlation between each feature and target.

    TODO(phase1): optionally add per-head and global correlations.
    TODO(phase1): add confidence intervals via bootstrap if needed for reporting.
    """
    required_cols = {"layer_id", TARGET_COLUMN, *FEATURE_COLUMNS}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows: list[dict] = []
    for layer_id, subdf in df.groupby("layer_id", sort=True):
        for feat_col in FEATURE_COLUMNS:
            corr, p_value = spearmanr(subdf[feat_col], subdf[TARGET_COLUMN], nan_policy="omit")
            rows.append(
                {
                    "layer_id": int(layer_id),
                    "feature": feat_col,
                    "spearman_corr": float(corr) if corr is not None else float("nan"),
                    "p_value": float(p_value) if p_value is not None else float("nan"),
                    "n": int(len(subdf)),
                }
            )
    return pd.DataFrame(rows)


def run_phase1_analysis(config: Phase1AnalysisConfig) -> Path:
    """Run scaffolded Phase 1 analysis and write CSV summary."""
    in_path = Path(config.feature_csv)
    out_path = Path(config.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # TODO(phase1): if your extraction also logs allocation vectors, add additional
    # analyses here:
    # - Jensen-Shannon divergence vs uniform allocations
    # - correlation with actual budget-per-head from Ada-KV pipeline
    # --------------------------------------------------------------------------
    print(f"[phase1] reading feature CSV from: {in_path}")
    df = pd.read_csv(in_path)
    summary = compute_layerwise_correlations(df)
    print(f"[phase1] writing correlation summary to: {out_path}")
    summary.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    cfg = Phase1AnalysisConfig()
    output = run_phase1_analysis(cfg)
    print(f"[phase1] wrote correlation summary to: {output}")

