"""
pipeline/eda.py — Automated Exploratory Data Analysis
Produces: null analysis, distributions, correlations, target analysis, quality score
"""
import os
import json
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
STORAGE_PATH = os.getenv("STORAGE_PATH", "./data_sessions")


def run_eda(dataset_path: str, session_id: str, cached: bool = False) -> dict:
    """Run full automated EDA on a parquet dataset."""
    cache_path = os.path.join(STORAGE_PATH, session_id, "eda.json")

    if cached and os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    df = pd.read_parquet(dataset_path)
    result = {}

    # Null Analysis
    null_rates = (df.isnull().sum() / len(df)).round(4).to_dict()
    result["null_analysis"] = {
        "per_column": null_rates,
        "total_missing_pct": round(df.isnull().mean().mean() * 100, 2),
        "columns_with_nulls": [c for c, r in null_rates.items() if r > 0],
    }

    # Distributions
    dist = {}
    for col in df.select_dtypes(include="number").columns:
        s = df[col].dropna()
        if len(s) < 4:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        outlier_count = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        dist[col] = {
            "skewness": round(float(s.skew()), 3),
            "kurtosis": round(float(s.kurtosis()), 3),
            "outlier_count_iqr": outlier_count,
            "outlier_pct_iqr": round(outlier_count / len(s) * 100, 2),
        }
    result["distributions"] = dist

    # Correlations
    numeric_df = df.select_dtypes(include="number")
    high_corr_pairs = []
    if len(numeric_df.columns) >= 2:
        corr_matrix = numeric_df.corr(method="pearson")
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                val = corr_matrix.iloc[i, j]
                if abs(val) > 0.85:
                    high_corr_pairs.append({
                        "col1": corr_matrix.columns[i],
                        "col2": corr_matrix.columns[j],
                        "pearson_r": round(float(val), 4),
                    })
    result["correlations"] = {"high_correlation_pairs": high_corr_pairs, "high_corr_threshold": 0.85}
    result["target_leakage_candidates"] = [p for p in high_corr_pairs if p["pearson_r"] > 0.99]

    # Target analysis
    candidate_targets = ["target", "label", "y", "class", df.columns[-1]]
    target_col = next((c for c in candidate_targets if c in df.columns), None)
    if target_col:
        vc = df[target_col].value_counts(normalize=True).round(4).to_dict()
        min_class_pct = min(vc.values()) * 100
        result["target_analysis"] = {
            "column": target_col,
            "distribution": {str(k): v for k, v in vc.items()},
            "n_classes": len(vc),
            "imbalanced": min_class_pct < 15,
            "minority_class_pct": round(min_class_pct, 2),
        }

    result["ai_summary"] = _build_eda_summary(result)

    with open(cache_path, "w") as f:
        json.dump(result, f, default=str)

    return result


def _build_eda_summary(eda: dict) -> str:
    lines = []
    total_missing = eda["null_analysis"]["total_missing_pct"]
    if total_missing > 20:
        lines.append(f"⚠️  High missing data rate ({total_missing:.1f}%). Imputation strategy critical.")
    elif total_missing > 0:
        lines.append(f"Missing data detected ({total_missing:.1f}% overall).")
    else:
        lines.append("✅ No missing values found.")

    hcp = eda["correlations"]["high_correlation_pairs"]
    if hcp:
        lines.append(f"⚠️  {len(hcp)} highly correlated feature pair(s) found.")
    ta = eda.get("target_analysis")
    if ta and ta.get("imbalanced"):
        lines.append(f"⚠️  Class imbalance detected: minority class {ta['minority_class_pct']}%.")
    return " ".join(lines) if lines else "Dataset looks clean and ready for preprocessing."
