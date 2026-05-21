"""
pipeline/drift.py — Data Drift Monitoring
Supports PSI (Population Stability Index) for numeric/categorical features
and KS test as a secondary method.
"""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats

DRIFT_THRESHOLD_PSI = float(os.getenv("DRIFT_THRESHOLD_PSI", "0.1"))
DRIFT_THRESHOLD_KS  = float(os.getenv("DRIFT_THRESHOLD_KS",  "0.05"))


def _psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index. >0.2 = significant drift."""
    eps = 1e-6
    expected_perc = np.histogram(expected, bins=buckets)[0] / len(expected) + eps
    actual_perc   = np.histogram(actual,   bins=np.histogram(expected, bins=buckets)[1])[0] / len(actual) + eps
    return float(np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc)))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (0 = identical, 1 = completely different)."""
    eps = 1e-10
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / (m + eps) + eps)) +
                 0.5 * np.sum(q * np.log(q / (m + eps) + eps)))


def compute_drift(training_dataset_path: str,
                  sample_data: list[dict],
                  model_id: str) -> dict:
    """
    Compare sample_data (recent prediction inputs) against training distribution.

    Returns per-feature dict:
    {
      "feature_name": {
        "method": "psi" | "ks" | "categorical",
        "score": float,
        "threshold": float,
        "drifted": bool,
        "training_mean": float | None,
        "sample_mean": float | None,
        "training_top_values": list | None,
        "sample_top_values": list | None,
      }
    }
    """
    if not os.path.exists(training_dataset_path):
        return {"error": "Training dataset not found for drift comparison."}

    ext = os.path.splitext(training_dataset_path)[1].lower()
    try:
        if ext == ".parquet":
            train_df = pd.read_parquet(training_dataset_path)
        elif ext in (".xls", ".xlsx"):
            train_df = pd.read_excel(training_dataset_path)
        else:
            train_df = pd.read_csv(training_dataset_path)
    except Exception as e:
        return {"error": f"Could not load training data: {e}"}

    if not sample_data:
        return {"error": "sample_data is empty."}

    sample_df = pd.DataFrame(sample_data)

    # Only check features present in both
    common_cols = [c for c in sample_df.columns if c in train_df.columns]
    if not common_cols:
        return {"error": "No common columns between sample and training data."}

    results = {}

    for col in common_cols:
        train_series  = train_df[col].dropna()
        sample_series = sample_df[col].dropna()

        if len(sample_series) < 5:
            results[col] = {"method": "skipped", "reason": "too few samples", "drifted": False}
            continue

        # Numeric column → PSI + KS
        if pd.api.types.is_numeric_dtype(train_series):
            psi_score = _psi(train_series.values, sample_series.values)
            ks_stat, ks_p = stats.ks_2samp(train_series.values, sample_series.values)

            drifted = psi_score > DRIFT_THRESHOLD_PSI or ks_p < DRIFT_THRESHOLD_KS
            results[col] = {
                "method": "psi+ks",
                "score": round(psi_score, 4),
                "ks_stat": round(float(ks_stat), 4),
                "ks_pvalue": round(float(ks_p), 4),
                "threshold": DRIFT_THRESHOLD_PSI,
                "drifted": drifted,
                "training_mean": round(float(train_series.mean()), 4),
                "sample_mean": round(float(sample_series.mean()), 4),
                "training_std": round(float(train_series.std()), 4),
                "sample_std": round(float(sample_series.std()), 4),
            }

        # Categorical column → JS divergence
        else:
            all_cats = set(train_series.unique()) | set(sample_series.unique())
            train_dist = np.array([
                (train_series == c).sum() for c in all_cats], dtype=float) + 1e-10
            sample_dist = np.array([
                (sample_series == c).sum() for c in all_cats], dtype=float) + 1e-10

            js_score = _js_divergence(train_dist, sample_dist)
            drifted  = js_score > 0.15

            results[col] = {
                "method": "js_divergence",
                "score": round(js_score, 4),
                "threshold": 0.15,
                "drifted": drifted,
                "training_top_values": train_series.value_counts().head(5).to_dict(),
                "sample_top_values":   sample_series.value_counts().head(5).to_dict(),
            }

    return results
