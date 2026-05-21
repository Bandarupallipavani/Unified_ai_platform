"""
pipeline/preprocess.py — AI-driven preprocessing pipeline
Steps: imputation, encoding, scaling, feature engineering, imbalance handling
"""
import os
import json
import logging
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)
STORAGE_PATH = os.getenv("STORAGE_PATH", "./data_sessions")

DOMAIN_RULES = {
    "healthcare": {"prioritize_recall": True, "check_protected": ["age", "race", "gender", "sex"]},
    "finance":    {"require_explainability": True, "check_protected": ["race", "gender", "age"]},
    "hr":         {"check_protected": ["gender", "age", "ethnicity", "race"]},
}


def build_preprocessing_plan(profile: dict, domain: str, mode: str) -> dict:
    """Build an AI-recommended preprocessing plan based on dataset profile."""
    columns = profile.get("columns", {})
    plan = {"steps": [], "feature_selection": True, "imbalance_strategy": None}
    imputation_steps = {}
    encoding_steps = {}
    scaling_steps = {}

    for col, info in columns.items():
        dtype = info.get("dtype", "object")
        null_rate = info.get("null_rate", 0)

        if null_rate > 0:
            if "float" in dtype or "int" in dtype:
                strategy = "knn" if null_rate < 0.3 and mode == "expert" else "median"
            else:
                strategy = "most_frequent"
            imputation_steps[col] = strategy

        unique_count = info.get("unique_count", 0)
        if "object" in dtype or "category" in dtype:
            if unique_count <= 10:
                encoding_steps[col] = "onehot"
            elif unique_count <= 50:
                encoding_steps[col] = "target"
            else:
                encoding_steps[col] = "label"

        if "float" in dtype or "int" in dtype:
            skewness = abs(info.get("skewness", 0) or 0)
            outlier_pct = info.get("outlier_pct_iqr", 0) or 0
            if outlier_pct > 10:
                scaling_steps[col] = "robust"
            elif skewness > 1.5:
                scaling_steps[col] = "minmax"
            else:
                scaling_steps[col] = "standard"

    plan["steps"].append({"type": "imputation", "config": imputation_steps})
    plan["steps"].append({"type": "encoding", "config": encoding_steps})
    plan["steps"].append({"type": "scaling", "config": scaling_steps})

    dr = DOMAIN_RULES.get(domain, {})
    protected = dr.get("check_protected", [])
    flagged = [c for c in protected if c in columns]
    if flagged:
        plan["protected_attribute_warning"] = (
            f"Columns {flagged} may be protected attributes. Fairness analysis will be applied."
        )

    target_info = profile.get("target_analysis", {})
    if target_info.get("imbalanced"):
        plan["imbalance_strategy"] = "smote"
        plan["steps"].append({"type": "imbalance", "strategy": "smote"})

    return plan


def apply_preprocessing(dataset_path: str, session_id: str, config: dict) -> list:
    """Apply the preprocessing plan to the dataset."""
    log = []
    save_dir = os.path.join(STORAGE_PATH, session_id)

    df = pd.read_parquet(dataset_path)
    log.append(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns.")

    target_col = config.get("target_column") or _detect_target(df)
    if target_col and target_col in df.columns:
        # Drop rows where target is null
        df = df.dropna(subset=[target_col])
        y = df[target_col]
        X = df.drop(columns=[target_col])
        log.append(f"Target column identified: '{target_col}'. Null rows in target dropped.")
    else:
        y = None
        X = df

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    impute_config = _get_step_config(config, "imputation")
    num_imputer = KNNImputer(n_neighbors=5) if "knn" in impute_config.values() else \
        SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    num_transformer = Pipeline([("imputer", num_imputer), ("scaler", StandardScaler())])
    cat_transformer = Pipeline([
        ("imputer", cat_imputer),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_transformer, numeric_cols),
        ("cat", cat_transformer, cat_cols),
    ], remainder="drop")

    X_processed = preprocessor.fit_transform(X)
    log.append(f"Imputed {len([c for c in X.columns if X[c].isnull().any()])} column(s) with nulls.")
    log.append(f"One-hot encoded {len(cat_cols)} categorical column(s).")
    log.append(f"Scaled {len(numeric_cols)} numeric column(s) with StandardScaler.")

    imbalance_strategy = _get_step_config(config, "imbalance", "strategy")
    if y is not None and imbalance_strategy == "smote":
        try:
            smote = SMOTE(random_state=42)
            X_processed, y_resampled = smote.fit_resample(X_processed, y)
            log.append(f"SMOTE applied: rebalanced to {len(y_resampled)} rows.")
            y = y_resampled
        except Exception as e:
            log.append(f"SMOTE skipped (reason: {e}).")

    feature_names = (
        numeric_cols +
        list(preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(cat_cols))
    )
    processed_df = pd.DataFrame(X_processed, columns=feature_names)
    if y is not None:
        processed_df["__target__"] = y.values if hasattr(y, "values") else y

    processed_path = os.path.join(save_dir, "processed.parquet")
    processed_df.to_parquet(processed_path, index=False)
    log.append(f"Preprocessed dataset saved: {processed_df.shape[0]} rows × {processed_df.shape[1]} columns.")

    preprocessor_path = os.path.join(save_dir, "preprocessor.pkl")
    with open(preprocessor_path, "wb") as f:
        pickle.dump(preprocessor, f)
    log.append("Preprocessor pipeline serialized for inference.")

    with open(os.path.join(save_dir, "preprocessing_log.json"), "w") as f:
        json.dump(log, f)

    return log


def _detect_target(df: pd.DataFrame) -> Optional[str]:
    for name in ["target", "label", "y", "class"]:
        if name in df.columns:
            return name
    return df.columns[-1]


def _get_step_config(config: dict, step_type: str, key: str = "config"):
    for step in config.get("steps", []):
        if step.get("type") == step_type:
            return step.get(key, {})
    return {}
