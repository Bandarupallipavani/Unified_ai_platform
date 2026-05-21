"""
pipeline/ingest.py — Data ingestion and profiling
Supports: CSV, JSON, Excel, Parquet, TSV, PostgreSQL, MySQL, SQLite, REST API
"""
import os
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from fastapi import UploadFile

from ai.nl_connector import interpret_connector

logger = logging.getLogger(__name__)

STORAGE_PATH = os.getenv("STORAGE_PATH", "./data_sessions")
os.makedirs(STORAGE_PATH, exist_ok=True)

SUPPORTED_EXTENSIONS = {".csv", ".json", ".xlsx", ".xls", ".parquet", ".tsv"}
MAX_FILE_SIZE_MB = 200


def _profile_dataframe(df: pd.DataFrame, dataset_path: str) -> dict:
    """Generate a rich statistical profile of a DataFrame."""
    profile = {
        "dataset_path": dataset_path,
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": {},
        "missing_values": {},
        "class_distribution": None,
        "data_quality_score": 0,
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
    }

    total_cells = len(df) * len(df.columns) if len(df.columns) > 0 else 1
    total_missing = 0

    for col in df.columns:
        series = df[col]
        null_count = int(series.isnull().sum())
        null_rate = round(null_count / len(df), 4) if len(df) > 0 else 0
        total_missing += null_count
        col_info = {
            "dtype": str(series.dtype),
            "null_count": null_count,
            "null_rate": null_rate,
            "unique_count": int(series.nunique()),
        }
        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            col_info.update({
                "mean": round(float(desc["mean"]), 4) if not np.isnan(desc["mean"]) else None,
                "std": round(float(desc["std"]), 4) if not np.isnan(desc.get("std", float("nan"))) else None,
                "min": round(float(desc["min"]), 4),
                "max": round(float(desc["max"]), 4),
                "skewness": round(float(series.skew()), 4),
            })
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
            col_info["top_values"] = series.value_counts().head(5).to_dict()

        profile["columns"][col] = col_info
        profile["missing_values"][col] = null_rate

    completeness = 1 - (total_missing / total_cells)
    profile["data_quality_score"] = round(completeness * 100, 1)
    return profile


async def ingest_file(file: UploadFile, session_id: str, content: bytes | None = None) -> dict:
    """Ingest an uploaded file and return its profile."""
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")

    if content is None:
        content = await file.read()
    size_mb = len(content) / 1e6
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"File too large ({size_mb:.1f} MB). Max {MAX_FILE_SIZE_MB} MB.")

    save_dir = os.path.join(STORAGE_PATH, session_id)
    os.makedirs(save_dir, exist_ok=True)
    raw_path = os.path.join(save_dir, f"raw{ext}")

    with open(raw_path, "wb") as f:
        f.write(content)

    df = _read_file(raw_path, ext)

    parquet_path = os.path.join(save_dir, "dataset.parquet")
    df.to_parquet(parquet_path, index=False)

    profile = _profile_dataframe(df, parquet_path)
    profile["original_filename"] = file.filename
    profile["file_size_mb"] = round(size_mb, 2)

    with open(os.path.join(save_dir, "profile.json"), "w") as f:
        json.dump(profile, f, default=str)

    logger.info(f"Ingested {file.filename}: {profile['shape']}")
    return profile


def _read_file(path: str, ext: str) -> pd.DataFrame:
    readers = {
        ".csv":     lambda p: pd.read_csv(p),
        ".tsv":     lambda p: pd.read_csv(p, sep="\t"),
        ".json":    lambda p: pd.read_json(p),
        ".xlsx":    lambda p: pd.read_excel(p),
        ".xls":     lambda p: pd.read_excel(p, engine="xlrd"),
        ".parquet": lambda p: pd.read_parquet(p),
    }
    return readers[ext](path)


async def ingest_nl_connector(description: str, session_id: str) -> dict:
    """Accept NL data source description and connect to it."""
    logger.info(f"NL Connector request: {description}")
    result = await interpret_connector(description, session_id)
    df = result["dataframe"]

    save_dir = os.path.join(STORAGE_PATH, session_id)
    os.makedirs(save_dir, exist_ok=True)
    parquet_path = os.path.join(save_dir, "dataset.parquet")
    df.to_parquet(parquet_path, index=False)

    profile = _profile_dataframe(df, parquet_path)
    profile["source_description"] = description
    profile["query_used"] = result.get("query")

    with open(os.path.join(save_dir, "profile.json"), "w") as f:
        json.dump(profile, f, default=str)

    return {"dataframe_path": parquet_path, "profile": profile, "query": result.get("query")}
