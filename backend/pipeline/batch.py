"""
pipeline/batch.py — Async batch prediction
Accepts CSV/Parquet, runs predictions in chunks, writes output CSV.
"""
import os
import logging
from datetime import datetime

import pandas as pd

from pipeline.deploy import run_prediction

logger = logging.getLogger(__name__)

BATCH_CHUNK_SIZE = int(os.getenv("BATCH_CHUNK_SIZE", "500"))
STORAGE_PATH     = os.getenv("STORAGE_PATH", "./data_sessions")


def run_batch_prediction(batch_id: str, model, input_path: str, ext: str, db):
    """
    Called as a BackgroundTask from main.py.
    Reads input file in chunks, runs model inference, writes results CSV.
    Updates BatchPrediction row throughout.
    """
    from models_db import BatchPrediction

    bp = db.query(BatchPrediction).filter(BatchPrediction.id == batch_id).first()
    if not bp:
        logger.error(f"BatchPrediction {batch_id} not found in DB.")
        return

    try:
        bp.status = "running"
        db.commit()

        # Load input
        if ext == ".parquet":
            df = pd.read_parquet(input_path)
        elif ext in (".tsv",):
            df = pd.read_csv(input_path, sep="\t")
        else:
            df = pd.read_csv(input_path)

        total_rows = len(df)
        bp.row_count = total_rows
        db.commit()

        all_preds = []
        framework = model.framework or "sklearn"

        # Process in chunks to avoid OOM on large files
        for start in range(0, total_rows, BATCH_CHUNK_SIZE):
            chunk = df.iloc[start : start + BATCH_CHUNK_SIZE]
            rows  = chunk.to_dict(orient="records")
            preds = run_prediction(
                model.artifact_path,
                rows,
                framework=framework,
                training_dataset_path=model.job.session.dataset_path if model.job and model.job.session else None,
                target_column=((model.job.config_json or {}).get("target_column") or None) if model.job else None,
            )
            all_preds.extend(preds)
            logger.info(f"Batch {batch_id}: processed {min(start + BATCH_CHUNK_SIZE, total_rows)}/{total_rows} rows")

        # Write output
        out_dir = os.path.join(STORAGE_PATH, "batch", batch_id)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "predictions.csv")

        df_out = df.copy()
        df_out["prediction"] = all_preds
        df_out.to_csv(out_path, index=False)

        bp.output_path  = out_path
        bp.status       = "completed"
        bp.completed_at = datetime.utcnow()
        db.commit()
        logger.info(f"Batch {batch_id} completed — {total_rows} rows → {out_path}")

    except Exception as e:
        logger.exception(f"Batch {batch_id} failed: {e}")
        if bp:
            bp.status = "failed"
            db.commit()
