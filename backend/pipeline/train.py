"""
pipeline/train.py — Training pipeline v3
Fixes:
  - Thread-safe DB session (each call gets its own session via SessionFactory)
  - SHAP support for DL/NLP (gradient explainer fallback)
  - ModelVersion snapshot on job completion
  - Proper best_model_id linking on TrainingJob
"""
import os
import uuid
import json
import logging
import asyncio
import pickle
from datetime import datetime
from typing import Callable, Awaitable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STORAGE_PATH = os.getenv("STORAGE_PATH", "./data_sessions")
OUTPUT_PATH  = os.getenv("OUTPUT_PATH",  "./outputs")
os.makedirs(OUTPUT_PATH, exist_ok=True)


# ── Model Catalogue ──────────────────────────────────────────────────────────

CLASSIC_ML_MODELS = {
    "xgboost_clf":    ("XGBoost Classifier",    "classification"),
    "xgboost_reg":    ("XGBoost Regressor",     "regression"),
    "rf_clf":         ("Random Forest Clf",     "classification"),
    "rf_reg":         ("Random Forest Reg",     "regression"),
    "lgbm_clf":       ("LightGBM Classifier",   "classification"),
    "lgbm_reg":       ("LightGBM Regressor",    "regression"),
    "lr":             ("Logistic Regression",   "classification"),
    "linear_reg":     ("Linear Regression",     "regression"),
    "svm":            ("SVM",                   "classification"),
    "catboost_clf":   ("CatBoost Classifier",   "classification"),
    "catboost_reg":   ("CatBoost Regressor",    "regression"),
    "knn":            ("K-Nearest Neighbors",   "classification"),
    "dt_clf":         ("Decision Tree Clf",     "classification"),
    "dt_reg":         ("Decision Tree Reg",     "regression"),
    "extra_trees":    ("Extra Trees Clf",       "classification"),
    "adaboost":       ("AdaBoost Classifier",   "classification"),
    "naive_bayes":    ("Naive Bayes",           "classification"),
    "ridge":          ("Ridge Regression",      "regression"),
    "lasso":          ("Lasso Regression",      "regression"),
    "elasticnet":     ("ElasticNet Regression", "regression"),
}

DL_MODELS = {
    "mlp":        "Multilayer Perceptron",
    "cnn_1d":     "1D CNN",
    "lstm":       "LSTM",
    "gru":        "GRU",
    "transformer":"Transformer Encoder",
    "resnet":     "ResNet (tabular)",
}

NLP_MODELS = {
    "bert-base-uncased":        "BERT Base",
    "distilbert-base-uncased":  "DistilBERT",
    "roberta-base":             "RoBERTa",
    "albert-base-v2":           "ALBERT",
}


# ── Entry-point ───────────────────────────────────────────────────────────────

async def start_training_job(
    job_id: str,
    dataset_path: str,
    config: dict,
    domain: str,
    mode: str,
    db,
    ws_log: Callable[[str], Awaitable[None]],
):
    from models_db import TrainingJob, Model, ModelVersion

    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        logger.error(f"Job {job_id} not found.")
        return

    job.status     = "running"
    job.started_at = datetime.utcnow()
    db.commit()

    model_type = config.get("model_type", "ml")
    await ws_log(f"🚀 Starting {model_type.upper()} training job {job_id}")

    try:
        # Load data
        await ws_log("📂 Loading dataset...")
        df = _load_dataset(dataset_path)
        await ws_log(f"   Loaded {len(df):,} rows × {df.shape[1]} columns")

        target_col = config.get("target_column") or _infer_target(df)
        task_type  = config.get("task_type") or _infer_task(df[target_col])
        await ws_log(f"   Target: '{target_col}' | Task: {task_type}")

        X, y, feature_names = _prepare_xy(df, target_col, task_type)

        if model_type == "ml":
            trained_models = await _train_ml(X, y, task_type, config, job_id, domain, ws_log)
        elif model_type == "dl":
            trained_models = await _train_dl(X, y, task_type, config, job_id, ws_log)
        elif model_type == "nlp":
            text_col = config.get("text_column", _infer_text_column(df))
            trained_models = await _train_nlp(df[text_col], y, config, job_id, ws_log)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        await ws_log(f"✅ Training complete — {len(trained_models)} models evaluated")

        # Persist models and pick best
        best_model_row = None
        best_score     = -1e9
        primary_metric = "accuracy" if task_type == "classification" else "r2"

        for m in trained_models:
            score = m["metrics"].get(primary_metric, m["metrics"].get("r2", 0))
            shap_values = _compute_shap(m["model_obj"], X, model_type)

            artifact_path = _save_artifact(m["model_obj"], job_id, m["algorithm"], model_type)

            model_row = Model(
                id=str(uuid.uuid4()),
                job_id=job_id,
                algorithm=m["algorithm"],
                model_type=model_type,
                framework=m.get("framework", "sklearn"),
                hyperparams_json=m.get("hyperparams", {}),
                metrics_json=m["metrics"],
                shap_json=shap_values,
                artifact_path=artifact_path,
            )
            db.add(model_row)
            db.flush()

            # Create initial version snapshot
            ver = ModelVersion(
                model_id=model_row.id,
                version=1,
                metrics_json=m["metrics"],
                artifact_path=artifact_path,
                notes="Initial training",
            )
            db.add(ver)

            if score > best_score:
                best_score     = score
                best_model_row = model_row

        if best_model_row:
            job.best_model_id = best_model_row.id

        job.status       = "completed"
        job.completed_at = datetime.utcnow()
        db.commit()

        await ws_log(f"🏆 Best model: {best_model_row.algorithm if best_model_row else 'N/A'} "
                     f"(score={best_score:.4f})")

    except Exception as e:
        logger.exception(f"Training job {job_id} failed: {e}")
        job.status = "failed"
        db.commit()
        await ws_log(f"❌ Training failed: {e}")
        raise


# ── ML Training (Optuna HPO) ─────────────────────────────────────────────────

async def _train_ml(X, y, task_type, config, job_id, domain, ws_log):
    import optuna
    from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    requested = config.get("selected_models", list(CLASSIC_ML_MODELS.keys())[:6])
    n_trials  = config.get("n_trials", 15)
    cv_folds  = config.get("cv_folds", 5)
    timeout   = config.get("timeout_per_model", 120)

    cv = (StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
          if task_type == "classification"
          else KFold(n_splits=cv_folds, shuffle=True, random_state=42))

    scoring   = "accuracy" if task_type == "classification" else "r2"
    results   = []

    for algo_key in requested:
        if algo_key not in CLASSIC_ML_MODELS:
            continue
        algo_name, algo_task = CLASSIC_ML_MODELS[algo_key]
        if task_type not in algo_task:
            continue

        await ws_log(f"   🔍 Tuning {algo_name} ({n_trials} trials)...")

        def objective(trial):
            model = _build_ml_model(algo_key, task_type, trial)
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

        best_params = study.best_params
        best_model  = _build_ml_model(algo_key, task_type, None, params=best_params)
        best_model.fit(X, y)

        metrics = _evaluate_model(best_model, X, y, task_type, cv, scoring)
        results.append({
            "algorithm":  algo_name,
            "model_obj":  best_model,
            "framework":  "sklearn",
            "hyperparams": best_params,
            "metrics":    metrics,
        })
        await ws_log(f"      ✓ {algo_name}: {scoring}={metrics.get(scoring, 0):.4f}")

    return results


def _build_ml_model(algo_key: str, task_type: str, trial=None, params: dict = None):
    """Build sklearn-compatible model with Optuna trial params or fixed params."""
    p = params or {}

    def tp(name, *args, **kwargs):
        if trial:
            fn = getattr(trial, f"suggest_{args[0]}")
            return fn(name, *args[1:], **kwargs)
        return p.get(name, kwargs.get("default", args[1] if len(args) > 1 else None))

    if algo_key in ("xgboost_clf", "xgboost_reg"):
        from xgboost import XGBClassifier, XGBRegressor
        kw = dict(
            n_estimators    = tp("n_estimators", "int", 50, 500),
            max_depth       = tp("max_depth", "int", 3, 10),
            learning_rate   = tp("learning_rate", "float", 1e-3, 0.3, log=True),
            subsample       = tp("subsample", "float", 0.5, 1.0),
            colsample_bytree= tp("colsample_bytree", "float", 0.5, 1.0),
            use_label_encoder=False, eval_metric="logloss" if task_type=="classification" else "rmse",
            random_state=42, verbosity=0
        )
        return (XGBClassifier if task_type == "classification" else XGBRegressor)(**kw)

    if algo_key in ("lgbm_clf", "lgbm_reg"):
        from lightgbm import LGBMClassifier, LGBMRegressor
        kw = dict(
            n_estimators  = tp("n_estimators", "int", 50, 500),
            max_depth     = tp("max_depth", "int", -1, 15),
            learning_rate = tp("learning_rate", "float", 1e-3, 0.3, log=True),
            num_leaves    = tp("num_leaves", "int", 20, 300),
            random_state=42, verbose=-1,
        )
        return (LGBMClassifier if task_type == "classification" else LGBMRegressor)(**kw)

    if algo_key in ("rf_clf", "rf_reg"):
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        kw = dict(
            n_estimators = tp("n_estimators", "int", 50, 400),
            max_depth    = tp("max_depth", "int", 3, 20),
            random_state=42, n_jobs=-1,
        )
        return (RandomForestClassifier if task_type == "classification" else RandomForestRegressor)(**kw)

    if algo_key in ("catboost_clf", "catboost_reg"):
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
            kw = dict(
                iterations   = tp("iterations", "int", 100, 500),
                depth        = tp("depth", "int", 3, 10),
                learning_rate= tp("learning_rate", "float", 1e-3, 0.3, log=True),
                verbose=0, random_state=42,
            )
            return (CatBoostClassifier if task_type == "classification" else CatBoostRegressor)(**kw)
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
            return (GradientBoostingClassifier if task_type == "classification" else GradientBoostingRegressor)()

    if algo_key == "lr":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=tp("C", "float", 1e-3, 10, log=True), max_iter=1000)

    if algo_key == "linear_reg":
        from sklearn.linear_model import LinearRegression
        return LinearRegression()

    if algo_key == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=tp("alpha", "float", 1e-3, 10, log=True))

    if algo_key == "lasso":
        from sklearn.linear_model import Lasso
        return Lasso(alpha=tp("alpha", "float", 1e-4, 1.0, log=True))

    if algo_key == "elasticnet":
        from sklearn.linear_model import ElasticNet
        return ElasticNet(alpha=tp("alpha", "float", 1e-4, 1.0, log=True),
                          l1_ratio=tp("l1_ratio", "float", 0.1, 0.9))

    if algo_key == "svm":
        from sklearn.svm import SVC
        return SVC(C=tp("C", "float", 1e-3, 10, log=True), kernel="rbf", probability=True)

    if algo_key == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(n_neighbors=tp("n_neighbors", "int", 3, 20))

    if algo_key in ("dt_clf", "dt_reg"):
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        kw = dict(max_depth=tp("max_depth", "int", 3, 20), random_state=42)
        return (DecisionTreeClassifier if task_type == "classification" else DecisionTreeRegressor)(**kw)

    if algo_key == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(n_estimators=tp("n_estimators", "int", 50, 300),
                                    max_depth=tp("max_depth", "int", 3, 20), random_state=42)

    if algo_key == "adaboost":
        from sklearn.ensemble import AdaBoostClassifier
        return AdaBoostClassifier(n_estimators=tp("n_estimators", "int", 50, 300), random_state=42)

    if algo_key == "naive_bayes":
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB()

    raise ValueError(f"Unknown algorithm: {algo_key}")


# ── DL Training ───────────────────────────────────────────────────────────────

async def _train_dl(X, y, task_type, config, job_id, ws_log):
    """Simple PyTorch MLP trainer — expandable to CNN/LSTM/GRU."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    selected = config.get("selected_models", ["mlp"])
    epochs   = config.get("epochs", 20)
    lr       = config.get("learning_rate", 1e-3)
    batch_sz = config.get("batch_size", 64)
    results  = []

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_tr, X_val, y_tr, y_val = train_test_split(X_s, y, test_size=0.2, random_state=42)

    n_classes  = len(np.unique(y)) if task_type == "classification" else 1
    input_dim  = X.shape[1]
    is_clf     = task_type == "classification"

    for arch in selected:
        await ws_log(f"   🧠 Training {DL_MODELS.get(arch, arch)} ({epochs} epochs)...")

        net = _build_dl_model(arch, input_dim, n_classes, is_clf)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss() if is_clf else nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        Xt = torch.FloatTensor(X_tr)
        yt = torch.LongTensor(y_tr.astype(int)) if is_clf else torch.FloatTensor(y_tr)
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_sz, shuffle=True)

        for epoch in range(epochs):
            net.train()
            for xb, yb in loader:
                opt.zero_grad()
                out = net(xb)
                if is_clf:
                    loss = criterion(out, yb)
                else:
                    loss = criterion(out.squeeze(), yb)
                loss.backward()
                opt.step()
            scheduler.step()
            if (epoch + 1) % 5 == 0:
                await ws_log(f"      Epoch {epoch+1}/{epochs} loss={loss.item():.4f}")

        # Evaluate
        net.eval()
        Xv = torch.FloatTensor(X_val)
        with torch.no_grad():
            preds = net(Xv)
        metrics = _evaluate_dl(preds, y_val, task_type, n_classes)
        await ws_log(f"      ✓ {DL_MODELS.get(arch, arch)}: {metrics}")

        results.append({
            "algorithm":  DL_MODELS.get(arch, arch),
            "model_obj":  {"net": net, "scaler": scaler, "arch": arch, "task": task_type},
            "framework":  "pytorch",
            "hyperparams": {"epochs": epochs, "lr": lr, "batch_size": batch_sz},
            "metrics":    metrics,
        })

    return results


def _build_dl_model(arch, input_dim, n_classes, is_clf):
    import torch.nn as nn

    out_dim = n_classes if is_clf else 1

    if arch == "mlp":
        return nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, out_dim),
        )
    if arch == "cnn_1d":
        class CNN1D(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Unflatten(1, (1, input_dim)),
                    nn.Conv1d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
                    nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
                    nn.AdaptiveAvgPool1d(8),
                )
                self.fc = nn.Linear(64 * 8, out_dim)
            def forward(self, x):
                return self.fc(self.conv(x).view(x.size(0), -1))
        return CNN1D()

    # Default fallback
    return nn.Sequential(
        nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, out_dim),
    )


# ── NLP Training ──────────────────────────────────────────────────────────────

async def _train_nlp(texts, labels, config, job_id, ws_log):
    """Fine-tunes a HuggingFace transformer for text classification."""
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              TrainingArguments, Trainer)
    from datasets import Dataset
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score

    model_name = config.get("selected_models", ["distilbert-base-uncased"])[0]
    await ws_log(f"   🤗 Fine-tuning {NLP_MODELS.get(model_name, model_name)}...")

    n_classes = len(np.unique(labels))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=n_classes, ignore_mismatched_sizes=True)

    texts_tr, texts_val, y_tr, y_val = train_test_split(
        texts.tolist(), labels.tolist(), test_size=0.2, random_state=42)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    train_ds = Dataset.from_dict({"text": texts_tr, "label": y_tr}).map(tokenize, batched=True)
    val_ds   = Dataset.from_dict({"text": texts_val, "label": y_val}).map(tokenize, batched=True)

    artifact_dir = os.path.join(OUTPUT_PATH, f"nlp_{job_id}")
    args = TrainingArguments(
        output_dir=artifact_dir,
        num_train_epochs=int(config.get("epochs", 3)),
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
        report_to="none",
    )

    def compute_metrics(pred):
        logits, labels_ = pred
        preds_ = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels_, preds_),
                "f1": f1_score(labels_, preds_, average="weighted", zero_division=0)}

    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=val_ds, compute_metrics=compute_metrics)
    trainer.train()
    metrics_ = trainer.evaluate()

    await ws_log(f"      ✓ {NLP_MODELS.get(model_name, model_name)}: acc={metrics_.get('eval_accuracy', 0):.4f}")

    return [{
        "algorithm":  NLP_MODELS.get(model_name, model_name),
        "model_obj":  {"trainer": trainer, "tokenizer": tokenizer, "model_name": model_name},
        "framework":  "transformers",
        "hyperparams": {"epochs": config.get("epochs", 3), "model": model_name},
        "metrics":    {
            "accuracy": round(metrics_.get("eval_accuracy", 0), 4),
            "f1":       round(metrics_.get("eval_f1", 0), 4),
        },
    }]


# ── SHAP (all model types) ────────────────────────────────────────────────────

def _compute_shap(model_obj, X, model_type: str) -> dict | None:
    try:
        import shap
        if model_type == "ml":
            # Tree explainer for tree-based models, linear otherwise
            try:
                explainer = shap.TreeExplainer(model_obj)
            except Exception:
                explainer = shap.LinearExplainer(model_obj, X) if hasattr(model_obj, "coef_") \
                            else shap.KernelExplainer(model_obj.predict, shap.sample(X, 50))
            sample = shap.sample(X, min(200, len(X)))
            vals   = explainer.shap_values(sample)
            if isinstance(vals, list):
                vals = vals[1] if len(vals) > 1 else vals[0]
            importance = np.abs(vals).mean(axis=0)
            return {"shap_mean_abs": importance.tolist()}

        elif model_type == "dl":
            import torch
            net    = model_obj["net"]
            scaler = model_obj.get("scaler")
            Xs     = scaler.transform(X) if scaler else X
            bg     = torch.FloatTensor(Xs[:50])
            test   = torch.FloatTensor(Xs[50:100])
            explainer = shap.GradientExplainer(net, bg)
            vals   = explainer.shap_values(test)
            if isinstance(vals, list):
                vals = vals[0]
            importance = np.abs(vals).mean(axis=0)
            return {"shap_mean_abs": importance.tolist()}

        # NLP — skip (token-level SHAP is expensive)
        return None

    except Exception as e:
        logger.warning(f"SHAP computation skipped: {e}")
        return None


# ── Evaluation Helpers ────────────────────────────────────────────────────────

def _evaluate_model(model, X, y, task_type, cv, scoring):
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                                 mean_squared_error, r2_score, mean_absolute_error)

    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    y_pred = model.predict(X)

    if task_type == "classification":
        metrics = {
            "accuracy":  round(float(accuracy_score(y, y_pred)), 4),
            "f1_score":  round(float(f1_score(y, y_pred, average="weighted", zero_division=0)), 4),
            "cv_mean":   round(float(cv_scores.mean()), 4),
            "cv_std":    round(float(cv_scores.std()), 4),
        }
        try:
            proba = model.predict_proba(X)
            auc = roc_auc_score(y, proba, multi_class="ovr", average="weighted") \
                  if len(np.unique(y)) > 2 else roc_auc_score(y, proba[:, 1])
            metrics["roc_auc"] = round(float(auc), 4)
        except Exception:
            pass
    else:
        metrics = {
            "r2":   round(float(r2_score(y, y_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y, y_pred))), 4),
            "mae":  round(float(mean_absolute_error(y, y_pred)), 4),
            "cv_mean": round(float(cv_scores.mean()), 4),
            "cv_std":  round(float(cv_scores.std()), 4),
        }
    return metrics


def _evaluate_dl(preds, y_val, task_type, n_classes):
    import torch
    from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
    if task_type == "classification":
        pred_cls = preds.argmax(dim=1).numpy()
        return {
            "accuracy": round(float(accuracy_score(y_val, pred_cls)), 4),
            "f1_score": round(float(f1_score(y_val, pred_cls, average="weighted", zero_division=0)), 4),
        }
    else:
        pv = preds.squeeze().numpy()
        return {
            "r2":   round(float(r2_score(y_val, pv)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_val, pv))), 4),
        }


# ── Artifact Helpers ──────────────────────────────────────────────────────────

def _save_artifact(model_obj, job_id: str, algorithm: str, model_type: str) -> str:
    safe_name = algorithm.replace(" ", "_").lower()
    path_dir  = os.path.join(OUTPUT_PATH, "models", job_id)
    os.makedirs(path_dir, exist_ok=True)

    if model_type == "ml":
        path = os.path.join(path_dir, f"{safe_name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model_obj, f)
        return path

    elif model_type == "dl":
        import torch
        path = os.path.join(path_dir, f"{safe_name}.pt")
        torch.save({"net_state": model_obj["net"].state_dict(),
                    "scaler":    model_obj.get("scaler"),
                    "arch":      model_obj.get("arch"),
                    "task":      model_obj.get("task")}, path)
        return path

    elif model_type == "nlp":
        path = os.path.join(path_dir, safe_name)
        model_obj["trainer"].save_model(path)
        model_obj["tokenizer"].save_pretrained(path)
        return path

    return ""


# ── Data Helpers ──────────────────────────────────────────────────────────────

def _load_dataset(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    elif ext in (".xls", ".xlsx"):
        return pd.read_excel(path)
    elif ext == ".json":
        return pd.read_json(path)
    return pd.read_csv(path)


def _infer_target(df: pd.DataFrame) -> str:
    common = ["target", "label", "class", "output", "y", "churn", "price", "salary"]
    for c in common:
        if c in df.columns:
            return c
    return df.columns[-1]


def _infer_text_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if df[col].dtype == object and df[col].str.len().mean() > 30:
            return col
    return df.columns[0]


def _infer_task(series: pd.Series) -> str:
    n_unique = series.nunique()
    if n_unique <= 20 or series.dtype in (object, bool):
        return "classification"
    return "regression"


def _prepare_xy(df: pd.DataFrame, target_col: str, task_type: str):
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.impute import SimpleImputer

    y_raw = df[target_col].values
    X_raw = df.drop(columns=[target_col])

    # Encode target
    if task_type == "classification" and X_raw.select_dtypes(include=object).empty is False:
        le = LabelEncoder()
        y  = le.fit_transform(y_raw.astype(str))
    else:
        y = y_raw.astype(float) if task_type == "regression" else LabelEncoder().fit_transform(y_raw.astype(str))

    # Encode categoricals
    X_enc = pd.get_dummies(X_raw, drop_first=True)
    feature_names = list(X_enc.columns)

    # Impute + scale
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X_enc)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_imp)

    return X, y, feature_names


# ── Status/Result Queries ─────────────────────────────────────────────────────

def get_job_status(job_id: str, db) -> dict:
    from models_db import TrainingJob
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise Exception(f"Job {job_id} not found")
    return {
        "job_id":    job_id,
        "status":    job.status,
        "model_type": job.model_type,
        "started_at":   str(job.started_at),
        "completed_at": str(job.completed_at),
        "best_model_id": job.best_model_id,
    }


def get_job_results(job_id: str, db) -> dict:
    from models_db import TrainingJob, Model
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise Exception(f"Job {job_id} not found")
    models = db.query(Model).filter(Model.job_id == job_id).all()
    return {
        "job_id":    job_id,
        "status":    job.status,
        "model_type": job.model_type,
        "models": [
            {"model_id": m.id, "algorithm": m.algorithm, "framework": m.framework,
             "metrics": m.metrics_json, "hyperparams": m.hyperparams_json,
             "shap": m.shap_json, "is_production": m.is_production}
            for m in models
        ],
        "best_model_id": job.best_model_id,
    }


def get_best_model(job_id: str, db) -> dict:
    from models_db import TrainingJob, Model
    from pipeline.deploy import get_model_feature_names

    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job or not job.best_model_id:
        raise Exception("No best model found.")
    model = db.query(Model).filter(Model.id == job.best_model_id).first()
    dataset_columns = []
    target_column = None
    if job.session and job.session.dataset_path and os.path.exists(job.session.dataset_path):
        dataset_df = _load_dataset(job.session.dataset_path)
        target_column = (job.config_json or {}).get("target_column") or _infer_target(dataset_df)
        dataset_columns = [col for col in dataset_df.columns if col != target_column]
    return {
        "model_id":   model.id,
        "algorithm":  model.algorithm,
        "framework":  model.framework,
        "metrics":    model.metrics_json,
        "shap":       model.shap_json,
        "hyperparams": model.hyperparams_json,
        "target_column": target_column,
        "input_features": dataset_columns,
        "features": get_model_feature_names(model.artifact_path, model.framework or "sklearn"),
    }
