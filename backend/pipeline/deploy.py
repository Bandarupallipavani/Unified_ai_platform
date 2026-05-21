"""
pipeline/deploy.py — Model deployment utilities
Generates: FastAPI server, Dockerfile, AWS/GCP/Azure export scripts, ONNX export, predictions
Supports ML (sklearn/pickle), DL (PyTorch .pt), NLP (HuggingFace directory)
"""
import os
import pickle
import re
import shutil
import textwrap
import logging
from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "./outputs")
os.makedirs(OUTPUT_PATH, exist_ok=True)


def prepare_model_bundle(model_id: str, model_path: str, framework: str = "sklearn") -> dict:
    """Copy the trained artifact into OUTPUT_PATH with a predictable deploy-friendly name."""
    if not model_path:
        raise FileNotFoundError(f"Model artifact path is missing for {model_id}.")

    source = Path(model_path)
    if not source.exists():
        raise FileNotFoundError(f"Model artifact not found for {model_id}: {model_path}")

    if source.is_dir():
        bundle_name = f"model_{model_id}"
        bundle_path = Path(OUTPUT_PATH) / bundle_name
        if bundle_path.exists():
            shutil.rmtree(bundle_path)
        shutil.copytree(source, bundle_path)
        return {"path": str(bundle_path), "name": bundle_name, "is_dir": True}

    suffix = source.suffix or (".pt" if framework == "pytorch" else ".pkl")
    bundle_name = f"model_{model_id}{suffix}"
    bundle_path = Path(OUTPUT_PATH) / bundle_name

    if source.resolve() != bundle_path.resolve():
        shutil.copyfile(source, bundle_path)

    return {"path": str(bundle_path), "name": bundle_name, "is_dir": False}


def get_model_feature_names(model_path: str, framework: str = "sklearn") -> List[str]:
    """Return saved feature names when the model exposes them."""
    if framework != "sklearn" or not model_path or not os.path.exists(model_path):
        return []

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception:
        return []

    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        return []

    return [str(name) for name in list(feature_names)]


def _load_sklearn_model(model_path: str):
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)

    if isinstance(artifact, dict) and "model" in artifact:
        return artifact["model"]
    return artifact


def _build_input_frame(data: list, columns: Optional[list] = None) -> pd.DataFrame:
    if data and isinstance(data[0], dict):
        return pd.DataFrame(data)
    return pd.DataFrame(data, columns=columns or None)


def _resolve_target_column(train_df: pd.DataFrame, target_column: Optional[str] = None) -> str:
    if target_column and target_column in train_df.columns:
        return target_column

    for candidate in ["target", "label", "class", "output", "y", "churn", "price", "salary"]:
        if candidate in train_df.columns:
            return candidate

    return train_df.columns[-1]


def _prepare_tabular_inference_frame(
    df: pd.DataFrame,
    training_dataset_path: str,
    target_column: Optional[str] = None,
) -> Tuple[np.ndarray, List[str]]:
    train_df = pd.read_parquet(training_dataset_path) if training_dataset_path.endswith(".parquet") else (
        pd.read_excel(training_dataset_path) if training_dataset_path.endswith((".xls", ".xlsx")) else (
            pd.read_json(training_dataset_path) if training_dataset_path.endswith(".json") else pd.read_csv(training_dataset_path)
        )
    )

    resolved_target = _resolve_target_column(train_df, target_column)
    if resolved_target not in train_df.columns:
        raise ValueError(f"Target column '{resolved_target}' not found in training dataset.")

    X_train_raw = train_df.drop(columns=[resolved_target])
    aligned_input = df.copy()
    train_columns = list(X_train_raw.columns)
    input_columns = list(aligned_input.columns)

    if input_columns and all(col is not None for col in input_columns):
        unknown_columns = [col for col in input_columns if col not in train_columns]
        if unknown_columns:
            raise ValueError(f"Unknown input column(s): {', '.join(map(str, unknown_columns[:8]))}.")
        aligned_input = aligned_input.reindex(columns=train_columns, fill_value=pd.NA)
    elif aligned_input.shape[1] == len(train_columns):
        aligned_input.columns = train_columns
    else:
        raise ValueError(
            f"Expected {len(train_columns)} raw input column(s) "
            f"({', '.join(map(str, train_columns[:8]))}), got {aligned_input.shape[1]}."
        )

    train_encoded = pd.get_dummies(X_train_raw, drop_first=True)
    input_encoded = pd.get_dummies(aligned_input, drop_first=True).reindex(columns=train_encoded.columns, fill_value=0)

    imputer = SimpleImputer(strategy="median")
    train_imputed = imputer.fit_transform(train_encoded)
    input_imputed = imputer.transform(input_encoded)

    scaler = StandardScaler()
    scaler.fit(train_imputed)
    return scaler.transform(input_imputed), list(train_encoded.columns)


def get_model_requirements(framework: str = "sklearn") -> list[str]:
    if framework == "transformers":
        return [
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
            "pydantic>=2.0.0",
            "transformers>=4.35.0",
            "torch>=2.0.0",
            "sentencepiece>=0.1.99",
        ]
    if framework == "pytorch":
        return [
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
            "pydantic>=2.0.0",
            "torch>=2.0.0",
            "numpy>=1.24.0",
        ]
    return [
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
    ]


def write_model_requirements(filename: str, framework: str = "sklearn") -> str:
    path = Path(OUTPUT_PATH) / filename
    path.write_text("\n".join(get_model_requirements(framework)) + "\n", encoding="utf-8")
    return str(path)


def build_azure_score_script(
    model_id: str,
    framework: str = "sklearn",
    bundled_model_name: Optional[str] = None,
) -> str:
    bundle_name = bundled_model_name or f"model_{model_id}.pkl"

    if framework == "pytorch":
        raise ValueError("Automated Azure deployment is currently supported for sklearn and transformers models.")

    if framework == "transformers":
        score_code = textwrap.dedent(f"""
            import json
            import os
            from pathlib import Path
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            TOKENIZER = None
            MODEL = None
            MODEL_NAME = "{bundle_name}"

            def _resolve_model_path() -> Path:
                model_root = Path(os.getenv("AZUREML_MODEL_DIR", Path(__file__).parent))
                direct = model_root / MODEL_NAME
                if direct.exists():
                    return direct
                for candidate in model_root.rglob(MODEL_NAME):
                    return candidate
                fallback = Path(__file__).with_name(MODEL_NAME)
                if fallback.exists():
                    return fallback
                raise FileNotFoundError(f"Could not locate model bundle '{{MODEL_NAME}}' in {{model_root}}")

            def init():
                global TOKENIZER, MODEL
                model_dir = _resolve_model_path()
                TOKENIZER = AutoTokenizer.from_pretrained(model_dir)
                MODEL = AutoModelForSequenceClassification.from_pretrained(model_dir)
                MODEL.eval()

            def run(raw_data):
                try:
                    payload = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
                    texts = payload.get("texts", [])
                    max_length = int(payload.get("max_length", 256))
                    enc = TOKENIZER(
                        texts,
                        truncation=True,
                        padding=True,
                        max_length=max_length,
                        return_tensors="pt",
                    )
                    with torch.no_grad():
                        logits = MODEL(**enc).logits
                    preds = logits.argmax(dim=-1).tolist()
                    proba = torch.softmax(logits, dim=-1).tolist()
                    return {{"predictions": preds, "probabilities": proba, "model_id": "{model_id}"}}
                except Exception as exc:
                    return {{"error": str(exc), "model_id": "{model_id}"}}
        """).strip()
    else:
        score_code = textwrap.dedent(f"""
            import json
            import os
            import pickle
            from pathlib import Path
            import pandas as pd

            MODEL = None
            MODEL_NAME = "{bundle_name}"

            def _resolve_model_path() -> Path:
                model_root = Path(os.getenv("AZUREML_MODEL_DIR", Path(__file__).parent))
                direct = model_root / MODEL_NAME
                if direct.exists():
                    return direct
                for candidate in model_root.rglob(MODEL_NAME):
                    return candidate
                fallback = Path(__file__).with_name(MODEL_NAME)
                if fallback.exists():
                    return fallback
                raise FileNotFoundError(f"Could not locate model bundle '{{MODEL_NAME}}' in {{model_root}}")

            def init():
                global MODEL
                with _resolve_model_path().open("rb") as f:
                    MODEL = pickle.load(f)

            def run(raw_data):
                try:
                    payload = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
                    rows = payload.get("data", [])
                    columns = payload.get("columns") or None
                    df = pd.DataFrame(rows, columns=columns)
                    preds = MODEL.predict(df).tolist()
                    return {{"predictions": preds, "model_id": "{model_id}"}}
                except Exception as exc:
                    return {{"error": str(exc), "model_id": "{model_id}"}}
        """).strip()

    path = Path(OUTPUT_PATH) / f"score_{model_id}.py"
    path.write_text(score_code, encoding="utf-8")
    return str(path)


def _get_azure_auth(config: dict):
    from azureml.core.authentication import AzureCliAuthentication, ServicePrincipalAuthentication

    tenant_id = config.get("tenant_id") or os.getenv("AZURE_TENANT_ID")
    client_id = config.get("service_principal_id") or os.getenv("AZURE_SERVICE_PRINCIPAL_ID")
    client_secret = config.get("service_principal_password") or os.getenv("AZURE_SERVICE_PRINCIPAL_PASSWORD")

    if tenant_id and client_id and client_secret:
        return ServicePrincipalAuthentication(
            tenant_id=tenant_id,
            service_principal_id=client_id,
            service_principal_password=client_secret,
        )

    return AzureCliAuthentication()


def _make_azure_service_name(model_id: str, requested_name: Optional[str] = None) -> str:
    raw = (requested_name or f"unified-ai-{model_id[:8]}").strip().lower()
    cleaned = re.sub(r"[^a-z0-9-]", "-", raw)
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    if len(cleaned) < 3:
        cleaned = f"uap-{model_id[:8]}"
    return cleaned[:32]


def deploy_azure_endpoint(
    model_id: str,
    model_path: str,
    model_type: str,
    framework: str,
    config: dict,
) -> dict:
    required = ["subscription_id", "resource_group", "workspace"]
    missing = [field for field in required if not (config.get(field) or "").strip()]
    if missing:
        raise ValueError(f"Azure deployment is missing required fields: {', '.join(missing)}")

    try:
        from azureml.core import Workspace, Model, Environment
        from azureml.core.model import InferenceConfig
        from azureml.core.webservice import AciWebservice
    except ImportError as exc:
        raise RuntimeError(
            "Azure deployment automation requires the 'azureml-core' package in the backend environment."
        ) from exc

    bundle = prepare_model_bundle(model_id, model_path, framework)
    score_path = build_azure_score_script(model_id, framework, bundled_model_name=bundle["name"])
    requirements_path = write_model_requirements(f"requirements_azure_{model_id}.txt", framework)
    auth = _get_azure_auth(config)

    workspace = Workspace.get(
        name=config["workspace"],
        subscription_id=config["subscription_id"],
        resource_group=config["resource_group"],
        auth=auth,
    )

    registered_model = Model.register(
        workspace=workspace,
        model_path=bundle["path"],
        model_name=config.get("model_name", f"unified-ai-{model_id[:8]}"),
        description=f"Unified AI Platform model {model_id}",
    )

    environment = Environment.from_pip_requirements(
        name=f"uap-env-{model_id[:8]}",
        file_path=requirements_path,
    )
    inference_config = InferenceConfig(entry_script=score_path, environment=environment)
    deployment_config = AciWebservice.deploy_configuration(
        cpu_cores=float(config.get("cpu_cores", 1)),
        memory_gb=float(config.get("memory_gb", 1)),
    )

    service_name = _make_azure_service_name(model_id, config.get("service_name"))
    service = Model.deploy(
        workspace=workspace,
        name=service_name,
        models=[registered_model],
        inference_config=inference_config,
        deployment_config=deployment_config,
        overwrite=True,
    )
    service.wait_for_deployment(show_output=False)

    state = getattr(service, "state", None) or getattr(service, "provisioning_state", None) or "Unknown"
    endpoint_url = getattr(service, "scoring_uri", None)
    if str(state).lower() in {"failed", "unhealthy"} or not endpoint_url:
        try:
            logs = service.get_logs()
        except Exception:
            logs = ""
        details = f"Azure deployment finished in state '{state}'."
        if logs:
            details = f"{details} Logs: {logs[:4000]}"
        raise RuntimeError(details)

    return {
        "target": "azure",
        "workspace": workspace.name,
        "service_name": service_name,
        "endpoint_url": endpoint_url,
        "state": state,
        "model_name": registered_model.name,
        "model_version": getattr(registered_model, "version", None),
        "score_script_path": score_path,
        "requirements_path": requirements_path,
        "artifact_name": bundle["name"],
    }


def build_fastapi_server(
    model_id: str,
    model_path: str,
    model_type: str = "ml",
    framework: str = "sklearn",
    bundled_model_name: Optional[str] = None,
) -> str:
    """Generate a standalone FastAPI prediction server script."""
    bundle = prepare_model_bundle(model_id, model_path, framework)
    bundle_name = bundled_model_name or bundle["name"]
    if framework == "transformers":
        server_code = textwrap.dedent(f"""
            \"\"\"
            Unified AI Platform — Auto-generated NLP Model Server
            Model ID: {model_id}
            \"\"\"
            from pathlib import Path
            from fastapi import FastAPI
            from pydantic import BaseModel
            from typing import List
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            app = FastAPI(title="NLP Model Server", version="1.0")

            MODEL_DIR = str(Path(__file__).with_name("{bundle_name}"))
            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
            model.eval()

            class PredictRequest(BaseModel):
                texts: List[str]
                max_length: int = 256

            @app.get("/health")
            def health():
                return {{"status": "healthy", "model_id": "{model_id}", "type": "nlp"}}

            @app.post("/predict")
            def predict(req: PredictRequest):
                enc = tokenizer(req.texts, truncation=True, padding=True,
                                max_length=req.max_length, return_tensors="pt")
                with torch.no_grad():
                    logits = model(**enc).logits
                preds = logits.argmax(dim=-1).tolist()
                proba = torch.softmax(logits, dim=-1).tolist()
                return {{"predictions": preds, "probabilities": proba, "model_id": "{model_id}"}}

            if __name__ == "__main__":
                import uvicorn
                uvicorn.run(app, host="0.0.0.0", port=8080)
        """).strip()
    elif framework == "pytorch":
        server_code = textwrap.dedent(f"""
            \"\"\"
            Unified AI Platform — Auto-generated DL Model Server
            Model ID: {model_id}
            \"\"\"
            from pathlib import Path
            import torch
            import numpy as np
            from fastapi import FastAPI
            from pydantic import BaseModel
            from typing import List, Any

            app = FastAPI(title="DL Model Server", version="1.0")

            MODEL_PATH = Path(__file__).with_name("{bundle_name}")
            checkpoint = torch.load(MODEL_PATH, map_location="cpu")

            @app.get("/health")
            def health():
                return {{"status": "healthy", "model_id": "{model_id}", "type": "dl"}}

            @app.post("/predict")
            def predict(data: dict):
                x = torch.tensor(data["data"], dtype=torch.float32)
                with torch.no_grad():
                    # reload model architecture here based on checkpoint["config"]
                    pass
                return {{"model_id": "{model_id}", "note": "Load architecture from config"}}

            if __name__ == "__main__":
                import uvicorn
                uvicorn.run(app, host="0.0.0.0", port=8080)
        """).strip()
    else:
        server_code = textwrap.dedent(f"""
            \"\"\"
            Unified AI Platform — Auto-generated ML Model Server
            Model ID: {model_id}
            \"\"\"
            from pathlib import Path
            import pickle
            import numpy as np
            import pandas as pd
            from fastapi import FastAPI
            from pydantic import BaseModel
            from typing import List, Any

            app = FastAPI(title="ML Model Server", version="1.0")

            MODEL_PATH = Path(__file__).with_name("{bundle_name}")

            with MODEL_PATH.open("rb") as f:
                MODEL = pickle.load(f)

            class PredictRequest(BaseModel):
                data: List[List[Any]]
                columns: List[str] = []

            @app.get("/health")
            def health():
                return {{"status": "healthy", "model_id": "{model_id}"}}

            @app.get("/model-info")
            def model_info():
                return {{"model_id": "{model_id}", "model_type": type(MODEL).__name__,
                        "features": getattr(MODEL, "feature_names_in_", [])}}

            @app.post("/predict")
            def predict(request: PredictRequest):
                df = pd.DataFrame(request.data, columns=request.columns or None)
                preds = MODEL.predict(df).tolist()
                return {{"predictions": preds, "model_id": "{model_id}"}}

            if __name__ == "__main__":
                import uvicorn
                uvicorn.run(app, host="0.0.0.0", port=8080)
        """).strip()

    path = os.path.join(OUTPUT_PATH, f"server_{model_id}.py")
    with open(path, "w") as f:
        f.write(server_code)
    return path


def build_dockerfile(model_id: str, framework: str = "sklearn", bundled_model_name: Optional[str] = None) -> str:
    """Generate a Dockerfile for containerizing the model server."""
    base_image = "python:3.11-slim"

    if framework == "transformers":
        requirements = [
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
            "pydantic>=2.0.0",
            "transformers>=4.35.0",
            "torch>=2.0.0",
            "sentencepiece>=0.1.99",
        ]
    elif framework == "pytorch":
        requirements = [
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
            "pydantic>=2.0.0",
            "torch>=2.0.0",
            "numpy>=1.24.0",
        ]
    else:
        requirements = [
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
            "pydantic>=2.0.0",
            "scikit-learn>=1.3.0",
            "xgboost>=2.0.0",
            "lightgbm>=4.0.0",
            "pandas>=2.0.0",
            "numpy>=1.24.0",
        ]

    bundle_name = bundled_model_name or f"model_{model_id}.pkl"
    copy_line = f"COPY {bundle_name} ./{bundle_name}"
    if "." not in Path(bundle_name).name:
        copy_line = f"COPY {bundle_name}/ ./{bundle_name}/"

    dockerfile = textwrap.dedent(f"""
        FROM {base_image}

        WORKDIR /app

        RUN pip install --no-cache-dir {" ".join(requirements)}

        COPY server_{model_id}.py ./server.py
        {copy_line}

        EXPOSE 8080

        HEALTHCHECK --interval=30s --timeout=10s --retries=3 \\
          CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

        CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
    """).strip()

    docker_path = os.path.join(OUTPUT_PATH, "Dockerfile")

    with open(docker_path, "w") as f:
        f.write(dockerfile)

    return docker_path


def export_onnx(model_id: str, model_path: str, framework: str, input_dim: int = 10) -> dict:
    """Export model to ONNX format for cross-platform inference."""
    onnx_path = os.path.join(OUTPUT_PATH, f"model_{model_id}.onnx")

    if framework == "pytorch":
        try:
            import torch
            import torch.onnx
            checkpoint = torch.load(model_path, map_location="cpu")
            logger.info(f"ONNX export: load checkpoint for {model_id}")
            dummy_input = torch.randn(1, input_dim)
            # Note: full ONNX export requires the model class to be available
            return {"onnx_path": onnx_path, "note": "Rebuild model from checkpoint config and re-export"}
        except Exception as e:
            return {"error": str(e)}
    elif framework == "sklearn":
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            initial_type = [("float_input", FloatTensorType([None, input_dim]))]
            onx = convert_sklearn(model, initial_types=initial_type)
            with open(onnx_path, "wb") as f:
                f.write(onx.SerializeToString())
            return {"onnx_path": onnx_path}
        except ImportError:
            return {"error": "skl2onnx not installed. pip install skl2onnx"}
        except Exception as e:
            return {"error": str(e)}
    return {"error": "ONNX export not supported for this framework."}


def export_cloud(model_id: str, model_path: str, target: str, config: dict) -> dict:
    """Generate cloud deployment scripts for AWS / GCP / Azure."""
    if target == "aws":
        return _export_aws(model_id, model_path, config)
    elif target == "gcp":
        return _export_gcp(model_id, model_path, config)
    elif target == "azure":
        return _export_azure(model_id, model_path, config)
    return {"error": f"Unknown deployment target: {target}"}


def build_render_blueprint(model_id: str, config: dict) -> dict:
    """Generate a Render Blueprint for public frontend + backend hosting."""
    def _slug(value: str, fallback: str) -> str:
        raw = (value or fallback).strip().lower()
        cleaned = re.sub(r"[^a-z0-9-]+", "-", raw).strip("-")
        return cleaned or fallback

    backend_name = _slug(config.get("backend_service_name", f"uap-api-{model_id[:8]}"), f"uap-api-{model_id[:8]}")
    frontend_name = _slug(config.get("frontend_service_name", f"uap-web-{model_id[:8]}"), f"uap-web-{model_id[:8]}")
    branch = (config.get("branch") or "main").strip()
    repo_url = (config.get("repo_url") or "").strip()

    backend_url = f"https://{backend_name}.onrender.com"
    frontend_url = f"https://{frontend_name}.onrender.com"

    repo_lines = ""
    if repo_url:
        repo_lines = textwrap.indent(
            textwrap.dedent(
                f"""
                repo: {repo_url}
                branch: {branch}
                """
            ).strip(),
            "            ",
        )
        repo_lines = f"\n{repo_lines}"

    blueprint = textwrap.dedent(
        f"""
        services:
          - type: web
            runtime: python
            name: {backend_name}
            plan: free
            rootDir: backend{repo_lines}
            autoDeploy: false
            buildCommand: pip install -r requirements.txt
            startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
            healthCheckPath: /health
            envVars:
              - key: SECRET_KEY
                value: change-me-before-production
              - key: DATABASE_URL
                value: sqlite:///./automl.db
              - key: ALLOWED_ORIGINS
                value: {frontend_url}
              - key: STORAGE_PATH
                value: ./data_sessions
              - key: REPORTS_PATH
                value: ./reports
              - key: OUTPUT_PATH
                value: ./outputs
              - key: MAX_CONCURRENT_JOBS
                value: "1"

          - type: web
            runtime: static
            name: {frontend_name}
            plan: free
            rootDir: frontend{repo_lines}
            autoDeploy: false
            buildCommand: npm ci && npm run build
            staticPublishPath: ./build
            envVars:
              - key: REACT_APP_API_URL
                value: {backend_url}
            routes:
              - type: rewrite
                source: /*
                destination: /index.html
        """
    ).strip() + "\n"

    path = os.path.join(OUTPUT_PATH, f"render_{model_id}.yaml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(blueprint)

    deploy_url = f"https://render.com/deploy?repo={repo_url}" if repo_url else "https://render.com/deploy"

    return {
        "target": "render_free",
        "blueprint_path": path,
        "frontend_url": frontend_url,
        "backend_url": backend_url,
        "deploy_url": deploy_url,
        "frontend_service_name": frontend_name,
        "backend_service_name": backend_name,
        "instructions": [
            "Commit render.yaml to the root of the same Git repo if you want one-click Deploy to Render.",
            "Open the Deploy to Render link, review both services, and approve the free deployment.",
            "After Render finishes, use the frontend onrender.com URL as the public app link.",
            "If you want current local-trained artifacts online, commit them or retrain inside the deployed app.",
        ],
    }


def _export_aws(model_id: str, model_path: str, config: dict) -> dict:
    region = config.get("region", "us-east-1")
    bucket = config.get("s3_bucket", "your-sagemaker-bucket")

    script = textwrap.dedent(f"""
        # Unified AI Platform — AWS SageMaker Deployment
        # Model: {model_id}
        import boto3, sagemaker
        from sagemaker.sklearn.model import SKLearnModel

        session = sagemaker.Session()
        role = "{config.get('iam_role', 'arn:aws:iam::ACCOUNT:role/SageMakerRole')}"

        s3 = boto3.client("s3", region_name="{region}")
        s3.upload_file("{model_path}", "{bucket}", "models/{model_id}/model.pkl")
        model_uri = "s3://{bucket}/models/{model_id}/model.pkl"

        sklearn_model = SKLearnModel(
            model_data=model_uri, role=role,
            entry_point="inference.py", framework_version="1.3-1",
        )
        predictor = sklearn_model.deploy(
            initial_instance_count=1, instance_type="{config.get('instance_type', 'ml.t2.medium')}",
            endpoint_name="unified-ai-{model_id[:8]}",
        )
        print(f"Endpoint: {{predictor.endpoint_name}}")
    """).strip()

    path = os.path.join(OUTPUT_PATH, f"deploy_aws_{model_id}.py")
    with open(path, "w") as f:
        f.write(script)

    return {
        "target": "aws", "script_path": path,
        "instructions": [
            "1. pip install boto3 sagemaker",
            f"2. aws configure (region: {region})",
            "3. Ensure IAM role has SageMakerFullAccess",
            "4. python deploy_aws_{model_id}.py",
        ],
    }


def _export_gcp(model_id: str, model_path: str, config: dict) -> dict:
    project = config.get("project_id", "your-gcp-project")
    region = config.get("region", "us-central1")
    bucket = config.get("gcs_bucket", "your-gcs-bucket")

    script = textwrap.dedent(f"""
        # Unified AI Platform — GCP Vertex AI Deployment
        # Model: {model_id}
        from google.cloud import aiplatform, storage

        aiplatform.init(project="{project}", location="{region}")

        gcs = storage.Client()
        blob = gcs.bucket("{bucket}").blob("models/{model_id}/model.pkl")
        blob.upload_from_filename("{model_path}")

        model = aiplatform.Model.upload(
            display_name="unified-ai-{model_id[:8]}",
            artifact_uri="gs://{bucket}/models/{model_id}/",
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest",
        )
        endpoint = model.deploy(machine_type="{config.get('machine_type', 'n1-standard-2')}")
        print(f"Endpoint: {{endpoint.resource_name}}")
    """).strip()

    path = os.path.join(OUTPUT_PATH, f"deploy_gcp_{model_id}.py")
    with open(path, "w") as f:
        f.write(script)

    return {"target": "gcp", "script_path": path}


def _export_azure(model_id: str, model_path: str, config: dict) -> dict:
    script = textwrap.dedent(f"""
        # Unified AI Platform — Azure ML Deployment
        # Model: {model_id}
        from azureml.core import Workspace, Model, Environment
        from azureml.core.webservice import AciWebservice
        from azureml.core.model import InferenceConfig

        ws = Workspace.from_config()

        model = Model.register(workspace=ws, model_path="{model_path}",
                               model_name="unified-ai-{model_id[:8]}")

        env = Environment.from_pip_requirements("env", "requirements_model.txt")
        ic  = InferenceConfig(entry_script="score.py", environment=env)
        aci = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

        svc = Model.deploy(ws, "unified-ai-{model_id[:8]}", [model], ic, aci)
        svc.wait_for_deployment(show_output=True)
        print(f"Scoring URI: {{svc.scoring_uri}}")
    """).strip()

    path = os.path.join(OUTPUT_PATH, f"deploy_azure_{model_id}.py")
    with open(path, "w") as f:
        f.write(script)

    return {"target": "azure", "script_path": path}


def run_prediction(
    model_path: str,
    data: list,
    framework: str = "sklearn",
    columns: Optional[list] = None,
    training_dataset_path: Optional[str] = None,
    target_column: Optional[str] = None,
) -> list:
    """Run inference with a saved model."""
    if framework == "sklearn":
        model = _load_sklearn_model(model_path)
        df = _build_input_frame(data, columns=columns)

        raw_feature_names = getattr(model, "feature_names_in_", None)
        expected_features = [str(name) for name in list(raw_feature_names)] if raw_feature_names is not None else []
        expected_feature_count = int(getattr(model, "n_features_in_", 0) or 0)
        needs_tabular_rebuild = bool(training_dataset_path) and (
            (expected_features and list(df.columns) != expected_features) or
            (expected_feature_count and df.shape[1] != expected_feature_count)
        )

        if needs_tabular_rebuild:
            prepared_data, prepared_columns = _prepare_tabular_inference_frame(df, training_dataset_path, target_column)
            if expected_features and len(expected_features) != len(prepared_columns):
                raise ValueError(
                    f"Prepared feature count {len(prepared_columns)} did not match model expectation {len(expected_features)}."
                )
            if expected_feature_count and expected_feature_count != len(prepared_columns):
                raise ValueError(
                    f"Prepared feature count {len(prepared_columns)} did not match model expectation {expected_feature_count}."
                )
            return model.predict(prepared_data).tolist()

        if expected_features:
            df = df.reindex(columns=expected_features, fill_value=0)
        return model.predict(df).tolist()
    elif framework == "pytorch":
        import torch
        checkpoint = torch.load(model_path, map_location="cpu")
        x = torch.tensor(data, dtype=torch.float32)
        # Inference requires rebuilding the model from checkpoint["config"]
        return []
    return []
