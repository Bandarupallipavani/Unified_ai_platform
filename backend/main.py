"""
main.py — Unified AI Platform Backend v3
Fixes: CORS lockdown, file validation, Redis-backed copilot sessions,
       CoPilot history restored from DB, async report generation,
       job rate limiting, batch prediction, drift monitoring,
       model registry/promotion, A-B deployment, scheduled retraining.
"""
import os
import uuid
import json
import asyncio
import logging
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import (FastAPI, Depends, HTTPException, UploadFile, File,
                     WebSocket, WebSocketDisconnect, BackgroundTasks, Query)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from db import engine, SessionLocal, Base, redis_set, redis_get, redis_delete, get_redis
from auth import (UserCreate, UserLogin, Token, hash_password, verify_password,
                  create_access_token, get_current_user)
from models_db import (User, Session, TrainingJob, Model, Report,
                       CopilotMessage, Deployment, DriftAlert,
                       ScheduledJob, BatchPrediction, ModelVersion)
from pipeline.ingest import ingest_file, ingest_nl_connector
from pipeline.eda import run_eda
from pipeline.preprocess import build_preprocessing_plan, apply_preprocessing
from pipeline.train import start_training_job, get_job_status, get_job_results, get_best_model
from pipeline.deploy import (
    build_fastapi_server,
    build_dockerfile,
    build_render_blueprint,
    deploy_azure_endpoint,
    export_cloud,
    export_onnx,
    prepare_model_bundle,
    run_prediction,
)
from pipeline.drift import compute_drift
from pipeline.batch import run_batch_prediction
from ai.copilot import CoPilot
from ai.report import generate_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Max concurrent training jobs (prevents OOM)
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "3"))
_running_jobs: set[str] = set()

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:3001"
).split(",")

# Allowed upload extensions + max size
ALLOWED_EXTENSIONS = {".csv", ".json", ".xlsx", ".xls", ".parquet", ".tsv"}
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "200"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created/verified.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Unified AI Platform",
    description="End-to-end ML/DL/NLP training, tuning, and deployment platform",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connections store (job_id → list[WebSocket])
ws_connections: dict[str, list[WebSocket]] = {}


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class ConnectRequest(BaseModel):
    description: str

class PreprocessRequest(BaseModel):
    session_id: str
    config: dict = {}

class TrainRequest(BaseModel):
    session_id: str
    config: dict = {}

class PredictRequest(BaseModel):
    data: list
    columns: list = []

class CopilotRequest(BaseModel):
    session_id: str
    message: str

class DeployRequest(BaseModel):
    target: str = "fastapi"
    config: dict = {}
    traffic_pct: int = 100

class PromoteModelRequest(BaseModel):
    notes: str = ""

class DriftCheckRequest(BaseModel):
    model_id: str
    sample_data: list   # list of dicts (recent prediction inputs)

class ScheduleRequest(BaseModel):
    session_id: str
    cron_expr: str       # e.g. "0 2 * * *"
    config: dict = {}

class BatchPredictRequest(BaseModel):
    model_id: str

class ABDeployRequest(BaseModel):
    model_id_a: str
    model_id_b: str
    traffic_pct_a: int = 50   # % sent to A; remainder to B


# ─────────────────────────────────────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/auth/register", response_model=Token)
def register(body: UserCreate, db=Depends(SessionLocal)):
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(400, "Email already registered.")
    user = User(
        email=body.email,
        hashed_password=hash_password(body.password),
        mode=body.mode,
        domain=body.domain,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token({"sub": user.id})
    return Token(access_token=token, token_type="bearer")


@app.post("/api/auth/login", response_model=Token)
def login(body: UserLogin, db=Depends(SessionLocal)):
    user = db.query(User).filter(User.email == body.email).first()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(401, "Invalid credentials.")
    token = create_access_token({"sub": user.id})
    return Token(access_token=token, token_type="bearer")


@app.get("/api/auth/me")
def me(user: User = Depends(get_current_user)):
    return {"id": user.id, "email": user.email, "mode": user.mode, "domain": user.domain}


# ─────────────────────────────────────────────────────────────────────────────
# Data Ingestion  (file validation fixed)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload(file: UploadFile = File(...),
                 user: User = Depends(get_current_user),
                 db=Depends(SessionLocal)):
    from pathlib import Path
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}")

    # Read and check size BEFORE processing
    content = await file.read()
    size_mb = len(content) / 1_000_000
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(413, f"File too large ({size_mb:.1f} MB). Max {MAX_FILE_SIZE_MB} MB.")

    session_id = str(uuid.uuid4())
    try:
        profile = await ingest_file(file, session_id, content=content)
    except ValueError as e:
        raise HTTPException(400, str(e))

    session = Session(id=session_id, user_id=user.id, profile_json=profile,
                      dataset_path=profile["dataset_path"], domain_mode=user.domain)
    db.add(session)
    db.commit()
    return {"session_id": session_id, "profile": profile}


@app.post("/api/connect")
async def connect_nl(body: ConnectRequest,
                     user: User = Depends(get_current_user),
                     db=Depends(SessionLocal)):
    session_id = str(uuid.uuid4())
    result = await ingest_nl_connector(body.description, session_id)
    session = Session(id=session_id, user_id=user.id,
                      profile_json=result["profile"],
                      dataset_path=result["dataframe_path"],
                      domain_mode=user.domain)
    db.add(session)
    db.commit()
    return {"session_id": session_id, "profile": result["profile"], "query": result.get("query")}


@app.get("/api/profile/{session_id}")
def get_profile(session_id: str, user: User = Depends(get_current_user), db=Depends(SessionLocal)):
    session = db.query(Session).filter(Session.id == session_id, Session.user_id == user.id).first()
    if not session:
        raise HTTPException(404, "Session not found.")
    return session.profile_json


# ─────────────────────────────────────────────────────────────────────────────
# EDA
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/eda/{session_id}")
def eda(session_id: str, user: User = Depends(get_current_user), db=Depends(SessionLocal)):
    session = db.query(Session).filter(Session.id == session_id, Session.user_id == user.id).first()
    if not session or not session.dataset_path:
        raise HTTPException(404, "Session or dataset not found.")
    return run_eda(session.dataset_path, session_id, cached=True)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/preprocess/plan/{session_id}")
def preprocessing_plan(session_id: str, user: User = Depends(get_current_user), db=Depends(SessionLocal)):
    session = db.query(Session).filter(Session.id == session_id, Session.user_id == user.id).first()
    if not session or not session.profile_json:
        raise HTTPException(404, "Session not found.")
    return build_preprocessing_plan(session.profile_json, user.domain, user.mode)


@app.post("/api/preprocess")
def preprocess(body: PreprocessRequest, user: User = Depends(get_current_user), db=Depends(SessionLocal)):
    session = db.query(Session).filter(Session.id == body.session_id, Session.user_id == user.id).first()
    if not session or not session.dataset_path:
        raise HTTPException(404, "Session or dataset not found.")
    config = body.config
    if not config.get("steps"):
        config = build_preprocessing_plan(session.profile_json or {}, user.domain, user.mode)
    log = apply_preprocessing(session.dataset_path, body.session_id, config)
    session.status = "preprocessed"
    db.commit()
    return {"session_id": body.session_id, "preprocessing_log": log}


# ─────────────────────────────────────────────────────────────────────────────
# Training  (job rate limit + background executor)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/train")
async def start_train(body: TrainRequest,
                      user: User = Depends(get_current_user),
                      db=Depends(SessionLocal)):
    # Rate limit
    if len(_running_jobs) >= MAX_CONCURRENT_JOBS:
        raise HTTPException(429, f"Server busy — max {MAX_CONCURRENT_JOBS} concurrent training jobs. Please wait.")

    session = db.query(Session).filter(Session.id == body.session_id, Session.user_id == user.id).first()
    if not session:
        raise HTTPException(404, "Session not found.")

    job_id = str(uuid.uuid4())
    model_type = body.config.get("model_type", "ml")
    job = TrainingJob(id=job_id, session_id=body.session_id,
                      config_json=body.config, model_type=model_type)
    db.add(job)
    db.commit()

    async def ws_log(msg: str):
        logger.info(f"[{job_id}] {msg}")
        connections = ws_connections.get(job_id, [])
        dead = []
        for ws in connections:
            try:
                await ws.send_text(json.dumps({"type": "log", "message": msg, "ts": datetime.utcnow().isoformat()}))
            except Exception:
                dead.append(ws)
        for ws in dead:
            connections.remove(ws)

    async def _run_and_cleanup():
        _running_jobs.add(job_id)
        try:
            # Run CPU-bound training in a thread pool so it doesn't block the event loop
            main_loop = asyncio.get_running_loop()
            await main_loop.run_in_executor(
                None,
                _sync_training_wrapper,
                job_id, session.dataset_path, body.config, user.domain, user.mode, main_loop
            )
            # Emit final WS log via async after executor completes
            await ws_log("🎉 Training job finished.")
        except Exception as e:
            logger.exception(f"Training job {job_id} crashed: {e}")
            await ws_log(f"❌ Training crashed: {e}")
        finally:
            _running_jobs.discard(job_id)

    # Kick off async task
    asyncio.create_task(_run_and_cleanup())

    return {"job_id": job_id, "status": "queued", "model_type": model_type}


def _emit_log_to_ws(job_id, message, main_loop):
    async def _broadcast():
        connections = ws_connections.get(job_id, [])
        dead = []
        for ws in list(connections):
            try:
                await ws.send_text(json.dumps({
                    "type": "log",
                    "message": message,
                    "ts": datetime.utcnow().isoformat(),
                }))
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in connections:
                connections.remove(ws)

    future = asyncio.run_coroutine_threadsafe(_broadcast(), main_loop)

    def _consume_result(done_future):
        try:
            done_future.result()
        except Exception:
            logger.debug("Skipping websocket log delivery for %s", job_id, exc_info=True)

    future.add_done_callback(_consume_result)


def _sync_training_wrapper(job_id, dataset_path, config, domain, mode, main_loop):
    """Run the entire training pipeline synchronously inside an executor thread."""
    import asyncio as _asyncio
    from pipeline.train import start_training_job as _train

    # Each thread needs its own DB session
    db_gen = SessionLocal()
    db = next(db_gen)

    # We need a sync ws_log inside the thread; we buffer and emit later
    buffered_logs = []

    async def _noop_log(msg: str):
        buffered_logs.append(msg)
        logger.info(f"[train][{job_id}] {msg}")
        _emit_log_to_ws(job_id, msg, main_loop)

    loop = _asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            _train(job_id=job_id, dataset_path=dataset_path, config=config,
                   domain=domain, mode=mode, db=db, ws_log=_noop_log)
        )
    finally:
        loop.close()
        try:
            db_gen.close()
        except Exception:
            pass


@app.websocket("/ws/train/{job_id}")
async def ws_train(websocket: WebSocket, job_id: str):
    await websocket.accept()
    ws_connections.setdefault(job_id, []).append(websocket)
    try:
        while True:
            await asyncio.sleep(30)
            await websocket.send_text(json.dumps({"type": "ping"}))
    except WebSocketDisconnect:
        ws_connections.get(job_id, []).remove(websocket)


@app.get("/api/train/status/{job_id}")
def train_status(job_id: str, user: User = Depends(get_current_user), db=Depends(SessionLocal)):
    return get_job_status(job_id, db)


@app.get("/api/train/results/{job_id}")
def train_results(job_id: str, user: User = Depends(get_current_user), db=Depends(SessionLocal)):
    return get_job_results(job_id, db)


@app.get("/api/train/best/{job_id}")
def best_model(job_id: str, user: User = Depends(get_current_user), db=Depends(SessionLocal)):
    return get_best_model(job_id, db)


@app.get("/api/train/history")
def training_history(user: User = Depends(get_current_user), db=Depends(SessionLocal)):
    sessions = db.query(Session).filter(Session.user_id == user.id).all()
    session_ids = [s.id for s in sessions]
    jobs = (db.query(TrainingJob)
            .filter(TrainingJob.session_id.in_(session_ids))
            .order_by(TrainingJob.created_at.desc())
            .limit(50).all())
    return [
        {"job_id": j.id, "model_type": j.model_type, "status": j.status,
         "created_at": str(j.created_at), "best_model_id": j.best_model_id,
         "session_id": j.session_id}
        for j in jobs
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Model Registry
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/models")
def list_models(user: User = Depends(get_current_user), db=Depends(SessionLocal),
                model_type: Optional[str] = Query(None),
                production_only: bool = Query(False)):
    sessions = db.query(Session).filter(Session.user_id == user.id).all()
    session_ids = [s.id for s in sessions]
    job_ids = [j.id for j in db.query(TrainingJob).filter(
        TrainingJob.session_id.in_(session_ids)).all()]
    q = db.query(Model).filter(Model.job_id.in_(job_ids))
    if model_type:
        q = q.filter(Model.model_type == model_type)
    if production_only:
        q = q.filter(Model.is_production == True)
    models = q.order_by(Model.created_at.desc()).all()
    return [
        {"model_id": m.id, "algorithm": m.algorithm, "model_type": m.model_type,
         "framework": m.framework, "metrics": m.metrics_json,
         "is_production": m.is_production, "created_at": str(m.created_at)}
        for m in models
    ]


@app.post("/api/models/{model_id}/promote")
def promote_model(model_id: str, body: PromoteModelRequest,
                  user: User = Depends(get_current_user), db=Depends(SessionLocal)):
    """Mark a model as production and create a version snapshot."""
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(404, "Model not found.")

    # Demote all other production models for same job
    db.query(Model).filter(
        Model.job_id == model.job_id, Model.id != model_id
    ).update({"is_production": False})

    model.is_production = True

    # Get next version number
    last_ver = db.query(ModelVersion).filter(ModelVersion.model_id == model_id).count()
    version = ModelVersion(
        model_id=model_id,
        version=last_ver + 1,
        metrics_json=model.metrics_json,
        artifact_path=model.artifact_path,
        notes=body.notes,
    )
    db.add(version)
    db.commit()
    return {"status": "promoted", "model_id": model_id, "version": last_ver + 1}


@app.get("/api/models/{model_id}/versions")
def model_versions(model_id: str, user: User = Depends(get_current_user), db=Depends(SessionLocal)):
    versions = (db.query(ModelVersion)
                .filter(ModelVersion.model_id == model_id)
                .order_by(ModelVersion.version.desc()).all())
    return [
        {"version": v.version, "metrics": v.metrics_json,
         "notes": v.notes, "is_active": v.is_active, "created_at": str(v.created_at)}
        for v in versions
    ]


@app.post("/api/models/{model_id}/compare")
def compare_models(model_id: str, body: dict,
                   user: User = Depends(get_current_user), db=Depends(SessionLocal)):
    """Compare two models side-by-side."""
    other_id = body.get("compare_with")
    m1 = db.query(Model).filter(Model.id == model_id).first()
    m2 = db.query(Model).filter(Model.id == other_id).first()
    if not m1 or not m2:
        raise HTTPException(404, "One or both models not found.")
    return {
        "model_a": {"id": m1.id, "algorithm": m1.algorithm, "metrics": m1.metrics_json,
                    "hyperparams": m1.hyperparams_json, "framework": m1.framework},
        "model_b": {"id": m2.id, "algorithm": m2.algorithm, "metrics": m2.metrics_json,
                    "hyperparams": m2.hyperparams_json, "framework": m2.framework},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Prediction + Batch Prediction
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/predict/{model_id}")
def predict(model_id: str, body: PredictRequest,
            user: User = Depends(get_current_user), db=Depends(SessionLocal)):
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(404, "Model not found.")
    job = model.job
    session = job.session if job else None
    try:
        preds = run_prediction(
            model.artifact_path,
            body.data,
            framework=model.framework or "sklearn",
            columns=body.columns or None,
            training_dataset_path=session.dataset_path if session else None,
            target_column=((job.config_json or {}).get("target_column") or None) if job else None,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"predictions": preds, "model_id": model_id}


@app.post("/api/predict/batch/upload")
async def batch_predict_upload(
    model_id: str,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    user: User = Depends(get_current_user),
    db=Depends(SessionLocal)
):
    """Upload a CSV and run batch predictions asynchronously."""
    from pathlib import Path
    ext = Path(file.filename).suffix.lower()
    if ext not in {".csv", ".tsv", ".parquet"}:
        raise HTTPException(400, "Batch input must be CSV, TSV, or Parquet.")

    content = await file.read()
    batch_id = str(uuid.uuid4())
    save_path = os.path.join(os.getenv("STORAGE_PATH", "./data_sessions"), "batch", batch_id)
    os.makedirs(save_path, exist_ok=True)
    input_path = os.path.join(save_path, f"input{ext}")
    with open(input_path, "wb") as f:
        f.write(content)

    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(404, "Model not found.")

    bp = BatchPrediction(id=batch_id, model_id=model_id, input_path=input_path, status="queued")
    db.add(bp)
    db.commit()

    background_tasks.add_task(run_batch_prediction, batch_id, model, input_path, ext, db)
    return {"batch_id": batch_id, "status": "queued"}


@app.get("/api/predict/batch/{batch_id}/status")
def batch_status(batch_id: str, user: User = Depends(get_current_user), db=Depends(SessionLocal)):
    bp = db.query(BatchPrediction).filter(BatchPrediction.id == batch_id).first()
    if not bp:
        raise HTTPException(404, "Batch job not found.")
    return {"batch_id": batch_id, "status": bp.status, "row_count": bp.row_count}


@app.get("/api/predict/batch/{batch_id}/download")
def batch_download(batch_id: str, user: User = Depends(get_current_user), db=Depends(SessionLocal)):
    bp = db.query(BatchPrediction).filter(BatchPrediction.id == batch_id).first()
    if not bp or bp.status != "completed":
        raise HTTPException(404, "Batch results not ready.")
    return FileResponse(bp.output_path, media_type="text/csv", filename=f"predictions_{batch_id}.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Deployment + A/B Testing
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/deploy/{model_id}")
def deploy_model(model_id: str, body: DeployRequest,
                 user: User = Depends(get_current_user), db=Depends(SessionLocal)):
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(404, "Model not found.")

    framework = model.framework or "sklearn"
    target, result = body.target, {}
    try:
        bundle = prepare_model_bundle(model_id, model.artifact_path, framework)
    except FileNotFoundError as exc:
        raise HTTPException(500, str(exc))

    if target in ("fastapi", "rest"):
        server_path = build_fastapi_server(
            model_id,
            model.artifact_path,
            model.model_type,
            framework,
            bundled_model_name=bundle["name"],
        )
        result = {"server_path": server_path, "artifact_name": bundle["name"], "target": "fastapi"}
    elif target == "docker":
        build_fastapi_server(
            model_id,
            model.artifact_path,
            model.model_type,
            framework,
            bundled_model_name=bundle["name"],
        )
        docker_path = build_dockerfile(model_id, framework, bundled_model_name=bundle["name"])
        result = {"dockerfile_path": docker_path, "artifact_name": bundle["name"], "target": "docker"}
    elif target == "onnx":
        result = export_onnx(model_id, model.artifact_path, framework)
        if result.get("error"):
            raise HTTPException(400, result["error"])
    elif target == "azure":
        try:
            result = deploy_azure_endpoint(
                model_id,
                model.artifact_path,
                model.model_type,
                framework,
                body.config,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except RuntimeError as exc:
            raise HTTPException(500, str(exc))
    elif target in ("aws", "gcp"):
        result = export_cloud(model_id, model.artifact_path, target, body.config)
        if result.get("error"):
            raise HTTPException(400, result["error"])
    elif target == "download":
        result = {
            "artifact_name": bundle["name"],
            "download_endpoint": f"/api/deploy/model/{model_id}",
            "target": "download",
        }
    elif target == "render_free":
        result = build_render_blueprint(model_id, body.config)
    else:
        raise HTTPException(400, f"Unknown deployment target: {target}")

    dep = Deployment(
        model_id=model_id,
        target=target,
        status="completed",
        endpoint_url=result.get("endpoint_url"),
        config_json=body.config,
        traffic_pct=body.traffic_pct,
    )
    db.add(dep)
    db.commit()
    return {"deployment_id": dep.id, "target": target, "result": result}


@app.post("/api/deploy/ab")
def ab_deploy(body: ABDeployRequest, user: User = Depends(get_current_user), db=Depends(SessionLocal)):
    """Set up an A/B traffic split between two deployed models."""
    if body.traffic_pct_a < 0 or body.traffic_pct_a > 100:
        raise HTTPException(400, "traffic_pct_a must be between 0 and 100.")

    for model_id, pct in [(body.model_id_a, body.traffic_pct_a),
                           (body.model_id_b, 100 - body.traffic_pct_a)]:
        dep = Deployment(model_id=model_id, target="ab_test",
                         status="active", traffic_pct=pct,
                         config_json={"ab_pair": [body.model_id_a, body.model_id_b]})
        db.add(dep)
    db.commit()
    return {
        "status": "ab_configured",
        "model_a": {"id": body.model_id_a, "traffic_pct": body.traffic_pct_a},
        "model_b": {"id": body.model_id_b, "traffic_pct": 100 - body.traffic_pct_a},
    }


@app.get("/api/deploy/fastapi/{model_id}")
def download_server(model_id: str, user: User = Depends(get_current_user)):
    path = os.path.join(os.getenv("OUTPUT_PATH", "./outputs"), f"server_{model_id}.py")
    if not os.path.exists(path):
        raise HTTPException(404, "Server file not generated yet.")
    return FileResponse(path, media_type="text/plain", filename=f"server_{model_id}.py")


@app.get("/api/deploy/dockerfile/{model_id}")
def download_dockerfile(model_id: str, user: User = Depends(get_current_user)):
    path = os.path.join(os.getenv("OUTPUT_PATH", "./outputs"), "Dockerfile")
    if not os.path.exists(path):
        raise HTTPException(404, "Dockerfile not generated yet.")
    return FileResponse(path, media_type="text/plain", filename="Dockerfile")


@app.get("/api/deploy/render-blueprint/{model_id}")
def download_render_blueprint(model_id: str, user: User = Depends(get_current_user)):
    path = os.path.join(os.getenv("OUTPUT_PATH", "./outputs"), f"render_{model_id}.yaml")
    if not os.path.exists(path):
        raise HTTPException(404, "Render blueprint not generated yet.")
    return FileResponse(path, media_type="text/yaml", filename=f"render_{model_id}.yaml")


@app.get("/api/deploy/model/{model_id}")
def download_model(model_id: str, user: User = Depends(get_current_user), db=Depends(SessionLocal)):
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model or not model.artifact_path or not os.path.exists(model.artifact_path):
        raise HTTPException(404, "Model artifact not found.")

    import shutil
    from pathlib import Path

    framework = model.framework or "sklearn"
    bundle = prepare_model_bundle(model_id, model.artifact_path, framework)
    bundle_path = Path(bundle["path"])

    if bundle_path.is_dir():
        archive_base = os.path.join(os.getenv("OUTPUT_PATH", "./outputs"), bundle["name"])
        archive_path = shutil.make_archive(archive_base, "zip", bundle_path.parent, bundle_path.name)
        return FileResponse(archive_path, filename=f"{bundle['name']}.zip")

    return FileResponse(str(bundle_path), filename=bundle["name"])


# ─────────────────────────────────────────────────────────────────────────────
# Data Drift Monitoring
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/drift/check")
def drift_check(body: DriftCheckRequest, user: User = Depends(get_current_user), db=Depends(SessionLocal)):
    """Compare recent prediction inputs against training distribution."""
    model = db.query(Model).filter(Model.id == body.model_id).first()
    if not model:
        raise HTTPException(404, "Model not found.")

    job = db.query(TrainingJob).filter(TrainingJob.id == model.job_id).first()
    session = db.query(Session).filter(Session.id == job.session_id).first() if job else None
    if not session:
        raise HTTPException(404, "Training session not found for this model.")

    drift_results = compute_drift(session.dataset_path, body.sample_data, model.id)

    # Persist alerts for features that exceeded threshold
    alerts = []
    for feat, info in drift_results.items():
        if info.get("drifted"):
            alert = DriftAlert(
                model_id=body.model_id,
                feature=feat,
                drift_score=info.get("score", 0),
                alert_type=info.get("method", "psi"),
            )
            db.add(alert)
            alerts.append(feat)

    db.commit()
    return {"model_id": body.model_id, "drift_results": drift_results,
            "drifted_features": alerts, "alert_count": len(alerts)}


@app.get("/api/drift/alerts/{model_id}")
def drift_alerts(model_id: str, user: User = Depends(get_current_user), db=Depends(SessionLocal)):
    alerts = (db.query(DriftAlert)
              .filter(DriftAlert.model_id == model_id, DriftAlert.resolved == False)
              .order_by(DriftAlert.created_at.desc()).all())
    return [
        {"id": a.id, "feature": a.feature, "score": a.drift_score,
         "type": a.alert_type, "created_at": str(a.created_at)}
        for a in alerts
    ]


@app.patch("/api/drift/alerts/{alert_id}/resolve")
def resolve_alert(alert_id: str, user: User = Depends(get_current_user), db=Depends(SessionLocal)):
    alert = db.query(DriftAlert).filter(DriftAlert.id == alert_id).first()
    if not alert:
        raise HTTPException(404, "Alert not found.")
    alert.resolved = True
    db.commit()
    return {"status": "resolved"}


# ─────────────────────────────────────────────────────────────────────────────
# Scheduled Retraining
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/schedule")
def create_schedule(body: ScheduleRequest,
                    user: User = Depends(get_current_user), db=Depends(SessionLocal)):
    from croniter import croniter
    from datetime import datetime
    if not croniter.is_valid(body.cron_expr):
        raise HTTPException(400, f"Invalid cron expression: {body.cron_expr}")

    cron = croniter(body.cron_expr, datetime.utcnow())
    next_run = cron.get_next(datetime)

    sched = ScheduledJob(
        session_id=body.session_id,
        cron_expr=body.cron_expr,
        config_json=body.config,
        next_run_at=next_run,
    )
    db.add(sched)
    db.commit()
    return {"schedule_id": sched.id, "next_run_at": str(next_run), "cron": body.cron_expr}


@app.get("/api/schedule")
def list_schedules(user: User = Depends(get_current_user), db=Depends(SessionLocal)):
    sessions = db.query(Session).filter(Session.user_id == user.id).all()
    session_ids = [s.id for s in sessions]
    scheds = db.query(ScheduledJob).filter(
        ScheduledJob.session_id.in_(session_ids),
        ScheduledJob.is_active == True
    ).all()
    return [
        {"id": s.id, "session_id": s.session_id, "cron_expr": s.cron_expr,
         "next_run_at": str(s.next_run_at), "last_run_at": str(s.last_run_at)}
        for s in scheds
    ]


@app.delete("/api/schedule/{schedule_id}")
def delete_schedule(schedule_id: str, user: User = Depends(get_current_user), db=Depends(SessionLocal)):
    sched = db.query(ScheduledJob).filter(ScheduledJob.id == schedule_id).first()
    if not sched:
        raise HTTPException(404, "Schedule not found.")
    sched.is_active = False
    db.commit()
    return {"status": "cancelled"}


# ─────────────────────────────────────────────────────────────────────────────
# AI Co-Pilot  (history restored from DB on session resume)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/copilot/chat")
async def copilot_chat(body: CopilotRequest,
                       user: User = Depends(get_current_user),
                       db=Depends(SessionLocal)):
    redis_key = f"copilot:{user.id}:{body.session_id}"

    # Try to restore history from Redis first, then DB
    cached = redis_get(redis_key)
    if cached:
        pilot = CoPilot(body.session_id, user.mode, user.domain)
        pilot.history = cached.get("history", [])
    else:
        pilot = CoPilot(body.session_id, user.mode, user.domain)
        # Restore from DB (fixes lost history on restart)
        db_msgs = (db.query(CopilotMessage)
                   .filter(CopilotMessage.session_id == body.session_id)
                   .order_by(CopilotMessage.created_at).all())
        pilot.history = [{"role": m.role, "content": m.content} for m in db_msgs]

    session = db.query(Session).filter(Session.id == body.session_id).first()
    jobs = db.query(TrainingJob).filter(TrainingJob.session_id == body.session_id).all()

    pipeline_context = {
        "status": session.status if session else "unknown",
        "model_type": "ml",
        "profile": session.profile_json if session else {},
        "selected_models": [],
        "best_model": {},
    }
    for job in jobs:
        pipeline_context["model_type"] = job.model_type
        if job.best_model_id:
            bm = db.query(Model).filter(Model.id == job.best_model_id).first()
            if bm:
                pipeline_context["best_model"] = {
                    "algorithm": bm.algorithm,
                    "score": list((bm.metrics_json or {}).values())[0] if bm.metrics_json else "N/A",
                }

    reply = await pilot.chat(body.message, pipeline_context, db)

    # Persist in DB
    db.add_all([
        CopilotMessage(session_id=body.session_id, role="user", content=body.message),
        CopilotMessage(session_id=body.session_id, role="assistant", content=reply),
    ])
    db.commit()

    # Update Redis cache
    pilot.history.extend([
        {"role": "user", "content": body.message},
        {"role": "assistant", "content": reply},
    ])
    redis_set(redis_key, {"history": pilot.history[-40:]}, ttl=86400)  # keep last 20 exchanges

    return {"reply": reply}


@app.get("/api/copilot/history/{session_id}")
def copilot_history(session_id: str, user: User = Depends(get_current_user), db=Depends(SessionLocal)):
    msgs = (db.query(CopilotMessage)
            .filter(CopilotMessage.session_id == session_id)
            .order_by(CopilotMessage.created_at).all())
    return [{"role": m.role, "content": m.content} for m in msgs]


@app.delete("/api/copilot/reset/{session_id}")
def copilot_reset(session_id: str, user: User = Depends(get_current_user), db=Depends(SessionLocal)):
    redis_delete(f"copilot:{user.id}:{session_id}")
    db.query(CopilotMessage).filter(CopilotMessage.session_id == session_id).delete()
    db.commit()
    return {"status": "reset"}


# ─────────────────────────────────────────────────────────────────────────────
# Reports  (async background task — no more blocking the request)
# ─────────────────────────────────────────────────────────────────────────────

_report_status: dict[str, str] = {}   # job_id -> "generating" | "done" | "error"
_report_ids: dict[str, str] = {}      # job_id -> report_id

@app.post("/api/report/generate/{job_id}")
async def report_generate(job_id: str,
                           background_tasks: BackgroundTasks,
                           user: User = Depends(get_current_user),
                           db=Depends(SessionLocal)):
    existing = (db.query(Report)
                  .filter(Report.job_id == job_id)
                  .order_by(Report.created_at.desc())
                  .first())
    if existing:
        _report_status[job_id] = "done"
        _report_ids[job_id] = existing.id
        return {
            "status": "done",
            "message": "Report already exists.",
            "report_id": existing.id,
        }

    if _report_status.get(job_id) == "generating":
        return {
            "status": "generating",
            "message": "Report is already being generated.",
            "report_id": _report_ids.get(job_id),
        }
    _report_status[job_id] = "generating"

    async def _gen():
        try:
            result = await generate_report(job_id, db)
            _report_status[job_id] = "done"
            if result.get("report_id"):
                _report_ids[job_id] = result["report_id"]
        except Exception as e:
            logger.exception(f"Report generation failed: {e}")
            _report_status[job_id] = "error"

    background_tasks.add_task(_gen)
    return {
        "status": "generating",
        "message": "Report generation started. Poll /api/report/status/{job_id}.",
        "report_id": _report_ids.get(job_id),
    }


@app.get("/api/report/status/{job_id}")
def report_status(job_id: str, user: User = Depends(get_current_user), db=Depends(SessionLocal)):
    status = _report_status.get(job_id, "not_started")
    report = (db.query(Report)
                .filter(Report.job_id == job_id)
                .order_by(Report.created_at.desc())
                .first())
    report_id = _report_ids.get(job_id) or (report.id if report else None)
    if report_id:
        _report_ids[job_id] = report_id
    if report and status == "not_started":
        status = "done"
        _report_status[job_id] = status
    return {"job_id": job_id, "status": status, "report_id": report_id}


@app.get("/api/report/download/{report_id}")
def report_download(report_id: str, fmt: str = "pdf",
                    user: User = Depends(get_current_user), db=Depends(SessionLocal)):
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(404, "Report not found.")
    path = report.pdf_path if fmt == "pdf" else report.docx_path
    if not path or not os.path.exists(path):
        raise HTTPException(404, "Report file not found.")
    media = "application/pdf" if fmt == "pdf" else \
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    return FileResponse(path, media_type=media, filename=f"report_{report_id}.{fmt}")


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health(db=Depends(SessionLocal)):
    redis_ok = get_redis() is not None
    try:
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    return {
        "status": "healthy" if db_ok else "degraded",
        "version": "3.0.0",
        "db": "ok" if db_ok else "error",
        "redis": "ok" if redis_ok else "unavailable",
        "running_jobs": len(_running_jobs),
        "max_jobs": MAX_CONCURRENT_JOBS,
    }
