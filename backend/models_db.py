"""
models_db.py — SQLAlchemy ORM models
New in v3: ModelVersion, DriftAlert, ScheduledJob, BatchPrediction
"""
import uuid
from datetime import datetime
from sqlalchemy import (Column, String, DateTime, JSON, Text, ForeignKey,
                        Boolean, Integer, Float)
from sqlalchemy.orm import relationship
from db import Base


def _uuid():
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=_uuid)
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    mode = Column(String, default="beginner")       # beginner | expert
    domain = Column(String, default="general")      # healthcare | finance | hr | retail
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    sessions = relationship("Session", back_populates="user")


class Session(Base):
    __tablename__ = "sessions"
    id = Column(String, primary_key=True, default=_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    dataset_path = Column(String)
    profile_json = Column(JSON)
    status = Column(String, default="created")      # created|profiled|preprocessed|trained
    domain_mode = Column(String, default="general")
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="sessions")
    jobs = relationship("TrainingJob", back_populates="session")
    messages = relationship("CopilotMessage", back_populates="session")


class TrainingJob(Base):
    __tablename__ = "training_jobs"
    id = Column(String, primary_key=True, default=_uuid)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    config_json = Column(JSON)
    model_type = Column(String, default="ml")       # ml | dl | nlp
    status = Column(String, default="queued")       # queued|running|completed|failed
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    best_model_id = Column(String, ForeignKey("models.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="jobs")
    models = relationship("Model", foreign_keys="[Model.job_id]", back_populates="job")
    reports = relationship("Report", back_populates="job")


class Model(Base):
    __tablename__ = "models"
    id = Column(String, primary_key=True, default=_uuid)
    job_id = Column(String, ForeignKey("training_jobs.id"), nullable=False)
    algorithm = Column(String)
    model_type = Column(String, default="ml")       # ml | dl | nlp
    framework = Column(String, default="sklearn")   # sklearn | pytorch | transformers
    hyperparams_json = Column(JSON)
    metrics_json = Column(JSON)
    shap_json = Column(JSON)
    artifact_path = Column(String)
    is_production = Column(Boolean, default=False)  # promoted to production
    created_at = Column(DateTime, default=datetime.utcnow)

    job = relationship("TrainingJob", foreign_keys=[job_id], back_populates="models")
    deployments = relationship("Deployment", back_populates="model")
    versions = relationship("ModelVersion", back_populates="model")
    drift_alerts = relationship("DriftAlert", back_populates="model")
    batch_preds = relationship("BatchPrediction", back_populates="model")


class ModelVersion(Base):
    """Tracks explicit versions of a model for rollback / A-B testing."""
    __tablename__ = "model_versions"
    id = Column(String, primary_key=True, default=_uuid)
    model_id = Column(String, ForeignKey("models.id"), nullable=False)
    version = Column(Integer, default=1)
    metrics_json = Column(JSON)
    artifact_path = Column(String)
    notes = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    model = relationship("Model", back_populates="versions")


class Report(Base):
    __tablename__ = "reports"
    id = Column(String, primary_key=True, default=_uuid)
    job_id = Column(String, ForeignKey("training_jobs.id"), nullable=False)
    pdf_path = Column(String)
    docx_path = Column(String)
    report_type = Column(String, default="full")
    created_at = Column(DateTime, default=datetime.utcnow)

    job = relationship("TrainingJob", back_populates="reports")


class CopilotMessage(Base):
    __tablename__ = "copilot_messages"
    id = Column(String, primary_key=True, default=_uuid)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    role = Column(String)       # user | assistant
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="messages")


class Deployment(Base):
    __tablename__ = "deployments"
    id = Column(String, primary_key=True, default=_uuid)
    model_id = Column(String, ForeignKey("models.id"), nullable=False)
    target = Column(String)     # fastapi | docker | aws | gcp | azure | onnx
    status = Column(String, default="pending")
    endpoint_url = Column(String)
    config_json = Column(JSON)
    traffic_pct = Column(Integer, default=100)      # for A/B testing
    created_at = Column(DateTime, default=datetime.utcnow)

    model = relationship("Model", back_populates="deployments")


class DriftAlert(Base):
    """Records data drift events detected post-deployment."""
    __tablename__ = "drift_alerts"
    id = Column(String, primary_key=True, default=_uuid)
    model_id = Column(String, ForeignKey("models.id"), nullable=False)
    feature = Column(String)
    drift_score = Column(Float)
    threshold = Column(Float, default=0.1)
    alert_type = Column(String, default="psi")      # psi | ks | js
    resolved = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    model = relationship("Model", back_populates="drift_alerts")


class ScheduledJob(Base):
    """Recurring training schedules."""
    __tablename__ = "scheduled_jobs"
    id = Column(String, primary_key=True, default=_uuid)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    cron_expr = Column(String)   # e.g. "0 2 * * *" = daily at 2am
    config_json = Column(JSON)
    last_run_at = Column(DateTime)
    next_run_at = Column(DateTime)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class BatchPrediction(Base):
    """Tracks batch prediction jobs."""
    __tablename__ = "batch_predictions"
    id = Column(String, primary_key=True, default=_uuid)
    model_id = Column(String, ForeignKey("models.id"), nullable=False)
    input_path = Column(String)
    output_path = Column(String)
    status = Column(String, default="queued")        # queued|running|completed|failed
    row_count = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)

    model = relationship("Model", back_populates="batch_preds")
