"""
Basic API tests for Unified AI Platform backend.
Run with: pytest tests/ -v
"""
import io
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Use in-memory SQLite for tests
TEST_DB_URL = "sqlite:///./test_automl.db"

import os
os.environ["DATABASE_URL"] = TEST_DB_URL
os.environ["ANTHROPIC_API_KEY"] = "test-key"
os.environ["SECRET_KEY"] = "test-secret"

from db import Base
from main import app, SessionLocal as _SessionLocal

engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)


def override_db():
    db = TestingSession()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[_SessionLocal] = override_db
client = TestClient(app)


# ── Auth ──────────────────────────────────────────────────────────────────────

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"


def test_register_and_login():
    res = client.post("/api/auth/register", json={
        "email": "test@example.com",
        "password": "password123",
        "mode": "beginner",
        "domain": "general",
    })
    assert res.status_code == 200
    token = res.json()["access_token"]
    assert token

    res2 = client.post("/api/auth/login", json={
        "email": "test@example.com",
        "password": "password123",
    })
    assert res2.status_code == 200
    return res2.json()["access_token"]


def test_duplicate_register():
    client.post("/api/auth/register", json={
        "email": "dup@example.com", "password": "pass", "mode": "beginner", "domain": "general"
    })
    res = client.post("/api/auth/register", json={
        "email": "dup@example.com", "password": "pass", "mode": "beginner", "domain": "general"
    })
    assert res.status_code == 400


def test_me_endpoint():
    token = test_register_and_login()
    res = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    assert res.json()["email"] == "test@example.com"


# ── Upload ────────────────────────────────────────────────────────────────────

def get_token():
    res = client.post("/api/auth/register", json={
        "email": f"u{os.urandom(4).hex()}@test.com",
        "password": "pass123", "mode": "beginner", "domain": "general",
    })
    return res.json()["access_token"]


def test_upload_csv():
    token = get_token()
    csv_data = "col1,col2,target\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n"
    res = client.post(
        "/api/upload",
        files={"file": ("test.csv", io.BytesIO(csv_data.encode()), "text/csv")},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "session_id" in data
    assert "profile" in data
    assert data["profile"]["shape"]["rows"] == 4
    return data["session_id"], token


def test_upload_invalid_extension():
    token = get_token()
    res = client.post(
        "/api/upload",
        files={"file": ("test.xyz", io.BytesIO(b"data"), "application/octet-stream")},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 400


def test_get_profile():
    session_id, token = test_upload_csv()
    res = client.get(f"/api/profile/{session_id}", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    assert "shape" in res.json()


def test_eda():
    session_id, token = test_upload_csv()
    res = client.get(f"/api/eda/{session_id}", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    data = res.json()
    assert "null_analysis" in data
    assert "distributions" in data


def test_preprocess_plan():
    session_id, token = test_upload_csv()
    res = client.get(f"/api/preprocess/plan/{session_id}", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    assert "steps" in res.json()


def test_preprocess():
    session_id, token = test_upload_csv()
    res = client.post("/api/preprocess",
        json={"session_id": session_id, "config": {}},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 200
    assert "preprocessing_log" in res.json()


def test_start_training():
    session_id, token = test_upload_csv()
    client.post("/api/preprocess", json={"session_id": session_id, "config": {}},
                headers={"Authorization": f"Bearer {token}"})
    res = client.post("/api/train",
        json={"session_id": session_id, "config": {
            "model_type": "ml", "models": ["logistic_regression"], "n_trials": 1,
            "time_budget_seconds": 30
        }},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "job_id" in data
    assert data["status"] == "queued"
    return data["job_id"], token


def test_training_history():
    token = get_token()
    res = client.get("/api/train/history", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_unauthorized_access():
    res = client.get("/api/train/history")
    assert res.status_code == 403
