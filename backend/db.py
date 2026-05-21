"""
db.py — Database + Redis configuration
PostgreSQL (primary), Redis (job queue / copilot session cache)
"""
import os
import json
import redis
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")
load_dotenv(Path(__file__).resolve().parent / ".env")

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./automl.db")

# Connection pooling for PostgreSQL; SQLite fallback for dev
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}
    engine = create_engine(DATABASE_URL, connect_args=connect_args)
else:
    engine = create_engine(
        DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,          # auto-reconnect on stale connections
        pool_recycle=3600,
    )

SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def SessionLocal():
    """FastAPI dependency — yields a DB session."""
    db = SessionFactory()
    try:
        yield db
    finally:
        db.close()


# ── Redis client (optional — falls back gracefully if not available) ──────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

def get_redis() -> redis.Redis | None:
    """Return a Redis client, or None if Redis is unavailable."""
    try:
        r = redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=2)
        r.ping()
        return r
    except Exception:
        return None


# Redis-backed key-value helpers used across the app
def redis_set(key: str, value: dict, ttl: int = 86400):
    r = get_redis()
    if r:
        r.setex(key, ttl, json.dumps(value))


def redis_get(key: str) -> dict | None:
    r = get_redis()
    if r:
        raw = r.get(key)
        if raw:
            return json.loads(raw)
    return None


def redis_delete(key: str):
    r = get_redis()
    if r:
        r.delete(key)
