.PHONY: help install dev docker-up docker-down migrate test lint

help:
	@echo "Unified AI Platform — Dev Commands"
	@echo ""
	@echo "  make install      Install all backend + frontend deps"
	@echo "  make dev          Run backend + frontend in dev mode"
	@echo "  make docker-up    Start full stack via Docker Compose"
	@echo "  make docker-down  Stop Docker Compose stack"
	@echo "  make migrate      Run Alembic DB migrations"
	@echo "  make test         Run backend tests"
	@echo "  make lint         Lint backend Python files"

install:
	cd backend && pip install -r requirements.txt
	cd frontend && npm install

dev:
	@echo "Starting backend and frontend..."
	cd backend && uvicorn main:app --reload --port 8000 &
	cd frontend && npm start

docker-up:
	docker-compose -f docker/docker-compose.yml up --build

docker-down:
	docker-compose -f docker/docker-compose.yml down

migrate:
	cd backend && alembic upgrade head

migrate-create:
	cd backend && alembic revision --autogenerate -m "$(MSG)"

test:
	cd backend && python -m pytest tests/ -v

lint:
	cd backend && ruff check . || flake8 .

seed-db:
	cd backend && python -c "
from db import engine, Base
from models_db import *
Base.metadata.create_all(bind=engine)
print('Tables created.')
"
