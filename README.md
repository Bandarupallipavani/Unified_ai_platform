# Unified AI Platform

End-to-end training, tuning, testing, and deployment platform for Classic ML, Deep Learning, and NLP/LLM models.

---

## Features

### Training Domains
| Domain | Models |
|--------|--------|
| **Classic ML** | XGBoost, LightGBM, Random Forest, Gradient Boosting, Logistic Regression, SVM, MLP, AdaBoost, Ridge, Lasso, ElasticNet |
| **Deep Learning** | PyTorch MLP, LSTM, GRU, 1D-CNN, TabNet, ResNet-50, EfficientNet-B3, Vision Transformer (ViT) |
| **NLP / LLM** | BERT-base, RoBERTa-large, DistilBERT, GPT-2, T5-small, XLM-RoBERTa |

### Automated Pipeline
```text
Data Upload -> EDA -> Preprocessing -> HPO -> Train -> Evaluate -> Test/Predict -> SHAP -> Report -> Deploy
```

### Key Capabilities
- Live training logs streamed via WebSocket
- Bayesian HPO with Optuna (ML) plus epoch-level training for DL/NLP
- Post-training prediction testing with manual sample inputs and batch file uploads
- SHAP explainability for ML models
- Auto-generated PDF and Word reports with Claude AI narratives
- AI Co-Pilot powered by Claude for context-aware pipeline guidance
- Natural language data connectors for describing data sources in plain English
- Six deployment targets: REST API, Docker, AWS SageMaker, GCP Vertex AI, Azure ML, ONNX

### Post-training Testing
- Open the Results page after a training job completes.
- Run a single prediction with custom JSON inputs to validate the trained model quickly.
- Use the Batch page to upload CSV, TSV, or Parquet files and test the model on many new inputs at once.

---

## Quick Start

### Option 1 - Docker Compose
```bash
git clone <repo>
cd unified-ai-platform
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
docker-compose -f docker/docker-compose.yml up
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API docs: http://localhost:8000/docs
- MLflow: http://localhost:5000

### Option 2 - Local Development

Backend:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp ../.env.example .env
# Edit .env
uvicorn main:app --reload --port 8000
```

Frontend:
```bash
cd frontend
npm install
npm start
```

---

## Project Structure

```text
unified-ai-platform/
|-- backend/
|   |-- main.py                 # FastAPI app + REST + WebSocket endpoints
|   |-- db.py                   # SQLAlchemy engine + session
|   |-- models_db.py            # ORM models
|   |-- auth.py                 # JWT auth + bcrypt
|   |-- requirements.txt
|   |-- Dockerfile
|   |-- pipeline/
|   |   |-- ingest.py           # File upload + NL data connector
|   |   |-- eda.py              # Automated EDA
|   |   |-- preprocess.py       # Preprocessing pipeline
|   |   |-- train.py            # Unified ML + DL + NLP training engine
|   |   `-- deploy.py           # Deployment helpers + prediction utilities
|   `-- ai/
|       |-- copilot.py          # Claude-powered AI Co-Pilot
|       |-- nl_connector.py     # NL to database/API connector
|       `-- report.py           # PDF + Word report generator
|
|-- frontend/
|   |-- package.json
|   |-- tailwind.config.js
|   |-- Dockerfile
|   |-- nginx.conf
|   `-- src/
|       |-- App.js              # Router + app shell
|       |-- api.js              # API client + WS helper
|       |-- constants/
|       |   `-- models.js       # Model, deploy, and pipeline definitions
|       |-- pages/
|       |   |-- LoginPage.jsx
|       |   |-- WizardPage.jsx
|       |   |-- ResultsPage.jsx
|       |   |-- BatchPredictPage.jsx
|       |   |-- DeployPage.jsx
|       |   `-- HistoryPage.jsx
|       |-- components/
|       |   `-- Layout.jsx
|       `-- copilot/
|           `-- CoPilotPanel.jsx
|
`-- docker/
    `-- docker-compose.yml
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/register` | Register new user |
| POST | `/api/auth/login` | Login and get JWT |
| POST | `/api/upload` | Upload dataset |
| POST | `/api/connect` | Natural-language data connector |
| GET | `/api/eda/{session_id}` | Run EDA |
| GET | `/api/preprocess/plan/{session_id}` | Get preprocessing plan |
| POST | `/api/preprocess` | Apply preprocessing |
| POST | `/api/train` | Start training job |
| WS | `/ws/train/{job_id}` | Live training log stream |
| GET | `/api/train/status/{job_id}` | Job status |
| GET | `/api/train/results/{job_id}` | All model results |
| GET | `/api/train/best/{job_id}` | Best model info |
| POST | `/api/predict/{model_id}` | Run a single prediction |
| POST | `/api/predict/batch/upload` | Start a batch prediction job |
| POST | `/api/deploy/{model_id}` | Deploy model |
| POST | `/api/copilot/chat` | Chat with AI Co-Pilot |
| POST | `/api/report/generate/{job_id}` | Generate PDF and Word report |

---

## Training Config Reference

```json
{
  "model_type": "ml",
  "models": ["xgboost", "lightgbm"],
  "n_trials": 20,
  "time_budget_seconds": 300,
  "cv_folds": 5,
  "target_column": "",
  "epochs": 10,
  "batch_size": 32,
  "learning_rate": 0.0002,
  "max_length": 256,
  "text_column": "",
  "enable_ensemble": false,
  "enable_mlflow": false
}
```

---

## Deployment Targets

| Target | Description |
|--------|-------------|
| `rest` | Auto-generated FastAPI server |
| `docker` | Dockerfile plus server script |
| `aws` | AWS SageMaker deploy script |
| `gcp` | GCP Vertex AI deploy script |
| `azure` | Azure ML deploy script |
| `onnx` | ONNX model export |
| `download` | Raw model artifact (`.pkl` / `.pt`) |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Claude API key for Co-Pilot and reports |
| `SECRET_KEY` | Yes | JWT signing secret |
| `DATABASE_URL` | Optional | Defaults to SQLite |
| `STORAGE_PATH` | Optional | Dataset storage path |
| `REPORTS_PATH` | Optional | Generated reports path |
| `OUTPUT_PATH` | Optional | Deployment artifact path |
