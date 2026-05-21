/**
 * Unified model registry — ML, Deep Learning, NLP/LLM
 */

export const MODEL_TYPES = [
  { id: "ml",  label: "Classic ML",    icon: "📊", sub: "Tabular / structured data" },
  { id: "dl",  label: "Deep Learning", icon: "🧠", sub: "Neural networks / vision / sequences" },
  { id: "nlp", label: "NLP / LLM",     icon: "💬", sub: "Text, language, transformers" },
];

export const ALL_MODELS = [
  // ── Classic ML ──────────────────────────────────────────────────────────────
  { id: "xgboost",             name: "XGBoost",              type: "ml",  tag: "ML",  desc: "Gradient boosting — fast, accurate, great default" },
  { id: "lightgbm",            name: "LightGBM",             type: "ml",  tag: "ML",  desc: "Microsoft boosting — excellent on large datasets" },
  { id: "random_forest",       name: "Random Forest",        type: "ml",  tag: "ML",  desc: "Ensemble of decision trees — robust & interpretable" },
  { id: "gradient_boosting",   name: "Gradient Boosting",    type: "ml",  tag: "ML",  desc: "sklearn GBM — stable, no external deps" },
  { id: "logistic_regression", name: "Logistic Regression",  type: "ml",  tag: "ML",  desc: "Linear baseline classifier — fast & interpretable" },
  { id: "svm",                 name: "SVM",                  type: "ml",  tag: "ML",  desc: "Support Vector Machine — strong on small datasets" },
  { id: "mlp",                 name: "MLP (sklearn)",         type: "ml",  tag: "ML",  desc: "Shallow neural net — good tabular baseline" },
  { id: "adaboost",            name: "AdaBoost",             type: "ml",  tag: "ML",  desc: "Adaptive boosting — simple ensemble" },
  { id: "ridge",               name: "Ridge Regression",     type: "ml",  tag: "ML",  desc: "L2-regularised linear regression" },
  { id: "lasso",               name: "Lasso Regression",     type: "ml",  tag: "ML",  desc: "L1-regularised — auto feature selection" },
  { id: "elasticnet",          name: "ElasticNet",           type: "ml",  tag: "ML",  desc: "L1+L2 combined — best of ridge and lasso" },

  // ── Deep Learning ───────────────────────────────────────────────────────────
  { id: "mlp_torch",  name: "MLP (PyTorch)",      type: "dl", tag: "DL", desc: "Deep feedforward net — configurable layers" },
  { id: "lstm",       name: "LSTM",               type: "dl", tag: "DL", desc: "Long short-term memory — sequences & time series" },
  { id: "gru",        name: "GRU",                type: "dl", tag: "DL", desc: "Gated recurrent unit — faster than LSTM" },
  { id: "cnn1d",      name: "1D-CNN",             type: "dl", tag: "DL", desc: "1D convnet — great for sensor/signal data" },
  { id: "tabnet",     name: "TabNet",             type: "dl", tag: "DL", desc: "Attention-based tabular model — interpretable DL" },
  { id: "resnet",     name: "ResNet-50",          type: "dl", tag: "CV", desc: "Deep residual network — image classification" },
  { id: "efficientnet", name: "EfficientNet-B3",  type: "dl", tag: "CV", desc: "Scalable CNN — SOTA accuracy/compute trade-off" },
  { id: "vit",        name: "Vision Transformer", type: "dl", tag: "CV", desc: "ViT-B/16 — attention-based image model" },

  // ── NLP / LLM ───────────────────────────────────────────────────────────────
  { id: "bert",       name: "BERT-base",          type: "nlp", tag: "NLP", desc: "Bidirectional encoder — fine-tune for classification" },
  { id: "roberta",    name: "RoBERTa-large",       type: "nlp", tag: "NLP", desc: "Optimised BERT — stronger pretraining, higher accuracy" },
  { id: "distilbert", name: "DistilBERT",          type: "nlp", tag: "NLP", desc: "Lightweight BERT — 60% faster, 97% accuracy" },
  { id: "gpt2",       name: "GPT-2 (fine-tune)",   type: "nlp", tag: "NLP", desc: "Generative pretrained transformer — text tasks" },
  { id: "t5",         name: "T5-small",            type: "nlp", tag: "NLP", desc: "Text-to-text transfer transformer — versatile" },
  { id: "xlm",        name: "XLM-RoBERTa",        type: "nlp", tag: "NLP", desc: "Multilingual model — 100 languages" },
];

export const DEPLOY_TARGETS = [
  { id: "render_free", name: "Render Full Stack", icon: "FREE", desc: "Free public frontend + backend links on Render" },
  { id: "rest",      name: "REST API",       icon: "🌐", desc: "FastAPI server, any cloud or on-prem" },
  { id: "docker",    name: "Docker",         icon: "🐳", desc: "Containerised & portable image" },
  { id: "aws",       name: "AWS SageMaker",  icon: "☁️", desc: "Managed ML endpoint on AWS" },
  { id: "gcp",       name: "GCP Vertex AI",  icon: "🔵", desc: "Google Cloud ML platform" },
  { id: "azure",     name: "Azure ML",       icon: "🟦", desc: "Microsoft Azure ML service" },
  { id: "onnx",      name: "ONNX Export",    icon: "📦", desc: "Cross-platform inference runtime" },
  { id: "download",  name: "Download Model", icon: "⬇️", desc: "Download .pkl / .pt file" },
];

export const PIPELINE_STEPS = [
  { id: 1, label: "Data",       icon: "🗄️" },
  { id: 2, label: "Models",     icon: "🧩" },
  { id: 3, label: "Configure",  icon: "⚙️" },
  { id: 4, label: "Train",      icon: "🚀" },
  { id: 5, label: "Results",    icon: "📈" },
  { id: 6, label: "Deploy",     icon: "🌐" },
];

export const TAG_COLORS = {
  ML:  { bg: "#E1F5EE", text: "#0F6E56" },
  DL:  { bg: "#EEEDFE", text: "#534AB7" },
  CV:  { bg: "#FAECE7", text: "#993C1D" },
  NLP: { bg: "#FAEEDA", text: "#854F0B" },
};
