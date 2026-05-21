import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { uploadFile, nlConnect, runPreprocess, startTrain, getTrainStatus } from "../api";
import CoPilotPanel from "../copilot/CoPilotPanel";
import { ALL_MODELS, MODEL_TYPES } from "../constants/models";

const DEFAULT_CONFIG = {
  model_type: "ml",
  models: ["xgboost", "lightgbm", "random_forest"],
  n_trials: 50,
  time_budget_seconds: 300,
  target_column: "",
  cv_folds: 5,
  epochs: 10,
  batch_size: 32,
  learning_rate: 0.0002,
  max_length: 256,
  enable_ensemble: false,
  enable_mlflow: false,
};

export default function ExpertDashboard() {
  const navigate = useNavigate();
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [sessionId, setSessionId] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [nlText, setNlText] = useState("");
  const [trainDone, setTrainDone] = useState(false);

  const update = (k, v) => setConfig(c => ({ ...c, [k]: v }));
  const toggleModel = (m) =>
    setConfig(c => ({
      ...c,
      models: c.models.includes(m) ? c.models.filter(x => x !== m) : [...c.models, m],
    }));

  const addLog = (msg) => setLogs(l => [...l, `[${new Date().toLocaleTimeString()}] ${msg}`]);

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setLoading(true);
    try {
      const res = await uploadFile(file);
      setSessionId(res.data.session_id);
      addLog(`✅ Loaded: ${file.name} — ${res.data.profile.shape.rows.toLocaleString()} rows × ${res.data.profile.shape.columns} cols`);
    } catch (err) {
      setError("Upload failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleNlConnect = async () => {
    setLoading(true);
    try {
      const res = await nlConnect(nlText);
      setSessionId(res.data.session_id);
      addLog(`✅ NL Connected: ${res.data.profile.shape.rows} rows`);
    } catch {
      setError("NL connect failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleRun = async () => {
    if (!sessionId) { setError("Load data first."); return; }
    setLoading(true); setError(""); setTrainDone(false);
    addLog(`⚙️  Preprocessing data...`);
    try {
      const ppRes = await runPreprocess(sessionId, config);
      addLog(`✅ Preprocessing: ${ppRes.data.preprocessing_log.length} steps`);
      addLog(`🚀 Starting ${config.model_type.toUpperCase()} training — ${config.models.length} model(s)...`);

      const trainRes = await startTrain(sessionId, config);
      const jid = trainRes.data.job_id;
      setJobId(jid);
      addLog(`📋 Job: ${jid}`);

      // Poll for training status
      const pollInterval = setInterval(async () => {
        try {
          const status = await getTrainStatus(jid);
          if (status.data.status === "completed" || status.data.status === "failed") {
            clearInterval(pollInterval);
            setTrainDone(true);
            addLog(`✅ Training ${status.data.status}`);
          }
        } catch (e) {
          clearInterval(pollInterval);
        }
      }, 2000);
    } catch (e) {
      setError(e.response?.data?.detail || "Run failed.");
    } finally {
      setLoading(false);
    }
  };

  const modelsByType = ALL_MODELS.filter(m => m.type === config.model_type);

  return (
    <div className="flex h-full gap-6">
      <div className="flex-1 overflow-auto space-y-5">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold text-gray-900">Expert Dashboard</h1>
          <span className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-xs font-medium">Expert Mode</span>
        </div>

        {error && <div className="bg-red-50 border border-red-200 text-red-700 rounded-lg p-3 text-sm">{error}</div>}

        {/* Data source */}
        <div className="bg-white rounded-xl border border-gray-200 p-5">
          <h3 className="font-semibold text-gray-800 mb-4">Data source</h3>
          <div className="flex gap-4">
            <label className="flex-1 cursor-pointer">
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center hover:border-indigo-400 transition-colors">
                <div className="text-2xl mb-1">📂</div>
                <div className="text-sm text-gray-600">Upload file (CSV / JSON / Excel / Parquet)</div>
              </div>
              <input type="file" className="hidden" onChange={handleUpload} accept=".csv,.json,.xlsx,.xls,.parquet" />
            </label>
            <div className="flex-1 flex gap-2">
              <input
                className="flex-1 border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                placeholder="Natural language connector…"
                value={nlText}
                onChange={e => setNlText(e.target.value)}
              />
              <button onClick={handleNlConnect} disabled={loading}
                className="bg-gray-800 text-white px-4 py-2 rounded-lg text-sm hover:bg-gray-700 disabled:opacity-50">
                Connect
              </button>
            </div>
          </div>
          {sessionId && <div className="mt-2 text-xs text-green-600">✅ Session: {sessionId}</div>}
        </div>

        {/* Model type selector */}
        <div className="bg-white rounded-xl border border-gray-200 p-5">
          <h3 className="font-semibold text-gray-800 mb-3">Training domain</h3>
          <div className="flex gap-2 mb-5">
            {MODEL_TYPES.map(t => (
              <button
                key={t.id}
                onClick={() => update("model_type", t.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg border text-sm font-medium transition-colors
                  ${config.model_type === t.id
                    ? "border-indigo-500 bg-indigo-50 text-indigo-700"
                    : "border-gray-200 text-gray-600 hover:bg-gray-50"}`}
              >
                <span>{t.icon}</span><span>{t.label}</span>
              </button>
            ))}
          </div>

          <h3 className="font-semibold text-gray-800 mb-3">
            Models — <span className="text-indigo-600">{config.models.length} selected</span>
          </h3>
          <div className="flex flex-wrap gap-2">
            {modelsByType.map(m => (
              <button
                key={m.id}
                onClick={() => toggleModel(m.id)}
                title={m.desc}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors border
                  ${config.models.includes(m.id)
                    ? "bg-indigo-600 text-white border-indigo-600"
                    : "bg-gray-100 text-gray-600 border-gray-200 hover:bg-gray-200"}`}
              >
                {m.name}
              </button>
            ))}
          </div>
        </div>

        {/* Config in 3 columns */}
        <div className="grid grid-cols-3 gap-5">
          {/* HPO / epochs */}
          <div className="bg-white rounded-xl border border-gray-200 p-5 space-y-4">
            <h3 className="font-semibold text-gray-800 text-sm">
              {config.model_type === "ml" ? "HPO (Optuna)" : "Training params"}
            </h3>
            {config.model_type === "ml" ? (
              <>
                {[
                  { key: "n_trials", label: "Trials", min: 5, max: 200 },
                  { key: "time_budget_seconds", label: "Time budget (s)", min: 60, max: 3600 },
                  { key: "cv_folds", label: "CV folds", min: 2, max: 10 },
                ].map(({ key, label, min, max }) => (
                  <div key={key}>
                    <label className="block text-xs text-gray-500 mb-1">{label}</label>
                    <input type="number" min={min} max={max}
                      className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500"
                      value={config[key]}
                      onChange={e => update(key, +e.target.value)} />
                  </div>
                ))}
              </>
            ) : (
              <>
                {[
                  { key: "epochs", label: "Epochs", min: 1, max: 100 },
                  { key: "batch_size", label: "Batch size", min: 4, max: 256 },
                ].map(({ key, label, min, max }) => (
                  <div key={key}>
                    <label className="block text-xs text-gray-500 mb-1">{label}</label>
                    <input type="number" min={min} max={max}
                      className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500"
                      value={config[key]}
                      onChange={e => update(key, +e.target.value)} />
                  </div>
                ))}
                <div>
                  <label className="block text-xs text-gray-500 mb-1">Learning rate</label>
                  <select className="w-full border border-gray-300 rounded px-3 py-2 text-sm"
                    value={config.learning_rate}
                    onChange={e => update("learning_rate", +e.target.value)}>
                    {[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3].map(v => (
                      <option key={v} value={v}>{v}</option>
                    ))}
                  </select>
                </div>
              </>
            )}
          </div>

          {/* Advanced options */}
          <div className="bg-white rounded-xl border border-gray-200 p-5 space-y-4 col-span-2">
            <h3 className="font-semibold text-gray-800 text-sm">Advanced options</h3>
            <div>
              <label className="block text-xs text-gray-500 mb-1">Target column</label>
              <input
                className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500"
                placeholder="Auto-detect if empty"
                value={config.target_column}
                onChange={e => update("target_column", e.target.value)} />
            </div>
            {config.model_type === "nlp" && (
              <div>
                <label className="block text-xs text-gray-500 mb-1">Max token length</label>
                <select className="w-full border border-gray-300 rounded px-3 py-2 text-sm"
                  value={config.max_length}
                  onChange={e => update("max_length", +e.target.value)}>
                  {[64, 128, 256, 512].map(v => <option key={v} value={v}>{v}</option>)}
                </select>
              </div>
            )}
            <div className="flex gap-6">
              {[
                { key: "enable_ensemble", label: "Ensemble (top-3 models)" },
                { key: "enable_mlflow",   label: "MLflow tracking" },
              ].map(({ key, label }) => (
                <label key={key} className="flex items-center gap-2 cursor-pointer">
                  <input type="checkbox" checked={config[key]}
                    onChange={e => update(key, e.target.checked)}
                    className="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500" />
                  <span className="text-sm text-gray-700">{label}</span>
                </label>
              ))}
            </div>

            {/* Config JSON preview */}
            <div className="bg-gray-900 rounded-xl p-3 mt-2">
              <div className="text-xs text-gray-400 uppercase tracking-wide mb-1">Config JSON</div>
              <pre className="text-green-400 text-xs overflow-auto max-h-28">
                {JSON.stringify(config, null, 2)}
              </pre>
            </div>
          </div>
        </div>

        {/* Run button */}
        <button
          onClick={handleRun}
          disabled={loading || !sessionId}
          className="w-full bg-indigo-600 text-white py-4 rounded-xl font-semibold text-lg hover:bg-indigo-700 disabled:opacity-50 transition-colors"
        >
          {loading ? "Running…" : `🚀 Run ${config.model_type.toUpperCase()} Pipeline`}
        </button>

        {/* Log console */}
        {logs.length > 0 && (
          <div className="bg-gray-950 rounded-xl p-5">
            <div className="text-xs text-gray-400 uppercase tracking-wide mb-3">Live pipeline log</div>
            <div className="font-mono text-sm text-green-400 space-y-0.5 max-h-60 overflow-auto">
              {logs.map((l, i) => (
                <div key={i} className={
                  l.includes("❌") || l.includes("failed") ? "text-red-400"
                  : l.includes("✅") || l.includes("🎉") ? "text-green-400"
                  : l.includes("⚠️") ? "text-yellow-400"
                  : l.includes("🚀") || l.includes("▶") ? "text-white"
                  : "text-gray-400"
                }>{l}</div>
              ))}
              {loading && <div className="text-gray-500 animate-pulse">{">"} █</div>}
            </div>
            {trainDone && jobId && (
              <div className="mt-4 flex gap-3">
                <button
                  onClick={() => navigate(`/results/${jobId}`)}
                  className="bg-green-600 text-white px-5 py-2 rounded-lg text-sm font-medium hover:bg-green-700"
                >
                  📈 View Results
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {sessionId && <CoPilotPanel sessionId={sessionId} />}
    </div>
  );
}
