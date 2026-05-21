import React, { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  uploadFile,
  nlConnect,
  getPreprocessPlan,
  runPreprocess,
  startTrain,
  getTrainStatus,
  getTrainResults,
  createTrainingWS,
} from "../api";
import CoPilotPanel from "../copilot/CoPilotPanel";
import { ALL_MODELS, DEPLOY_TARGETS, MODEL_TYPES, TAG_COLORS } from "../constants/models";

const LAST_JOB_KEY = "unifiedai:lastJobId";
const LAST_MODEL_KEY = "unifiedai:lastModelId";

const STEP_LABELS = [
  { id: 1, label: "Data source", code: "DB" },
  { id: 2, label: "Model selection", code: "AI" },
  { id: 3, label: "Configure", code: "CFG" },
  { id: 4, label: "Train", code: "RUN" },
  { id: 5, label: "Results", code: "RPT" },
  { id: 6, label: "Deploy", code: "DEP" },
];

const PIPELINE_STAGES = ["Data", "EDA", "Prep", "Search", "Train", "Score", "Explain", "Ship"];

const SOURCE_PRESETS = [
  { title: "PostgreSQL", prompt: "Connect to PostgreSQL customer churn table." },
  { title: "BigQuery", prompt: "Connect to BigQuery weekly sales dataset." },
  { title: "Object store", prompt: "Connect to the latest parquet files in cloud storage." },
];

function classNames(...parts) {
  return parts.filter(Boolean).join(" ");
}

function shortMetric(value) {
  if (typeof value === "number") {
    if (value >= 100) return value.toFixed(0);
    if (value >= 10) return value.toFixed(1);
    return value.toFixed(3);
  }
  return value ?? "--";
}

function firstMetricValue(model) {
  const values = Object.values(model?.metrics || {});
  const first = values.find((value) => typeof value === "number") ?? values[0];
  return typeof first === "number" ? first : Number(first) || 0;
}

function orderModels(models = [], bestModelId) {
  return [...models].sort((left, right) => {
    if (left.model_id === bestModelId) return -1;
    if (right.model_id === bestModelId) return 1;
    return firstMetricValue(right) - firstMetricValue(left);
  });
}

function formatDuration(totalSeconds) {
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  return [hours, minutes, seconds].map((value) => String(value).padStart(2, "0")).join(":");
}

function GlassCard({ className = "", children }) {
  return <div className={classNames("glass-panel rounded-2xl", className)}>{children}</div>;
}

function SoftCard({ className = "", children }) {
  return <div className={classNames("soft-card rounded-xl", className)}>{children}</div>;
}

function SectionTitle({ eyebrow, title, body, action }) {
  return (
    <div className="mb-5 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
      <div>
        {eyebrow && (
          <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-[var(--muted)]">
            {eyebrow}
          </div>
        )}
        <h2 className="mt-2 text-[28px] font-semibold tracking-tight text-[var(--ink)]">{title}</h2>
        {body && <p className="mt-2 max-w-2xl text-sm leading-6 text-[var(--muted)]">{body}</p>}
      </div>
      {action}
    </div>
  );
}

function StepIndicator({ current }) {
  return (
    <div className="overflow-x-auto">
      <div className="flex min-w-max items-center gap-2">
        {STEP_LABELS.map((step) => {
          const status = step.id < current ? "done" : step.id === current ? "active" : "pending";
          return (
            <div
              key={step.id}
              className={classNames(
                "min-w-[126px] rounded-2xl border px-4 py-3",
                status === "done" && "border-[rgba(21,128,61,0.18)] bg-[var(--success-soft)]",
                status === "active" && "border-[rgba(37,99,235,0.18)] bg-[var(--accent-soft)]",
                status === "pending" && "border-[var(--border)] bg-[var(--surface-solid)]"
              )}
            >
              <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--muted)]">
                Step {String(step.id).padStart(2, "0")}
              </div>
              <div className="mt-1 text-sm font-semibold text-[var(--ink)]">{step.label}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function RailItem({ step, current, done, onClick, badge }) {
  return (
    <button
      type="button"
      onClick={() => onClick(step.id)}
      className={classNames(
        "flex w-full items-center gap-3 rounded-[16px] border px-3.5 py-3 text-left",
        current
          ? "border-[rgba(37,99,235,0.18)] bg-[var(--accent-soft)]"
          : done
            ? "border-[var(--border)] bg-[var(--surface-alt)]"
              : "border-transparent bg-transparent hover:border-[var(--border)] hover:bg-[var(--surface-alt)]"
      )}
    >
      <div
        className={classNames(
          "flex h-9 w-9 items-center justify-center rounded-xl text-[11px] font-bold tracking-[0.08em]",
          current && "bg-[var(--accent)] text-[var(--surface-solid)]",
          !current && done && "bg-[var(--success-soft)] text-[var(--success)]",
          !current && !done && "bg-[var(--surface-alt)] text-[var(--muted)]"
        )}
      >
        {String(step.id).padStart(2, "0")}
      </div>
      <div className="min-w-0 flex-1">
        <div className="text-sm font-semibold text-[var(--ink)]">{step.label}</div>
        <div className="text-xs text-[var(--muted)]">
          {current ? "Current focus" : done ? "Ready" : "Queued"}
        </div>
      </div>
      {badge ? (
        <span className="rounded-full bg-[var(--surface-solid)] px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--muted)]">
          {badge}
        </span>
      ) : null}
    </button>
  );
}

function MetricTile({ label, value, hint }) {
  return (
    <div className="rounded-2xl border border-[var(--border)] bg-[var(--surface-alt)] px-4 py-4">
      <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--muted)]">{label}</div>
      <div className="mt-2 text-[22px] font-semibold text-[var(--ink)]">{value}</div>
      {hint && <div className="mt-1 text-xs text-[var(--muted)]">{hint}</div>}
    </div>
  );
}

function TypeBadge({ label }) {
  const colors = TAG_COLORS[label] || { bg: "#ecfdf5", text: "#0f766e" };
  return (
    <span
      className="rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.14em]"
      style={{ background: `${colors.bg}66`, color: colors.text, borderColor: `${colors.text}22` }}
    >
      {label}
    </span>
  );
}

export default function WizardPage({ mode = "beginner" }) {
  const navigate = useNavigate();
  const logRef = useRef(null);

  const [step, setStep] = useState(1);
  const [sessionId, setSessionId] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [sourceLabel, setSourceLabel] = useState("");
  const [dataProfile, setDataProfile] = useState(null);
  const [preprocessPlan, setPreprocessPlan] = useState(null);
  const [modelType, setModelType] = useState("ml");
  const [selectedModels, setSelectedModels] = useState(["xgboost", "lightgbm", "random_forest"]);
  const [config, setConfig] = useState(() => ({
    n_trials: mode === "expert" ? 50 : 20,
    time_budget_seconds: 300,
    cv_folds: 5,
    target_column: "",
    epochs: 10,
    batch_size: 32,
    learning_rate: 0.0002,
    max_length: 256,
    enable_ensemble: mode === "expert",
    enable_mlflow: mode === "expert",
  }));
  const [deployTarget, setDeployTarget] = useState("render_free");
  const [logs, setLogs] = useState([]);
  const [results, setResults] = useState(null);
  const [trainRunning, setTrainRunning] = useState(false);
  const [trainDone, setTrainDone] = useState(false);
  const [connected, setConnected] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [timerAnchor, setTimerAnchor] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [nlText, setNlText] = useState("");

  useEffect(() => {
    const available = ALL_MODELS.filter((model) => model.type === modelType).map((model) => model.id);
    setSelectedModels((previous) => {
      const kept = previous.filter((id) => available.includes(id));
      return kept.length ? kept : available.slice(0, 3);
    });
  }, [modelType]);

  useEffect(() => {
    if (!timerAnchor) {
      setElapsedSeconds(0);
      return undefined;
    }

    const interval = window.setInterval(() => {
      setElapsedSeconds(Math.max(0, Math.floor((Date.now() - timerAnchor) / 1000)));
    }, 1000);

    return () => window.clearInterval(interval);
  }, [timerAnchor]);

  useEffect(() => {
    if (!jobId || !trainRunning) return undefined;

    const socket = createTrainingWS(
      jobId,
      (message) => {
        setConnected(true);
        setLogs((previous) => [...previous, message]);
      },
      () => {
        setConnected(false);
      }
    );

    return () => socket.close();
  }, [jobId, trainRunning]);

  useEffect(() => {
    if (!jobId || !trainRunning) return undefined;

    const interval = window.setInterval(async () => {
      try {
        const statusResponse = await getTrainStatus(jobId);
        const status = statusResponse.data.status;

        if (status === "completed" || status === "failed") {
          window.clearInterval(interval);
          setTrainRunning(false);
          setTrainDone(status === "completed");
          setConnected(false);
          setLogs((previous) => [...previous, `Training ${status}.`]);

          if (status === "completed") {
            const resultsResponse = await getTrainResults(jobId);
            const ordered = orderModels(resultsResponse.data.models, resultsResponse.data.best_model_id);
            const normalized = { ...resultsResponse.data, models: ordered };
            setResults(normalized);
            const bestModelId = statusResponse.data.best_model_id || normalized.best_model_id || ordered[0]?.model_id;
            localStorage.setItem(LAST_JOB_KEY, jobId);
            if (bestModelId) localStorage.setItem(LAST_MODEL_KEY, bestModelId);
          }
        }
      } catch (pollError) {
        window.clearInterval(interval);
        setTrainRunning(false);
        setConnected(false);
        setError(pollError.response?.data?.detail || "Could not refresh training status.");
      }
    }, 2000);

    return () => window.clearInterval(interval);
  }, [jobId, trainRunning]);

  useEffect(() => {
    if (!logRef.current) return;
    logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logs]);

  const activeModels = ALL_MODELS.filter((model) => selectedModels.includes(model.id));
  const availableModels = ALL_MODELS.filter((model) => model.type === modelType);
  const orderedResults = results ? orderModels(results.models, results.best_model_id) : [];
  const bestModel = orderedResults[0];
  const topMetricEntries = Object.entries(bestModel?.metrics || {}).slice(0, 4);
  const maxComparison = Math.max(...orderedResults.map((model) => firstMetricValue(model)), 1);
  const topFeatures = (bestModel?.shap?.top_features || []).slice(0, 5);
  const pipelineStageIndex = trainDone
    ? PIPELINE_STAGES.length - 1
    : trainRunning
      ? Math.min(PIPELINE_STAGES.length - 2, Math.max(1, Math.floor(logs.length / 4)))
      : -1;
  const primaryModelId = bestModel?.model_id || results?.best_model_id || localStorage.getItem(LAST_MODEL_KEY);

  const completedSteps = {
    1: Boolean(dataProfile),
    2: selectedModels.length > 0,
    3: Boolean(jobId),
    4: trainDone,
    5: Boolean(bestModel),
    6: Boolean(primaryModelId),
  };

  const persistProfile = async (payload, label) => {
    setSessionId(payload.session_id);
    setSourceLabel(label);
    setDataProfile(payload.profile);
    setTimerAnchor(Date.now());
    setError("");
    setTrainDone(false);
    setTrainRunning(false);
    setResults(null);
    setJobId(null);
    setLogs([]);
    localStorage.removeItem(LAST_JOB_KEY);
    localStorage.removeItem(LAST_MODEL_KEY);

    try {
      const planResponse = await getPreprocessPlan(payload.session_id);
      setPreprocessPlan(planResponse.data);
    } catch {
      setPreprocessPlan(null);
    }
  };

  const handleUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setLoading(true);
    setError("");

    try {
      const response = await uploadFile(file);
      await persistProfile(response.data, file.name);
    } catch (uploadError) {
      setError(uploadError.response?.data?.detail || "Upload failed.");
    } finally {
      setLoading(false);
      event.target.value = "";
    }
  };

  const handleNlConnect = async (prompt = nlText) => {
    if (!prompt.trim()) return;

    setLoading(true);
    setError("");

    try {
      const response = await nlConnect(prompt);
      await persistProfile(response.data, "Live connector");
      setNlText(prompt);
    } catch (connectError) {
      setError(connectError.response?.data?.detail || "Connection failed.");
    } finally {
      setLoading(false);
    }
  };

  const toggleModel = (modelId) => {
    setSelectedModels((previous) =>
      previous.includes(modelId)
        ? previous.filter((id) => id !== modelId)
        : [...previous, modelId]
    );
  };

  const selectAll = () => {
    setSelectedModels(availableModels.map((model) => model.id));
  };

  const clearModels = () => {
    setSelectedModels([]);
  };

  const handleTrain = async () => {
    if (!sessionId) {
      setError("Load a dataset before training.");
      return;
    }

    if (selectedModels.length === 0) {
      setError("Choose at least one model.");
      return;
    }

    const payload = {
      ...config,
      model_type: modelType,
      models: selectedModels,
    };

    setStep(4);
    setLoading(true);
    setError("");
    setTrainDone(false);
    setTrainRunning(true);
    setConnected(false);
    setResults(null);
    setJobId(null);
    setLogs([]);
    setTimerAnchor(Date.now());

    try {
      const preprocessResponse = await runPreprocess(sessionId, {});
      const preprocessLogs = preprocessResponse.data.preprocessing_log || [];
      setLogs(preprocessLogs.length ? preprocessLogs : ["Preprocessing pipeline prepared."]);

      const trainResponse = await startTrain(sessionId, payload);
      setJobId(trainResponse.data.job_id);
      localStorage.setItem(LAST_JOB_KEY, trainResponse.data.job_id);
      setLogs((previous) => [
        ...previous,
        `Training job queued: ${trainResponse.data.job_id}`,
        `Domain: ${modelType.toUpperCase()} | Models: ${selectedModels.length}`,
      ]);
    } catch (trainError) {
      setTrainRunning(false);
      setConnected(false);
      setError(trainError.response?.data?.detail || "Training failed.");
    } finally {
      setLoading(false);
    }
  };

  const previewConfig = JSON.stringify(
    {
      model_type: modelType,
      models: selectedModels,
      ...config,
    },
    null,
    2
  );

  const resultTarget = DEPLOY_TARGETS.find((target) => target.id === deployTarget);

  return (
    <div className="grid gap-5">
      <div className="grid gap-5 lg:grid-cols-[280px_minmax(0,1fr)]">
        <GlassCard className="h-fit p-4">
          <div className="soft-card rounded-[18px] p-4">
            <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-[var(--muted)]">
              Pipeline
            </div>
            <div className="mt-4 space-y-2">
              {STEP_LABELS.map((item) => (
                <RailItem
                  key={item.id}
                  step={item}
                  current={step === item.id}
                  done={completedSteps[item.id] || item.id < step}
                  badge={item.id === 4 && trainRunning ? "LIVE" : item.id === 4 && trainDone ? "DONE" : ""}
                  onClick={setStep}
                />
              ))}
            </div>
          </div>

          <div className="mt-4 soft-card rounded-[18px] p-4">
            <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-[var(--muted)]">
              Models active
            </div>
            <div className="mt-4 space-y-3">
              {activeModels.length === 0 ? (
                <div className="text-sm text-[var(--muted)]">Select a model family to build your run sheet.</div>
              ) : (
                activeModels.slice(0, 5).map((model) => (
                  <div key={model.id} className="flex items-center justify-between gap-3">
                    <div>
                      <div className="text-sm font-semibold text-[var(--ink)]">{model.name}</div>
                      <div className="text-xs text-[var(--muted)]">{model.desc}</div>
                    </div>
                    <TypeBadge label={model.tag} />
                  </div>
                ))
              )}
            </div>
          </div>

          <div className="mt-4 rounded-[18px] border border-[var(--border)] bg-[var(--surface-alt)] p-4">
            <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-[var(--muted)]">Session</div>
            <div className="mt-4 text-[28px] font-semibold text-[var(--ink)]">{formatDuration(elapsedSeconds)}</div>
            <div className="mt-2 text-sm text-[var(--muted)]">
              {sourceLabel ? `Source: ${sourceLabel}` : "Waiting for your first dataset."}
            </div>
            <div className="mt-4 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">
              <span
                className={classNames(
                  "h-2.5 w-2.5 rounded-full",
                  trainRunning ? "bg-[var(--accent)]" : trainDone ? "bg-[var(--success)]" : "bg-[var(--border-strong)]"
                )}
              />
              {trainRunning ? "Training live" : trainDone ? "Training complete" : "Idle"}
            </div>
          </div>
        </GlassCard>

        <div className="space-y-5">
          <GlassCard className="p-5 lg:p-6">
            <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
              <div>
                <h1 className="text-2xl font-semibold text-[var(--ink)]">Train a model</h1>
                <p className="mt-2 max-w-3xl text-sm leading-6 text-[var(--muted)]">
                  Move through data, model selection, configuration, training, results, and deployment in one flow.
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-2">
                {sessionId ? <CoPilotPanel sessionId={sessionId} triggerMode="inline" /> : null}
                <span className="rounded-lg border border-[var(--border)] bg-[var(--surface-alt)] px-3 py-2 text-xs font-semibold text-[var(--ink)]">
                  {mode} mode
                </span>
                <span className="rounded-lg border border-[var(--border)] bg-[var(--surface-alt)] px-3 py-2 text-xs font-semibold text-[var(--muted)]">
                  {modelType.toUpperCase()}
                </span>
                <span className="rounded-lg border border-[var(--border)] bg-[var(--surface-alt)] px-3 py-2 text-xs font-semibold text-[var(--muted)]">
                  {selectedModels.length} models
                </span>
              </div>
            </div>

            <div className="mt-6">
              <StepIndicator current={step} />
            </div>
          </GlassCard>

          {error ? (
            <div className="rounded-[18px] border border-red-200 bg-red-50 px-5 py-4 text-sm text-red-700">
              {error}
            </div>
          ) : null}

          {step === 1 ? (
            <GlassCard className="p-6">
              <SectionTitle
                eyebrow="Step 01"
                title="Connect your data source"
                body="Upload a file or kick off a live connector prompt. Once we have the dataset profile, the wizard will build a preprocessing plan automatically."
              />

              <div className="grid gap-4 lg:grid-cols-3">
                {SOURCE_PRESETS.map((preset) => (
                  <button
                    key={preset.title}
                    type="button"
                    onClick={() => {
                      setNlText(preset.prompt);
                      handleNlConnect(preset.prompt);
                    }}
                    className="rounded-[18px] border border-[var(--border)] bg-[var(--surface-solid)] p-4 text-left hover:border-[var(--border-strong)]"
                  >
                    <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--muted)]">Live source</div>
                    <div className="mt-3 text-base font-semibold text-[var(--ink)]">{preset.title}</div>
                    <div className="mt-2 text-sm leading-6 text-[var(--muted)]">{preset.prompt}</div>
                  </button>
                ))}
              </div>

              <div className="mt-6 grid gap-6 xl:grid-cols-[minmax(0,1fr)_340px]">
                <label className="flex cursor-pointer flex-col items-center justify-center rounded-[22px] border border-dashed border-[var(--border-strong)] bg-[var(--surface-alt)] px-6 py-10 text-center hover:border-[var(--accent)] hover:bg-[var(--accent-soft)]">
                  <div className="rounded-full bg-[var(--surface-solid)] px-4 py-2 text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--accent)]">
                    Upload dataset
                  </div>
                  <div className="mt-5 max-w-full break-all text-[22px] font-semibold tracking-tight text-[var(--ink)] sm:text-[24px] lg:text-[26px]">
                    {dataProfile ? dataProfile.original_filename || "Dataset ready" : "Drag, drop, or browse a file"}
                  </div>
                  <div className="mt-3 max-w-md text-sm leading-6 text-[var(--muted)]">
                    CSV, JSON, Excel, TSV, or Parquet. We profile structure, missing values, and quality before training starts.
                  </div>
                  <input
                    type="file"
                    className="hidden"
                    onChange={handleUpload}
                    disabled={loading}
                    accept=".csv,.json,.xlsx,.xls,.parquet,.tsv"
                  />
                </label>

                <SoftCard className="p-5">
                  <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--muted)]">Natural language connector</div>
                  <textarea
                    className="mt-4 min-h-[150px] w-full resize-none rounded-[18px] border border-[var(--border)] bg-[var(--surface-alt)] px-4 py-4 text-sm text-[var(--ink)] outline-none focus:border-[var(--accent)] focus:bg-[var(--surface-solid)]"
                    placeholder="Connect to the customer churn table in PostgreSQL and pull the latest 90 days."
                    value={nlText}
                    onChange={(event) => setNlText(event.target.value)}
                  />
                  <button
                    type="button"
                    onClick={() => handleNlConnect()}
                    disabled={loading || !nlText.trim()}
                    className="mt-4 rounded-xl bg-[var(--ink)] px-5 py-2.5 text-sm font-semibold text-[var(--surface-solid)] disabled:opacity-45"
                  >
                    {loading ? "Connecting..." : "Connect source"}
                  </button>
                </SoftCard>
              </div>

              {dataProfile ? (
                <div className="mt-6 rounded-[22px] border border-[var(--border)] bg-[var(--surface-alt)] p-6">
                  <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
                    <div>
                      <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--muted)]">Dataset profile</div>
                      <div className="mt-2 break-all text-2xl font-semibold tracking-tight text-[var(--ink)]">
                        {dataProfile.original_filename || sourceLabel || "Connected dataset"}
                      </div>
                      <div className="mt-2 text-sm text-[var(--muted)]">
                        {Object.keys(dataProfile.columns || {}).length} columns profiled with an automation plan ready.
                      </div>
                    </div>
                    <div className="rounded-full bg-[var(--surface-solid)] px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.16em] text-[var(--accent)]">
                      {preprocessPlan?.steps?.length || 0} prep stages
                    </div>
                  </div>

                  <div className="mt-6 grid gap-4 md:grid-cols-4">
                    <MetricTile label="Rows" value={dataProfile.shape?.rows?.toLocaleString() || "--"} />
                    <MetricTile label="Columns" value={dataProfile.shape?.columns || "--"} />
                    <MetricTile label="Quality" value={`${dataProfile.data_quality_score || 0}/100`} />
                    <MetricTile label="Memory" value={`${dataProfile.memory_usage_mb || dataProfile.file_size_mb || "--"} MB`} />
                  </div>

                  <div className="mt-5 h-2 overflow-hidden rounded-full bg-[rgba(23,33,38,0.08)]">
                    <div
                      className="h-full rounded-full bg-[linear-gradient(90deg,#0f766e_0%,#34d399_100%)]"
                      style={{ width: `${Math.min(Number(dataProfile.data_quality_score || 0), 100)}%` }}
                    />
                  </div>
                </div>
              ) : null}

              <div className="mt-6 flex justify-end">
                <button
                  type="button"
                  onClick={() => setStep(2)}
                  disabled={!dataProfile}
                  className="rounded-xl bg-[var(--ink)] px-5 py-2.5 text-sm font-semibold text-[var(--surface-solid)] disabled:opacity-40"
                >
                  Continue to models
                </button>
              </div>
            </GlassCard>
          ) : null}

          {step === 2 ? (
            <GlassCard className="p-6">
              <SectionTitle
                eyebrow="Step 02"
                title="Pick the model family"
                body="Switch between classical ML, deep learning, and NLP tracks. The grid only shows models that match the active domain."
                action={
                  <div className="flex items-center gap-3 text-sm text-[var(--muted)]">
                    <button type="button" onClick={selectAll} className="font-semibold text-[var(--accent)]">
                      Select all
                    </button>
                    <span>/</span>
                    <button type="button" onClick={clearModels} className="font-semibold text-[var(--muted)]">
                      Clear
                    </button>
                  </div>
                }
              />

              <div className="grid gap-4 lg:grid-cols-3">
                {MODEL_TYPES.map((type) => (
                  <button
                    key={type.id}
                    type="button"
                    onClick={() => setModelType(type.id)}
                    className={classNames(
                      "rounded-[24px] border p-5 text-left",
                      modelType === type.id
                        ? "border-[var(--border-strong)] bg-[var(--accent-soft)]"
                        : "border-[var(--border)] bg-[var(--surface-solid)] hover:border-[var(--border-strong)]"
                    )}
                  >
                    <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--muted)]">{type.id}</div>
                    <div className="mt-3 text-xl font-semibold text-[var(--ink)]">{type.label}</div>
                    <div className="mt-2 text-sm leading-6 text-[var(--muted)]">{type.sub}</div>
                  </button>
                ))}
              </div>

              <div className="mt-6 grid gap-4 xl:grid-cols-2">
                {availableModels.map((model) => {
                  const active = selectedModels.includes(model.id);
                  return (
                    <button
                      key={model.id}
                      type="button"
                      onClick={() => toggleModel(model.id)}
                      className={classNames(
                        "relative rounded-[24px] border p-5 text-left",
                        active
                          ? "border-[var(--border-strong)] bg-[var(--accent-soft)]"
                          : "border-[var(--border)] bg-[var(--surface-solid)] hover:border-[var(--border-strong)]"
                      )}
                    >
                      <div className="flex items-start justify-between gap-4">
                        <div>
                          <TypeBadge label={model.tag} />
                          <div className="mt-4 text-lg font-semibold text-[var(--ink)]">{model.name}</div>
                          <div className="mt-2 text-sm leading-6 text-[var(--muted)]">{model.desc}</div>
                        </div>
                        <div
                          className={classNames(
                            "flex h-7 w-7 items-center justify-center rounded-full border text-[11px] font-bold",
                            active
                              ? "border-[var(--accent)] bg-[var(--accent)] text-[var(--surface-solid)]"
                              : "border-[var(--border)] bg-transparent text-[var(--muted)]"
                          )}
                        >
                          {active ? "OK" : "+"}
                        </div>
                      </div>
                    </button>
                  );
                })}
              </div>

              <div className="mt-6 flex flex-col gap-3 sm:flex-row sm:justify-between">
                <button
                  type="button"
                  onClick={() => setStep(1)}
                  className="rounded-full border border-[var(--border)] px-5 py-2.5 text-sm font-semibold text-[var(--muted)]"
                >
                  Back
                </button>
                <button
                  type="button"
                  onClick={() => setStep(3)}
                  disabled={selectedModels.length === 0}
                  className="rounded-xl bg-[var(--ink)] px-5 py-2.5 text-sm font-semibold text-[var(--surface-solid)] disabled:opacity-40"
                >
                  Configure run
                </button>
              </div>
            </GlassCard>
          ) : null}

          {step === 3 ? (
            <GlassCard className="p-6">
              <SectionTitle
                eyebrow="Step 03"
                title="Dial in the training recipe"
                body="Tune search depth, training budget, and extra automation before you launch the pipeline."
              />

              <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_360px]">
                <div className="space-y-5">
                  <SoftCard className="p-5">
                    <div className="text-lg font-semibold text-[var(--ink)]">
                      {modelType === "ml" ? "Search controls" : modelType === "dl" ? "Deep learning controls" : "NLP fine-tuning controls"}
                    </div>
                    <div className="mt-5 space-y-4">
                      {modelType === "ml" ? (
                        <>
                          {[
                            ["n_trials", "Trials per model", 5, 100, 1],
                            ["cv_folds", "Cross validation folds", 3, 10, 1],
                            ["time_budget_seconds", "Time budget (seconds)", 60, 3600, 60],
                          ].map(([key, label, min, max, stepValue]) => (
                            <div key={key}>
                              <div className="mb-2 flex items-center justify-between text-sm">
                                <span className="text-[var(--muted)]">{label}</span>
                                <span className="font-semibold text-[var(--ink)]">{config[key]}</span>
                              </div>
                              <input
                                type="range"
                                min={min}
                                max={max}
                                step={stepValue}
                                value={config[key]}
                                onChange={(event) =>
                                  setConfig((previous) => ({ ...previous, [key]: Number(event.target.value) }))
                                }
                                className="w-full accent-[var(--accent)]"
                              />
                            </div>
                          ))}
                        </>
                      ) : (
                        <>
                          {[
                            ["epochs", "Epochs", 1, 50, 1],
                            ["batch_size", "Batch size", 8, 128, 8],
                          ].map(([key, label, min, max, stepValue]) => (
                            <div key={key}>
                              <div className="mb-2 flex items-center justify-between text-sm">
                                <span className="text-[var(--muted)]">{label}</span>
                                <span className="font-semibold text-[var(--ink)]">{config[key]}</span>
                              </div>
                              <input
                                type="range"
                                min={min}
                                max={max}
                                step={stepValue}
                                value={config[key]}
                                onChange={(event) =>
                                  setConfig((previous) => ({ ...previous, [key]: Number(event.target.value) }))
                                }
                                className="w-full accent-[var(--accent)]"
                              />
                            </div>
                          ))}

                          <div className="grid gap-4 md:grid-cols-2">
                            <label className="text-sm text-[var(--muted)]">
                              Learning rate
                              <select
                                className="mt-2 w-full rounded-[18px] border border-[var(--border)] bg-[var(--surface-solid)] px-4 py-3 text-[var(--ink)]"
                                value={config.learning_rate}
                                onChange={(event) =>
                                  setConfig((previous) => ({ ...previous, learning_rate: Number(event.target.value) }))
                                }
                              >
                                {[0.00001, 0.00002, 0.00005, 0.0001, 0.0002].map((value) => (
                                  <option key={value} value={value}>
                                    {value}
                                  </option>
                                ))}
                              </select>
                            </label>

                            {modelType === "nlp" ? (
                              <label className="text-sm text-[var(--muted)]">
                                Max token length
                                <select
                                  className="mt-2 w-full rounded-[18px] border border-[var(--border)] bg-[var(--surface-solid)] px-4 py-3 text-[var(--ink)]"
                                  value={config.max_length}
                                  onChange={(event) =>
                                    setConfig((previous) => ({ ...previous, max_length: Number(event.target.value) }))
                                  }
                                >
                                  {[64, 128, 256, 512].map((value) => (
                                    <option key={value} value={value}>
                                      {value}
                                    </option>
                                  ))}
                                </select>
                              </label>
                            ) : null}
                          </div>
                        </>
                      )}
                    </div>
                  </SoftCard>

                  <SoftCard className="p-5">
                    <div className="text-lg font-semibold text-[var(--ink)]">Pipeline automation</div>
                    <div className="mt-5 grid gap-4 md:grid-cols-2">
                      {[
                        ["enable_ensemble", "Blend top models"],
                        ["enable_mlflow", "Track in MLflow"],
                      ].map(([key, label]) => (
                        <label
                          key={key}
                          className="flex items-center justify-between rounded-[20px] border border-[var(--border)] bg-[rgba(247,245,239,0.6)] px-4 py-3"
                        >
                          <span className="text-sm font-medium text-[var(--ink)]">{label}</span>
                          <input
                            type="checkbox"
                            checked={Boolean(config[key])}
                            onChange={(event) =>
                              setConfig((previous) => ({ ...previous, [key]: event.target.checked }))
                            }
                            className="h-4 w-4 rounded border-[var(--border)] text-[var(--accent)] accent-[var(--accent)]"
                          />
                        </label>
                      ))}
                    </div>

                    <label className="mt-4 block text-sm text-[var(--muted)]">
                      Target column
                      <input
                        type="text"
                        className="mt-2 w-full rounded-[18px] border border-[var(--border)] bg-[var(--surface-solid)] px-4 py-3 text-[var(--ink)] outline-none focus:border-[var(--accent)]"
                        placeholder="Leave blank to auto-detect"
                        value={config.target_column}
                        onChange={(event) =>
                          setConfig((previous) => ({ ...previous, target_column: event.target.value }))
                        }
                      />
                    </label>
                  </SoftCard>
                </div>

                <SoftCard className="p-5">
                  <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-[var(--muted)]">Run blueprint</div>
                  <div className="mt-4 rounded-[22px] bg-[var(--surface-dark)] p-4 text-sm text-emerald-300">
                    <pre className="max-h-[420px] overflow-auto whitespace-pre-wrap break-words font-mono">
                      {previewConfig}
                    </pre>
                  </div>
                  <div className="mt-5 grid gap-3">
                    <MetricTile label="Session" value={sessionId ? "Linked" : "Waiting"} />
                    <MetricTile label="Selected" value={`${selectedModels.length} models`} />
                    <MetricTile label="Mode" value={mode} />
                  </div>
                </SoftCard>
              </div>

              <div className="mt-6 flex flex-col gap-3 sm:flex-row sm:justify-between">
                <button
                  type="button"
                  onClick={() => setStep(2)}
                  className="rounded-full border border-[var(--border)] px-5 py-2.5 text-sm font-semibold text-[var(--muted)]"
                >
                  Back
                </button>
                <button
                  type="button"
                  onClick={handleTrain}
                  disabled={loading}
                  className="rounded-xl bg-[var(--ink)] px-5 py-2.5 text-sm font-semibold text-[var(--surface-solid)] disabled:opacity-45"
                >
                  {loading ? "Preparing..." : "Launch training"}
                </button>
              </div>
            </GlassCard>
          ) : null}

          {step === 4 ? (
            <GlassCard className="p-6">
              <SectionTitle
                eyebrow="Step 04"
                title="Live training monitor"
                body="This panel tracks preprocessing, search, training, evaluation, and artifact generation in the same flow as your mockup."
                action={
                  <div className="rounded-full border border-[var(--border)] bg-[var(--surface-solid)] px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.16em] text-[var(--muted)]">
                    {trainRunning ? connected ? "streaming" : "polling" : trainDone ? "complete" : "ready"}
                  </div>
                }
              />

              <div className="overflow-x-auto">
                <div className="flex min-w-max items-center gap-2 pb-2">
                  {PIPELINE_STAGES.map((stage, index) => {
                    const done = trainDone || (trainRunning && index < pipelineStageIndex);
                    const active = trainRunning && index === pipelineStageIndex;
                    return (
                      <React.Fragment key={stage}>
                        <div className="flex flex-col items-center gap-2">
                          <div
                            className={classNames(
                              "flex h-11 w-11 items-center justify-center rounded-2xl text-[10px] font-bold uppercase tracking-[0.16em]",
                              done && "bg-[var(--success-soft)] text-[var(--success)]",
                              active && "bg-[var(--accent-soft)] text-[var(--accent)]",
                              !done && !active && "bg-[rgba(23,33,38,0.06)] text-[var(--muted)]"
                            )}
                          >
                            {stage}
                          </div>
                          <span className="text-[11px] font-medium text-[var(--muted)]">{stage}</span>
                        </div>
                        {index < PIPELINE_STAGES.length - 1 ? (
                          <div className="h-px w-8 bg-[var(--border-strong)]" />
                        ) : null}
                      </React.Fragment>
                    );
                  })}
                </div>
              </div>

              <div className="mt-6 grid gap-6 xl:grid-cols-[minmax(0,1fr)_280px]">
                <div className="rounded-[28px] terminal-surface p-5 text-sm text-slate-200">
                  <div className="mb-4 flex items-center justify-between">
                    <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-white/55">Training log</div>
                    <div className="flex items-center gap-2 text-xs uppercase tracking-[0.16em] text-white/55">
                      <span
                        className={classNames(
                          "h-2.5 w-2.5 rounded-full",
                          trainRunning ? "bg-cyan-400 shadow-[0_0_18px_rgba(34,211,238,0.8)]" : trainDone ? "bg-emerald-400" : "bg-white/20"
                        )}
                      />
                      {trainRunning ? "Running" : trainDone ? "Finished" : "Standby"}
                    </div>
                  </div>

                  <div ref={logRef} className="max-h-[360px] overflow-y-auto font-mono text-xs leading-7">
                    {logs.length === 0 ? (
                      <div className="text-slate-500">Waiting for pipeline output...</div>
                    ) : (
                      logs.map((line, index) => (
                        <div key={`${line}-${index}`} className="text-slate-300">
                          <span className="mr-3 select-none text-slate-600">&gt;</span>
                          {line}
                        </div>
                      ))
                    )}
                    {trainRunning ? (
                      <div className="text-slate-500">
                        <span className="mr-3 select-none text-slate-600">&gt;</span>
                        <span className="animate-pulse">...</span>
                      </div>
                    ) : null}
                  </div>
                </div>

                <div className="space-y-4">
                  <MetricTile label="Socket" value={connected ? "Live" : "Idle"} />
                  <MetricTile label="Models" value={selectedModels.length} />
                  <MetricTile label="Timer" value={formatDuration(elapsedSeconds)} />

                  <SoftCard className="p-4">
                    <div className="text-sm font-semibold text-[var(--ink)]">Selected run set</div>
                    <div className="mt-3 space-y-3">
                      {activeModels.slice(0, 3).map((model, index) => (
                        <div key={model.id}>
                          <div className="mb-2 flex items-center justify-between text-xs text-[var(--muted)]">
                            <span>{model.name}</span>
                            <span>{trainDone ? "100%" : trainRunning ? `${Math.min(92, 22 + index * 21 + logs.length * 2)}%` : "0%"}</span>
                          </div>
                          <div className="h-2 overflow-hidden rounded-full bg-[rgba(23,33,38,0.08)]">
                            <div
                              className="h-full rounded-full bg-[linear-gradient(90deg,#0f766e_0%,#34d399_100%)]"
                              style={{
                                width: trainDone ? "100%" : trainRunning ? `${Math.min(92, 22 + index * 21 + logs.length * 2)}%` : "0%",
                              }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </SoftCard>
                </div>
              </div>

              <div className="mt-6 flex flex-col gap-3 sm:flex-row sm:justify-between">
                <button
                  type="button"
                  onClick={() => setStep(3)}
                  className="rounded-full border border-[var(--border)] px-5 py-2.5 text-sm font-semibold text-[var(--muted)]"
                >
                  Back
                </button>
                {trainDone ? (
                  <button
                    type="button"
                    onClick={() => setStep(5)}
                    className="rounded-full bg-[var(--success)] px-5 py-2.5 text-sm font-semibold text-[var(--surface-solid)]"
                  >
                    View results
                  </button>
                ) : null}
              </div>
            </GlassCard>
          ) : null}

          {step === 5 ? (
            <GlassCard className="p-6">
              <SectionTitle
                eyebrow="Step 05"
                title="Results snapshot"
                body="A polished in-wizard summary before you jump into the dedicated results route."
              />

              {bestModel ? (
                <>
                  <SoftCard className="p-6">
                    <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
                      <div>
                        <div className="text-sm font-semibold text-[var(--muted)]">Best model</div>
                        <div className="mt-2 text-3xl font-semibold text-[var(--ink)]">{bestModel.algorithm}</div>
                        <div className="mt-2 text-sm text-[var(--muted)]">{bestModel.framework} stack</div>
                      </div>
                      <div className="grid gap-3 md:grid-cols-4">
                        {topMetricEntries.map(([metric, value]) => (
                          <MetricTile key={metric} label={metric.replace(/_/g, " ")} value={shortMetric(value)} />
                        ))}
                      </div>
                    </div>
                  </SoftCard>

                  <div className="mt-6 grid gap-6 xl:grid-cols-[minmax(0,1fr)_320px]">
                    <SoftCard className="p-5">
                      <div className="text-lg font-semibold text-[var(--ink)]">Model comparison</div>
                      <div className="mt-5 space-y-4">
                        {orderedResults.slice(0, 6).map((model, index) => {
                          const score = firstMetricValue(model);
                          const width = `${Math.max(12, Math.min((score / maxComparison) * 100, 100))}%`;
                          return (
                            <div key={model.model_id}>
                              <div className="mb-2 flex items-center justify-between text-sm">
                                <span className="font-medium text-[var(--ink)]">{model.algorithm}</span>
                                <span className="text-[var(--muted)]">{shortMetric(score)}</span>
                              </div>
                              <div className="h-5 overflow-hidden rounded-full bg-[rgba(23,33,38,0.08)]">
                                <div
                                  className={classNames(
                                    "flex h-full items-center rounded-full px-3 text-[10px] font-semibold uppercase tracking-[0.14em]",
                                    index === 0 ? "bg-[var(--accent)] text-[var(--surface-solid)]" : "bg-[var(--accent-soft)] text-[var(--accent)]"
                                  )}
                                  style={{ width }}
                                >
                                  {index === 0 ? "Best" : "Alt"}
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </SoftCard>

                    <SoftCard className="p-5">
                      <div className="text-lg font-semibold text-[var(--ink)]">Top SHAP features</div>
                      <div className="mt-5 space-y-4">
                        {topFeatures.length === 0 ? (
                          <div className="text-sm text-[var(--muted)]">Feature importance will appear here when SHAP output is available.</div>
                        ) : (
                          topFeatures.map((feature) => (
                            <div key={feature.feature}>
                              <div className="mb-2 flex items-center justify-between text-sm">
                                <span className="font-medium text-[var(--ink)]">{feature.feature}</span>
                                <span className="text-[var(--muted)]">{shortMetric(feature.importance)}</span>
                              </div>
                              <div className="h-3 overflow-hidden rounded-full bg-[rgba(23,33,38,0.08)]">
                                <div
                                  className="h-full rounded-full bg-[var(--accent)]"
                                  style={{
                                    width: `${Math.max(12, (feature.importance / (topFeatures[0]?.importance || 1)) * 100)}%`,
                                  }}
                                />
                              </div>
                            </div>
                          ))
                        )}
                      </div>
                    </SoftCard>
                  </div>

                  <div className="mt-6 flex flex-col gap-3 sm:flex-row sm:justify-between">
                    <button
                      type="button"
                      onClick={() => setStep(4)}
                      className="rounded-full border border-[var(--border)] px-5 py-2.5 text-sm font-semibold text-[var(--muted)]"
                    >
                      Back
                    </button>
                    <div className="flex flex-col gap-3 sm:flex-row">
                      <button
                        type="button"
                        onClick={() => navigate(`/results/${jobId}`)}
                        className="rounded-xl border border-[var(--border)] bg-[var(--surface-solid)] px-5 py-2.5 text-sm font-semibold text-[var(--ink)]"
                      >
                        Open results and test
                      </button>
                      <button
                        type="button"
                        onClick={() => setStep(6)}
                        className="rounded-xl bg-[var(--ink)] px-5 py-2.5 text-sm font-semibold text-[var(--surface-solid)]"
                      >
                        Continue to deploy
                      </button>
                    </div>
                  </div>
                </>
              ) : (
                <SoftCard className="p-8 text-center">
                  <div className="text-lg font-semibold text-[var(--ink)]">Results unlock after training completes</div>
                  <div className="mt-2 text-sm text-[var(--muted)]">Run the pipeline first, then come back here for the summary.</div>
                </SoftCard>
              )}
            </GlassCard>
          ) : null}

          {step === 6 ? (
            <GlassCard className="p-6">
              <SectionTitle
                eyebrow="Step 06"
                title="Deployment handoff"
                body="Choose the output shape you want, then jump into the dedicated deployment console with the best model preselected."
              />

              <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                {DEPLOY_TARGETS.map((target) => (
                  <button
                    key={target.id}
                    type="button"
                    onClick={() => setDeployTarget(target.id)}
                    className={classNames(
                      "rounded-xl border p-4 text-left",
                      deployTarget === target.id
                        ? "border-[var(--accent)] bg-[var(--accent-soft)]"
                        : "border-[var(--border)] bg-[var(--surface-solid)] hover:border-[var(--border-strong)]"
                    )}
                  >
                    <div className="text-lg font-semibold text-[var(--ink)]">{target.name}</div>
                    <div className="mt-2 text-sm leading-6 text-[var(--muted)]">{target.desc}</div>
                  </button>
                ))}
              </div>

              <div className="mt-6 grid gap-6 xl:grid-cols-[minmax(0,1fr)_320px]">
                <SoftCard className="p-5">
                  <div className="text-lg font-semibold text-[var(--ink)]">Selected target</div>
                  <div className="mt-4 rounded-xl border border-[var(--border)] bg-[var(--surface-alt)] p-5">
                    <div className="text-sm font-semibold text-[var(--ink)]">{resultTarget?.name}</div>
                    <div className="mt-2 text-sm leading-6 text-[var(--muted)]">{resultTarget?.desc}</div>
                  </div>
                </SoftCard>

                <SoftCard className="p-5">
                  <div className="text-lg font-semibold text-[var(--ink)]">Ready artifact</div>
                  <div className="mt-4 space-y-3">
                    <MetricTile label="Best model" value={bestModel?.algorithm || "--"} />
                    <MetricTile label="Model ID" value={primaryModelId ? `${primaryModelId.slice(0, 8)}...` : "--"} />
                    <MetricTile label="Target" value={deployTarget.toUpperCase()} />
                  </div>
                </SoftCard>
              </div>

              <div className="mt-6 flex flex-col gap-3 sm:flex-row sm:justify-between">
                <button
                  type="button"
                  onClick={() => setStep(5)}
                  className="rounded-full border border-[var(--border)] px-5 py-2.5 text-sm font-semibold text-[var(--muted)]"
                >
                  Back
                </button>
                <button
                  type="button"
                  onClick={() => navigate(`/deploy/${primaryModelId || "demo"}`)}
                  className="rounded-xl bg-[var(--ink)] px-5 py-2.5 text-sm font-semibold text-[var(--surface-solid)]"
                >
                  Open deploy console
                </button>
              </div>
            </GlassCard>
          ) : null}
        </div>
      </div>

    </div>
  );
}
