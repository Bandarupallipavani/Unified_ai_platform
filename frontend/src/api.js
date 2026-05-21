const BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";

const getToken = () => localStorage.getItem("token");
const clearToken = () => {
  localStorage.removeItem("token");
  window.dispatchEvent(new Event("auth:unauthorized"));
};

const wrapResponse = (payload, res) => {
  const wrapped = {
    data: payload,
    status: res.status,
    headers: res.headers,
    ok: res.ok,
  };

  if (payload && typeof payload === "object" && !Array.isArray(payload) && !(payload instanceof Blob)) {
    Object.assign(wrapped, payload);
  }

  return wrapped;
};

async function req(method, path, body, isFormData = false) {
  const headers = {};
  const token = getToken();

  if (token) headers.Authorization = `Bearer ${token}`;
  if (!isFormData) headers["Content-Type"] = "application/json";

  const res = await fetch(`${BASE}${path}`, {
    method,
    headers,
    body: isFormData ? body : body ? JSON.stringify(body) : undefined,
  });

  if (!res.ok) {
    const payload = await res.json().catch(() => ({ detail: res.statusText }));
    if (res.status === 401) clearToken();
    const error = new Error(payload.detail || `HTTP ${res.status}`);
    error.response = { status: res.status, data: payload };
    throw error;
  }

  const contentType = res.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    const payload = await res.json();
    return wrapResponse(payload, res);
  }

  const payload = await res.blob();
  return wrapResponse(payload, res);
}

export const register = (body) => req("POST", "/api/auth/register", body);
export const login = (body) => req("POST", "/api/auth/login", body);
export const getMe = () => req("GET", "/api/auth/me");

export const uploadFile = (file) => {
  const data = new FormData();
  data.append("file", file);
  return req("POST", "/api/upload", data, true);
};

export const connectNL = (description) => req("POST", "/api/connect", { description });
export const getProfile = (sessionId) => req("GET", `/api/profile/${sessionId}`);

export const getEDA = (sessionId) => req("GET", `/api/eda/${sessionId}`);

export const getPreprocessPlan = (sessionId) => req("GET", `/api/preprocess/plan/${sessionId}`);
export const runPreprocess = (sessionId, config) =>
  req("POST", "/api/preprocess", { session_id: sessionId, config });

export const startTrain = (sessionId, config) =>
  req("POST", "/api/train", { session_id: sessionId, config });
export const getTrainStatus = (jobId) => req("GET", `/api/train/status/${jobId}`);
export const getTrainResults = (jobId) => req("GET", `/api/train/results/${jobId}`);
export const getBestModel = (jobId) => req("GET", `/api/train/best/${jobId}`);
export const getTrainHistory = () => req("GET", "/api/train/history");

export const listModels = (params = {}) => {
  const query = new URLSearchParams(params).toString();
  return req("GET", `/api/models${query ? `?${query}` : ""}`);
};
export const promoteModel = (modelId, notes) =>
  req("POST", `/api/models/${modelId}/promote`, { notes });
export const getModelVersions = (modelId) => req("GET", `/api/models/${modelId}/versions`);
export const compareModels = (modelId, other) =>
  req("POST", `/api/models/${modelId}/compare`, { compare_with: other });

export const predict = (modelId, payload) =>
  req("POST", `/api/predict/${modelId}`, Array.isArray(payload) ? { data: payload } : payload);

export const batchPredictUpload = (modelId, file) => {
  const data = new FormData();
  data.append("file", file);
  return req("POST", `/api/predict/batch/upload?model_id=${modelId}`, data, true);
};
export const getBatchStatus = (batchId) => req("GET", `/api/predict/batch/${batchId}/status`);
export const downloadBatchCSV = (batchId) =>
  `${BASE}/api/predict/batch/${batchId}/download?token=${getToken()}`;

export const deployModel = (modelId, target, config, traffic_pct) =>
  req("POST", `/api/deploy/${modelId}`, { target, config, traffic_pct });
export const abDeploy = (modelIdA, modelIdB, trafficPctA) =>
  req("POST", "/api/deploy/ab", {
    model_id_a: modelIdA,
    model_id_b: modelIdB,
    traffic_pct_a: trafficPctA,
  });
export const downloadServer = (modelId) => req("GET", `/api/deploy/fastapi/${modelId}`);
export const downloadDockerfile = (modelId) => req("GET", `/api/deploy/dockerfile/${modelId}`);
export const downloadRenderBlueprint = (modelId) => req("GET", `/api/deploy/render-blueprint/${modelId}`);
export const downloadModel = (modelId) => req("GET", `/api/deploy/model/${modelId}`);

export const checkDrift = (modelId, sampleData) =>
  req("POST", "/api/drift/check", { model_id: modelId, sample_data: sampleData });
export const getDriftAlerts = (modelId) => req("GET", `/api/drift/alerts/${modelId}`);
export const resolveAlert = (alertId) => req("PATCH", `/api/drift/alerts/${alertId}/resolve`);

export const createSchedule = (sessionId, cronExpr, config) =>
  req("POST", "/api/schedule", { session_id: sessionId, cron_expr: cronExpr, config });
export const listSchedules = () => req("GET", "/api/schedule");
export const deleteSchedule = (scheduleId) => req("DELETE", `/api/schedule/${scheduleId}`);

export const copilotChat = (sessionId, message) =>
  req("POST", "/api/copilot/chat", { session_id: sessionId, message });
export const copilotHistory = (sessionId) => req("GET", `/api/copilot/history/${sessionId}`);
export const copilotReset = (sessionId) => req("DELETE", `/api/copilot/reset/${sessionId}`);

export const generateReport = (jobId) => req("POST", `/api/report/generate/${jobId}`);
export const getReportStatus = (jobId) => req("GET", `/api/report/status/${jobId}`);
export const downloadReport = (reportId, format) =>
  req("GET", `/api/report/download/${reportId}?fmt=${format}`);

export const healthCheck = () => req("GET", "/health");

export const createTrainingWS = (jobId, onLog, onClose) => {
  const wsUrl = `${BASE.replace(/^http/, "ws")}/ws/train/${jobId}`;
  const ws = new WebSocket(wsUrl);

  ws.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data);
      if (payload.type === "log" && payload.message) onLog?.(payload.message, payload);
    } catch {
      if (event.data) onLog?.(event.data, { type: "raw" });
    }
  };

  ws.onclose = () => onClose?.();
  ws.onerror = () => onClose?.();

  return ws;
};

export const resetCopilot = copilotReset;
export const downloadFastapiServer = downloadServer;
export const nlConnect = connectNL;
