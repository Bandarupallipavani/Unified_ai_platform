import { useEffect, useState } from "react";
import { checkDrift, getDriftAlerts, listModels, resolveAlert } from "../api";

export default function DriftMonitorPage() {
  const [models, setModels] = useState([]);
  const [modelId, setModelId] = useState("");
  const [alerts, setAlerts] = useState([]);
  const [results, setResults] = useState(null);
  const [jsonInput, setJsonInput] = useState('[\n  {"age": 35, "income": 55000, "tenure": 3}\n]');
  const [loading, setLoading] = useState(false);
  const [toast, setToast] = useState("");

  useEffect(() => {
    listModels()
      .then((response) => setModels(response.data))
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!modelId) {
      setAlerts([]);
      return;
    }

    getDriftAlerts(modelId)
      .then((response) => setAlerts(response.data))
      .catch(() => {});
  }, [modelId]);

  const handleCheck = async () => {
    let sampleData;
    try {
      sampleData = JSON.parse(jsonInput);
    } catch {
      setToast("Invalid JSON input.");
      return;
    }

    if (!modelId) {
      setToast("Please select a model.");
      return;
    }

    setLoading(true);

    try {
      const response = await checkDrift(modelId, sampleData);
      setResults(response.data);
      const alertsResponse = await getDriftAlerts(modelId);
      setAlerts(alertsResponse.data);
      setToast(
        response.data.alert_count > 0
          ? `Drift detected in ${response.data.alert_count} feature(s).`
          : "No significant drift detected."
      );
    } catch (error) {
      setToast(`Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleResolve = async (alertId) => {
    try {
      await resolveAlert(alertId);
      setAlerts((previous) => previous.filter((alert) => alert.id !== alertId));
      setToast("Alert resolved.");
    } catch (error) {
      setToast(`Error: ${error.message}`);
    }
  };

  return (
    <div className="space-y-6">
      <div className="glass-panel rounded-[28px] p-6">
        <div className="text-[11px] font-semibold uppercase tracking-[0.24em] text-[var(--muted)]">Drift monitor</div>
        <h1 className="mt-2 text-3xl font-semibold text-[var(--ink)]">Production input drift</h1>
        <p className="mt-3 max-w-3xl text-sm leading-6 text-[var(--muted)]">
          Compare recent inference inputs against the training distribution and review any active drift alerts.
        </p>
      </div>

      {toast ? (
        <div className="rounded-[22px] border border-[var(--border)] bg-[var(--surface-solid)] px-4 py-3 text-sm text-[var(--ink)]">
          {toast}
        </div>
      ) : null}

      <div className="grid gap-6 xl:grid-cols-[320px_minmax(0,1fr)]">
        <div className="space-y-6">
          <div className="soft-card rounded-[24px] p-5">
            <div className="text-sm font-semibold text-[var(--ink)]">Model selection</div>
            <select
              value={modelId}
              onChange={(event) => setModelId(event.target.value)}
              className="mt-4 w-full rounded-[18px] border border-[var(--border)] bg-[var(--surface-solid)] px-4 py-3 text-sm text-[var(--ink)]"
            >
              <option value="">Choose a model</option>
              {models.map((model) => (
                <option key={model.model_id} value={model.model_id}>
                  {model.algorithm} ({model.model_type})
                </option>
              ))}
            </select>
          </div>

          <div className="soft-card rounded-[24px] p-5">
            <div className="text-sm font-semibold text-[var(--ink)]">Recent input sample</div>
            <textarea
              rows={10}
              value={jsonInput}
              onChange={(event) => setJsonInput(event.target.value)}
              className="mt-4 w-full rounded-[18px] border border-[var(--border)] bg-[var(--surface-solid)] px-4 py-3 font-mono text-xs text-[var(--ink)]"
            />
            <button
              type="button"
              onClick={handleCheck}
              disabled={loading}
              className="mt-4 rounded-full bg-[var(--accent)] px-5 py-2.5 text-sm font-semibold text-[var(--surface-solid)] disabled:opacity-45"
            >
              {loading ? "Checking..." : "Run drift check"}
            </button>
          </div>
        </div>

        <div className="space-y-6">
          <div className="soft-card rounded-[24px] p-5">
            <div className="text-sm font-semibold text-[var(--ink)]">Drift results</div>
            {!results ? (
              <div className="mt-4 text-sm text-[var(--muted)]">Run a drift check to see feature-by-feature results.</div>
            ) : (
              <div className="mt-4 overflow-x-auto">
                <table className="min-w-full text-left text-sm">
                  <thead>
                    <tr className="border-b border-[var(--border)] text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--muted)]">
                      <th className="pb-3">Feature</th>
                      <th className="pb-3">Method</th>
                      <th className="pb-3">Score</th>
                      <th className="pb-3">Threshold</th>
                      <th className="pb-3">State</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(results.drift_results || {}).map(([feature, info]) => (
                      <tr key={feature} className="border-b border-[rgba(23,33,38,0.06)]">
                        <td className="py-3 font-semibold text-[var(--ink)]">{feature}</td>
                        <td className="py-3 text-[var(--muted)]">{info.method}</td>
                        <td className="py-3 font-mono text-[var(--ink)]">
                          {typeof info.score === "number" ? info.score.toFixed(4) : info.score}
                        </td>
                        <td className="py-3 font-mono text-[var(--muted)]">
                          {typeof info.threshold === "number" ? info.threshold.toFixed(2) : "--"}
                        </td>
                        <td className="py-3">
                          <span
                            className={`rounded-full px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] ${
                              info.drifted ? "bg-red-50 text-red-600" : "bg-[var(--success-soft)] text-[var(--success)]"
                            }`}
                          >
                            {info.drifted ? "Drift" : "Stable"}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          <div className="soft-card rounded-[24px] p-5">
            <div className="text-sm font-semibold text-[var(--ink)]">Active alerts</div>
            {alerts.length === 0 ? (
              <div className="mt-4 text-sm text-[var(--muted)]">No active drift alerts for this model.</div>
            ) : (
              <div className="mt-4 space-y-3">
                {alerts.map((alert) => (
                  <div
                    key={alert.id}
                    className="flex flex-col gap-3 rounded-[20px] border border-red-200 bg-red-50 px-4 py-4 sm:flex-row sm:items-center"
                  >
                    <div className="flex-1">
                      <div className="font-semibold text-red-700">{alert.feature}</div>
                      <div className="mt-1 text-xs uppercase tracking-[0.16em] text-red-500">{alert.alert_type}</div>
                    </div>
                    <div className="font-mono text-sm text-red-700">{Number(alert.drift_score || 0).toFixed(4)}</div>
                    <button
                      type="button"
                      onClick={() => handleResolve(alert.id)}
                      className="rounded-full border border-red-200 bg-white px-4 py-2 text-xs font-semibold text-red-700"
                    >
                      Resolve
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
