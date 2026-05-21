import { useCallback, useEffect, useState } from "react";
import { compareModels, getModelVersions, listModels, promoteModel } from "../api";

const METRIC_KEYS = ["accuracy", "f1_score", "roc_auc", "r2", "rmse", "mae", "cv_mean"];

function primaryMetric(metrics) {
  if (!metrics) return "N/A";
  for (const key of METRIC_KEYS) {
    if (metrics[key] !== undefined) return `${key}: ${metrics[key]}`;
  }
  return JSON.stringify(metrics).slice(0, 40);
}

function typeBadgeClass(modelType) {
  if (modelType === "ml") return "border border-[var(--border)] bg-[var(--surface-alt)] text-[var(--muted)]";
  if (modelType === "dl") return "border border-[var(--border)] bg-[var(--surface-alt)] text-[var(--muted)]";
  return "border border-[var(--border)] bg-[var(--surface-alt)] text-[var(--muted)]";
}

export default function ModelRegistryPage() {
  const [models, setModels] = useState([]);
  const [filter, setFilter] = useState({ model_type: "", production_only: false });
  const [selected, setSelected] = useState(null);
  const [compareWith, setCompare] = useState(null);
  const [comparison, setComparison] = useState(null);
  const [versions, setVersions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [toast, setToast] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const params = {};
      if (filter.model_type) params.model_type = filter.model_type;
      if (filter.production_only) params.production_only = "true";
      const response = await listModels(params);
      setModels(response.data);
    } catch (error) {
      setToast(`Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  }, [filter]);

  useEffect(() => {
    load();
  }, [load]);

  const handlePromote = async (modelId) => {
    const notes = prompt("Add a note for this version (optional):", "Promoted to production");
    try {
      await promoteModel(modelId, notes || "");
      setToast("Model promoted to production.");
      load();
    } catch (error) {
      setToast(`Error: ${error.message}`);
    }
  };

  const handleVersions = async (modelId) => {
    try {
      const response = await getModelVersions(modelId);
      setVersions(response.data);
      setSelected(modelId);
    } catch (error) {
      setToast(`Error: ${error.message}`);
    }
  };

  const handleCompare = async () => {
    if (!selected || !compareWith) {
      setToast("Select two models to compare.");
      return;
    }

    try {
      const response = await compareModels(selected, compareWith);
      setComparison(response.data);
    } catch (error) {
      setToast(`Error: ${error.message}`);
    }
  };

  return (
    <div className="space-y-6">
      <div className="rounded-2xl border border-[var(--border)] bg-white p-6">
        <h1 className="text-2xl font-semibold text-[var(--ink)]">Model registry</h1>
        <p className="mt-2 text-sm text-[var(--muted)]">
          Review stored models, compare versions, and promote the right model to production.
        </p>
      </div>

      {toast ? (
        <div className="rounded-xl border border-blue-200 bg-blue-50 p-3 text-sm text-blue-800">
          {toast}
          <button onClick={() => setToast("")} className="ml-3 font-bold text-blue-500">x</button>
        </div>
      ) : null}

      <div className="rounded-2xl border border-[var(--border)] bg-white p-5">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center">
          <select
            value={filter.model_type}
            onChange={(event) => setFilter({ ...filter, model_type: event.target.value })}
            className="rounded-xl border border-[var(--border)] px-3 py-2 text-sm"
          >
            <option value="">All Types</option>
            <option value="ml">Classic ML</option>
            <option value="dl">Deep Learning</option>
            <option value="nlp">NLP</option>
          </select>

          <label className="flex items-center gap-2 text-sm text-gray-600">
            <input
              type="checkbox"
              checked={filter.production_only}
              onChange={(event) => setFilter({ ...filter, production_only: event.target.checked })}
            />
            Production only
          </label>

          <button
            onClick={load}
            className="rounded-xl border border-[var(--border)] bg-white px-4 py-2 text-sm font-medium text-[var(--ink)] lg:ml-auto"
          >
            Refresh
          </button>
        </div>
      </div>

      {loading ? (
        <div className="py-12 text-center text-gray-500">Loading models...</div>
      ) : (
        <div className="overflow-x-auto rounded-2xl border border-[var(--border)] bg-white">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 text-xs uppercase text-gray-600">
              <tr>
                {["", "Algorithm", "Type", "Framework", "Primary Metric", "Production", "Actions"].map((heading) => (
                  <th key={heading} className="px-4 py-3 text-left font-medium">{heading}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {models.length === 0 ? (
                <tr>
                  <td colSpan={7} className="py-8 text-center text-gray-400">No models found.</td>
                </tr>
              ) : null}

              {models.map((model) => (
                <tr
                  key={model.model_id}
                  className={selected === model.model_id ? "bg-indigo-50" : "hover:bg-gray-50"}
                >
                  <td className="px-4 py-3">
                    <input
                      type="radio"
                      name="selected"
                      checked={selected === model.model_id}
                      onChange={() => setSelected(model.model_id)}
                    />
                  </td>
                  <td className="px-4 py-3 font-medium text-gray-900">{model.algorithm}</td>
                  <td className="px-4 py-3">
                    <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${typeBadgeClass(model.model_type)}`}>
                      {model.model_type?.toUpperCase()}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-gray-500">{model.framework}</td>
                  <td className="px-4 py-3 font-mono text-xs text-gray-700">{primaryMetric(model.metrics)}</td>
                  <td className="px-4 py-3">
                    {model.is_production ? (
                      <span className="rounded-full border border-[var(--border)] bg-[var(--surface-alt)] px-2 py-0.5 text-xs font-bold text-[var(--ink)]">PROD</span>
                    ) : (
                      <span className="text-xs text-gray-400">-</span>
                    )}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex gap-2">
                      <button
                        onClick={() => handlePromote(model.model_id)}
                        className="rounded-lg bg-green-600 px-2 py-1 text-xs text-[var(--surface-solid)]"
                      >
                        Promote
                      </button>
                      <button
                        onClick={() => handleVersions(model.model_id)}
                        className="rounded-lg border border-[var(--border)] bg-white px-2 py-1 text-xs text-gray-700"
                      >
                        Versions
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="rounded-2xl border border-[var(--border)] bg-white p-5">
        <h2 className="mb-3 font-semibold text-gray-800">Compare two models</h2>
        <div className="flex flex-col gap-3 lg:flex-row lg:items-end">
          <div className="flex-1">
            <label className="mb-1 block text-xs text-gray-500">Model A</label>
            <div className="min-h-[40px] rounded-xl border border-[var(--border)] bg-[var(--surface-alt)] px-3 py-2 text-sm text-gray-700">
              {selected ? models.find((model) => model.model_id === selected)?.algorithm || selected : "Select a model above"}
            </div>
          </div>

          <div className="flex-1">
            <label className="mb-1 block text-xs text-gray-500">Model B</label>
            <select
              value={compareWith || ""}
              onChange={(event) => setCompare(event.target.value)}
              className="w-full rounded-xl border border-[var(--border)] px-3 py-2 text-sm"
            >
              <option value="">Select model B...</option>
              {models
                .filter((model) => model.model_id !== selected)
                .map((model) => (
                  <option key={model.model_id} value={model.model_id}>
                    {model.algorithm} ({model.model_type})
                  </option>
                ))}
            </select>
          </div>

          <button
            onClick={handleCompare}
            className="rounded-xl bg-[var(--accent)] px-4 py-2 text-sm font-semibold text-[var(--surface-solid)]"
          >
            Compare
          </button>
        </div>

        {comparison ? (
          <div className="mt-4 grid gap-4 lg:grid-cols-2">
            {[comparison.model_a, comparison.model_b].map((model, index) => (
              <div key={index} className="rounded-xl border border-[var(--border)] bg-[var(--surface-alt)] p-4">
                <div className="mb-2 font-semibold text-gray-800">
                  {["Model A", "Model B"][index]}: {model.algorithm}
                </div>
                <div className="mb-2 text-xs text-gray-500">{model.framework}</div>
                <table className="w-full text-xs">
                  <tbody>
                    {model.metrics && Object.entries(model.metrics).map(([key, value]) => (
                      <tr key={key} className="border-b border-gray-200">
                        <td className="py-1 text-gray-600">{key}</td>
                        <td className="py-1 text-right font-mono font-bold text-gray-900">{value}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ))}
          </div>
        ) : null}
      </div>

      {versions.length > 0 ? (
        <div className="rounded-2xl border border-[var(--border)] bg-white p-5">
          <h2 className="mb-3 font-semibold text-gray-800">Version history - {selected}</h2>
          <div className="space-y-2">
            {versions.map((version) => (
              <div
                key={version.version}
                className={`flex items-center gap-4 rounded-lg border p-3 text-sm ${
                  version.is_active ? "border-[var(--border)] bg-[var(--surface-alt)]" : "border-gray-200 bg-gray-50"
                }`}
              >
                <span className="w-16 font-bold text-gray-700">v{version.version}</span>
                <span className="flex-1 text-gray-500">{version.notes || "-"}</span>
                <span className="font-mono text-xs text-gray-600">
                  {version.metrics ? Object.entries(version.metrics).map(([key, value]) => `${key}=${value}`).join(" | ") : ""}
                </span>
                <span className="text-xs text-gray-400">{new Date(version.created_at).toLocaleDateString()}</span>
                {version.is_active ? <span className="rounded border border-[var(--border)] bg-[var(--surface-solid)] px-2 py-0.5 text-xs text-[var(--ink)]">Active</span> : null}
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}
