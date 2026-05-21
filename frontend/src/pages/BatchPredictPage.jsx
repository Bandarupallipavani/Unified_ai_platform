import { useEffect, useRef, useState } from "react";
import { batchPredictUpload, downloadBatchCSV, getBatchStatus, listModels } from "../api";

const STATUS_COLORS = {
  queued: "bg-amber-50 text-amber-700",
  running: "bg-[var(--accent-soft)] text-[var(--accent)]",
  completed: "bg-[var(--success-soft)] text-[var(--success)]",
  failed: "bg-red-50 text-red-600",
};
const LAST_MODEL_KEY = "unifiedai:lastModelId";

export default function BatchPredictPage() {
  const [models, setModels] = useState([]);
  const [modelId, setModelId] = useState(() => localStorage.getItem(LAST_MODEL_KEY) || "");
  const [file, setFile] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [toast, setToast] = useState("");
  const pollerRef = useRef(null);

  useEffect(() => {
    listModels()
      .then((response) => setModels(response.data))
      .catch(() => {});

    return () => window.clearInterval(pollerRef.current);
  }, []);

  useEffect(() => {
    window.clearInterval(pollerRef.current);

    const pending = jobs.filter((job) => ["queued", "running"].includes(job.status));
    if (pending.length === 0) return undefined;

    pollerRef.current = window.setInterval(async () => {
      const nextJobs = await Promise.all(
        jobs.map(async (job) => {
          if (!["queued", "running"].includes(job.status)) return job;

          try {
            const response = await getBatchStatus(job.batchId);
            return {
              ...job,
              status: response.data.status,
              rowCount: response.data.row_count,
            };
          } catch {
            return job;
          }
        })
      );

      setJobs(nextJobs);
    }, 3000);

    return () => window.clearInterval(pollerRef.current);
  }, [jobs]);

  const handleSubmit = async () => {
    if (!modelId) {
      setToast("Please select a model.");
      return;
    }

    if (!file) {
      setToast("Please select a CSV or Parquet file.");
      return;
    }

    setLoading(true);

    try {
      const response = await batchPredictUpload(modelId, file);
      setJobs((previous) => [
        {
          batchId: response.data.batch_id,
          status: response.data.status,
          rowCount: null,
          modelId,
        },
        ...previous,
      ]);
      setToast(`Batch job queued: ${response.data.batch_id}`);
      setFile(null);
    } catch (error) {
      setToast(`Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="glass-panel rounded-[28px] p-6">
        <div className="text-[11px] font-semibold uppercase tracking-[0.24em] text-[var(--muted)]">Batch scoring</div>
        <h1 className="mt-2 text-3xl font-semibold text-[var(--ink)]">Run offline predictions</h1>
        <p className="mt-3 max-w-3xl text-sm leading-6 text-[var(--muted)]">
          Use this after training to test a selected model with many different inputs at once, then download the enriched CSV when the batch job completes.
        </p>
      </div>

      {toast ? (
        <div className="rounded-[22px] border border-[var(--border)] bg-[var(--surface-solid)] px-4 py-3 text-sm text-[var(--ink)]">
          {toast}
        </div>
      ) : null}

      <div className="soft-card rounded-[24px] p-5">
        <div className="text-sm font-semibold text-[var(--ink)]">New batch job</div>
        <div className="mt-5 grid gap-4 md:grid-cols-2">
          <label className="text-sm text-[var(--muted)]">
            Model
            <select
              value={modelId}
              onChange={(event) => {
                setModelId(event.target.value);
                if (event.target.value) localStorage.setItem(LAST_MODEL_KEY, event.target.value);
              }}
              className="mt-2 w-full rounded-[18px] border border-[var(--border)] bg-[var(--surface-solid)] px-4 py-3 text-[var(--ink)]"
            >
              <option value="">Choose a model</option>
              {models.map((model) => (
                <option key={model.model_id} value={model.model_id}>
                  {model.algorithm} ({model.model_type})
                </option>
              ))}
            </select>
          </label>

          <label className="text-sm text-[var(--muted)]">
            Input file
            <input
              type="file"
              accept=".csv,.tsv,.parquet"
              onChange={(event) => setFile(event.target.files?.[0] || null)}
              className="mt-2 block w-full text-sm text-[var(--muted)] file:mr-3 file:rounded-full file:border-0 file:bg-[var(--accent-soft)] file:px-4 file:py-2 file:font-semibold file:text-[var(--accent)]"
            />
          </label>
        </div>

        {file ? (
          <div className="mt-4 rounded-[18px] border border-[var(--border)] bg-[rgba(247,245,239,0.6)] px-4 py-3 text-sm text-[var(--ink)]">
            {file.name} / {(file.size / 1024).toFixed(1)} KB
          </div>
        ) : null}

        <button
          type="button"
          onClick={handleSubmit}
          disabled={loading}
          className="mt-5 rounded-full bg-[var(--accent)] px-5 py-2.5 text-sm font-semibold text-[var(--surface-solid)] disabled:opacity-45"
        >
          {loading ? "Submitting..." : "Run batch prediction"}
        </button>
      </div>

      <div className="soft-card rounded-[24px] p-5">
        <div className="text-sm font-semibold text-[var(--ink)]">Batch jobs</div>
        {jobs.length === 0 ? (
          <div className="mt-4 text-sm text-[var(--muted)]">No batch jobs yet.</div>
        ) : (
          <div className="mt-4 space-y-3">
            {jobs.map((job) => (
              <div
                key={job.batchId}
                className="flex flex-col gap-4 rounded-[20px] border border-[var(--border)] bg-[rgba(247,245,239,0.6)] px-4 py-4 lg:flex-row lg:items-center"
              >
                <div className="flex-1">
                  <div className="font-mono text-sm font-semibold text-[var(--ink)]">{job.batchId}</div>
                  <div className="mt-1 text-xs text-[var(--muted)]">
                    {job.rowCount ? `${job.rowCount.toLocaleString()} rows processed` : "Awaiting row count"}
                  </div>
                </div>

                <span className={`rounded-full px-4 py-2 text-xs font-semibold uppercase tracking-[0.16em] ${STATUS_COLORS[job.status] || "bg-gray-100 text-gray-600"}`}>
                  {job.status}
                </span>

                {job.status === "completed" ? (
                  <a
                    href={downloadBatchCSV(job.batchId)}
                    className="rounded-full bg-[var(--success)] px-4 py-2 text-xs font-semibold text-[var(--surface-solid)]"
                    download
                  >
                    Download CSV
                  </a>
                ) : null}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
