import React, { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { downloadReport, generateReport, getBestModel, getReportStatus, getTrainResults, predict } from "../api";

const LAST_JOB_KEY = "unifiedai:lastJobId";
const LAST_MODEL_KEY = "unifiedai:lastModelId";

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

function numericMetric(model) {
  const values = Object.values(model?.metrics || {});
  const first = values.find((value) => typeof value === "number") ?? values[0];
  return typeof first === "number" ? first : Number(first) || 0;
}

function orderModels(models = [], bestModelId) {
  return [...models].sort((left, right) => {
    if (left.model_id === bestModelId) return -1;
    if (right.model_id === bestModelId) return 1;
    return numericMetric(right) - numericMetric(left);
  });
}

function GlassCard({ className = "", children }) {
  return <div className={classNames("glass-panel rounded-2xl", className)}>{children}</div>;
}

function SoftCard({ className = "", children }) {
  return <div className={classNames("soft-card rounded-xl", className)}>{children}</div>;
}

function MetricTile({ label, value }) {
  return (
    <div className="rounded-xl border border-[var(--border)] bg-[var(--surface-alt)] px-4 py-4">
      <div className="text-xl font-semibold text-[var(--ink)]">{value}</div>
      <div className="mt-1 text-sm text-[var(--muted)]">{label}</div>
    </div>
  );
}

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function buildTestDefaults(modelType, champion) {
  const inputFeatures = Array.isArray(champion?.input_features) ? champion.input_features.filter(Boolean) : [];
  const featureNames = Array.isArray(champion?.features) ? champion.features.filter(Boolean) : [];
  const shapFeatures = Array.isArray(champion?.shap?.top_features)
    ? champion.shap.top_features.map((item) => item.feature).filter(Boolean)
    : [];
  const columns = inputFeatures.length ? inputFeatures : (featureNames.length ? featureNames : shapFeatures).slice(0, 8);

  if (modelType === "nlp") {
    return {
      columnsText: "",
      payloadText: JSON.stringify(["Example text for prediction"], null, 2),
    };
  }

  const safeColumns = columns.length ? columns : ["feature_1", "feature_2", "feature_3", "feature_4"];
  return {
    columnsText: safeColumns.join(", "),
    payloadText: JSON.stringify([safeColumns.map(() => 0)], null, 2),
  };
}

function formatDisplayValue(value) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "number") {
    return value.toLocaleString(undefined, {
      maximumFractionDigits: value >= 100 ? 2 : 4,
    });
  }
  if (typeof value === "boolean") return value ? "Yes" : "No";
  return String(value);
}

function normalizeSubmittedRows(payload, columns, modelType) {
  if (!Array.isArray(payload)) return [];

  if (modelType === "nlp") {
    return payload.map((value, index) => ({
      id: index + 1,
      values: { text: value },
    }));
  }

  return payload.map((row, index) => {
    if (row && typeof row === "object" && !Array.isArray(row)) {
      return { id: index + 1, values: row };
    }

    const mapped = {};
    (columns || []).forEach((column, columnIndex) => {
      mapped[column] = Array.isArray(row) ? row[columnIndex] : undefined;
    });
    return { id: index + 1, values: mapped };
  });
}

export default function ResultsPage() {
  const { jobId } = useParams();
  const navigate = useNavigate();

  const [results, setResults] = useState(null);
  const [bestModel, setBestModel] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [reportLoading, setReportLoading] = useState(false);
  const [reportId, setReportId] = useState(null);
  const [testColumnsText, setTestColumnsText] = useState("");
  const [testPayloadText, setTestPayloadText] = useState("");
  const [testLoading, setTestLoading] = useState(false);
  const [testError, setTestError] = useState("");
  const [testResult, setTestResult] = useState(null);

  useEffect(() => {
    if (!jobId) return;

    setLoading(true);
    setError("");

    Promise.all([getTrainResults(jobId), getBestModel(jobId)])
      .then(([resultsResponse, bestResponse]) => {
        const ordered = orderModels(resultsResponse.data.models, resultsResponse.data.best_model_id);
        setResults({ ...resultsResponse.data, models: ordered });
        setBestModel(bestResponse.data);
        localStorage.setItem(LAST_JOB_KEY, jobId);
        if (bestResponse.data?.model_id) localStorage.setItem(LAST_MODEL_KEY, bestResponse.data.model_id);
      })
      .catch((requestError) => {
        setError(requestError.response?.data?.detail || "Could not load results.");
      })
      .finally(() => setLoading(false));
  }, [jobId]);

  useEffect(() => {
    const ordered = orderModels(results?.models || [], results?.best_model_id || bestModel?.model_id);
    const champion = bestModel || ordered[0];
    if (!champion) return;

    const defaults = buildTestDefaults(results?.model_type || "ml", champion);
    setTestColumnsText(defaults.columnsText);
    setTestPayloadText(defaults.payloadText);
    setTestResult(null);
    setTestError("");
  }, [bestModel, results]);

  const handleGenerateReport = async () => {
    if (!jobId) return;

    setReportLoading(true);
    setError("");

    try {
      const startResponse = await generateReport(jobId);

      if (startResponse.data.report_id && startResponse.data.status === "done") {
        setReportId(startResponse.data.report_id);
        return;
      }

      for (let attempt = 0; attempt < 30; attempt += 1) {
        await sleep(2000);
        const statusResponse = await getReportStatus(jobId);
        if (statusResponse.data.status === "done" && statusResponse.data.report_id) {
          setReportId(statusResponse.data.report_id);
          return;
        }
        if (statusResponse.data.status === "error") {
          throw new Error("Report generation failed.");
        }
      }

      throw new Error("Report generation timed out.");
    } catch (reportError) {
      setError(reportError.response?.data?.detail || reportError.message || "Report generation failed.");
    } finally {
      setReportLoading(false);
    }
  };

  const handleDownloadReport = async (format) => {
    if (!reportId) return;

    try {
      const response = await downloadReport(reportId, format);
      const url = window.URL.createObjectURL(response.data);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `report_${jobId}.${format}`;
      anchor.click();
      window.URL.revokeObjectURL(url);
    } catch (downloadError) {
      setError(downloadError.response?.data?.detail || "Download failed.");
    }
  };

  const handleRunTest = async (champion, modelType) => {
    if (!champion?.model_id) return;

    if (champion.framework !== "sklearn") {
      setTestError(`Interactive testing is currently available for sklearn models. This champion uses ${champion.framework || "an unsupported framework"}.`);
      setTestResult(null);
      return;
    }

    let parsedPayload;
    try {
      parsedPayload = JSON.parse(testPayloadText);
    } catch {
      setTestError("Sample input must be valid JSON.");
      setTestResult(null);
      return;
    }

    if (!Array.isArray(parsedPayload)) {
      setTestError("Sample input must be a JSON array.");
      setTestResult(null);
      return;
    }

    const columns =
      modelType === "nlp"
        ? []
        : testColumnsText
            .split(",")
            .map((column) => column.trim())
            .filter(Boolean);

    setTestLoading(true);
    setTestError("");
    setTestResult(null);

    try {
      const response = await predict(champion.model_id, {
        data: parsedPayload,
        columns,
      });
      setTestResult({
        ...response.data,
        submittedPayload: parsedPayload,
        submittedColumns: columns,
      });
    } catch (requestError) {
      setTestError(requestError.response?.data?.detail || "Prediction failed.");
    } finally {
      setTestLoading(false);
    }
  };

  if (loading) {
    return (
      <GlassCard className="p-10 text-center">
        <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-[var(--muted)]">Loading</div>
        <div className="mt-3 text-2xl font-semibold text-[var(--ink)]">Fetching model results</div>
      </GlassCard>
    );
  }

  if (error && !results) {
    return (
      <GlassCard className="p-10 text-center">
        <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-red-600">Results unavailable</div>
        <div className="mt-3 text-2xl font-semibold text-[var(--ink)]">{error}</div>
        <button
          type="button"
          onClick={() => navigate(-1)}
          className="mt-6 rounded-full bg-[var(--ink)] px-5 py-2.5 text-sm font-semibold text-[var(--surface-solid)]"
        >
          Go back
        </button>
      </GlassCard>
    );
  }

  const models = orderModels(results?.models || [], results?.best_model_id || bestModel?.model_id);
  const champion = bestModel || models[0];
  const metrics = Object.entries(champion?.metrics || {});
  const topFeatures = (champion?.shap?.top_features || []).slice(0, 6);
  const modelType = results?.model_type || "ml";
  const maxScore = Math.max(...models.map((model) => numericMetric(model)), 1);
  const supportsInteractiveTest = champion?.framework === "sklearn";
  const targetColumn = champion?.target_column || "";
  const predictionCount = Array.isArray(testResult?.predictions) ? testResult.predictions.length : 0;
  const submittedRows = normalizeSubmittedRows(testResult?.submittedPayload, testResult?.submittedColumns, modelType);
  const firstPrediction = predictionCount ? testResult.predictions[0] : null;

  return (
    <div className="space-y-5">
      <GlassCard className="p-5 lg:p-6">
        <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <h1 className="text-2xl font-semibold text-[var(--ink)]">Training results</h1>
            <p className="mt-2 text-sm leading-6 text-[var(--muted)]">
              Review the best model, compare scores, test it with different inputs, and export the report.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={handleGenerateReport}
              disabled={reportLoading}
              className="rounded-full border border-[var(--border)] bg-[var(--surface-solid)] px-5 py-2.5 text-sm font-semibold text-[var(--ink)] disabled:opacity-45"
            >
              {reportLoading ? "Generating report..." : reportId ? "Refresh report" : "Generate report"}
            </button>
            {reportId ? (
              <>
                <button
                  type="button"
                  onClick={() => handleDownloadReport("pdf")}
                  className="rounded-full bg-[var(--ink)] px-5 py-2.5 text-sm font-semibold text-[var(--surface-solid)]"
                >
                  Download PDF
                </button>
                <button
                  type="button"
                  onClick={() => handleDownloadReport("docx")}
                  className="rounded-full border border-[var(--border)] bg-[var(--surface-solid)] px-5 py-2.5 text-sm font-semibold text-[var(--ink)]"
                >
                  Download DOCX
                </button>
              </>
            ) : null}
            {champion?.model_id ? (
              <button
                type="button"
                onClick={() => {
                  localStorage.setItem(LAST_MODEL_KEY, champion.model_id);
                  navigate("/batch");
                }}
                className="rounded-full border border-[var(--border)] bg-[var(--surface-solid)] px-5 py-2.5 text-sm font-semibold text-[var(--ink)]"
              >
                Batch test inputs
              </button>
            ) : null}
            {champion?.model_id ? (
              <button
                type="button"
                onClick={() => navigate(`/deploy/${champion.model_id}`)}
                className="rounded-full bg-[var(--accent)] px-5 py-2.5 text-sm font-semibold text-[var(--surface-solid)]"
              >
                Deploy champion
              </button>
            ) : null}
          </div>
        </div>

        {error ? (
          <div className="mt-5 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-700">
            {error}
          </div>
        ) : null}
      </GlassCard>

      <GlassCard className="p-5 lg:p-6">
        <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <div className="text-sm font-semibold text-[var(--muted)]">Best model</div>
            <div className="mt-2 text-3xl font-semibold text-[var(--ink)]">{champion?.algorithm || "Unavailable"}</div>
            <div className="mt-2 text-sm text-[var(--muted)]">
              {modelType} / {champion?.framework || "framework"}
            </div>
          </div>

          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {metrics.slice(0, 4).map(([metric, value]) => (
              <MetricTile key={metric} label={metric.replace(/_/g, " ")} value={shortMetric(value)} />
            ))}
          </div>
        </div>
      </GlassCard>

      <div className="grid items-start gap-5 lg:gap-6 xl:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
        <div className="space-y-5 lg:space-y-6">
          <SoftCard className="p-5">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
            <div>
              <div className="text-lg font-semibold text-[var(--ink)]">Test trained model with different inputs</div>
              <div className="mt-1 text-sm leading-6 text-[var(--muted)]">
                Enter sample inputs as JSON, run a quick prediction, and validate the trained model before deployment.
              </div>
            </div>

            <span className="rounded-full border border-[var(--border)] bg-[var(--surface-alt)] px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.16em] text-[var(--muted)]">
              {supportsInteractiveTest ? "Ready to test" : "Framework limited"}
            </span>
          </div>

          {!supportsInteractiveTest ? (
            <div className="mt-5 rounded-xl border border-[var(--border)] bg-[var(--surface-alt)] px-4 py-3 text-sm leading-6 text-[var(--muted)]">
              Interactive testing is currently available for sklearn models. This champion uses {champion?.framework || "another framework"}.
            </div>
          ) : (
            <>
              {modelType !== "nlp" ? (
                <label className="mt-5 block text-sm text-[var(--muted)]">
                  Feature columns
                  <input
                    type="text"
                    value={testColumnsText}
                    onChange={(event) => setTestColumnsText(event.target.value)}
                    className="mt-2 w-full rounded-xl border border-[var(--border)] bg-[var(--surface-solid)] px-4 py-3 text-[var(--ink)] outline-none focus:border-[var(--accent)]"
                    placeholder="feature_1, feature_2, feature_3"
                  />
                </label>
              ) : null}

              <label className="mt-5 block text-sm text-[var(--muted)]">
                Sample input JSON
                <textarea
                  value={testPayloadText}
                  onChange={(event) => setTestPayloadText(event.target.value)}
                  className="mt-2 min-h-[160px] w-full rounded-xl border border-[var(--border)] bg-[var(--surface-solid)] px-4 py-3 font-mono text-sm text-[var(--ink)] outline-none focus:border-[var(--accent)]"
                  placeholder='[[0, 0, 0, 0]]'
                />
              </label>

              <div className="mt-3 text-xs leading-6 text-[var(--muted)]">
                Use one array per row for tabular models. If your model was trained with named columns, keep the column order aligned with the payload values. For larger validation runs, open batch testing and upload a file of new inputs.
              </div>

              <div className="mt-5 flex flex-wrap gap-3">
                <button
                  type="button"
                  onClick={() => handleRunTest(champion, modelType)}
                  disabled={testLoading}
                  className="rounded-xl bg-[var(--ink)] px-5 py-2.5 text-sm font-semibold text-[var(--surface-solid)] disabled:opacity-45"
                >
                  {testLoading ? "Running test..." : "Run test prediction"}
                </button>
                <button
                  type="button"
                  onClick={() => {
                    if (champion?.model_id) localStorage.setItem(LAST_MODEL_KEY, champion.model_id);
                    navigate("/batch");
                  }}
                  className="rounded-xl border border-[var(--border)] bg-[var(--surface-solid)] px-5 py-2.5 text-sm font-semibold text-[var(--ink)]"
                >
                  Open batch testing
                </button>
                <button
                  type="button"
                  onClick={() => {
                    const defaults = buildTestDefaults(modelType, champion);
                    setTestColumnsText(defaults.columnsText);
                    setTestPayloadText(defaults.payloadText);
                    setTestError("");
                    setTestResult(null);
                  }}
                  className="rounded-xl border border-[var(--border)] bg-[var(--surface-solid)] px-5 py-2.5 text-sm font-semibold text-[var(--ink)]"
                >
                  Reset sample
                </button>
              </div>
            </>
          )}
          </SoftCard>

          <SoftCard className="p-5">
            <div className="text-lg font-semibold text-[var(--ink)]">Leaderboard</div>
            <div className="mt-5 space-y-4">
              {models.map((model, index) => {
                const score = numericMetric(model);
                return (
                  <div key={model.model_id}>
                    <div className="mb-2 flex items-center justify-between text-sm">
                      <span className="font-medium text-[var(--ink)]">{model.algorithm}</span>
                      <span className="text-[var(--muted)]">{shortMetric(score)}</span>
                    </div>
                    <div className="h-6 overflow-hidden rounded-full bg-[rgba(23,33,38,0.08)]">
                      <div
                        className={classNames(
                          "flex h-full items-center rounded-full px-3 text-[10px] font-semibold",
                          index === 0 ? "bg-[var(--accent)] text-[var(--surface-solid)]" : "bg-[var(--accent-soft)] text-[var(--accent)]"
                        )}
                        style={{ width: `${Math.max(12, Math.min((score / maxScore) * 100, 100))}%` }}
                      >
                        {index === 0 ? "Best" : "Candidate"}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </SoftCard>

          <SoftCard className="p-5">
          <div className="text-lg font-semibold text-[var(--ink)]">SHAP feature importance</div>
          <div className="mt-5 space-y-4">
            {topFeatures.length === 0 ? (
              <div className="text-sm text-[var(--muted)]">Explainability artifacts were not attached to this model.</div>
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

        <div className="space-y-5 lg:space-y-6">
          <SoftCard className="p-5 xl:max-h-[42rem] xl:overflow-y-auto">
            <div className="text-lg font-semibold text-[var(--ink)]">Prediction output</div>
            <div className="mt-1 text-sm leading-6 text-[var(--muted)]">
              {targetColumn
                ? `Review the predicted ${targetColumn} from the trained champion model.`
                : "Review the live output from the trained champion model."}
            </div>

            {targetColumn ? (
              <div className="mt-4 inline-flex rounded-full border border-[var(--border)] bg-[var(--surface-alt)] px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">
                Output column: {targetColumn}
              </div>
            ) : null}

            {testError ? (
              <div className="mt-5 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                {testError}
              </div>
            ) : null}

            {testResult ? (
              <div className="mt-5 space-y-4">
                <div className="grid gap-3 sm:grid-cols-2">
                  <MetricTile label="Predictions returned" value={predictionCount} />
                  <MetricTile
                    label={predictionCount === 1 ? `Predicted ${targetColumn || "output"}` : `First predicted ${targetColumn || "output"}`}
                    value={firstPrediction !== null ? formatDisplayValue(firstPrediction) : "--"}
                  />
                </div>

                {submittedRows.length ? (
                  <div className="space-y-3">
                    {submittedRows.map((row, index) => (
                      <div key={row.id} className="rounded-xl border border-[var(--border)] bg-[var(--surface-alt)] p-4">
                        <div className="flex items-center justify-between gap-3">
                          <div className="text-sm font-semibold text-[var(--ink)]">
                            {predictionCount > 1 ? `Input row ${row.id}` : "Submitted input"}
                          </div>
                          <div className="rounded-full bg-[var(--surface-solid)] px-3 py-1 text-xs font-semibold text-[var(--accent)]">
                            Predicted {targetColumn || "output"}: {formatDisplayValue(testResult.predictions?.[index])}
                          </div>
                        </div>

                        <div className="mt-3 grid gap-2 sm:grid-cols-2">
                          {Object.entries(row.values).map(([key, value]) => (
                            <div key={`${row.id}-${key}`} className="rounded-lg bg-[var(--surface-solid)] px-3 py-2">
                              <div className="text-[11px] font-semibold uppercase tracking-[0.12em] text-[var(--muted)]">
                                {key.replace(/_/g, " ")}
                              </div>
                              <div className="mt-1 text-sm leading-5 text-[var(--ink)] break-words">
                                {formatDisplayValue(value)}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : null}

                <details className="rounded-xl border border-[var(--border)] bg-[var(--surface-alt)]">
                  <summary className="cursor-pointer list-none px-4 py-3 text-sm font-semibold text-[var(--ink)]">
                    Raw response
                  </summary>
                  <div className="border-t border-[var(--border)] p-3">
                    <div className="max-h-[220px] overflow-auto rounded-xl bg-[var(--surface-dark)] p-4 text-xs leading-7 text-emerald-300">
                      <pre className="whitespace-pre-wrap break-words font-mono">
                        {JSON.stringify(testResult, null, 2)}
                      </pre>
                    </div>
                  </div>
                </details>
              </div>
            ) : (
              <div className="mt-5 rounded-xl border border-[var(--border)] bg-[var(--surface-alt)] px-4 py-4 text-sm leading-6 text-[var(--muted)]">
                Run a sample prediction with new inputs to verify the model before deployment.
              </div>
            )}
          </SoftCard>

          <SoftCard className="p-5">
            <div className="text-lg font-semibold text-[var(--ink)]">Metrics and configuration</div>
            <div className="mt-5 space-y-3">
              {metrics.length === 0 ? (
                <div className="text-sm text-[var(--muted)]">No metrics were recorded for this run.</div>
              ) : (
                metrics.map(([metric, value]) => (
                  <div
                    key={metric}
                    className="flex items-center justify-between rounded-xl border border-[var(--border)] bg-[var(--surface-alt)] px-4 py-3 text-sm"
                  >
                    <span className="capitalize text-[var(--muted)]">{metric.replace(/_/g, " ")}</span>
                    <span className="font-semibold text-[var(--ink)]">{shortMetric(value)}</span>
                  </div>
                ))
              )}
            </div>

            <div className="mt-5 text-sm font-semibold text-[var(--ink)]">Best hyperparameters</div>
            <div className="mt-3 space-y-3">
              {Object.keys(champion?.hyperparams || {}).length === 0 ? (
                <div className="text-sm text-[var(--muted)]">No hyperparameters saved for the champion.</div>
              ) : (
                Object.entries(champion.hyperparams).map(([key, value]) => (
                  <div
                    key={key}
                    className="flex items-center justify-between rounded-xl border border-[var(--border)] bg-[var(--surface-alt)] px-4 py-3 text-sm"
                  >
                    <span className="font-mono text-[var(--muted)]">{key}</span>
                    <span className="font-mono font-semibold text-[var(--ink)]">{String(value)}</span>
                  </div>
                ))
              )}
            </div>
          </SoftCard>

          <SoftCard className="p-5">
          <div className="text-lg font-semibold text-[var(--ink)]">Full model inventory</div>
          <div className="mt-5 overflow-x-auto">
            <table className="min-w-full text-left text-sm">
              <thead>
                <tr className="border-b border-[var(--border)] text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--muted)]">
                  <th className="pb-3">Algorithm</th>
                  <th className="pb-3">Framework</th>
                  <th className="pb-3">Key score</th>
                  <th className="pb-3">Action</th>
                </tr>
              </thead>
              <tbody>
                {models.map((model, index) => (
                  <tr key={model.model_id} className="border-b border-[rgba(23,33,38,0.06)]">
                    <td className="py-4">
                      <div className="font-semibold text-[var(--ink)]">{model.algorithm}</div>
                      <div className="text-xs text-[var(--muted)]">{index === 0 ? "Champion model" : "Alternative candidate"}</div>
                    </td>
                    <td className="py-4 text-[var(--muted)]">{model.framework}</td>
                    <td className="py-4 font-mono text-[var(--ink)]">{shortMetric(numericMetric(model))}</td>
                    <td className="py-4">
                      <button
                        type="button"
                        onClick={() => navigate(`/deploy/${model.model_id}`)}
                        className="rounded-full border border-[var(--border)] px-3 py-1.5 text-xs font-semibold text-[var(--ink)]"
                      >
                        Deploy
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          </SoftCard>
        </div>
      </div>
    </div>
  );
}
