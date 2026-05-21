import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getTrainHistory } from "../api";

const STATUS_STYLES = {
  completed: {
    badge: "bg-[var(--success-soft)] text-[var(--success)]",
    dot: "bg-[var(--success)]",
    label: "Completed",
  },
  running: {
    badge: "bg-[var(--accent-soft)] text-[var(--accent)]",
    dot: "bg-[var(--accent)]",
    label: "Running",
  },
  failed: {
    badge: "bg-red-50 text-red-600",
    dot: "bg-red-500",
    label: "Failed",
  },
  queued: {
    badge: "bg-amber-50 text-amber-700",
    dot: "bg-amber-500",
    label: "Queued",
  },
};

function classNames(...parts) {
  return parts.filter(Boolean).join(" ");
}

function GlassCard({ className = "", children }) {
  return <div className={classNames("glass-panel rounded-[28px]", className)}>{children}</div>;
}

export default function HistoryPage() {
  const navigate = useNavigate();
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    getTrainHistory()
      .then((response) => setJobs(response.data))
      .catch((requestError) => {
        setError(requestError.response?.data?.detail || "Could not load history.");
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <GlassCard className="p-10 text-center">
        <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-[var(--muted)]">Loading</div>
        <div className="mt-3 text-2xl font-semibold text-[var(--ink)]">Collecting recent training runs</div>
      </GlassCard>
    );
  }

  return (
    <div className="space-y-6">
      <GlassCard className="p-6">
        <div className="flex flex-col gap-5 xl:flex-row xl:items-end xl:justify-between">
          <div>
            <div className="text-[11px] font-semibold uppercase tracking-[0.24em] text-[var(--muted)]">Run history</div>
            <h1 className="mt-2 text-3xl font-semibold text-[var(--ink)]">Training timeline</h1>
            <p className="mt-3 max-w-3xl text-sm leading-6 text-[var(--muted)]">
              Browse queued, running, completed, and failed jobs, then jump back into results or deployment for any completed run.
            </p>
          </div>

          <button
            type="button"
            onClick={() => navigate("/train")}
            className="rounded-full bg-[var(--accent)] px-5 py-2.5 text-sm font-semibold text-[var(--surface-solid)]"
          >
            Start new project
          </button>
        </div>

        {error ? (
          <div className="mt-5 rounded-[22px] border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {error}
          </div>
        ) : null}
      </GlassCard>

      {jobs.length === 0 && !error ? (
        <GlassCard className="p-12 text-center">
          <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-[var(--muted)]">No runs yet</div>
          <div className="mt-3 text-2xl font-semibold text-[var(--ink)]">Your history will appear here</div>
          <div className="mt-2 text-sm text-[var(--muted)]">
            Kick off a training job and this page will turn into a project timeline.
          </div>
        </GlassCard>
      ) : null}

      <div className="space-y-4">
        {jobs.map((job) => {
          const status = STATUS_STYLES[job.status] || STATUS_STYLES.queued;
          const date = new Date(job.created_at);
          const dateLabel = Number.isNaN(date.getTime()) ? job.created_at : date.toLocaleString();

          return (
            <GlassCard key={job.job_id} className="p-5">
              <div className="flex flex-col gap-4 xl:flex-row xl:items-center">
                <div className="flex items-start gap-4">
                  <div className={classNames("mt-2 h-3 w-3 rounded-full", status.dot)} />
                  <div>
                    <div className="flex flex-wrap items-center gap-2">
                      <span className={classNames("rounded-full px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.16em]", status.badge)}>
                        {status.label}
                      </span>
                      <span className="rounded-full border border-[var(--border)] bg-[var(--surface-solid)] px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-[var(--muted)]">
                        {job.model_type}
                      </span>
                    </div>
                    <div className="mt-3 text-lg font-semibold text-[var(--ink)]">{job.job_id}</div>
                    <div className="mt-1 text-sm text-[var(--muted)]">{dateLabel}</div>
                  </div>
                </div>

                <div className="xl:ml-auto flex flex-wrap items-center gap-3">
                  {job.status === "completed" ? (
                    <>
                      <button
                        type="button"
                        onClick={() => navigate(`/results/${job.job_id}`)}
                        className="rounded-full border border-[var(--border)] bg-[var(--surface-solid)] px-4 py-2 text-sm font-semibold text-[var(--ink)]"
                      >
                        Results
                      </button>
                      {job.best_model_id ? (
                        <button
                          type="button"
                          onClick={() => navigate(`/deploy/${job.best_model_id}`)}
                          className="rounded-full bg-[var(--accent)] px-4 py-2 text-sm font-semibold text-[var(--surface-solid)]"
                        >
                          Deploy
                        </button>
                      ) : null}
                    </>
                  ) : null}

                  {job.status === "running" ? (
                    <span className="rounded-full bg-[var(--accent-soft)] px-4 py-2 text-sm font-semibold text-[var(--accent)]">
                      Training in progress
                    </span>
                  ) : null}

                  {job.status === "failed" ? (
                    <span className="rounded-full bg-red-50 px-4 py-2 text-sm font-semibold text-red-600">
                      Check logs in the original run
                    </span>
                  ) : null}
                </div>
              </div>
            </GlassCard>
          );
        })}
      </div>
    </div>
  );
}
