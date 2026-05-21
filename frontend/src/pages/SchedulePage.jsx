import { useEffect, useState } from "react";
import { createSchedule, deleteSchedule, listSchedules } from "../api";

const CRON_PRESETS = [
  { label: "Every day at 2 AM", value: "0 2 * * *" },
  { label: "Every Monday at 9 AM", value: "0 9 * * 1" },
  { label: "Every hour", value: "0 * * * *" },
  { label: "Every 6 hours", value: "0 */6 * * *" },
  { label: "First day of month", value: "0 0 1 * *" },
  { label: "Custom", value: "custom" },
];

export default function SchedulePage() {
  const [schedules, setSchedules] = useState([]);
  const [sessionId, setSessionId] = useState("");
  const [cronPreset, setCronPreset] = useState(CRON_PRESETS[0].value);
  const [customCron, setCustomCron] = useState("");
  const [config, setConfig] = useState('{"model_type":"ml"}');
  const [loading, setLoading] = useState(false);
  const [toast, setToast] = useState("");

  const load = () =>
    listSchedules()
      .then((response) => setSchedules(response.data))
      .catch(() => {});

  useEffect(() => {
    load();
  }, []);

  const cronValue = cronPreset === "custom" ? customCron : cronPreset;

  const handleCreate = async () => {
    if (!sessionId) {
      setToast("Please enter a session ID.");
      return;
    }

    if (!cronValue) {
      setToast("Please choose a cron expression.");
      return;
    }

    let parsedConfig;
    try {
      parsedConfig = JSON.parse(config);
    } catch {
      setToast("Invalid JSON config.");
      return;
    }

    setLoading(true);

    try {
      await createSchedule(sessionId, cronValue, parsedConfig);
      setToast("Schedule created.");
      setSessionId("");
      load();
    } catch (error) {
      setToast(`Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (scheduleId) => {
    if (!window.confirm("Cancel this schedule?")) return;

    try {
      await deleteSchedule(scheduleId);
      setToast("Schedule cancelled.");
      load();
    } catch (error) {
      setToast(`Error: ${error.message}`);
    }
  };

  return (
    <div className="space-y-6">
      <div className="glass-panel rounded-[28px] p-6">
        <div className="text-[11px] font-semibold uppercase tracking-[0.24em] text-[var(--muted)]">Automation</div>
        <h1 className="mt-2 text-3xl font-semibold text-[var(--ink)]">Scheduled retraining</h1>
        <p className="mt-3 max-w-3xl text-sm leading-6 text-[var(--muted)]">
          Create recurring jobs so fresh data can retrain your pipeline without manually relaunching the wizard.
        </p>
      </div>

      {toast ? (
        <div className="rounded-[22px] border border-[var(--border)] bg-[var(--surface-solid)] px-4 py-3 text-sm text-[var(--ink)]">
          {toast}
        </div>
      ) : null}

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_360px]">
        <div className="soft-card rounded-[24px] p-5">
          <div className="text-sm font-semibold text-[var(--ink)]">Create a schedule</div>
          <div className="mt-5 grid gap-4 md:grid-cols-2">
            <label className="text-sm text-[var(--muted)]">
              Session ID
              <input
                type="text"
                value={sessionId}
                onChange={(event) => setSessionId(event.target.value)}
                placeholder="Paste the session_id from a training project"
                className="mt-2 w-full rounded-[18px] border border-[var(--border)] bg-[var(--surface-solid)] px-4 py-3 text-[var(--ink)]"
              />
            </label>

            <label className="text-sm text-[var(--muted)]">
              Frequency
              <select
                value={cronPreset}
                onChange={(event) => setCronPreset(event.target.value)}
                className="mt-2 w-full rounded-[18px] border border-[var(--border)] bg-[var(--surface-solid)] px-4 py-3 text-[var(--ink)]"
              >
                {CRON_PRESETS.map((preset) => (
                  <option key={preset.value} value={preset.value}>
                    {preset.label}
                  </option>
                ))}
              </select>
            </label>

            {cronPreset === "custom" ? (
              <label className="text-sm text-[var(--muted)] md:col-span-2">
                Custom cron
                <input
                  type="text"
                  value={customCron}
                  onChange={(event) => setCustomCron(event.target.value)}
                  placeholder="0 3 * * 1-5"
                  className="mt-2 w-full rounded-[18px] border border-[var(--border)] bg-[var(--surface-solid)] px-4 py-3 font-mono text-[var(--ink)]"
                />
              </label>
            ) : null}

            <label className="text-sm text-[var(--muted)] md:col-span-2">
              Training config JSON
              <textarea
                rows={5}
                value={config}
                onChange={(event) => setConfig(event.target.value)}
                className="mt-2 w-full rounded-[18px] border border-[var(--border)] bg-[var(--surface-solid)] px-4 py-3 font-mono text-xs text-[var(--ink)]"
              />
            </label>
          </div>

          <div className="mt-5 flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={handleCreate}
              disabled={loading}
              className="rounded-full bg-[var(--accent)] px-5 py-2.5 text-sm font-semibold text-[var(--surface-solid)] disabled:opacity-45"
            >
              {loading ? "Creating..." : "Create schedule"}
            </button>
            <span className="rounded-full border border-[var(--border)] bg-[var(--surface-solid)] px-4 py-2 text-xs font-semibold text-[var(--muted)]">
              {cronValue || "--"}
            </span>
          </div>
        </div>

        <div className="soft-card rounded-[24px] p-5">
          <div className="text-sm font-semibold text-[var(--ink)]">What happens next</div>
          <div className="mt-4 space-y-3 text-sm text-[var(--muted)]">
            <div className="rounded-[18px] border border-[var(--border)] bg-[var(--surface-alt)] px-4 py-3">
              The scheduler keeps the original session context and training config together.
            </div>
            <div className="rounded-[18px] border border-[var(--border)] bg-[var(--surface-alt)] px-4 py-3">
              Each trigger launches a fresh training job using the stored configuration payload.
            </div>
            <div className="rounded-[18px] border border-[var(--border)] bg-[var(--surface-alt)] px-4 py-3">
              Use this for nightly refreshes, weekly score updates, or monthly benchmark runs.
            </div>
          </div>
        </div>
      </div>

      <div className="soft-card rounded-[24px] p-5">
        <div className="mb-4 flex items-center justify-between">
          <div className="text-sm font-semibold text-[var(--ink)]">Active schedules</div>
          <button type="button" onClick={load} className="text-sm font-semibold text-[var(--accent)]">
            Refresh
          </button>
        </div>

        {schedules.length === 0 ? (
          <div className="text-sm text-[var(--muted)]">No schedules have been created yet.</div>
        ) : (
          <div className="space-y-3">
            {schedules.map((schedule) => (
              <div
                key={schedule.id}
                className="flex flex-col gap-4 rounded-[20px] border border-[var(--border)] bg-[var(--surface-alt)] px-4 py-4 lg:flex-row lg:items-center"
              >
                <div className="flex-1">
                  <div className="font-mono text-sm font-semibold text-[var(--ink)]">{schedule.cron_expr}</div>
                  <div className="mt-1 text-xs text-[var(--muted)]">
                    Session {schedule.session_id.slice(0, 8)}... / Next run{" "}
                    {schedule.next_run_at && schedule.next_run_at !== "None"
                      ? new Date(schedule.next_run_at).toLocaleString()
                      : "--"}
                  </div>
                </div>

                <button
                  type="button"
                  onClick={() => handleDelete(schedule.id)}
                  className="rounded-full border border-[var(--border)] bg-[var(--surface-solid)] px-4 py-2 text-xs font-semibold text-[var(--muted)]"
                >
                  Cancel
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
