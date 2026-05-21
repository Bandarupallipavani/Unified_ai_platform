import React, { useState } from "react";
import { Link } from "react-router-dom";
import { register } from "../api";

const HIGHLIGHTS = [
  ["Guided setup", "Start with sensible defaults and a step-by-step training flow."],
  ["Flexible modes", "Switch between beginner and expert depth without changing tools."],
  ["Faster handoff", "Move from dataset to deploy-ready artifacts in one workspace."],
];

export default function RegisterPage({ onLogin }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [mode, setMode] = useState("beginner");
  const [domain, setDomain] = useState("general");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!email || !password) return;

    setLoading(true);
    setError("");

    try {
      const res = await register({ email, password, mode, domain });
      if (res && res.access_token) {
        onLogin(res.access_token);
      } else {
        throw new Error("Registration succeeded but no access token returned.");
      }
    } catch (err) {
      setError(err.message || "Failed to create account. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[var(--page)]">
      <div className="mx-auto flex min-h-screen max-w-5xl flex-col justify-center gap-10 px-4 py-10 lg:grid lg:grid-cols-[minmax(0,1fr)_460px] lg:items-center lg:px-6">
        <div className="max-w-xl">
          <div className="inline-flex items-center rounded-lg border border-[var(--border)] bg-white px-3 py-2 text-xs font-semibold text-[var(--muted)]">
            Unified AI Platform
          </div>
          <h1 className="mt-6 text-4xl font-semibold tracking-tight text-[var(--ink)]">
            Create your workspace.
          </h1>
          <p className="mt-4 max-w-lg text-base leading-7 text-[var(--muted)]">
            Choose a starting mode and get into training, results, and deployment quickly.
          </p>

          <div className="mt-8 space-y-3">
            {HIGHLIGHTS.map(([title, description]) => (
              <div key={title} className="flex gap-3 rounded-xl border border-[var(--border)] bg-white px-4 py-3">
                <div className="mt-0.5 h-2.5 w-2.5 rounded-full bg-[var(--accent)]" />
                <div>
                  <div className="text-sm font-semibold text-[var(--ink)]">{title}</div>
                  <div className="mt-1 text-sm leading-6 text-[var(--muted)]">{description}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-2xl border border-[var(--border)] bg-white p-8">
          <div>
            <div className="text-sm font-semibold text-[var(--muted)]">Create account</div>
            <h2 className="mt-3 text-2xl font-semibold text-[var(--ink)]">Create your workspace</h2>
            <p className="mt-2 text-sm leading-6 text-[var(--muted)]">Choose a starting mode now. You can keep refining it inside the app.</p>
          </div>

          {error ? (
            <div className="mt-6 rounded-2xl border border-red-200 bg-red-50 p-3 text-sm leading-relaxed text-red-700">
              {error}
            </div>
          ) : null}

          <form onSubmit={handleSubmit} className="mt-6 space-y-4">
            <div>
              <label className="mb-1.5 block text-sm font-medium text-[var(--ink)]">Email Address</label>
              <input
                type="email"
                required
                className="w-full rounded-xl border border-[var(--border)] bg-[var(--surface-alt)] px-4 py-3 text-sm text-[var(--ink)] outline-none focus:border-[var(--accent)] focus:bg-white"
                placeholder="you@company.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>

            <div>
              <label className="mb-1.5 block text-sm font-medium text-[var(--ink)]">Password</label>
              <input
                type="password"
                required
                className="w-full rounded-xl border border-[var(--border)] bg-[var(--surface-alt)] px-4 py-3 text-sm text-[var(--ink)] outline-none focus:border-[var(--accent)] focus:bg-white"
                placeholder="Create a password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>

            <div>
              <label className="mb-1.5 block text-sm font-medium text-[var(--ink)]">Experience Level</label>
              <select
                className="w-full rounded-xl border border-[var(--border)] bg-[var(--surface-alt)] px-4 py-3 text-sm text-[var(--ink)] outline-none focus:border-[var(--accent)] focus:bg-white"
                value={mode}
                onChange={(e) => setMode(e.target.value)}
              >
                <option value="beginner">Beginner - Guided setup and plain-language defaults</option>
                <option value="expert">Expert - More control and technical detail</option>
              </select>
            </div>

            <div>
              <label className="mb-1.5 block text-sm font-medium text-[var(--ink)]">Industry Domain</label>
              <select
                className="w-full rounded-xl border border-[var(--border)] bg-[var(--surface-alt)] px-4 py-3 text-sm text-[var(--ink)] outline-none focus:border-[var(--accent)] focus:bg-white"
                value={domain}
                onChange={(e) => setDomain(e.target.value)}
              >
                {["general", "healthcare", "finance", "retail", "manufacturing", "hr", "marketing"].map((d) => (
                  <option key={d} value={d}>
                    {d.charAt(0).toUpperCase() + d.slice(1)}
                  </option>
                ))}
              </select>
            </div>

            <button
              type="submit"
              disabled={loading || !email || !password}
              className="flex w-full items-center justify-center gap-2 rounded-xl bg-[var(--ink)] py-3 text-sm font-semibold text-[var(--surface-solid)] disabled:opacity-50"
            >
              {loading ? (
                <>
                  <svg className="h-4 w-4 animate-spin text-[var(--surface-solid)]" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  <span>Creating Account...</span>
                </>
              ) : (
                "Create Account"
              )}
            </button>
          </form>

          <div className="pt-6 text-center">
            <p className="text-sm text-[var(--muted)]">
              Already have an account?{" "}
              <Link to="/login" className="font-semibold text-[var(--accent)]">
                Sign In
              </Link>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
