import React, { useState } from "react";
import { Link } from "react-router-dom";
import { login } from "../api";

const HIGHLIGHTS = [
  ["Upload", "Profile data and build a preprocessing plan."],
  ["Train", "Run ML, DL, and NLP experiments from one flow."],
  ["Deploy", "Hand off the best model with fewer steps."],
];

export default function LoginPage({ onLogin }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!email || !password) return;

    setLoading(true);
    setError("");

    try {
      const res = await login({ email, password });
      if (res && res.access_token) {
        onLogin(res.access_token);
      } else {
        throw new Error("Invalid response from server.");
      }
    } catch (err) {
      setError(err.message || "Failed to sign in. Please check your credentials.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[var(--page)]">
      <div className="mx-auto flex min-h-screen max-w-5xl flex-col justify-center gap-10 px-4 py-10 lg:grid lg:grid-cols-[minmax(0,1fr)_420px] lg:items-center lg:px-6">
        <div className="max-w-xl">
          <div className="inline-flex items-center rounded-lg border border-[var(--border)] bg-white px-3 py-2 text-xs font-semibold text-[var(--muted)]">
            Unified AI Platform
          </div>
          <h1 className="mt-6 text-4xl font-semibold tracking-tight text-[var(--ink)]">
            Sign in to continue your work.
          </h1>
          <p className="mt-4 max-w-lg text-base leading-7 text-[var(--muted)]">
            Open your training runs, review results, and deploy models from one workspace.
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
            <div className="text-sm font-semibold text-[var(--muted)]">Sign in</div>
            <h2 className="mt-3 text-2xl font-semibold text-[var(--ink)]">Welcome back</h2>
            <p className="mt-2 text-sm leading-6 text-[var(--muted)]">Use your workspace credentials to continue.</p>
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
                placeholder="Enter your password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
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
                  <span>Signing In...</span>
                </>
              ) : (
                "Sign In"
              )}
            </button>
          </form>

          <div className="pt-6 text-center">
            <p className="text-sm text-[var(--muted)]">
              Don&apos;t have an account?{" "}
              <Link to="/register" className="font-semibold text-[var(--accent)]">
                Create one now
              </Link>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
