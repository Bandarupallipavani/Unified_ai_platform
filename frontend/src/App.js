import { BrowserRouter, Link, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { useEffect, useState } from "react";

import LoginPage from "./pages/LoginPage";
import RegisterPage from "./pages/RegisterPage";
import WizardPage from "./pages/WizardPage";
import ResultsPage from "./pages/ResultsPage";
import DeployPage from "./pages/DeployPage";
import HistoryPage from "./pages/HistoryPage";
import ModelRegistryPage from "./pages/ModelRegistryPage";
import DriftMonitorPage from "./pages/DriftMonitorPage";
import BatchPredictPage from "./pages/BatchPredictPage";
import SchedulePage from "./pages/SchedulePage";
import { getMe } from "./api";

const LAST_JOB_KEY = "unifiedai:lastJobId";
const LAST_MODEL_KEY = "unifiedai:lastModelId";
const SIDEBAR_KEY = "unifiedai:sidebar";
const THEME_KEY = "unifiedai:theme";

const toolLinks = [
  { to: "/registry", label: "Registry" },
  { to: "/batch", label: "Batch Test" },
  { to: "/drift", label: "Drift" },
  { to: "/schedule", label: "Schedule" },
];

const PAGE_META = [
  { match: "/train", title: "Training", description: "Create, configure, and run model pipelines." },
  { match: "/results", title: "Results", description: "Review scores, test new inputs, and compare best models." },
  { match: "/deploy", title: "Deploy", description: "Export or deploy a trained model." },
  { match: "/history", title: "History", description: "Browse previous training jobs." },
  { match: "/registry", title: "Registry", description: "Manage model versions and production status." },
  { match: "/batch", title: "Batch", description: "Run offline predictions on different input files." },
  { match: "/drift", title: "Drift", description: "Check production input drift and alerts." },
  { match: "/schedule", title: "Schedule", description: "Set up recurring retraining jobs." },
];

function readRecentTargets() {
  return {
    lastJobId: localStorage.getItem(LAST_JOB_KEY),
    lastModelId: localStorage.getItem(LAST_MODEL_KEY),
  };
}

function navClass(active, disabled = false) {
  if (active) {
    return "rounded-xl border border-[var(--ink)] bg-[var(--ink)] px-3 py-2 text-sm font-semibold text-[var(--surface-solid)]";
  }
  if (disabled) {
    return "rounded-xl border border-[var(--border)] bg-white px-3 py-2 text-sm font-medium text-[var(--muted)] opacity-45";
  }
  return "rounded-xl border border-[var(--border)] bg-white px-3 py-2 text-sm font-medium text-[var(--muted)] hover:text-[var(--ink)]";
}

function toolClass(active) {
  return active
    ? "rounded-xl border border-[var(--accent)] bg-[var(--accent-soft)] px-3 py-2 text-sm font-semibold text-[var(--accent)]"
    : "rounded-xl border border-[var(--border)] bg-white px-3 py-2 text-sm font-medium text-[var(--muted)] hover:text-[var(--ink)]";
}

function sidebarLinkClass(active, disabled = false) {
  if (active) {
    return "flex items-center rounded-xl border border-[var(--accent)] bg-[var(--accent-soft)] px-3 py-2.5 text-sm font-semibold text-[var(--accent)]";
  }
  if (disabled) {
    return "flex items-center rounded-xl border border-[var(--border)] bg-[var(--surface-alt)] px-3 py-2.5 text-sm font-medium text-[var(--muted)] opacity-45";
  }
  return "flex items-center rounded-xl border border-transparent px-3 py-2.5 text-sm font-medium text-[var(--muted)] hover:border-[var(--border)] hover:bg-[var(--surface-alt)] hover:text-[var(--ink)]";
}

function getPageMeta(pathname) {
  return PAGE_META.find((item) => pathname.startsWith(item.match)) || PAGE_META[0];
}

function classNames(...parts) {
  return parts.filter(Boolean).join(" ");
}

function ThemeToggleButton({ theme, onToggle }) {
  const isDark = theme === "dark";
  return (
    <button
      type="button"
      onClick={onToggle}
      aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
      title={isDark ? "Light mode" : "Dark mode"}
      className="inline-flex h-9 w-9 items-center justify-center rounded-xl border border-[var(--border)] bg-white text-[var(--ink)]"
    >
      {isDark ? (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="4" />
          <path d="M12 2v2" />
          <path d="M12 20v2" />
          <path d="m4.93 4.93 1.41 1.41" />
          <path d="m17.66 17.66 1.41 1.41" />
          <path d="M2 12h2" />
          <path d="M20 12h2" />
          <path d="m6.34 17.66-1.41 1.41" />
          <path d="m19.07 4.93-1.41 1.41" />
        </svg>
      ) : (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9z" />
        </svg>
      )}
    </button>
  );
}

function SidebarContent({
  coreLinks,
  toolLinks,
  pathname,
  lastJobId,
  lastModelId,
  mode,
  onLogout,
  onNavigate,
}) {
  return (
    <>
      <div className="border-b border-[var(--border)] p-5">
        <Link to="/train" className="flex min-w-0 items-center gap-3" onClick={onNavigate}>
          <div className="flex h-11 w-11 items-center justify-center rounded-xl border border-[var(--border)] bg-[var(--surface-alt)] text-sm font-bold text-[var(--ink)]">
            UA
          </div>
          <div className="min-w-0">
            <div className="truncate text-base font-semibold text-[var(--ink)]">Unified AI Platform</div>
            <div className="text-sm text-[var(--muted)]">Compact ML workspace</div>
          </div>
        </Link>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        <div className="space-y-2">
          <div className="px-3 pb-1 text-xs font-semibold uppercase tracking-[0.12em] text-[var(--muted)]">
            Workspace
          </div>
          {coreLinks.map((link) =>
            link.disabled && !link.active ? (
              <span key={link.label} className={sidebarLinkClass(link.active, true)}>
                {link.label}
              </span>
            ) : (
              <Link key={link.label} to={link.to} className={sidebarLinkClass(link.active, false)} onClick={onNavigate}>
                {link.label}
              </Link>
            )
          )}
        </div>

        <div className="mt-6 space-y-2">
          <div className="px-3 pb-1 text-xs font-semibold uppercase tracking-[0.12em] text-[var(--muted)]">
            Tools
          </div>
          {toolLinks.map((link) => (
            <Link
              key={link.to}
              to={link.to}
              className={sidebarLinkClass(pathname.startsWith(link.to))}
              onClick={onNavigate}
            >
              {link.label}
            </Link>
          ))}
        </div>

        <div className="mt-6 rounded-2xl border border-[var(--border)] bg-[var(--surface-alt)] p-4">
          <div className="text-sm font-semibold text-[var(--ink)]">Recent project</div>
          <div className="mt-3 space-y-2 text-sm text-[var(--muted)]">
            <div className="flex items-center justify-between gap-3">
              <span>Job</span>
              <span className="font-mono text-xs text-[var(--ink)]">{lastJobId ? `${lastJobId.slice(0, 8)}...` : "--"}</span>
            </div>
            <div className="flex items-center justify-between gap-3">
              <span>Model</span>
              <span className="font-mono text-xs text-[var(--ink)]">{lastModelId ? `${lastModelId.slice(0, 8)}...` : "--"}</span>
            </div>
          </div>
        </div>
      </div>

      <div className="border-t border-[var(--border)] p-4">
        <div className="mb-3 rounded-xl border border-[var(--border)] bg-[var(--surface-alt)] px-3 py-2 text-sm font-semibold capitalize text-[var(--ink)]">
          {mode} mode
        </div>
        <button
          onClick={onLogout}
          className="w-full rounded-xl border border-[var(--border)] bg-white px-3 py-2.5 text-sm font-medium text-[var(--muted)] hover:text-[var(--ink)]"
        >
          Sign out
        </button>
      </div>
    </>
  );
}

function AppChrome({ mode, onLogout, children }) {
  const location = useLocation();
  const [{ lastJobId, lastModelId }, setRecentTargets] = useState(readRecentTargets);
  const [sidebarOpen, setSidebarOpen] = useState(() => localStorage.getItem(SIDEBAR_KEY) !== "closed");
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [theme, setTheme] = useState(() => localStorage.getItem(THEME_KEY) || "light");

  useEffect(() => {
    setRecentTargets(readRecentTargets());
  }, [location.pathname]);

  useEffect(() => {
    localStorage.setItem(SIDEBAR_KEY, sidebarOpen ? "open" : "closed");
  }, [sidebarOpen]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.body.dataset.theme = theme;
    localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  useEffect(() => {
    setMobileSidebarOpen(false);
  }, [location.pathname]);

  const coreLinks = [
    {
      to: "/train",
      label: "New Project",
      active: location.pathname.startsWith("/train"),
    },
    {
      to: lastJobId ? `/results/${lastJobId}` : "/history",
      label: "Results",
      active: location.pathname.startsWith("/results"),
      disabled: !lastJobId,
    },
    {
      to: lastModelId ? `/deploy/${lastModelId}` : "/history",
      label: "Deploy",
      active: location.pathname.startsWith("/deploy"),
      disabled: !lastModelId,
    },
    {
      to: "/history",
      label: "History",
      active: location.pathname.startsWith("/history"),
    },
  ];

  const pageMeta = getPageMeta(location.pathname);
  const toggleSidebar = () => {
    if (typeof window !== "undefined" && window.innerWidth >= 1024) {
      setSidebarOpen((previous) => !previous);
      return;
    }
    setMobileSidebarOpen((previous) => !previous);
  };

  return (
    <div className="min-h-screen bg-[var(--page)] text-[var(--ink)] lg:flex">
      {mobileSidebarOpen ? (
        <>
          <button
            type="button"
            aria-label="Close sidebar overlay"
            onClick={() => setMobileSidebarOpen(false)}
            className="fixed inset-0 z-30 bg-black/20 lg:hidden"
          />
          <aside className="fixed inset-y-0 left-0 z-40 flex w-72 flex-col border-r border-[var(--border)] bg-white shadow-lg lg:hidden">
            <SidebarContent
              coreLinks={coreLinks}
              toolLinks={toolLinks}
              pathname={location.pathname}
              lastJobId={lastJobId}
              lastModelId={lastModelId}
              mode={mode}
              onLogout={onLogout}
              onNavigate={() => setMobileSidebarOpen(false)}
            />
          </aside>
        </>
      ) : null}

      <aside
        className={classNames(
          "hidden shrink-0 overflow-hidden bg-white transition-[width,border-color] duration-200 lg:flex lg:flex-col",
          sidebarOpen ? "lg:w-72 lg:border-r lg:border-[var(--border)]" : "lg:w-20 lg:border-r lg:border-[var(--border)]"
        )}
      >
        {sidebarOpen ? (
          <div className="flex h-full flex-col">
            <SidebarContent
              coreLinks={coreLinks}
              toolLinks={toolLinks}
              pathname={location.pathname}
              lastJobId={lastJobId}
              lastModelId={lastModelId}
              mode={mode}
              onLogout={onLogout}
            />
          </div>
        ) : (
          <div className="flex h-full flex-col items-center gap-3 p-3">
            <button
              type="button"
              onClick={() => setSidebarOpen(true)}
              aria-label="Open sidebar"
              className="inline-flex h-11 w-11 items-center justify-center rounded-xl border border-[var(--border)] bg-white text-[var(--ink)]"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <path d="M9 18l6-6-6-6" />
              </svg>
            </button>

            <div className="mt-auto rounded-xl border border-[var(--border)] bg-[var(--surface-alt)] px-2 py-3 text-[10px] font-semibold uppercase tracking-[0.12em] text-[var(--muted)] [writing-mode:vertical-rl] [text-orientation:mixed]">
              {mode}
            </div>
          </div>
        )}
      </aside>

      <div className="flex min-h-screen min-w-0 flex-1 flex-col">
        <header className="border-b border-[var(--border)] bg-white">
          <div className="flex flex-col gap-3 px-4 py-3 lg:px-6">
            <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
              <div className="flex min-w-0 items-start gap-3">
                <button
                  type="button"
                  onClick={toggleSidebar}
                  aria-label="Toggle sidebar"
                  className="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-[var(--border)] bg-white text-[var(--ink)]"
                >
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                    <path d="M4 7h16" />
                    <path d="M4 12h16" />
                    <path d="M4 17h16" />
                  </svg>
                </button>
                <div className="min-w-0">
                  <div className="text-sm font-semibold text-[var(--ink)]">Workspace</div>
                  <div className="text-xs text-[var(--muted)]">{pageMeta.title}</div>
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-2">
                <ThemeToggleButton theme={theme} onToggle={() => setTheme((previous) => (previous === "dark" ? "light" : "dark"))} />
                <span className="rounded-xl border border-[var(--border)] bg-[var(--surface-alt)] px-3 py-2 text-xs font-semibold capitalize text-[var(--ink)] lg:hidden">
                  {mode} mode
                </span>
                <span className="rounded-xl border border-[var(--border)] bg-[var(--surface-alt)] px-3 py-2 text-xs font-medium text-[var(--muted)]">
                  {lastJobId ? "Recent run ready" : "Start a new run"}
                </span>
                <button
                  onClick={onLogout}
                  className="rounded-xl border border-[var(--border)] bg-white px-3 py-2 text-xs font-medium text-[var(--muted)] hover:text-[var(--ink)] lg:hidden"
                >
                  Sign out
                </button>
              </div>
            </div>

            <div className="flex flex-wrap gap-2 lg:hidden">
              {coreLinks.map((link) =>
                link.disabled && !link.active ? (
                  <span key={link.label} className={navClass(link.active, true)}>
                    {link.label}
                  </span>
                ) : (
                  <Link key={link.label} to={link.to} className={navClass(link.active, false)}>
                    {link.label}
                  </Link>
                )
              )}
              {toolLinks.map((link) => (
                <Link
                  key={link.to}
                  to={link.to}
                  className={toolClass(location.pathname.startsWith(link.to))}
                >
                  {link.label}
                </Link>
              ))}
            </div>
          </div>
        </header>

        <main className="min-w-0 flex-1 px-4 py-4 lg:px-6 lg:py-5">{children}</main>
      </div>
    </div>
  );
}

export default function App() {
  const [token, setToken] = useState(localStorage.getItem("token"));
  const [mode, setMode] = useState("beginner");

  const handleLogout = () => {
    localStorage.removeItem("token");
    setToken(null);
  };

  useEffect(() => {
    if (!token) return;

    getMe()
      .then((response) => setMode(response.mode || response.data?.mode || "beginner"))
      .catch((error) => {
        if (error?.response?.status === 401) handleLogout();
      });
  }, [token]);

  useEffect(() => {
    window.addEventListener("auth:unauthorized", handleLogout);
    return () => window.removeEventListener("auth:unauthorized", handleLogout);
  }, []);

  const handleLogin = (newToken) => {
    localStorage.setItem("token", newToken);
    setToken(newToken);
  };

  if (!token) {
    return (
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<LoginPage onLogin={handleLogin} />} />
          <Route path="/register" element={<RegisterPage onLogin={handleLogin} />} />
          <Route path="*" element={<Navigate to="/login" replace />} />
        </Routes>
      </BrowserRouter>
    );
  }

  return (
    <BrowserRouter>
      <AppChrome mode={mode} onLogout={handleLogout}>
        <Routes>
          <Route path="/train" element={<WizardPage mode={mode} />} />
          <Route path="/results/:jobId" element={<ResultsPage />} />
          <Route path="/deploy/:modelId" element={<DeployPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/registry" element={<ModelRegistryPage />} />
          <Route path="/batch" element={<BatchPredictPage />} />
          <Route path="/drift" element={<DriftMonitorPage />} />
          <Route path="/schedule" element={<SchedulePage />} />
          <Route path="*" element={<Navigate to="/train" replace />} />
        </Routes>
      </AppChrome>
    </BrowserRouter>
  );
}
