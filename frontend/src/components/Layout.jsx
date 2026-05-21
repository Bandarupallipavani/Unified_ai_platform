import React from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { useAuth } from "../App";

const NAV = [
  { to: "/wizard",  icon: "🧙", label: "Wizard" },
  { to: "/expert",  icon: "⚙️", label: "Expert" },
  { to: "/history", icon: "🕐", label: "History" },
];

export default function Layout({ children }) {
  const { user, signOut } = useAuth();
  const navigate = useNavigate();

  return (
    <div className="flex h-screen bg-gray-50 overflow-hidden">
      <aside className="w-56 bg-white border-r border-gray-200 flex flex-col flex-shrink-0">
        {/* Brand */}
        <div className="px-5 py-5 border-b border-gray-100">
          <div className="flex items-center gap-2">
            <span className="text-2xl">🤖</span>
            <div>
              <div className="font-bold text-gray-900 text-sm">Unified AI</div>
              <div className="text-[10px] text-gray-400 uppercase tracking-wide">ML · DL · NLP</div>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-4 space-y-1">
          {NAV.map(({ to, icon, label }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors
                ${isActive
                  ? "bg-indigo-50 text-indigo-700"
                  : "text-gray-600 hover:bg-gray-50 hover:text-gray-900"}`
              }
            >
              <span>{icon}</span><span>{label}</span>
            </NavLink>
          ))}

          <div className="pt-4 pb-1 px-3">
            <div className="text-[10px] uppercase tracking-wider text-gray-400">Resources</div>
          </div>
          {[
            { href: "http://localhost:8000/docs", icon: "📚", label: "API Docs" },
            { href: "http://localhost:5000",      icon: "📊", label: "MLflow" },
          ].map(({ href, icon, label }) => (
            <a
              key={href}
              href={href}
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-gray-600 hover:bg-gray-50 hover:text-gray-900"
            >
              <span>{icon}</span><span>{label}</span>
            </a>
          ))}
        </nav>

        {/* Mode badges */}
        <div className="px-4 pb-2">
          <div className="flex gap-1 flex-wrap">
            {["ML", "DL", "NLP"].map(t => (
              <span key={t} className="text-[10px] px-2 py-0.5 rounded-full bg-indigo-50 text-indigo-600 font-medium">{t}</span>
            ))}
          </div>
        </div>

        {/* User */}
        <div className="px-4 py-4 border-t border-gray-100">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-8 h-8 bg-indigo-100 text-indigo-600 rounded-full flex items-center justify-center text-sm font-bold">
              {user?.mode === "expert" ? "E" : "B"}
            </div>
            <div>
              <div className="text-xs font-medium text-gray-700 capitalize">{user?.mode} mode</div>
              <div className="text-[10px] text-gray-400">Active session</div>
            </div>
          </div>
          <button
            onClick={() => { signOut(); navigate("/login"); }}
            className="w-full text-left text-xs text-gray-500 hover:text-gray-700 px-2 py-1.5 rounded hover:bg-gray-50"
          >
            Sign out →
          </button>
        </div>
      </aside>

      <main className="flex-1 overflow-auto">
        <div className="p-8 h-full">{children}</div>
      </main>
    </div>
  );
}
