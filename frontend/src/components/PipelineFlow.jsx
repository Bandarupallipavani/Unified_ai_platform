/**
 * PipelineFlow — visual pipeline stages with done/active/pending state
 * Props:
 *   running: boolean
 *   done: boolean
 *   activeStep?: number (0-indexed, default auto from running/done)
 */
import React from "react";

const STAGES = [
  { label: "Data",       icon: "🗄️" },
  { label: "EDA",        icon: "🔍" },
  { label: "Preprocess", icon: "⚙️" },
  { label: "HPO",        icon: "🎯" },
  { label: "Train",      icon: "🧠" },
  { label: "Evaluate",   icon: "📊" },
  { label: "SHAP",       icon: "👁️" },
  { label: "Report",     icon: "📄" },
];

export default function PipelineFlow({ running = false, done = false, activeStep = -1 }) {
  return (
    <div className="flex items-center gap-0 overflow-x-auto pb-1">
      {STAGES.map((stage, i) => {
        const isDone   = done || i < (running ? 3 : 0);
        const isActive = running && (activeStep >= 0 ? i === activeStep : i === 4);
        return (
          <React.Fragment key={stage.label}>
            <div className="flex flex-col items-center gap-1 flex-shrink-0 w-16">
              <div
                className={`w-9 h-9 rounded-lg flex items-center justify-center text-base transition-colors
                  ${isDone   ? "bg-green-100 text-green-600"
                  : isActive ? "bg-indigo-100 text-indigo-600 ring-2 ring-indigo-300"
                  :            "bg-gray-100 text-gray-400"}`}
              >
                {stage.icon}
              </div>
              <span
                className={`text-[10px] text-center leading-tight
                  ${isDone ? "text-green-600" : isActive ? "text-indigo-600 font-medium" : "text-gray-400"}`}
              >
                {stage.label}
              </span>
            </div>
            {i < STAGES.length - 1 && (
              <div className={`w-4 h-px flex-shrink-0 ${isDone ? "bg-green-300" : "bg-gray-200"}`} />
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
}
