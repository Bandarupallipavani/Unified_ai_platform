/**
 * LiveLogTerminal — reusable terminal-style log viewer
 * Props:
 *   logs: string[]
 *   running: boolean
 *   height?: string (default "h-64")
 */
import React, { useRef, useEffect } from "react";

function classifyLog(log) {
  if (log.includes("❌") || log.includes("failed") || log.includes("Error"))
    return "text-red-400";
  if (log.includes("✅") || log.includes("🎉") || log.includes("complete") || log.includes("Best"))
    return "text-green-400";
  if (log.includes("⚠️") || log.includes("warning") || log.includes("skipped"))
    return "text-yellow-400";
  if (log.includes("▶") || log.includes("🚀") || log.includes("Training") || log.includes("Starting"))
    return "text-white font-semibold";
  if (log.includes("Epoch") || log.includes("Trial") || log.includes("score"))
    return "text-cyan-300";
  return "text-gray-400";
}

export default function LiveLogTerminal({ logs = [], running = false, height = "h-64" }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  return (
    <div className={`bg-gray-950 rounded-xl p-4 font-mono text-xs leading-relaxed ${height} overflow-y-auto border border-gray-800`}>
      {logs.length === 0 && (
        <span className="text-gray-600 italic">Waiting for training to start…</span>
      )}
      {logs.map((log, i) => (
        <div key={i} className={classifyLog(log)}>
          <span className="text-gray-600 select-none mr-2">›</span>
          {log}
        </div>
      ))}
      {running && (
        <div className="text-gray-500 mt-1">
          <span className="text-gray-600 mr-2">›</span>
          <span className="animate-pulse">█</span>
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  );
}
