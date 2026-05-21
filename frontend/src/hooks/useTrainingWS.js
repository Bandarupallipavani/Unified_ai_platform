/**
 * useTrainingWS — React hook for live training log streaming via WebSocket
 * Usage:
 *   const { logs, connected, done } = useTrainingWS(jobId, { onComplete });
 */
import { useState, useEffect, useRef, useCallback } from "react";
import { createTrainingWS } from "../api";

export default function useTrainingWS(jobId, { onComplete } = {}) {
  const [logs, setLogs] = useState([]);
  const [connected, setConnected] = useState(false);
  const [done, setDone] = useState(false);
  const wsRef = useRef(null);

  const addLog = useCallback((msg) => {
    setLogs((prev) => [...prev, msg]);
    if (
      msg.includes("🎉") ||
      msg.includes("Training complete") ||
      msg.includes("complete")
    ) {
      setDone(true);
      onComplete?.();
    }
  }, [onComplete]);

  useEffect(() => {
    if (!jobId) return;

    setLogs([]);
    setDone(false);
    setConnected(false);

    wsRef.current = createTrainingWS(
      jobId,
      (msg) => {
        setConnected(true);
        addLog(msg);
      },
      () => {
        setConnected(false);
      }
    );

    return () => {
      wsRef.current?.close();
    };
  }, [jobId, addLog]);

  const clearLogs = () => setLogs([]);

  return { logs, connected, done, clearLogs };
}
