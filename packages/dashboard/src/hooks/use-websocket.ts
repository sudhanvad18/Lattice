"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { AgentEvent, createWebSocket, normalizeAgentEvent } from "@/lib/api";

export function useActivityFeed() {
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const ws = createWebSocket();
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);

    ws.onmessage = (event) => {
      try {
        const data = normalizeAgentEvent(JSON.parse(event.data));
        if (!data) return;
        setEvents((prev) => [data, ...prev].slice(0, 100));
      } catch {
        // ignore malformed messages
      }
    };

    return () => {
      ws.close();
    };
  }, []);

  const clear = useCallback(() => setEvents([]), []);

  return { events, connected, clear };
}

export function useTaskStream(taskId: string | null) {
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [status, setStatus] = useState<string>("idle");
  const [streamId, setStreamId] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  if (taskId !== streamId) {
    setStreamId(taskId);
    setEvents([]);
    setStatus(taskId ? "connecting" : "idle");
  }

  useEffect(() => {
    if (!taskId) return;

    const ws = createWebSocket(taskId);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus((current) => (current === "error" || current === "done" ? current : "running"));
    };
    ws.onerror = () => setStatus("error");
    ws.onclose = () => {
      setStatus((current) => {
        if (current === "error" || current === "done") return current;
        if (current === "connecting") return "error";
        return "done";
      });
    };

    ws.onmessage = (event) => {
      try {
        const data = normalizeAgentEvent(JSON.parse(event.data), taskId);
        if (!data) return;
        if (data.event_type === "completed") setStatus("done");
        if (data.event_type === "error") setStatus("error");
        setEvents((prev) => [...prev, data]);
      } catch {
        // ignore
      }
    };

    return () => {
      ws.close();
    };
  }, [taskId]);

  return { events, status };
}
