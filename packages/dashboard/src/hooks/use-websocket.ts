"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { AgentEvent, createWebSocket } from "@/lib/api";

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
        const data = JSON.parse(event.data) as AgentEvent;
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
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!taskId) return;

    setEvents([]);
    setStatus("connecting");

    const ws = createWebSocket(taskId);
    wsRef.current = ws;

    ws.onopen = () => setStatus("running");
    ws.onclose = () => setStatus("done");

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as AgentEvent;
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
