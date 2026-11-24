"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";
import { createWebSocket, getHealth, getKBStatus, HealthResponse, KBStatus } from "./api";

interface KBContextValue {
  kb: KBStatus | null;
  health: HealthResponse | null;
  apiReachable: boolean;
  refresh: () => void;
}

const KBContext = createContext<KBContextValue>({
  kb: null,
  health: null,
  apiReachable: false,
  refresh: () => undefined,
});

export function KBProvider({ children }: { children: React.ReactNode }) {
  const [kb, setKB] = useState<KBStatus | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [apiReachable, setApiReachable] = useState(false);

  const refresh = useCallback(() => {
    Promise.all([getKBStatus(), getHealth()])
      .then(([kbData, healthData]) => {
        setKB(kbData);
        setHealth(healthData);
        setApiReachable(true);
      })
      .catch(() => {
        setApiReachable(false);
      });
  }, []);

  // Polling at 5s intervals
  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, [refresh]);

  // Keep a ref to the latest refresh for the WebSocket handler
  const refreshRef = useRef(refresh);
  useEffect(() => {
    refreshRef.current = refresh;
  }, [refresh]);

  // WebSocket subscription to trigger immediate refresh on kb_ingest_complete
  useEffect(() => {
    let ws: WebSocket | null = null;
    let retryTimeout: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      ws = createWebSocket();
      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data as string) as Record<string, unknown>;
          if (msg.event === "kb_ingest_complete") {
            refreshRef.current();
          }
        } catch { /* ignore */ }
      };
      ws.onclose = () => {
        retryTimeout = setTimeout(connect, 5000);
      };
    };

    connect();
    return () => {
      ws?.close();
      if (retryTimeout) clearTimeout(retryTimeout);
    };
  }, []);

  return (
    <KBContext.Provider value={{ kb, health, apiReachable, refresh }}>
      {children}
    </KBContext.Provider>
  );
}

export function useKB(): KBContextValue {
  return useContext(KBContext);
}
