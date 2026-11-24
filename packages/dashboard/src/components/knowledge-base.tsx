"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  Database,
  FolderOpen,
  RefreshCw,
  Trash2,
  AlertTriangle,
  CheckCircle2,
  Loader2,
  Search,
  X,
} from "lucide-react";
import { ApiError, ingestPath, resetKB, createWebSocket } from "@/lib/api";
import { useKB } from "@/lib/kb-context";

interface IngestMessage {
  event: string;
  message: string;
  timestamp: Date;
}

export function KnowledgeBaseView() {
  const { kb, refresh } = useKB();

  // Ingest form
  const [path, setPath] = useState("");
  const [reset, setReset] = useState(false);
  const [ingesting, setIngesting] = useState(false);
  const [ingestError, setIngestError] = useState<string | null>(null);
  const [ingestMessages, setIngestMessages] = useState<IngestMessage[]>([]);

  // Reset confirm
  const [resetting, setResetting] = useState(false);
  const [confirmReset, setConfirmReset] = useState(false);
  const [resetError, setResetError] = useState<string | null>(null);

  // Source filter
  const [filter, setFilter] = useState("");

  // WebSocket for ingest progress
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const ws = createWebSocket();
    wsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data as string) as Record<string, unknown>;
        const ev = typeof msg.event === "string" ? msg.event : "";
        if (!ev.startsWith("kb_ingest_")) return;

        const data =
          msg.data && typeof msg.data === "object"
            ? (msg.data as Record<string, unknown>)
            : {};
        const message =
          typeof data.message === "string" && data.message ? data.message : ev.replace(/_/g, " ");

        setIngestMessages((prev) =>
          [...prev, { event: ev, message, timestamp: new Date() }].slice(-50),
        );

        if (ev === "kb_ingest_complete" || ev === "kb_ingest_failed") {
          setIngesting(false);
          if (ev === "kb_ingest_complete") {
            refresh();
          }
        }
      } catch { /* ignore */ }
    };

    return () => ws.close();
  }, [refresh]);

  const handleIngest = useCallback(async () => {
    if (!path.trim()) return;
    setIngesting(true);
    setIngestError(null);
    setIngestMessages([]);
    try {
      await ingestPath(path.trim(), reset);
      // Progress comes via WebSocket; setIngesting(false) happens on kb_ingest_complete
    } catch (err) {
      setIngesting(false);
      if (err instanceof ApiError && err.code) {
        if (err.code === "bad_path") {
          setIngestError(`Path not found on the API host: ${err.message}`);
        } else if (err.code === "busy") {
          setIngestError("Another ingest is already running. Please wait.");
        } else {
          setIngestError(err.message);
        }
      } else {
        setIngestError(err instanceof Error ? err.message : "Ingest failed");
      }
    }
  }, [path, reset]);

  const handleReset = useCallback(async () => {
    setResetting(true);
    setResetError(null);
    try {
      await resetKB();
      refresh();
      setConfirmReset(false);
    } catch (err) {
      setResetError(err instanceof Error ? err.message : "Reset failed");
    } finally {
      setResetting(false);
    }
  }, [refresh]);

  const filteredSources = kb?.sources.filter((s) =>
    !filter || s.source.toLowerCase().includes(filter.toLowerCase()),
  ) ?? [];

  const lastIngested = kb?.last_ingested_at
    ? new Date(kb.last_ingested_at).toLocaleString()
    : "Never";

  return (
    <div className="h-full flex flex-col p-6 overflow-y-auto scrollbar-thin">
      <div className="mb-6">
        <h1 className="text-2xl font-bold">Knowledge Base</h1>
        <p className="text-zinc-400 text-sm mt-1">
          Ingest documents to build the knowledge base that powers every view.
        </p>
      </div>

      {/* Stats cards */}
      {kb && (
        <div className="grid grid-cols-5 gap-3 mb-6">
          <StatCard label="Chunks" value={kb.chunks.toLocaleString()} />
          <StatCard label="Entities" value={kb.entities.toLocaleString()} />
          <StatCard label="Relations" value={kb.relations.toLocaleString()} />
          <StatCard label="Communities" value={kb.communities.toLocaleString()} />
          <StatCard label="Last ingested" value={lastIngested} small />
        </div>
      )}

      {/* Ingest form */}
      <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 mb-6">
        <h2 className="text-sm font-semibold text-zinc-200 mb-4 flex items-center gap-2">
          <FolderOpen className="w-4 h-4 text-emerald-400" />
          Ingest documents
        </h2>
        <div className="flex gap-3 mb-3">
          <input
            type="text"
            value={path}
            onChange={(e) => setPath(e.target.value)}
            placeholder="e.g. /home/user/project or ./examples/aerospace"
            className="flex-1 bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-200 placeholder-zinc-500 outline-none focus:border-zinc-500"
          />
          <button
            onClick={handleIngest}
            disabled={ingesting || !path.trim()}
            className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:bg-zinc-700 disabled:text-zinc-500 text-white rounded-lg text-sm font-medium transition-colors"
          >
            {ingesting ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Database className="w-4 h-4" />
            )}
            {ingesting ? "Ingesting..." : "Ingest"}
          </button>
        </div>
        <label className="flex items-center gap-2 text-sm text-zinc-400 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={reset}
            onChange={(e) => setReset(e.target.checked)}
            className="rounded"
          />
          Replace existing (reset KB before ingesting)
        </label>
        <p className="mt-2 text-xs text-zinc-600">
          Paths resolve on the API host (e.g. the repo root when running locally).
        </p>

        {ingestError && (
          <div className="mt-3 flex items-start gap-2 px-3 py-2 bg-red-950/30 border border-red-800 rounded-lg text-xs text-red-300">
            <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
            {ingestError}
          </div>
        )}

        {/* Ingest progress */}
        {ingestMessages.length > 0 && (
          <div className="mt-3 space-y-1 max-h-32 overflow-y-auto scrollbar-thin">
            {ingestMessages.map((m, i) => (
              <div key={i} className="flex items-start gap-2 text-xs">
                {m.event === "kb_ingest_complete" ? (
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 mt-0.5 shrink-0" />
                ) : m.event === "kb_ingest_failed" ? (
                  <AlertTriangle className="w-3.5 h-3.5 text-red-400 mt-0.5 shrink-0" />
                ) : (
                  <Loader2 className="w-3.5 h-3.5 text-zinc-400 mt-0.5 shrink-0 animate-spin" />
                )}
                <span className={
                  m.event === "kb_ingest_complete" ? "text-emerald-300" :
                  m.event === "kb_ingest_failed" ? "text-red-300" : "text-zinc-400"
                }>
                  {m.message}
                </span>
                <span className="text-zinc-600 ml-auto shrink-0">
                  {m.timestamp.toLocaleTimeString()}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Sources table */}
      {kb && kb.sources.length > 0 && (
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden mb-6">
          <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800">
            <h2 className="text-sm font-semibold text-zinc-200">Sources</h2>
            <div className="relative">
              <Search className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-zinc-500" />
              <input
                type="text"
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                placeholder="Filter paths..."
                className="pl-7 pr-6 py-1.5 bg-zinc-800 border border-zinc-700 rounded-lg text-xs text-zinc-200 placeholder-zinc-500 outline-none focus:border-zinc-500 w-48"
              />
              {filter && (
                <button
                  onClick={() => setFilter("")}
                  className="absolute right-2 top-2 text-zinc-500 hover:text-zinc-300"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          </div>
          <div className="max-h-64 overflow-y-auto scrollbar-thin">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-zinc-500 text-xs">
                  <th className="px-4 py-2 text-left font-medium">Path</th>
                  <th className="px-4 py-2 text-right font-medium">Chunks</th>
                </tr>
              </thead>
              <tbody>
                {filteredSources.map((s, i) => (
                  <tr key={i} className="border-t border-zinc-800/50 hover:bg-zinc-800/30">
                    <td className="px-4 py-2 font-mono text-xs text-zinc-300 truncate max-w-xs">
                      {s.source}
                    </td>
                    <td className="px-4 py-2 text-right text-zinc-400">{s.chunks}</td>
                  </tr>
                ))}
                {filteredSources.length === 0 && (
                  <tr>
                    <td colSpan={2} className="px-4 py-4 text-xs text-zinc-500 text-center">
                      No sources match the filter.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Reset section */}
      <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-4">
        <h2 className="text-sm font-semibold text-zinc-200 mb-2 flex items-center gap-2">
          <Trash2 className="w-4 h-4 text-red-400" />
          Reset knowledge base
        </h2>
        <p className="text-xs text-zinc-500 mb-3">
          Permanently deletes all chunks, entities, relations, and communities. This cannot be
          undone.
        </p>

        {resetError && (
          <p className="mb-2 text-xs text-red-300">{resetError}</p>
        )}

        {confirmReset ? (
          <div className="flex items-center gap-3">
            <span className="text-xs text-amber-300">Are you sure? This cannot be undone.</span>
            <button
              onClick={handleReset}
              disabled={resetting}
              className="flex items-center gap-1 px-3 py-1.5 bg-red-700 hover:bg-red-600 disabled:opacity-50 text-white rounded-lg text-xs font-medium"
            >
              {resetting ? <Loader2 className="w-3 h-3 animate-spin" /> : <Trash2 className="w-3 h-3" />}
              Confirm reset
            </button>
            <button
              onClick={() => setConfirmReset(false)}
              className="text-xs text-zinc-400 hover:text-zinc-200"
            >
              Cancel
            </button>
          </div>
        ) : (
          <button
            onClick={() => setConfirmReset(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded-lg text-xs"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Reset knowledge base
          </button>
        )}
      </div>
    </div>
  );
}

function StatCard({
  label,
  value,
  small,
}: {
  label: string;
  value: string;
  small?: boolean;
}) {
  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-4">
      <p className="text-xs text-zinc-500 mb-1">{label}</p>
      <p className={`font-bold ${small ? "text-sm" : "text-xl"} truncate`}>{value}</p>
    </div>
  );
}
