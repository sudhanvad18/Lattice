"use client";

import { useState } from "react";
import {
  Activity,
  MessageSquare,
  Network,
  BarChart3,
  History,
  Zap,
  Database,
  AlertTriangle,
} from "lucide-react";
import { AskView } from "@/components/ask";
import { ActivityFeed } from "@/components/activity-feed";
import { KnowledgeGraphView } from "@/components/kg-visualizer";
import { EvalDashboard } from "@/components/eval-dashboard";
import { KnowledgeBaseView } from "@/components/knowledge-base";
import { TaskHistory } from "@/components/task-history";
import { ViewErrorBoundary } from "@/components/error-boundary";
import { useKB } from "@/lib/kb-context";

type View = "kb" | "ask" | "activity" | "graph" | "eval" | "history";

const NAV_ITEMS: { id: View; label: string; icon: React.ReactNode }[] = [
  { id: "kb", label: "Knowledge Base", icon: <Database className="w-4 h-4" /> },
  { id: "ask", label: "Ask", icon: <MessageSquare className="w-4 h-4" /> },
  { id: "activity", label: "Activity", icon: <Activity className="w-4 h-4" /> },
  { id: "graph", label: "Graph", icon: <Network className="w-4 h-4" /> },
  { id: "eval", label: "Evaluations", icon: <BarChart3 className="w-4 h-4" /> },
  { id: "history", label: "History", icon: <History className="w-4 h-4" /> },
];

export default function Dashboard() {
  const { kb, health, apiReachable } = useKB();
  const [manualView, setView] = useState<View | null>(null);

  // Derive active view: manual override wins; until kb loads show "kb"; then pick based on readiness.
  const activeView: View = manualView ?? (kb === null ? "kb" : kb.ready ? "ask" : "kb");

  const providerReachable = apiReachable && !!health && !health.provider.detail;
  const providerLabel =
    health ? `${health.provider.name} / ${health.provider.model}` : null;

  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      <aside className="w-56 border-r border-zinc-800 bg-zinc-900/50 flex flex-col">
        <div className="p-4 border-b border-zinc-800">
          <div className="flex items-center gap-2">
            <Zap className="w-5 h-5 text-emerald-400" />
            <span className="font-semibold text-lg">Lattice</span>
          </div>
          <p className="text-xs text-zinc-500 mt-1">Agent Platform</p>
        </div>
        <nav className="flex-1 p-2 space-y-1">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              onClick={() => setView(item.id)}
              className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${
                activeView === item.id
                  ? "bg-zinc-800 text-emerald-400"
                  : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50"
              }`}
            >
              {item.icon}
              {item.label}
            </button>
          ))}
        </nav>

        {/* Live status footer */}
        <div className="p-3 border-t border-zinc-800 space-y-1.5">
          {/* API / model status */}
          <div className="flex items-center gap-2 min-w-0">
            <div
              className={`w-2 h-2 rounded-full shrink-0 ${
                !apiReachable
                  ? "bg-red-400"
                  : providerReachable
                    ? "bg-emerald-400"
                    : "bg-amber-400"
              }`}
            />
            {!apiReachable ? (
              <span className="text-xs text-red-400">API offline</span>
            ) : providerLabel ? (
              <span
                title={health?.provider.detail}
                className={`text-xs truncate ${
                  providerReachable ? "text-emerald-400" : "text-amber-400"
                }`}
              >
                {providerLabel}
              </span>
            ) : (
              <span className="text-xs text-zinc-500">Connecting...</span>
            )}
            {!providerReachable && health?.provider.detail && (
              <AlertTriangle className="w-3 h-3 text-amber-400 shrink-0" />
            )}
          </div>

          {/* KB counts */}
          {kb && (
            <p className="text-xs text-zinc-600">
              {kb.chunks.toLocaleString()} chunks · {kb.entities.toLocaleString()} entities
            </p>
          )}
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-hidden">
        <ViewErrorBoundary
          key={activeView}
          label={NAV_ITEMS.find((item) => item.id === activeView)?.label ?? activeView}
        >
          {activeView === "kb" && <KnowledgeBaseView />}
          {activeView === "ask" && <AskView onNavigateToKB={() => setView("kb")} />}
          {activeView === "activity" && <ActivityFeed />}
          {activeView === "graph" && <KnowledgeGraphView />}
          {activeView === "eval" && <EvalDashboard />}
          {activeView === "history" && <TaskHistory />}
        </ViewErrorBoundary>
      </main>
    </div>
  );
}
