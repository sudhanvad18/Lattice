"use client";

import { useState } from "react";
import {
  Activity,
  MessageSquare,
  Network,
  BarChart3,
  History,
  Send,
  Zap,
} from "lucide-react";
import { TaskSubmission } from "@/components/task-submission";
import { ActivityFeed } from "@/components/activity-feed";
import { KnowledgeGraphView } from "@/components/kg-visualizer";
import { EvalDashboard } from "@/components/eval-dashboard";
import { ChatInterface } from "@/components/chat-interface";
import { TaskHistory } from "@/components/task-history";

type View = "tasks" | "activity" | "graph" | "eval" | "chat" | "history";

const NAV_ITEMS: { id: View; label: string; icon: React.ReactNode }[] = [
  { id: "tasks", label: "Submit Task", icon: <Send className="w-4 h-4" /> },
  { id: "activity", label: "Activity", icon: <Activity className="w-4 h-4" /> },
  { id: "graph", label: "Knowledge Graph", icon: <Network className="w-4 h-4" /> },
  { id: "eval", label: "Evaluations", icon: <BarChart3 className="w-4 h-4" /> },
  { id: "chat", label: "Chat", icon: <MessageSquare className="w-4 h-4" /> },
  { id: "history", label: "History", icon: <History className="w-4 h-4" /> },
];

export default function Dashboard() {
  const [view, setView] = useState<View>("tasks");

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
                view === item.id
                  ? "bg-zinc-800 text-emerald-400"
                  : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50"
              }`}
            >
              {item.icon}
              {item.label}
            </button>
          ))}
        </nav>
        <div className="p-3 border-t border-zinc-800">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-xs text-zinc-500">System Online</span>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-hidden">
        {view === "tasks" && <TaskSubmission />}
        {view === "activity" && <ActivityFeed />}
        {view === "graph" && <KnowledgeGraphView />}
        {view === "eval" && <EvalDashboard />}
        {view === "chat" && <ChatInterface />}
        {view === "history" && <TaskHistory />}
      </main>
    </div>
  );
}
