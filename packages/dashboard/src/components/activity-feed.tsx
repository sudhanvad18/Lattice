"use client";

import { Activity, Wifi, WifiOff, Trash2 } from "lucide-react";
import { useActivityFeed } from "@/hooks/use-websocket";

export function ActivityFeed() {
  const { events, connected, clear } = useActivityFeed();

  return (
    <div className="h-full flex flex-col p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">Agent Activity</h1>
          <p className="text-zinc-400 text-sm mt-1">
            Live feed of all agent actions across all tasks
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={clear}
            className="flex items-center gap-1 px-3 py-1.5 text-xs text-zinc-400 hover:text-zinc-200 bg-zinc-800 rounded-lg transition-colors"
          >
            <Trash2 className="w-3 h-3" /> Clear
          </button>
          <div className="flex items-center gap-1.5">
            {connected ? (
              <>
                <Wifi className="w-4 h-4 text-emerald-400" />
                <span className="text-xs text-emerald-400">Connected</span>
              </>
            ) : (
              <>
                <WifiOff className="w-4 h-4 text-red-400" />
                <span className="text-xs text-red-400">Disconnected</span>
              </>
            )}
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto scrollbar-thin space-y-1">
        {events.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 text-zinc-600">
            <Activity className="w-8 h-8 mb-2" />
            <p className="text-sm">No activity yet. Submit a task to see agents at work.</p>
          </div>
        ) : (
          events.map((event, i) => (
            <div
              key={i}
              className="flex items-start gap-4 px-4 py-3 bg-zinc-900/30 hover:bg-zinc-900/60 rounded-lg transition-colors"
            >
              <EventIcon type={event.event_type} />
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <AgentTag agent={event.agent} />
                  <span className="text-xs text-zinc-500 font-mono">
                    {event.task_id.slice(0, 8)}
                  </span>
                </div>
                <p className="text-sm text-zinc-300 mt-0.5 truncate">
                  {event.details}
                </p>
              </div>
              <span className="text-xs text-zinc-600 shrink-0">
                {new Date(event.timestamp).toLocaleTimeString()}
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

function EventIcon({ type }: { type: string }) {
  const base = "w-5 h-5 mt-0.5";
  switch (type) {
    case "started":
      return <div className={`${base} rounded-full bg-blue-500/20 border border-blue-500 flex items-center justify-center`}><div className="w-1.5 h-1.5 rounded-full bg-blue-400" /></div>;
    case "completed":
      return <div className={`${base} rounded-full bg-emerald-500/20 border border-emerald-500 flex items-center justify-center`}><div className="w-1.5 h-1.5 rounded-full bg-emerald-400" /></div>;
    case "error":
      return <div className={`${base} rounded-full bg-red-500/20 border border-red-500 flex items-center justify-center`}><div className="w-1.5 h-1.5 rounded-full bg-red-400" /></div>;
    case "tool_call":
      return <div className={`${base} rounded-full bg-amber-500/20 border border-amber-500 flex items-center justify-center`}><div className="w-1.5 h-1.5 rounded-full bg-amber-400" /></div>;
    default:
      return <div className={`${base} rounded-full bg-zinc-700 border border-zinc-600 flex items-center justify-center`}><div className="w-1.5 h-1.5 rounded-full bg-zinc-400" /></div>;
  }
}

function AgentTag({ agent }: { agent: string }) {
  const colors: Record<string, string> = {
    orchestrator: "text-purple-400",
    researcher: "text-blue-400",
    writer: "text-green-400",
    reviewer: "text-amber-400",
    code: "text-cyan-400",
    ingestion: "text-rose-400",
  };
  return (
    <span className={`text-xs font-medium ${colors[agent] || "text-zinc-400"}`}>
      {agent}
    </span>
  );
}
