"use client";

import { useState } from "react";
import { Send, Loader2, CheckCircle2, AlertCircle } from "lucide-react";
import { submitTask, Task } from "@/lib/api";
import { useTaskStream } from "@/hooks/use-websocket";

export function TaskSubmission() {
  const [goal, setGoal] = useState("");
  const [approvalMode, setApprovalMode] = useState("on_low_confidence");
  const [submitting, setSubmitting] = useState(false);
  const [currentTask, setCurrentTask] = useState<Task | null>(null);
  const [error, setError] = useState<string | null>(null);

  const { events, status } = useTaskStream(currentTask?.id ?? null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!goal.trim()) return;

    setSubmitting(true);
    setError(null);
    try {
      const task = await submitTask(goal, approvalMode);
      setCurrentTask(task);
      setGoal("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to submit task");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="h-full flex flex-col p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold">Submit Task</h1>
        <p className="text-zinc-400 text-sm mt-1">
          Describe what you need. The agent team will plan, research, write, review, and deliver.
        </p>
      </div>

      {/* Submission Form */}
      <form onSubmit={handleSubmit} className="mb-6">
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-4">
          <textarea
            value={goal}
            onChange={(e) => setGoal(e.target.value)}
            placeholder="e.g., Write a technical design doc for migrating our auth service to OAuth2..."
            className="w-full bg-transparent text-zinc-100 placeholder-zinc-600 resize-none outline-none min-h-[120px] text-sm"
            rows={4}
          />
          <div className="flex items-center justify-between mt-3 pt-3 border-t border-zinc-800">
            <div className="flex items-center gap-2">
              <label className="text-xs text-zinc-500">Approval:</label>
              <select
                value={approvalMode}
                onChange={(e) => setApprovalMode(e.target.value)}
                className="bg-zinc-800 border border-zinc-700 rounded-md px-2 py-1 text-xs text-zinc-300"
              >
                <option value="never">Auto (No Review)</option>
                <option value="on_low_confidence">On Low Confidence</option>
                <option value="always">Always Review</option>
              </select>
            </div>
            <button
              type="submit"
              disabled={submitting || !goal.trim()}
              className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:bg-zinc-700 disabled:text-zinc-500 text-white rounded-lg text-sm font-medium transition-colors"
            >
              {submitting ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
              Submit
            </button>
          </div>
        </div>
      </form>

      {error && (
        <div className="mb-4 p-3 bg-red-950/30 border border-red-800 rounded-lg flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-red-400" />
          <span className="text-sm text-red-300">{error}</span>
        </div>
      )}

      {/* Live Task Progress */}
      {currentTask && (
        <div className="flex-1 overflow-hidden flex flex-col">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <h2 className="text-sm font-medium text-zinc-300">Task Progress</h2>
              <span className="text-xs px-2 py-0.5 rounded-full bg-zinc-800 text-zinc-400">
                {currentTask.id.slice(0, 8)}
              </span>
            </div>
            <StatusBadge status={status} />
          </div>
          <div className="flex-1 overflow-y-auto scrollbar-thin bg-zinc-900/50 border border-zinc-800 rounded-xl p-4 space-y-2">
            {events.length === 0 && status === "running" && (
              <div className="flex items-center gap-2 text-zinc-500 text-sm">
                <Loader2 className="w-3 h-3 animate-spin" />
                Waiting for agent activity...
              </div>
            )}
            {events.map((event, i) => (
              <div key={i} className="flex items-start gap-3 text-sm">
                <span className="text-zinc-600 font-mono text-xs mt-0.5 shrink-0">
                  {new Date(event.timestamp).toLocaleTimeString()}
                </span>
                <AgentBadge agent={event.agent} />
                <span className="text-zinc-300">{event.details}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  if (status === "running") {
    return (
      <span className="flex items-center gap-1 text-xs text-amber-400">
        <Loader2 className="w-3 h-3 animate-spin" /> Running
      </span>
    );
  }
  if (status === "done") {
    return (
      <span className="flex items-center gap-1 text-xs text-emerald-400">
        <CheckCircle2 className="w-3 h-3" /> Complete
      </span>
    );
  }
  return <span className="text-xs text-zinc-500">{status}</span>;
}

function AgentBadge({ agent }: { agent: string }) {
  const colors: Record<string, string> = {
    orchestrator: "bg-purple-900/50 text-purple-300 border-purple-700",
    researcher: "bg-blue-900/50 text-blue-300 border-blue-700",
    writer: "bg-green-900/50 text-green-300 border-green-700",
    reviewer: "bg-amber-900/50 text-amber-300 border-amber-700",
    code: "bg-cyan-900/50 text-cyan-300 border-cyan-700",
    ingestion: "bg-rose-900/50 text-rose-300 border-rose-700",
  };
  const style = colors[agent] || "bg-zinc-800 text-zinc-400 border-zinc-700";
  return (
    <span className={`px-2 py-0.5 rounded text-xs border shrink-0 ${style}`}>
      {agent}
    </span>
  );
}
