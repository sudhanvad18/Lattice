"use client";

import { useEffect, useState } from "react";
import { History, ChevronRight, FileText, Code, Search, CheckCircle2, XCircle, Clock } from "lucide-react";
import { getTasks, getTask, Task, Artifact } from "@/lib/api";

export function TaskHistory() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [selectedTask, setSelectedTask] = useState<Task | null>(null);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedArtifact, setSelectedArtifact] = useState<Artifact | null>(null);

  useEffect(() => {
    loadTasks();
  }, []);

  const loadTasks = async () => {
    try {
      const data = await getTasks();
      setTasks(data);
    } catch {
      setTasks(getDemoTasks());
    } finally {
      setLoading(false);
    }
  };

  const selectTask = async (task: Task) => {
    setSelectedTask(task);
    setSelectedArtifact(null);
    try {
      const detail = await getTask(task.id);
      setArtifacts(detail.artifacts || []);
    } catch {
      setArtifacts(getDemoArtifacts());
    }
  };

  if (loading) {
    return <div className="h-full flex items-center justify-center text-zinc-500">Loading history...</div>;
  }

  return (
    <div className="h-full flex">
      {/* Task List */}
      <div className="w-80 border-r border-zinc-800 overflow-y-auto scrollbar-thin">
        <div className="p-4 border-b border-zinc-800">
          <h2 className="text-lg font-bold">Task History</h2>
          <p className="text-xs text-zinc-500 mt-1">{tasks.length} tasks</p>
        </div>
        <div className="p-2 space-y-1">
          {tasks.map((task) => (
            <button
              key={task.id}
              onClick={() => selectTask(task)}
              className={`w-full text-left px-3 py-3 rounded-lg transition-colors ${
                selectedTask?.id === task.id
                  ? "bg-zinc-800 border border-zinc-700"
                  : "hover:bg-zinc-900"
              }`}
            >
              <div className="flex items-start justify-between">
                <p className="text-sm text-zinc-200 line-clamp-2">{task.goal}</p>
                <TaskStatusIcon status={task.status} />
              </div>
              <div className="flex items-center gap-2 mt-1.5">
                <span className="text-xs text-zinc-600 font-mono">{task.id.slice(0, 8)}</span>
                <span className="text-xs text-zinc-600">
                  {new Date(task.created_at).toLocaleDateString()}
                </span>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Task Detail / Artifact Browser */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {!selectedTask ? (
          <div className="flex-1 flex items-center justify-center text-zinc-600">
            <div className="text-center">
              <History className="w-8 h-8 mx-auto mb-2" />
              <p className="text-sm">Select a task to view artifacts</p>
            </div>
          </div>
        ) : (
          <>
            {/* Task Header */}
            <div className="p-4 border-b border-zinc-800">
              <div className="flex items-center gap-2">
                <TaskStatusIcon status={selectedTask.status} />
                <h2 className="font-medium">{selectedTask.goal}</h2>
              </div>
              <div className="flex items-center gap-4 mt-2 text-xs text-zinc-500">
                <span>ID: {selectedTask.id.slice(0, 12)}</span>
                <span>Created: {new Date(selectedTask.created_at).toLocaleString()}</span>
                <span>Approval: {selectedTask.approval_mode}</span>
              </div>
            </div>

            {/* Artifacts */}
            <div className="flex-1 flex overflow-hidden">
              {/* Artifact List */}
              <div className="w-56 border-r border-zinc-800 overflow-y-auto scrollbar-thin p-2">
                <p className="px-2 py-1 text-xs text-zinc-500 font-medium uppercase">
                  Artifacts ({artifacts.length})
                </p>
                {artifacts.map((artifact) => (
                  <button
                    key={artifact.id}
                    onClick={() => setSelectedArtifact(artifact)}
                    className={`w-full text-left px-3 py-2 rounded-lg flex items-center gap-2 text-sm transition-colors ${
                      selectedArtifact?.id === artifact.id
                        ? "bg-zinc-800 text-zinc-200"
                        : "text-zinc-400 hover:bg-zinc-900 hover:text-zinc-300"
                    }`}
                  >
                    <ArtifactIcon type={artifact.artifact_type} />
                    <div className="min-w-0">
                      <p className="truncate text-xs">{artifact.artifact_type}</p>
                      <p className="text-xs text-zinc-600">{artifact.source_agent}</p>
                    </div>
                    <ChevronRight className="w-3 h-3 ml-auto shrink-0 text-zinc-600" />
                  </button>
                ))}
              </div>

              {/* Artifact Content */}
              <div className="flex-1 overflow-y-auto scrollbar-thin p-4">
                {!selectedArtifact ? (
                  <div className="h-full flex items-center justify-center text-zinc-600 text-sm">
                    Select an artifact to view its content
                  </div>
                ) : (
                  <div>
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <h3 className="font-medium capitalize">{selectedArtifact.artifact_type}</h3>
                        <p className="text-xs text-zinc-500">
                          by {selectedArtifact.source_agent} — confidence: {(selectedArtifact.confidence * 100).toFixed(0)}%
                        </p>
                      </div>
                    </div>
                    <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-4">
                      <pre className="text-sm text-zinc-300 whitespace-pre-wrap font-mono leading-relaxed">
                        {selectedArtifact.content}
                      </pre>
                    </div>
                    {selectedArtifact.citations.length > 0 && (
                      <div className="mt-3">
                        <p className="text-xs text-zinc-500 mb-1">Citations:</p>
                        <div className="flex flex-wrap gap-1">
                          {selectedArtifact.citations.map((c, i) => (
                            <span key={i} className="text-xs px-2 py-0.5 bg-zinc-800 rounded text-zinc-400">
                              {c}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function TaskStatusIcon({ status }: { status: string }) {
  switch (status) {
    case "completed":
      return <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />;
    case "failed":
      return <XCircle className="w-4 h-4 text-red-400 shrink-0" />;
    case "running":
      return <Clock className="w-4 h-4 text-amber-400 shrink-0 animate-pulse" />;
    default:
      return <Clock className="w-4 h-4 text-zinc-500 shrink-0" />;
  }
}

function ArtifactIcon({ type }: { type: string }) {
  switch (type) {
    case "code":
      return <Code className="w-4 h-4 text-cyan-400 shrink-0" />;
    case "research":
      return <Search className="w-4 h-4 text-blue-400 shrink-0" />;
    default:
      return <FileText className="w-4 h-4 text-zinc-400 shrink-0" />;
  }
}

function getDemoTasks(): Task[] {
  return [
    { id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890", goal: "Generate turbine maintenance documentation", status: "completed", created_at: "2026-07-28T10:00:00Z", approval_mode: "on_low_confidence" },
    { id: "b2c3d4e5-f6a7-8901-bcde-f12345678901", goal: "Code review: authentication middleware refactor", status: "completed", created_at: "2026-07-27T14:30:00Z", approval_mode: "always" },
    { id: "c3d4e5f6-a7b8-9012-cdef-123456789012", goal: "Draft API migration plan from REST to GraphQL", status: "running", created_at: "2026-07-28T16:00:00Z", approval_mode: "on_low_confidence" },
    { id: "d4e5f6a7-b8c9-0123-defa-234567890123", goal: "Analyze thermal fatigue patterns in Q3 data", status: "failed", created_at: "2026-07-26T09:15:00Z", approval_mode: "never" },
  ];
}

function getDemoArtifacts(): Artifact[] {
  return [
    {
      id: "art-1",
      artifact_type: "document",
      content: "# Turbine Maintenance Documentation\n\n## Overview\nThis document outlines the maintenance procedures for the PW4000 series turbofan engines.\n\n## Inspection Schedule\n- Visual inspection: Every 500 flight hours\n- Boroscope inspection: Every 2,000 flight hours\n- Full teardown: Every 10,000 flight hours\n\n## Common Issues\n1. Blade erosion in the first-stage compressor\n2. Thermal fatigue cracking in combustion liners\n3. Bearing wear in the high-pressure turbine section",
      source_agent: "writer",
      confidence: 0.88,
      citations: ["chunk_001", "chunk_014", "chunk_032"],
      created_at: "2026-07-28T10:05:00Z",
    },
    {
      id: "art-2",
      artifact_type: "research",
      content: "Found 8 relevant documents covering turbine maintenance procedures.\n\nKey findings:\n- PW4000 maintenance intervals are well-documented in source_014\n- Blade erosion patterns are described in source_032\n- Recent field data shows 15% reduction in unscheduled maintenance",
      source_agent: "researcher",
      confidence: 0.92,
      citations: ["chunk_001", "chunk_014", "chunk_032", "chunk_045"],
      created_at: "2026-07-28T10:02:00Z",
    },
  ];
}
