"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  Send,
  Loader2,
  AlertCircle,
  BookOpen,
  Users,
  ChevronDown,
  ChevronRight,
  AlertTriangle,
} from "lucide-react";
import {
  chatQuery,
  submitTask,
  getTaskResult,
  createWebSocket,
  normalizeAgentEvent,
  ChatCitation,
  AgentEvent,
  Artifact,
} from "@/lib/api";
import { useKB } from "@/lib/kb-context";

type AskMode = "answer" | "agent";

interface ConversationMessage {
  role: "user" | "assistant";
  content: string;
  citations?: ChatCitation[];
  chatMode?: string;
  notice?: string | null;
  agentRun?: AgentRun;
  timestamp: Date;
}

interface AgentRun {
  taskId: string;
  events: AgentEvent[];
  status: "running" | "done" | "error";
  artifact?: Artifact;
  errorMessage?: string;
}

export function AskView({ onNavigateToKB }: { onNavigateToKB: () => void }) {
  const { kb, health, apiReachable } = useKB();
  const [mode, setMode] = useState<AskMode>("answer");
  const [approvalMode, setApprovalMode] = useState("on_low_confidence");
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  const kbReady = kb?.ready ?? false;
  const modelReachable = apiReachable && !!health && !!health.provider?.reachable;

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  // --- Answer mode ---

  const sendAnswer = useCallback(
    async (userMessage: string) => {
      const history = messages.map((m) => ({ role: m.role, content: m.content }));
      try {
        const res = await chatQuery(userMessage, history);
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: res.response,
            citations: res.citations,
            chatMode: res.mode,
            notice: res.notice,
            timestamp: new Date(),
          },
        ]);
      } catch (err) {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: err instanceof Error ? err.message : "Could not get an answer.",
            timestamp: new Date(),
          },
        ]);
      }
    },
    [messages],
  );

  // --- Agent team mode ---

  const sendAgentTask = useCallback(async (userMessage: string) => {
    let taskId = "";
    try {
      const task = await submitTask(userMessage, approvalMode);
      taskId = task.task_id;
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: err instanceof Error ? err.message : "Failed to submit task.",
          timestamp: new Date(),
        },
      ]);
      setLoading(false);
      return;
    }

    const runState: AgentRun = { taskId, events: [], status: "running" };

    setMessages((prev) => [
      ...prev,
      {
        role: "assistant",
        content: "",
        agentRun: runState,
        timestamp: new Date(),
      },
    ]);
    setLoading(false);

    const ws = createWebSocket(taskId);

    ws.onmessage = (event) => {
      try {
        const data = normalizeAgentEvent(JSON.parse(event.data as string), taskId);
        if (!data) return;

        setMessages((prev) => {
          const idx = prev.findLastIndex((m) => m.agentRun?.taskId === taskId);
          if (idx === -1) return prev;
          const updated = { ...prev[idx] };
          const run: AgentRun = { ...updated.agentRun!, events: [...updated.agentRun!.events, data] };
          if (data.event_type === "completed") run.status = "done";
          if (data.event_type === "error") {
            run.status = "error";
            run.errorMessage = data.details;
          }
          updated.agentRun = run;
          return [...prev.slice(0, idx), updated, ...prev.slice(idx + 1)];
        });

        if (data.event_type === "completed") {
          ws.close();
          getTaskResult(taskId)
            .then((result) => {
              const artifact =
                result.artifacts.find((a) => a.artifact_type === "document") ??
                result.artifacts[result.artifacts.length - 1];
              setMessages((prev) => {
                const idx = prev.findLastIndex((m) => m.agentRun?.taskId === taskId);
                if (idx === -1) return prev;
                const updated = { ...prev[idx] };
                updated.agentRun = { ...updated.agentRun!, status: "done", artifact };
                return [...prev.slice(0, idx), updated, ...prev.slice(idx + 1)];
              });
            })
            .catch(() => { /* artifact display is best-effort */ });
        }
        if (data.event_type === "error") ws.close();
      } catch { /* ignore */ }
    };

    ws.onclose = () => {
      setMessages((prev) => {
        const idx = prev.findLastIndex((m) => m.agentRun?.taskId === taskId);
        if (idx === -1) return prev;
        if (prev[idx].agentRun?.status === "running") {
          const updated = { ...prev[idx] };
          updated.agentRun = { ...updated.agentRun!, status: "done" };
          return [...prev.slice(0, idx), updated, ...prev.slice(idx + 1)];
        }
        return prev;
      });
    };
  }, [approvalMode]);

  const handleSend = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      const text = input.trim();
      if (!text || loading) return;

      setMessages((prev) => [...prev, { role: "user", content: text, timestamp: new Date() }]);
      setInput("");
      setLoading(true);

      if (mode === "answer") {
        await sendAnswer(text);
        setLoading(false);
      } else {
        await sendAgentTask(text);
        // loading is set false inside sendAgentTask after task is submitted
      }
    },
    [input, loading, mode, sendAnswer, sendAgentTask],
  );

  // --- Render ---

  const composerDisabled = !kbReady;

  return (
    <div className="h-full flex flex-col p-6">
      {/* Header + mode switch */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-2xl font-bold">Ask</h1>
          <p className="text-zinc-400 text-sm mt-1">
            Query your knowledge base or launch the agent team.
          </p>
        </div>
        <div className="flex bg-zinc-800 rounded-lg p-0.5" role="group">
          <ModeButton active={mode === "answer"} onClick={() => setMode("answer")}>
            <BookOpen className="w-3.5 h-3.5" /> Answer
          </ModeButton>
          <ModeButton
            active={mode === "agent"}
            onClick={() => setMode("agent")}
            disabled={!modelReachable}
            title={
              !modelReachable ? (health?.provider?.detail ?? "Model unavailable") : undefined
            }
          >
            <Users className="w-3.5 h-3.5" /> Agent team
          </ModeButton>
        </div>
      </div>

      {/* Retrieval-only notice for Answer mode when model is down */}
      {mode === "answer" && !modelReachable && apiReachable && (
        <div className="mb-3 px-3 py-2 bg-amber-950/30 border border-amber-800 rounded-lg text-xs text-amber-300 flex items-center gap-2">
          <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
          Model unavailable — answers will use retrieval only
        </div>
      )}

      {/* Thread */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto scrollbar-thin space-y-4 mb-4">
        {messages.length === 0 && kbReady && (
          <div className="flex flex-col items-center justify-center h-full text-zinc-600">
            <BookOpen className="w-10 h-10 mb-3" />
            <p className="text-sm">Ask anything about your knowledge base.</p>
          </div>
        )}

        {!kbReady && (
          <div className="flex flex-col items-center justify-center h-full text-zinc-600">
            <AlertCircle className="w-10 h-10 mb-3" />
            <p className="text-sm mb-2">The knowledge base is empty.</p>
            <button
              onClick={onNavigateToKB}
              className="text-sm text-emerald-400 hover:underline"
            >
              Go to Knowledge Base to ingest documents
            </button>
          </div>
        )}

        {messages.map((msg, i) => (
          <MessageBubble key={i} msg={msg} />
        ))}

        {loading && mode === "answer" && (
          <div className="flex justify-start">
            <div className="bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-3">
              <Loader2 className="w-4 h-4 animate-spin text-zinc-500" />
            </div>
          </div>
        )}
      </div>

      {/* Composer */}
      <form onSubmit={handleSend} className="shrink-0">
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-3">
          {composerDisabled ? (
            <div className="flex items-center gap-2 text-sm text-zinc-500 py-2">
              <AlertCircle className="w-4 h-4" />
              <span>
                Knowledge base is empty.{" "}
                <button
                  type="button"
                  onClick={onNavigateToKB}
                  className="text-emerald-400 hover:underline"
                >
                  Ingest documents
                </button>{" "}
                to enable asking.
              </span>
            </div>
          ) : (
            <>
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    void handleSend(e as unknown as React.FormEvent);
                  }
                }}
                placeholder={
                  mode === "answer"
                    ? "Ask a question... (Enter to send)"
                    : "Describe a task for the agent team..."
                }
                rows={3}
                className="w-full bg-transparent text-zinc-100 placeholder-zinc-600 resize-none outline-none text-sm"
              />
              <div className="flex items-center justify-between mt-2 pt-2 border-t border-zinc-800">
                {mode === "agent" ? (
                  <div className="flex items-center gap-2">
                    <label className="text-xs text-zinc-500">Approval:</label>
                    <select
                      value={approvalMode}
                      onChange={(e) => setApprovalMode(e.target.value)}
                      className="bg-zinc-800 border border-zinc-700 rounded-md px-2 py-1 text-xs text-zinc-300"
                    >
                      <option value="never">Auto</option>
                      <option value="on_low_confidence">On Low Confidence</option>
                      <option value="always">Always Review</option>
                    </select>
                  </div>
                ) : (
                  <div />
                )}
                <button
                  type="submit"
                  disabled={loading || !input.trim()}
                  className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:bg-zinc-700 disabled:text-zinc-500 text-white rounded-lg text-sm font-medium transition-colors"
                >
                  {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                  {mode === "answer" ? "Ask" : "Run"}
                </button>
              </div>
            </>
          )}
        </div>
      </form>
    </div>
  );
}

// --- Sub-components ---

function ModeButton({
  active,
  onClick,
  disabled,
  title,
  children,
}: {
  active: boolean;
  onClick: () => void;
  disabled?: boolean;
  title?: string;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs transition-colors ${
        active
          ? "bg-zinc-700 text-emerald-400"
          : disabled
            ? "text-zinc-600 cursor-not-allowed"
            : "text-zinc-400 hover:text-zinc-200"
      }`}
    >
      {children}
    </button>
  );
}

function MessageBubble({ msg }: { msg: ConversationMessage }) {
  const [expandedCitation, setExpandedCitation] = useState<number | null>(null);

  if (msg.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[75%] rounded-xl px-4 py-3 bg-emerald-900/30 border border-emerald-800 text-zinc-200">
          <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
          <span className="text-xs text-zinc-600 mt-1 block">
            {msg.timestamp.toLocaleTimeString()}
          </span>
        </div>
      </div>
    );
  }

  // Assistant: agent run
  if (msg.agentRun) {
    return <AgentRunBubble run={msg.agentRun} timestamp={msg.timestamp} />;
  }

  // Assistant: answer
  const citations = msg.citations ?? [];
  const chatMode = msg.chatMode;
  const openCitation = citations.find((c) => c.index === expandedCitation);

  return (
    <div className="flex justify-start">
      <div className="max-w-[80%] rounded-xl px-4 py-3 bg-zinc-900 border border-zinc-800 text-zinc-300">
        {chatMode === "retrieval_only" && msg.notice && (
          <div className="mb-2 px-2 py-1.5 bg-amber-950/30 border border-amber-800 rounded-lg text-xs text-amber-300 flex items-center gap-1.5">
            <AlertTriangle className="w-3 h-3 shrink-0" />
            {msg.notice}
          </div>
        )}

        <AnswerText content={msg.content} citations={citations}
          expandedCitation={expandedCitation} onToggle={setExpandedCitation} />

        {openCitation && (
          <div className="mt-2 p-3 bg-zinc-800 border border-zinc-700 rounded-lg text-xs">
            <p className="font-medium text-zinc-300 mb-1">
              [{openCitation.index}] {openCitation.source}
            </p>
            <p className="text-zinc-400">{openCitation.snippet}</p>
          </div>
        )}

        <span className="text-xs text-zinc-600 mt-1 block">
          {msg.timestamp.toLocaleTimeString()}
        </span>
      </div>
    </div>
  );
}

function AnswerText({
  content,
  citations,
  expandedCitation,
  onToggle,
}: {
  content: string;
  citations: ChatCitation[];
  expandedCitation: number | null;
  onToggle: (n: number) => void;
}) {
  // Split on [n] markers, alternating between text and marker indices
  const parts = content.split(/\[(\d+)\]/g);
  return (
    <p className="text-sm whitespace-pre-wrap">
      {parts.map((part, i) => {
        if (i % 2 === 0) return part;
        const n = parseInt(part, 10);
        if (!citations.some((c) => c.index === n)) return `[${part}]`;
        const isOpen = expandedCitation === n;
        return (
          <button
            key={i}
            onClick={() => onToggle(isOpen ? -1 : n)}
            className={`inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-xs font-medium mx-0.5 transition-colors ${
              isOpen
                ? "bg-emerald-700 text-white"
                : "bg-zinc-700 text-emerald-400 hover:bg-zinc-600"
            }`}
          >
            {n}
            {isOpen ? (
              <ChevronDown className="w-2.5 h-2.5" />
            ) : (
              <ChevronRight className="w-2.5 h-2.5" />
            )}
          </button>
        );
      })}
    </p>
  );
}

function AgentRunBubble({ run, timestamp }: { run: AgentRun; timestamp: Date }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="flex justify-start">
      <div className="max-w-[80%] rounded-xl bg-zinc-900 border border-zinc-800 overflow-hidden">
        {/* Step list header */}
        <button
          onClick={() => setExpanded((v) => !v)}
          className="w-full flex items-center justify-between px-4 py-3 text-sm hover:bg-zinc-800/50 transition-colors"
        >
          <div className="flex items-center gap-2">
            {run.status === "running" ? (
              <Loader2 className="w-4 h-4 animate-spin text-amber-400" />
            ) : run.status === "error" ? (
              <AlertCircle className="w-4 h-4 text-red-400" />
            ) : (
              <Users className="w-4 h-4 text-emerald-400" />
            )}
            <span className="text-zinc-300">
              {run.status === "running"
                ? "Agent team running..."
                : run.status === "error"
                  ? "Agent team failed"
                  : "Agent team complete"}
            </span>
            <span className="text-xs text-zinc-600 font-mono">{run.taskId.slice(0, 8)}</span>
          </div>
          <div className="flex items-center gap-1 text-xs text-zinc-500">
            {run.events.length} steps
            {expanded ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
          </div>
        </button>

        {/* Steps */}
        {expanded && run.events.length > 0 && (
          <div className="border-t border-zinc-800 px-4 py-2 space-y-1 max-h-48 overflow-y-auto scrollbar-thin">
            {run.events.map((e, i) => (
              <div key={i} className="flex items-start gap-2 text-xs py-0.5">
                <span className="text-zinc-600 font-mono shrink-0">
                  {new Date(e.timestamp).toLocaleTimeString()}
                </span>
                <span
                  className={`shrink-0 ${
                    e.event_type === "error"
                      ? "text-red-400"
                      : e.event_type === "completed"
                        ? "text-emerald-400"
                        : "text-zinc-500"
                  }`}
                >
                  [{e.agent}]
                </span>
                <span className="text-zinc-400">{e.details}</span>
              </div>
            ))}
          </div>
        )}

        {/* Error */}
        {run.status === "error" && run.errorMessage && (
          <div className="border-t border-zinc-800 px-4 py-2">
            <p className="text-xs text-red-300">{run.errorMessage}</p>
          </div>
        )}

        {/* Final artifact */}
        {run.artifact && (
          <div className="border-t border-zinc-800 p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-zinc-500 capitalize">{run.artifact.artifact_type}</span>
              <span className="text-xs text-zinc-500">
                confidence: {(run.artifact.confidence * 100).toFixed(0)}%
              </span>
            </div>
            <pre className="text-sm text-zinc-300 whitespace-pre-wrap font-mono leading-relaxed bg-zinc-950 rounded-lg p-3 max-h-64 overflow-y-auto scrollbar-thin">
              {run.artifact.content}
            </pre>
          </div>
        )}

        <div className="px-4 pb-2">
          <span className="text-xs text-zinc-600">{timestamp.toLocaleTimeString()}</span>
        </div>
      </div>
    </div>
  );
}
