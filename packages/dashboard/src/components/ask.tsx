"use client";

import { useCallback, useEffect, useEffectEvent, useRef, useState, useSyncExternalStore } from "react";
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
  streamChat,
  submitTask,
  getTask,
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
  id?: string;
  streaming?: boolean;
  streamStatus?: string;
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
  artifacts?: Artifact[];
  progress?: string;
  resultPending?: boolean;
  errorMessage?: string;
}

interface Conversation {
  mode: AskMode;
  approvalMode: string;
  input: string;
  messages: ConversationMessage[];
}

const storageKey = "lattice.ask.v1";
const storageLimit = 1_000_000;
const messageLimit = 100;
const eventLimit = 200; // Retain the full server replay window for deduplication.
const emptyConversation: Conversation = {
  mode: "answer", approvalMode: "on_low_confidence", input: "", messages: [],
};

function record(x: unknown): x is Record<string, unknown> {
  return x !== null && typeof x === "object" && !Array.isArray(x);
}

function citation(x: unknown): x is ChatCitation {
  return record(x) && Number.isInteger(x.index) && (x.kind === "chunk" || x.kind === "entity")
    && typeof x.id === "string" && typeof x.source === "string" && typeof x.snippet === "string";
}

function artifact(x: unknown): x is Artifact {
  return record(x) && typeof x.content === "string" && typeof x.artifact_type === "string"
    && typeof x.id === "string" && typeof x.source_agent === "string"
    && typeof x.created_at === "string" && typeof x.confidence === "number"
    && Array.isArray(x.citations) && x.citations.every((id) => typeof id === "string");
}

function restoreMessage(x: unknown): ConversationMessage[] {
  if (!record(x) || (x.role !== "user" && x.role !== "assistant") || typeof x.content !== "string"
    || typeof x.timestamp !== "string" || !Number.isFinite(Date.parse(x.timestamp))) return [];
  const message: ConversationMessage = {
    role: x.role, content: x.content, timestamp: new Date(x.timestamp),
    citations: Array.isArray(x.citations) ? x.citations.filter(citation) : undefined,
    notice: x.streaming === true ? "Answer interrupted by page reload. You can ask again."
      : typeof x.notice === "string" ? x.notice : undefined,
    chatMode: typeof x.chatMode === "string" ? x.chatMode : undefined,
  };
  if (x.agentRun !== undefined) {
    const run = x.agentRun;
    if (x.role !== "assistant" || !record(run) || typeof run.taskId !== "string" || !/^[a-zA-Z0-9_-]{1,128}$/.test(run.taskId)
      || (run.status !== "running" && run.status !== "done" && run.status !== "error")) return [];
    message.agentRun = {
      taskId: run.taskId, status: run.status, resultPending: run.resultPending === true,
      events: Array.isArray(run.events) ? run.events.slice(-eventLimit).flatMap((raw) => {
        const event = normalizeAgentEvent(raw, run.taskId as string);
        return event && event.task_id === run.taskId ? [event] : [];
      }) : [],
      artifacts: Array.isArray(run.artifacts) ? run.artifacts.filter(artifact) : undefined,
      errorMessage: typeof run.errorMessage === "string" ? run.errorMessage : undefined,
    };
  }
  return [message];
}

function readConversation(): Conversation {
  try {
    const raw = localStorage.getItem(storageKey);
    if (!raw || raw.length > storageLimit) return emptyConversation;
    const data: unknown = JSON.parse(raw);
    if (!record(data) || !Array.isArray(data.messages)) return emptyConversation;
    const seen = new Set<string>();
    const messages = data.messages.slice(-messageLimit).flatMap(restoreMessage).filter((message) => {
      const id = message.agentRun?.taskId;
      if (!id) return true;
      if (seen.has(id)) return false;
      seen.add(id);
      return true;
    });
    return {
      mode: data.mode === "agent" ? "agent" : "answer",
      approvalMode: data.approvalMode === "always" || data.approvalMode === "never"
        ? data.approvalMode : "on_low_confidence",
      input: typeof data.input === "string" ? data.input : "",
      messages,
    };
  } catch {
    return emptyConversation;
  }
}

function saveConversation(state: Conversation) {
  try {
    const snapshot = { ...state, input: state.input.slice(0, 64_000), messages: [] as ConversationMessage[] };
    let remaining = storageLimit - JSON.stringify(snapshot).length;
    // Reserve room for recoverable task IDs before filling the remaining space
    // with recent messages. Restore the original conversation order afterward.
    const recent = state.messages.map((message, index) => ({ message, index })).reverse();
    const pending = (m: ConversationMessage) => m.agentRun?.status === "running" || m.agentRun?.resultPending;
    const active = recent.filter(({ message }) => pending(message)).slice(0, messageLimit);
    const ordered = [...active, ...recent.filter(({ message }) => !pending(message)).slice(0, messageLimit)];
    const activeBudget = Math.floor(remaining / Math.max(active.length, 1));
    const saved: typeof recent = [];
    for (const row of ordered) {
      if (saved.length >= messageLimit) break;
      let message = row.message;
      let size = JSON.stringify(message).length + 1;
      if (size > (pending(message) ? activeBudget : remaining)) {
        if (message.agentRun) {
          message = { ...message, agentRun: { ...message.agentRun, events: [], artifacts: undefined,
            resultPending: message.agentRun.status !== "running" } };
        } else {
          message = { ...message, content: message.content.slice(0, 100_000), citations: [],
            notice: "This message exceeded the local history limit." };
        }
        size = JSON.stringify(message).length + 1;
        if (size > remaining) continue;
      }
      saved.push({ index: row.index, message });
      remaining -= size;
    }
    snapshot.messages = saved.sort((a, b) => a.index - b.index).map((row) => row.message);
    localStorage.setItem(storageKey, JSON.stringify(snapshot));
  } catch { /* Storage may be disabled or full. Keep the live conversation. */ }
}

const subscribeToHydration = () => () => {};
export function AskView(props: { onNavigateToKB: () => void }) {
  const hydrated = useSyncExternalStore(subscribeToHydration, () => true, () => false);
  return hydrated ? <AskConversation {...props} /> : null;
}

function AskConversation({ onNavigateToKB }: { onNavigateToKB: () => void }) {
  const { kb, health, apiReachable } = useKB();
  const [initial] = useState(readConversation);
  const [mode, setMode] = useState<AskMode>(initial.mode);
  const [approvalMode, setApprovalMode] = useState(initial.approvalMode);
  const [input, setInput] = useState(initial.input);
  const [messages, setMessages] = useState(initial.messages);
  const pendingSave = useRef<Conversation | null>(null);
  const saveTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const flush = useCallback(() => {
    clearTimeout(saveTimer.current);
    saveTimer.current = undefined;
    if (pendingSave.current) saveConversation(pendingSave.current);
    pendingSave.current = null;
  }, []);
  useEffect(() => {
    pendingSave.current = { mode, approvalMode, input, messages };
    saveTimer.current ??= setTimeout(flush, 300);
  }, [mode, approvalMode, input, messages, flush]);
  useEffect(() => {
    window.addEventListener("pagehide", flush);
    return () => { window.removeEventListener("pagehide", flush); flush(); };
  }, [flush]);
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const answerController = useRef<AbortController | null>(null);
  const followAnswer = useRef(true);

  const submission = useRef<AbortController | null>(null);
  useEffect(() => () => { answerController.current?.abort(); submission.current?.abort(); }, []);

  const kbReady = kb?.ready ?? false;
  const modelReachable = apiReachable && !!health && !!health.provider?.reachable;

  useEffect(() => {
    if (followAnswer.current && scrollRef.current) {
      scrollRef.current.scrollTo({ top: scrollRef.current.scrollHeight,
        behavior: messages.some((m) => m.streaming) ? "auto" : "smooth" });
    }
  }, [messages]);

  // --- Answer mode ---

  const sendAnswer = useCallback(
    async (userMessage: string) => {
      const history = messages.filter((m) => !m.streaming && !m.agentRun)
        .map((m) => ({ role: m.role, content: m.content }));
      const id = crypto.randomUUID();
      const controller = new AbortController();
      answerController.current = controller;
      setMessages((prev) => [...prev, {
        id, role: "assistant", content: "", streaming: true,
        streamStatus: "Searching knowledge base…", timestamp: new Date(),
      }]);
      const update = (change: (message: ConversationMessage) => ConversationMessage) => {
        if (!controller.signal.aborted) {
          setMessages((prev) => prev.map((m) => m.id === id ? change(m) : m));
        }
      };
      try {
        await streamChat(userMessage, history, ({ event, data }) => {
          switch (event) {
            case "status":
              update((m) => ({ ...m, streamStatus: data.message }));
              break;
            case "sources":
              update((m) => ({ ...m, citations: data.citations }));
              break;
            case "delta":
              update((m) => ({ ...m, content: m.content + data.text,
                streamStatus: "Generating answer…" }));
              break;
            case "done":
              update((m) => ({ ...m, content: data.response, citations: data.citations,
                chatMode: data.mode, notice: data.notice, streaming: false,
                streamStatus: undefined }));
              break;
          }
        }, controller.signal);
      } catch (err) {
        update((m) => ({ ...m, streaming: false, streamStatus: undefined,
          notice: err instanceof Error ? err.message : "Could not get an answer." }));
      } finally {
        if (answerController.current === controller) answerController.current = null;
      }
    },
    [messages],
  );

  // --- Agent team mode ---

  const sendAgentTask = useCallback(async (userMessage: string) => {
    const controller = new AbortController();
    submission.current = controller;
    try {
      const task = await submitTask(userMessage, approvalMode, controller.signal);
      if (!controller.signal.aborted) setMessages((prev) => [...prev, {
        role: "assistant", content: "", timestamp: new Date(),
        agentRun: { taskId: task.task_id, events: [], status: "running" },
      }]);
    } catch (err) {
      if (!controller.signal.aborted) setMessages((prev) => [...prev, {
        role: "assistant", timestamp: new Date(),
        content: err instanceof Error ? err.message : "Failed to submit task.",
      }]);
    } finally {
      submission.current = null;
      setLoading(false);
    }
  }, [approvalMode]);

  const updateRun = useCallback((taskId: string, change: (run: AgentRun) => AgentRun) => {
    setMessages((prev) => {
      const index = prev.findIndex((m) => m.agentRun?.taskId === taskId);
      const run = prev[index]?.agentRun;
      if (!run) return prev;
      const next = change(run);
      return next === run ? prev : prev.with(index, { ...prev[index], agentRun: next });
    });
  }, []);

  const handleSend = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      const text = input.trim();
      if (!text || loading || answerController.current || submission.current || (mode === "agent" && !modelReachable)) return;

      setMessages((prev) => [...prev, { role: "user", content: text, timestamp: new Date() }]);
      setInput("");
      followAnswer.current = true;
      setLoading(true);

      if (mode === "answer") {
        await sendAnswer(text);
        setLoading(false);
      } else {
        await sendAgentTask(text);
        // loading is set false inside sendAgentTask after task is submitted
      }
    },
    [input, loading, mode, modelReachable, sendAnswer, sendAgentTask],
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
            {mode === "agent" ? "The agent team uses your current knowledge base for evidence."
              : "Query your knowledge base or launch the agent team."}
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
      <div ref={scrollRef} onScroll={(event) => {
        const node = event.currentTarget;
        followAnswer.current = node.scrollHeight - node.scrollTop - node.clientHeight < 80;
      }} className="flex-1 overflow-y-auto scrollbar-thin space-y-4 mb-4">
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
          <MessageBubble key={msg.agentRun?.taskId ?? i} msg={msg} updateRun={updateRun} />
        ))}


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
                  disabled={loading || !input.trim() || (mode === "agent" && !modelReachable)}
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

type UpdateRun = (id: string, change: (run: AgentRun) => AgentRun) => void;

function MessageBubble({ msg, updateRun }: { msg: ConversationMessage; updateRun: UpdateRun }) {
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
    return <AgentRunBubble updateRun={updateRun} run={msg.agentRun} timestamp={msg.timestamp} />;
  }

  // Assistant: answer
  const citations = msg.citations ?? [];
  const openCitation = citations.find((c) => c.index === expandedCitation);

  return (
    <div className="flex justify-start">
      <div aria-busy={msg.streaming ?? false}
        className="max-w-[80%] rounded-xl px-4 py-3 bg-zinc-900 border border-zinc-800 text-zinc-300">
        {msg.notice && (
          <div className="mb-2 px-2 py-1.5 bg-amber-950/30 border border-amber-800 rounded-lg text-xs text-amber-300 flex items-center gap-1.5">
            <AlertTriangle className="w-3 h-3 shrink-0" />
            {msg.notice}
          </div>
        )}

        {msg.streaming && (
          <div role="status" className="flex items-center gap-2 text-xs text-zinc-400 mb-2">
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
            {msg.streamStatus}
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

function terminalStatus(status?: string) {
  if (status === "completed") return "done";
  if (status === "failed" || status === "rejected") return "error";
}

function monitorTask(initial: AgentRun, update: UpdateRun, pause: (reason: string) => void) {
  const id = initial.taskId;
  const controller = new AbortController();
  let socket: WebSocket | undefined;
  let timer: ReturnType<typeof setTimeout>;
  let busy = false;
  let wake = false;
  let stopped = false;
  let failures = 0;
  let connections = 0;
  let terminal = initial.status === "running" ? undefined : initial.status;
  let terminalError = initial.errorMessage ?? "";

  function markTerminal() {
    if (!terminal) return;
    const status = terminal;
    update(id, (run) => ({ ...run, status, resultPending: true,
      errorMessage: status === "error" ? terminalError || "The agent task failed." : undefined }));
  }
  function disconnect() {
    if (!socket) return;
    socket.onmessage = socket.onclose = socket.onerror = null;
    socket.close();
    socket = undefined;
  }
  function stop() {
    stopped = true;
    clearTimeout(timer);
    controller.abort();
    disconnect();
  }
  function schedule(delay: number) {
    if (stopped) return;
    clearTimeout(timer);
    if (busy) wake = true;
    else timer = setTimeout(reconcile, delay);
  }
  function connect() {
    // The server replays its event log on connection. After three attempts,
    // continue with authoritative status polling alone.
    if (connections >= 3 || socket || terminal) return;
    connections++;
    try { socket = createWebSocket(id); } catch { return; }
    socket.onclose = () => { socket = undefined; schedule(0); };
    socket.onerror = () => socket?.close();
    socket.onmessage = ({ data }) => {
      let raw;
      try { raw = JSON.parse(data); } catch { return; }
      const event = normalizeAgentEvent(raw, id);
      if (!event || event.task_id !== id) return;
      update(id, (run) => {
        if (run.events.some((e) => e.timestamp === event.timestamp && e.agent === event.agent
          && e.event_type === event.event_type && e.details === event.details)) return run;
        return { ...run, events: [...run.events, event].slice(-eventLimit), progress: event.details };
      });
      if (raw.event === "task_complete" || raw.event === "error") {
        terminal = raw.event === "error" ? "error" : "done";
        if (terminal === "error") terminalError = event.details;
        markTerminal();
        disconnect();
        schedule(0);
      }
    };
  }

  async function reconcile() {
    busy = true;
    wake = false;
    // Bound each request even if the API accepts a connection and goes silent.
    const signal = () => AbortSignal.any([controller.signal, AbortSignal.timeout(10_000)]);
    try {
      const task = await getTask(id, signal()).catch((error) => {
        if (!terminal) throw error;
        return undefined;
      });
      if (stopped) return;
      terminal = terminalStatus(task?.status) ?? terminal;
      terminalError = task?.errors?.filter((s) => s.trim()).join("\n") || terminalError;
      if (terminal) {
        markTerminal();
        disconnect();
        const result = await getTaskResult(id, signal());
        if (stopped) return;
        const status = terminalStatus(result.status) ?? terminal;
        const errorMessage = result.errors?.filter((s) => s.trim()).join("\n") || terminalError
          || "The agent task failed without an error description.";
        update(id, (run) => ({ ...run, status, resultPending: false, artifacts: result.artifacts,
          errorMessage: status === "error" ? errorMessage : undefined }));
        stop();
      } else {
        if (task?.current_agent) {
          const progress = [task.current_agent, task.iteration_count != null && `iteration ${task.iteration_count}`]
            .filter(Boolean).join(" · ");
          update(id, (run) => run.progress === progress ? run : { ...run, progress });
        }
        connect();
      }
      failures = 0;
    } catch (error) {
      if (stopped) return;
      if (++failures >= 5) {
        stop();
        const detail = error instanceof Error ? error.message : "API unavailable";
        pause([terminalError, `Could not load task ${terminal ? "result" : "status"} (${detail}).`]
          .filter(Boolean).join(" "));
      }
    } finally {
      busy = false;
      schedule(wake ? 0 : Math.min(30_000, 3000 * 2 ** failures));
    }
  }

  timer = setTimeout(() => { connect(); void reconcile(); }, 0);
  return stop;
}

function AgentRunBubble({ run, timestamp, updateRun }: {
  run: AgentRun; timestamp: Date; updateRun: UpdateRun;
}) {
  const [expanded, setExpanded] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const [monitorError, setMonitorError] = useState("");
  const monitoring = run.status === "running" || run.resultPending === true;
  const startMonitor = useEffectEvent(() => monitorTask(run, updateRun, setMonitorError));
  useEffect(() => {
    if (monitoring) return startMonitor();
  }, [run.taskId, monitoring, attempt]);

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
                ? monitorError ? "Task monitoring paused" : "Agent team running..."
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

        {monitorError && (
          <div className="border-t border-zinc-800 px-4 py-2 text-xs text-amber-300">
            {monitorError} <button className="underline" onClick={() => {
              setMonitorError(""); setAttempt((n) => n + 1);
            }}>Retry monitoring</button>
          </div>
        )}
        {run.status === "running" && run.progress && (
          <p className="px-4 pb-2 text-xs text-zinc-400">{run.progress}</p>
        )}

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
        {run.artifacts?.map((artifact) => (
          <div key={artifact.id} className="border-t border-zinc-800 p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-zinc-500 capitalize">{artifact.artifact_type}</span>
              <span className="text-xs text-zinc-500">
                confidence: {(artifact.confidence * 100).toFixed(0)}%
              </span>
            </div>
            <pre className="text-sm text-zinc-300 whitespace-pre-wrap font-mono leading-relaxed bg-zinc-950 rounded-lg p-3 max-h-64 overflow-y-auto scrollbar-thin">
              {artifact.content}
            </pre>
          </div>
        ))}

        <div className="px-4 pb-2">
          <span className="text-xs text-zinc-600">{timestamp.toLocaleTimeString()}</span>
        </div>
      </div>
    </div>
  );
}
