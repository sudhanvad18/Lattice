const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// --- Errors ---

export class ApiError extends Error {
  code?: string;
  constructor(public status: number, path: string, message?: string, code?: string) {
    super(message ?? `Request to ${path} failed (${status})`);
    this.code = code;
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store", ...options });
  if (!res.ok) {
    let code: string | undefined;
    let message: string | undefined;
    try {
      const body = await res.json() as unknown;
      if (body && typeof body === "object" && "detail" in body) {
        const detail = (body as Record<string, unknown>).detail;
        if (detail && typeof detail === "object" && "code" in detail && "message" in detail) {
          code = String((detail as Record<string, unknown>).code);
          message = String((detail as Record<string, unknown>).message);
        }
      }
    } catch { /* ignore parse errors */ }
    throw new ApiError(res.status, path, message, code);
  }
  return res.json() as Promise<T>;
}

// --- Task types ---

export interface Task {
  task_id: string;
  original_task: string;
  status: string;
  created_at: string;
  current_agent?: string | null;
  iteration_count?: number;
  artifact_count?: number;
  review_count?: number;
  updated_at?: string | null;
  errors?: string[];
  approval_mode?: string;
}

export interface TaskResult {
  task_id: string;
  status: string;
  original_task: string;
  artifacts: Artifact[];
  reviews?: Record<string, unknown>[];
  iteration_count?: number;
}

export interface Artifact {
  id: string;
  artifact_type: string;
  content: string;
  source_agent: string;
  confidence: number;
  citations: string[];
  created_at: string;
}

export interface AgentEvent {
  timestamp: string;
  agent: string;
  event_type: "started" | "completed" | "error" | "tool_call" | "llm_call";
  details: string;
  task_id: string;
}

// --- Eval types ---

export interface EvalResult {
  task_id: string;
  task_name: string;
  status: string;
  completed: boolean;
  passed: boolean;
  quality_score: number;
  keyword_recall: number;
  keywords_missing: string[];
  latency_seconds: number;
  iterations: number;
  artifact_count: number;
  errors: string[];
  groundedness: number;
  category?: string;
}

export interface EvalSummary {
  total_tasks: number;
  completion_rate: number;
  pass_rate: number;
  avg_iterations: number;
  avg_latency_seconds: number;
  avg_keyword_recall: number;
  avg_review_confidence: number;
  avg_groundedness?: number;
}

export interface EvalRun {
  run_at: string | null;
  provider: string | null;
  model: string | null;
  summary: EvalSummary | null;
  results: EvalResult[];
  kb?: { chunks: number; entities: number; sources: number } | null;
}

export interface EvalStatus {
  running: boolean;
  run_id: string | null;
  started_at: string | null;
  completed: number;
  total: number;
  error: string | null;
}

// --- Health types ---

export interface HealthProvider {
  name: string;
  model: string;
  reachable: boolean;
  /** Actionable explanation when not reachable. */
  detail: string | null;
}

export interface HealthResponse {
  status: string;
  provider: HealthProvider | null;
}

// --- Knowledge base types ---

export interface KBSource {
  source: string;
  chunks: number;
}

export interface KBStatus {
  ready: boolean;
  chunks: number;
  entities: number;
  relations: number;
  communities: number;
  last_ingested_at: string | null;
  sources: KBSource[];
  ingesting: boolean;
  data_dir: string;
}

// --- Chat types ---

export interface ChatCitation {
  /** Matches the `[n]` markers in the response text. */
  index: number;
  kind: "chunk" | "entity";
  id: string;
  source: string;
  snippet: string;
}

export interface ChatResponse {
  response: string;
  citations: ChatCitation[];
  mode: "answer" | "retrieval_only";
  notice: string | null;
}

// --- Graph types ---

export interface KGNode {
  id: string;
  name: string;
  entity_type: string;
  description?: string | null;
  community?: number | null;
}

export interface KGEdge {
  id: string;
  confidence: number;
  source: string;
  target: string;
  relation_type: string;
  provenance?: string;
}

export interface KnowledgeGraph {
  nodes: KGNode[];
  edges: KGEdge[];
  truncated: boolean;
}

export interface GraphStats {
  backend: string;
  chunks_stored: number;
  kg_entities: number;
  kg_relations: number;
  kg_entity_types: Record<string, number>;
  kg_relation_types: Record<string, number>;
}

export interface EntityDetail {
  entity: KGNode;
  neighbors: KGNode[];
  relations: KGEdge[];
}

export interface Communities {
  communities: { id: number; size: number; members: { id: string; name: string }[] }[];
  god_nodes: { id: string; name: string }[];
  surprising_connections: { source: string; target: string; explanation: string }[];
}

// --- Task fetchers ---

export function submitTask(task: string, approvalMode: string = "on_low_confidence"): Promise<Task> {
  return request("/tasks", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ task, approval_mode: approvalMode }),
  });
}

export function getTasks(): Promise<Task[]> {
  return request("/tasks");
}

export function getTask(id: string): Promise<Task> {
  return request(`/tasks/${id}`);
}

export function getTaskResult(id: string): Promise<TaskResult> {
  return request(`/tasks/${id}/result`);
}

export function getAuditTrail(taskId: string): Promise<AgentEvent[]> {
  return request(`/tasks/${taskId}/audit`);
}

// --- Health / KB fetchers ---

export function getHealth(): Promise<HealthResponse> {
  return request("/health");
}

export function getKBStatus(): Promise<KBStatus> {
  return request("/kb/status");
}

export function ingestPath(path: string, reset: boolean): Promise<{ job_id: string; path: string }> {
  return request("/kb/ingest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, reset }),
  });
}

export function resetKB(): Promise<KBStatus> {
  return request("/kb/reset", { method: "POST" });
}

// --- Eval fetchers ---

export function getEvalResults(): Promise<EvalRun> {
  return request("/eval/results");
}

export function getEvalStatus(): Promise<EvalStatus> {
  return request("/eval/status");
}

export function runEval(limit?: number): Promise<{ run_id: string; tasks: number }> {
  return request("/eval/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(limit !== undefined ? { limit } : {}),
  });
}

// --- Chat fetcher ---

export function chatQuery(
  message: string,
  history: { role: string; content: string }[] = [],
): Promise<ChatResponse> {
  return request("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history }),
  });
}

// --- Graph fetchers ---

function graphRequest<T>(path: string, options?: RequestInit): Promise<T> {
  return request(`/kg/${path}`, options);
}

export function getKnowledgeGraph(limit = 500): Promise<KnowledgeGraph> {
  return graphRequest(`graph?limit=${limit}`);
}

export function getGraphStats(): Promise<GraphStats> {
  return graphRequest("stats");
}

export function getEntityDetail(id: string): Promise<EntityDetail> {
  return graphRequest(`entities/${encodeURIComponent(id)}`);
}

export function getCommunities(): Promise<Communities> {
  return graphRequest("communities");
}

export function ingestText(
  text: string,
  source = "api",
): Promise<{ chunks: number; stats: Omit<GraphStats, "backend"> }> {
  return graphRequest("ingest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, source }),
  });
}

// --- WebSocket helpers ---

export function createWebSocket(taskId?: string): WebSocket {
  const wsBase = API_BASE.startsWith("https://")
    ? `wss://${API_BASE.slice("https://".length)}`
    : API_BASE.startsWith("http://")
      ? `ws://${API_BASE.slice("http://".length)}`
      : API_BASE.replace(/^http/, "ws");
  const path = taskId ? `/ws/${encodeURIComponent(taskId)}` : "/ws/activity";
  return new WebSocket(`${wsBase}${path}`);
}

export function normalizeAgentEvent(raw: unknown, fallbackTaskId = ""): AgentEvent | null {
  if (!raw || typeof raw !== "object") return null;
  const msg = raw as Record<string, unknown>;
  if (msg.event === "ping") return null;

  if (typeof msg.event === "string") {
    const data =
      msg.data && typeof msg.data === "object" ? (msg.data as Record<string, unknown>) : {};
    const taskId =
      typeof msg.task_id === "string" && msg.task_id ? msg.task_id : fallbackTaskId;
    return {
      timestamp: typeof msg.timestamp === "string" ? msg.timestamp : new Date().toISOString(),
      agent: typeof data.agent === "string" && data.agent ? data.agent : "system",
      event_type: mapEventType(msg.event),
      details: describeEvent(msg.event, data),
      task_id: taskId,
    };
  }

  if (typeof msg.event_type === "string" && typeof msg.details === "string") {
    return {
      timestamp: typeof msg.timestamp === "string" ? msg.timestamp : new Date().toISOString(),
      agent: typeof msg.agent === "string" && msg.agent ? msg.agent : "system",
      event_type: msg.event_type as AgentEvent["event_type"],
      details: msg.details,
      task_id: typeof msg.task_id === "string" && msg.task_id ? msg.task_id : fallbackTaskId,
    };
  }
  return null;
}

function mapEventType(event: string): AgentEvent["event_type"] {
  if (
    event === "error" ||
    event.endsWith("_failed") ||
    event.endsWith("_rejected") ||
    event.endsWith("_error")
  )
    return "error";
  if (event === "llm_call") return "llm_call";
  if (event === "tool_call") return "tool_call";
  if (
    event === "task_complete" ||
    event === "task_completed" ||
    event === "agent_completed" ||
    event.endsWith("_complete") ||
    event.endsWith("_completed")
  )
    return "completed";
  return "started";
}

function describeEvent(event: string, data: Record<string, unknown>): string {
  if (typeof data.summary === "string" && data.summary) return data.summary;

  // KB ingest events
  if (event === "kb_ingest_started") return "Knowledge base ingestion started";
  if (event === "kb_ingest_complete") {
    const n = (k: string) => (typeof data[k] === "number" ? data[k] : 0);
    return `Ingested ${n("chunks")} chunks: ${n("entities")} entities, ${n("relations")} relations`;
  }
  if (event === "kb_ingest_failed")
    return typeof data.message === "string" && data.message
      ? `Ingestion failed: ${data.message}`
      : "Knowledge base ingestion failed";
  if (event === "kb_ingest_progress")
    return typeof data.message === "string" && data.message ? data.message : "Ingesting...";

  // Eval events
  if (event === "eval_started") return "Evaluation benchmark started";
  if (event === "eval_complete")
    return typeof data.pass_rate === "number"
      ? `Benchmark complete: ${Math.round(data.pass_rate * 100)}% passed`
      : "Evaluation benchmark complete";
  if (event === "eval_failed")
    return typeof data.message === "string" && data.message
      ? `Benchmark failed: ${data.message}`
      : "Evaluation benchmark failed";
  if (event === "eval_progress") {
    const completed =
      typeof data.completed === "number" ? data.completed : null;
    const total = typeof data.total === "number" ? data.total : null;
    if (completed !== null && total !== null) return `Benchmark: ${completed}/${total} tasks`;
    return typeof data.message === "string" && data.message ? data.message : "Running benchmark...";
  }

  if (typeof data.message === "string" && data.message) return data.message;
  if (typeof data.task === "string" && data.task) return data.task;
  if (event === "task_complete" || event === "task_completed") {
    const status = typeof data.status === "string" ? data.status : "finished";
    const artifacts = typeof data.artifacts === "number" ? data.artifacts : 0;
    return `Task ${status} (${artifacts} artifacts)`;
  }

  // Show error data.message for error events
  if (event === "error" && typeof data.message === "string" && data.message) return data.message;

  return event.replace(/_/g, " ");
}
