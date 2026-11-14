const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface Task {
  id: string;
  goal: string;
  status: "pending" | "running" | "completed" | "failed";
  created_at: string;
  completed_at?: string;
  approval_mode: string;
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
}

export interface EvalSummary {
  total_tasks: number;
  completion_rate: number;
  pass_rate: number;
  avg_iterations: number;
  avg_latency_seconds: number;
  avg_keyword_recall: number;
  avg_review_confidence: number;
}

export interface EvalRun {
  run_at: string | null;
  provider: string | null;
  model: string | null;
  summary: EvalSummary | null;
  results: EvalResult[];
}

export class ApiError extends Error {
  constructor(public status: number, path: string) {
    super(`Request to ${path} failed (${status})`);
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store", ...options });
  if (!res.ok) throw new ApiError(res.status, path);
  return res.json();
}

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

export function submitTask(goal: string, approvalMode: string = "on_low_confidence"): Promise<Task> {
  return request("/tasks", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ goal, approval_mode: approvalMode }),
  });
}

export function getTasks(): Promise<Task[]> {
  return request("/tasks");
}

export function getTask(id: string): Promise<Task & { artifacts: Artifact[] }> {
  return request(`/tasks/${id}`);
}

export function getAuditTrail(taskId: string): Promise<AgentEvent[]> {
  return request(`/tasks/${taskId}/audit`);
}

export function getEvalResults(): Promise<EvalRun> {
  return request("/eval/results");
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

export function ingestText(text: string, source = "api"): Promise<{ chunks: number; stats: Omit<GraphStats, "backend"> }> {
  return graphRequest("ingest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, source }),
  });
}

export function chatQuery(message: string): Promise<{ response: string; citations: string[] }> {
  return request("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
}

export function createWebSocket(taskId?: string): WebSocket {
  const wsBase = API_BASE.replace("http", "ws");
  const path = taskId ? `/ws/tasks/${taskId}` : "/ws/activity";
  return new WebSocket(`${wsBase}${path}`);
}
