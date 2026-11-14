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
  task_name: string;
  completed: boolean;
  quality_score: number;
  citation_accuracy: number;
  latency_seconds: number;
  iterations: number;
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

export async function submitTask(goal: string, approvalMode: string = "on_low_confidence"): Promise<Task> {
  const res = await fetch(`${API_BASE}/tasks`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ goal, approval_mode: approvalMode }),
  });
  return res.json();
}

export async function getTasks(): Promise<Task[]> {
  const res = await fetch(`${API_BASE}/tasks`);
  return res.json();
}

export async function getTask(id: string): Promise<Task & { artifacts: Artifact[] }> {
  const res = await fetch(`${API_BASE}/tasks/${id}`);
  return res.json();
}

export async function getAuditTrail(taskId: string): Promise<AgentEvent[]> {
  const res = await fetch(`${API_BASE}/tasks/${taskId}/audit`);
  return res.json();
}

export async function getEvalResults(): Promise<EvalResult[]> {
  const res = await fetch(`${API_BASE}/eval/results`);
  return res.json();
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

async function graphRequest<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}/kg/${path}`, options);
  if (!res.ok) throw new Error(`Knowledge graph request failed (${res.status})`);
  return res.json();
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

export async function chatQuery(message: string): Promise<{ response: string; citations: string[] }> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  return res.json();
}

export function createWebSocket(taskId?: string): WebSocket {
  const wsBase = API_BASE.replace("http", "ws");
  const path = taskId ? `/ws/tasks/${taskId}` : "/ws/activity";
  return new WebSocket(`${wsBase}${path}`);
}
