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
  description?: string;
}

export interface KGEdge {
  source: string;
  target: string;
  relation_type: string;
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

export async function getKnowledgeGraph(): Promise<{ nodes: KGNode[]; edges: KGEdge[] }> {
  const res = await fetch(`${API_BASE}/kg/graph`);
  return res.json();
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
