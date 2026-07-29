"""
Prometheus metrics for the Lattice platform.

Exposes:
- Task throughput (submitted, completed, failed)
- Agent utilization (calls per agent type, latency histograms)
- LLM usage (tokens per model, generation latency)
- Write-back stats (attempts, successes, rejections)
- Error rates (by type, by agent)
- Knowledge base stats (entities, chunks, queries)
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any

from prometheus_client import Counter, Gauge, Histogram, Info

# --- Task Metrics ---

TASKS_SUBMITTED = Counter(
    "lattice_tasks_submitted_total",
    "Total tasks submitted",
    ["approval_mode"],
)

TASKS_COMPLETED = Counter(
    "lattice_tasks_completed_total",
    "Total tasks completed",
    ["status"],  # completed, failed
)

TASK_DURATION = Histogram(
    "lattice_task_duration_seconds",
    "Task execution duration",
    ["status"],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600],
)

TASKS_IN_PROGRESS = Gauge(
    "lattice_tasks_in_progress",
    "Tasks currently being executed",
)

# --- Agent Metrics ---

AGENT_CALLS = Counter(
    "lattice_agent_calls_total",
    "Total agent invocations",
    ["agent", "outcome"],  # outcome: success, error
)

AGENT_LATENCY = Histogram(
    "lattice_agent_latency_seconds",
    "Agent execution latency",
    ["agent"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30],
)

AGENT_TOOL_CALLS = Counter(
    "lattice_agent_tool_calls_total",
    "Tool calls made by agents",
    ["agent", "tool", "success"],
)

# --- LLM Metrics ---

LLM_CALLS = Counter(
    "lattice_llm_calls_total",
    "Total LLM API calls",
    ["provider", "model"],
)

LLM_TOKENS = Counter(
    "lattice_llm_tokens_total",
    "Total tokens used",
    ["provider", "model", "type"],  # type: prompt, completion
)

LLM_LATENCY = Histogram(
    "lattice_llm_latency_seconds",
    "LLM generation latency",
    ["provider", "model"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60],
)

LLM_ERRORS = Counter(
    "lattice_llm_errors_total",
    "LLM API errors",
    ["provider", "model", "error_type"],
)

# --- Write-Back Metrics ---

WRITEBACK_ATTEMPTS = Counter(
    "lattice_writeback_attempts_total",
    "Write-back attempts",
    ["target", "outcome"],  # outcome: success, rejected, failed
)

# --- Review Metrics ---

REVIEWS_TOTAL = Counter(
    "lattice_reviews_total",
    "Total reviews issued",
    ["verdict"],  # approved, rejected
)

REVIEW_CONFIDENCE = Histogram(
    "lattice_review_confidence",
    "Review confidence scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# --- Knowledge Base Metrics ---

KB_ENTITIES = Gauge(
    "lattice_kb_entities_total",
    "Total entities in knowledge graph",
)

KB_CHUNKS = Gauge(
    "lattice_kb_chunks_total",
    "Total chunks in vector store",
)

KB_SEARCHES = Counter(
    "lattice_kb_searches_total",
    "Knowledge base search queries",
    ["source"],  # vector_store, knowledge_graph
)


# --- Helper Context Managers ---


@contextmanager
def track_agent_execution(agent_name: str):
    """Track agent execution time and outcome."""
    TASKS_IN_PROGRESS.inc()
    start = time.time()
    try:
        yield
        AGENT_CALLS.labels(agent=agent_name, outcome="success").inc()
    except Exception:
        AGENT_CALLS.labels(agent=agent_name, outcome="error").inc()
        raise
    finally:
        duration = time.time() - start
        AGENT_LATENCY.labels(agent=agent_name).observe(duration)
        TASKS_IN_PROGRESS.dec()


@contextmanager
def track_llm_call(provider: str, model: str):
    """Track LLM call latency and count."""
    LLM_CALLS.labels(provider=provider, model=model).inc()
    start = time.time()
    try:
        yield
    except Exception as e:
        LLM_ERRORS.labels(provider=provider, model=model, error_type=type(e).__name__).inc()
        raise
    finally:
        LLM_LATENCY.labels(provider=provider, model=model).observe(time.time() - start)


def record_llm_usage(provider: str, model: str, prompt_tokens: int, completion_tokens: int):
    """Record token usage."""
    LLM_TOKENS.labels(provider=provider, model=model, type="prompt").inc(prompt_tokens)
    LLM_TOKENS.labels(provider=provider, model=model, type="completion").inc(completion_tokens)


def record_review(approved: bool, confidence: float):
    """Record a review verdict."""
    REVIEWS_TOTAL.labels(verdict="approved" if approved else "rejected").inc()
    REVIEW_CONFIDENCE.observe(confidence)


def record_writeback(target: str, outcome: str):
    """Record a write-back attempt."""
    WRITEBACK_ATTEMPTS.labels(target=target, outcome=outcome).inc()
