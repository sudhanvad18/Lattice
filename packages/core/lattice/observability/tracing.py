"""
LangFuse integration for LLM observability.

Wraps agent LLM calls with LangFuse traces, providing:
- Full call chains (orchestrator → researcher → writer → reviewer)
- Token usage tracking per agent
- Latency measurements
- Input/output logging
- Cost estimation

Works in two modes:
- Live: sends traces to a LangFuse server (production)
- Local: records traces in-memory for testing/demo without a server
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

import structlog

logger = structlog.get_logger()


@dataclass
class Span:
    """A single span in a trace (one LLM call or agent execution)."""

    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    trace_id: str = ""
    parent_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    input_data: dict[str, Any] = field(default_factory=dict)
    output_data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    level: str = "DEFAULT"  # DEFAULT, DEBUG, WARNING, ERROR
    status: str = "ok"  # ok, error

    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0


@dataclass
class Generation(Span):
    """A span specifically for LLM generation calls."""

    model: str = ""
    provider: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class Trace:
    """A complete trace of a task execution."""

    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    task_id: str = ""
    spans: list[Span] = field(default_factory=list)
    generations: list[Generation] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(g.total_tokens for g in self.generations)

    @property
    def total_duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0

    @property
    def generation_count(self) -> int:
        return len(self.generations)

    def summary(self) -> dict[str, Any]:
        return {
            "trace_id": self.id,
            "name": self.name,
            "task_id": self.task_id,
            "duration_ms": round(self.total_duration_ms, 1),
            "total_tokens": self.total_tokens,
            "generations": self.generation_count,
            "spans": len(self.spans),
            "tags": self.tags,
        }


class TracingProvider:
    """Local tracing provider that records traces in-memory.

    For production, this would forward to LangFuse's API.
    For development/testing, provides the same interface without network calls.
    """

    def __init__(self) -> None:
        self._traces: dict[str, Trace] = {}
        self._active_trace: Optional[Trace] = None
        self._span_stack: list[str] = []

    @property
    def traces(self) -> list[Trace]:
        return list(self._traces.values())

    def get_trace(self, trace_id: str) -> Trace | None:
        return self._traces.get(trace_id)

    @contextmanager
    def trace(self, name: str, task_id: str = "", tags: list[str] | None = None):
        """Start a new trace context."""
        t = Trace(name=name, task_id=task_id, tags=tags or [])
        self._traces[t.id] = t
        prev_trace = self._active_trace
        self._active_trace = t
        logger.debug("trace_started", trace_id=t.id, name=name)
        try:
            yield t
        finally:
            t.end_time = time.time()
            self._active_trace = prev_trace
            logger.debug("trace_ended", trace_id=t.id, duration_ms=t.total_duration_ms)

    @contextmanager
    def span(self, name: str, input_data: dict[str, Any] | None = None, **metadata):
        """Record a span within the current trace."""
        if not self._active_trace:
            yield None
            return

        s = Span(
            name=name,
            trace_id=self._active_trace.id,
            parent_id=self._span_stack[-1] if self._span_stack else None,
            input_data=input_data or {},
            metadata=metadata,
        )
        self._active_trace.spans.append(s)
        self._span_stack.append(s.id)
        try:
            yield s
        except Exception as e:
            s.status = "error"
            s.metadata["error"] = str(e)
            raise
        finally:
            s.end_time = time.time()
            self._span_stack.pop()

    def record_generation(
        self,
        name: str,
        model: str,
        provider: str,
        input_messages: list[dict] | None = None,
        output: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        **metadata,
    ) -> Generation | None:
        """Record an LLM generation within the current trace."""
        if not self._active_trace:
            return None

        gen = Generation(
            name=name,
            trace_id=self._active_trace.id,
            parent_id=self._span_stack[-1] if self._span_stack else None,
            model=model,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            input_data={"messages": input_messages or []},
            output_data={"content": output[:500]},
            metadata=metadata,
        )
        gen.end_time = time.time()
        self._active_trace.generations.append(gen)
        return gen


class LangFuseProvider(TracingProvider):
    """Production LangFuse integration.

    Extends TracingProvider to also send traces to a LangFuse server.
    Falls back gracefully if the server is unreachable.
    """

    def __init__(
        self,
        public_key: str = "",
        secret_key: str = "",
        host: str = "https://cloud.langfuse.com",
    ) -> None:
        super().__init__()
        self._public_key = public_key
        self._secret_key = secret_key
        self._host = host
        self._client = None

        if public_key and secret_key:
            try:
                from langfuse import Langfuse
                self._client = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host,
                )
                logger.info("langfuse_connected", host=host)
            except Exception as e:
                logger.warning("langfuse_init_failed", error=str(e))

    def record_generation(self, name: str, model: str, provider: str, **kwargs) -> Generation | None:
        gen = super().record_generation(name=name, model=model, provider=provider, **kwargs)

        # Forward to LangFuse if connected
        if self._client and self._active_trace:
            try:
                self._client.generation(
                    trace_id=self._active_trace.id,
                    name=name,
                    model=model,
                    input=kwargs.get("input_messages"),
                    output=kwargs.get("output", "")[:500],
                    usage={
                        "prompt_tokens": kwargs.get("prompt_tokens", 0),
                        "completion_tokens": kwargs.get("completion_tokens", 0),
                    },
                )
            except Exception as e:
                logger.debug("langfuse_generation_failed", error=str(e))

        return gen
