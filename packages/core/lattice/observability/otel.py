"""
OpenTelemetry instrumentation for distributed tracing.

Provides:
- Automatic span creation for API requests, agent executions, LLM calls
- Trace context propagation across service boundaries
- Export to OTLP-compatible backends (Jaeger, Grafana Tempo, etc.)
- Graceful degradation when no collector is configured

Usage:
    from lattice.observability.otel import init_telemetry, instrument

    init_telemetry(service_name="lattice-api")

    @instrument("process_task")
    async def handle_task(task_id: str):
        ...
"""

from __future__ import annotations

import functools
import os
from contextlib import contextmanager
from typing import Any, Callable

import structlog

logger = structlog.get_logger()

_TRACER = None
_INITIALIZED = False


def init_telemetry(
    service_name: str = "lattice",
    endpoint: str | None = None,
    enabled: bool = True,
) -> None:
    """Initialize OpenTelemetry with OTLP export.

    If no endpoint is provided, checks OTEL_EXPORTER_OTLP_ENDPOINT env var.
    If neither is set, telemetry is disabled (no-op mode).
    """
    global _TRACER, _INITIALIZED

    if not enabled:
        _INITIALIZED = True
        logger.info("otel_disabled")
        return

    endpoint = endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        if endpoint:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            exporter = OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")
            provider.add_span_processor(BatchSpanProcessor(exporter))
            logger.info("otel_otlp_configured", endpoint=endpoint)
        elif os.getenv("LATTICE_OTEL_CONSOLE", "").lower() == "true":
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
            logger.info("otel_console_configured")
        else:
            logger.info("otel_noop", reason="No endpoint configured")

        trace.set_tracer_provider(provider)
        _TRACER = trace.get_tracer(service_name)
        _INITIALIZED = True
        logger.info("otel_initialized", service=service_name)

    except ImportError as e:
        logger.warning("otel_import_failed", error=str(e))
        _INITIALIZED = True


def get_tracer():
    """Get the configured tracer (or None if not initialized)."""
    return _TRACER


@contextmanager
def otel_span(name: str, attributes: dict[str, Any] | None = None):
    """Create an OpenTelemetry span context manager.

    Falls back to no-op if OTEL is not configured.
    """
    if _TRACER:
        with _TRACER.start_as_current_span(name) as span:
            if attributes:
                for k, v in attributes.items():
                    span.set_attribute(k, str(v) if not isinstance(v, (bool, int, float)) else v)
            try:
                yield span
            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                raise
    else:
        yield None


def instrument(name: str = "", attributes: dict[str, Any] | None = None):
    """Decorator to instrument a function with OpenTelemetry spans.

    Works on both sync and async functions.
    """
    def decorator(func: Callable) -> Callable:
        span_name = name or f"{func.__module__}.{func.__qualname__}"

        if _is_coroutine_function(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with otel_span(span_name, attributes):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with otel_span(span_name, attributes):
                    return func(*args, **kwargs)
            return sync_wrapper

    return decorator


def _is_coroutine_function(func: Callable) -> bool:
    """Check if a function is async."""
    import asyncio
    return asyncio.iscoroutinefunction(func)


# --- Structured Logging Integration ---


def configure_structured_logging(json_output: bool = False, log_level: str = "INFO"):
    """Configure structlog with OpenTelemetry trace context injection.

    Adds trace_id and span_id to all log records when OTEL is active.
    """
    import logging

    import structlog

    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        _add_otel_context,
        structlog.processors.StackInfoRenderer(),
    ]

    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper(), logging.INFO),
    )


def _add_otel_context(logger, method, event_dict):
    """Structlog processor that injects OTEL trace context into log records."""
    if _TRACER:
        try:
            from opentelemetry import trace
            span = trace.get_current_span()
            ctx = span.get_span_context()
            if ctx.trace_id:
                event_dict["trace_id"] = format(ctx.trace_id, "032x")
                event_dict["span_id"] = format(ctx.span_id, "016x")
        except Exception:
            pass
    return event_dict
