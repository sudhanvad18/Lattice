"""
Tests for OpenTelemetry instrumentation.
"""

import asyncio

import pytest
import structlog

from lattice.observability.otel import (
    _add_otel_context,
    configure_structured_logging,
    get_tracer,
    init_telemetry,
    instrument,
    otel_span,
)


@pytest.fixture(autouse=True)
def reset_structlog():
    """Reset structlog config after tests that modify it."""
    yield
    structlog.reset_defaults()


class TestOtelInit:
    def test_init_disabled(self):
        init_telemetry(enabled=False)
        # Should not crash, tracer may or may not be set

    def test_init_without_endpoint_is_noop(self):
        init_telemetry(service_name="test-lattice")
        # Should initialize without error even with no collector

    def test_get_tracer_after_init(self):
        init_telemetry(service_name="test-lattice")
        tracer = get_tracer()
        assert tracer is not None


class TestOtelSpan:
    def test_span_context_no_tracer(self):
        with otel_span("test-span") as span:
            # Should work even if tracer yields None
            pass

    def test_span_with_attributes(self):
        init_telemetry(service_name="test-lattice")
        with otel_span("attributed-span", attributes={"task_id": "t-123", "agent": "researcher"}) as span:
            if span:
                assert span is not None


class TestInstrumentDecorator:
    def test_sync_function(self):
        init_telemetry(service_name="test-lattice")

        @instrument("test.sync_add")
        def add(a, b):
            return a + b

        result = add(2, 3)
        assert result == 5

    @pytest.mark.asyncio
    async def test_async_function(self):
        init_telemetry(service_name="test-lattice")

        @instrument("test.async_fetch")
        async def fetch_data():
            return {"data": [1, 2, 3]}

        result = await fetch_data()
        assert result == {"data": [1, 2, 3]}

    def test_instrument_preserves_exceptions(self):
        init_telemetry(service_name="test-lattice")

        @instrument("test.failing")
        def failing():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            failing()


class TestStructuredLogging:
    def test_configure_console(self):
        configure_structured_logging(json_output=False, log_level="DEBUG")

    def test_configure_json(self):
        configure_structured_logging(json_output=True, log_level="INFO")

    def test_otel_context_processor_no_tracer(self):
        event_dict = {"event": "test"}
        result = _add_otel_context(None, None, event_dict)
        assert result == event_dict
