"""
Tests for the tracing and metrics observability layer.
"""

import time

import pytest

from lattice.observability.tracing import (
    Generation,
    LangFuseProvider,
    Span,
    Trace,
    TracingProvider,
)


class TestTracingProvider:
    """Test the local tracing provider."""

    @pytest.fixture
    def provider(self):
        return TracingProvider()

    def test_create_trace(self, provider):
        with provider.trace("test-task", task_id="t-123", tags=["test"]) as t:
            assert t.id
            assert t.name == "test-task"
            assert t.task_id == "t-123"

        assert len(provider.traces) == 1
        assert provider.traces[0].end_time is not None

    def test_trace_duration(self, provider):
        with provider.trace("timed") as t:
            time.sleep(0.01)

        assert t.total_duration_ms >= 10

    def test_nested_spans(self, provider):
        with provider.trace("parent-trace") as t:
            with provider.span("orchestrator") as s1:
                with provider.span("researcher") as s2:
                    pass

        assert len(t.spans) == 2
        assert t.spans[0].name == "orchestrator"
        assert t.spans[1].name == "researcher"
        # Researcher's parent should be orchestrator
        assert t.spans[1].parent_id == t.spans[0].id

    def test_span_captures_input_data(self, provider):
        with provider.trace("test") as t:
            with provider.span("agent", input_data={"task": "Write docs"}) as s:
                pass

        assert t.spans[0].input_data == {"task": "Write docs"}

    def test_span_records_errors(self, provider):
        with provider.trace("test") as t:
            try:
                with provider.span("failing") as s:
                    raise ValueError("something broke")
            except ValueError:
                pass

        assert t.spans[0].status == "error"
        assert "something broke" in t.spans[0].metadata["error"]

    def test_record_generation(self, provider):
        with provider.trace("test") as t:
            gen = provider.record_generation(
                name="researcher_llm_call",
                model="llama3.2",
                provider="ollama",
                input_messages=[{"role": "user", "content": "Research turbines"}],
                output="Found 3 relevant documents.",
                prompt_tokens=50,
                completion_tokens=30,
            )

        assert len(t.generations) == 1
        assert t.generations[0].model == "llama3.2"
        assert t.generations[0].provider == "ollama"
        assert t.generations[0].total_tokens == 80
        assert t.total_tokens == 80

    def test_multiple_generations_tracked(self, provider):
        with provider.trace("multi") as t:
            provider.record_generation("call1", "model_a", "ollama", prompt_tokens=10, completion_tokens=20)
            provider.record_generation("call2", "model_b", "anthropic", prompt_tokens=100, completion_tokens=50)

        assert t.generation_count == 2
        assert t.total_tokens == 180

    def test_trace_summary(self, provider):
        with provider.trace("summary-test", task_id="task-abc", tags=["benchmark"]) as t:
            provider.record_generation("call", "llama3.2", "ollama", prompt_tokens=50, completion_tokens=25)

        s = t.summary()
        assert s["name"] == "summary-test"
        assert s["task_id"] == "task-abc"
        assert s["total_tokens"] == 75
        assert s["generations"] == 1
        assert "benchmark" in s["tags"]

    def test_get_trace_by_id(self, provider):
        with provider.trace("findme") as t:
            pass

        found = provider.get_trace(t.id)
        assert found is not None
        assert found.name == "findme"

    def test_no_trace_context_returns_none(self, provider):
        gen = provider.record_generation("orphan", "model", "prov")
        assert gen is None

    def test_span_outside_trace_yields_none(self, provider):
        with provider.span("orphan") as s:
            pass
        assert s is None


class TestLangFuseProvider:
    """Test LangFuse provider (without real server)."""

    def test_init_without_keys_works(self):
        provider = LangFuseProvider()
        assert provider._client is None

    def test_local_tracing_still_works(self):
        provider = LangFuseProvider()
        with provider.trace("test") as t:
            provider.record_generation("call", "model", "prov", prompt_tokens=10, completion_tokens=5)

        assert t.total_tokens == 15
