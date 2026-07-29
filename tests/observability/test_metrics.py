"""
Tests for Prometheus metrics.
"""

import pytest

from lattice.observability.metrics import (
    AGENT_CALLS,
    AGENT_LATENCY,
    KB_SEARCHES,
    LLM_CALLS,
    LLM_TOKENS,
    REVIEWS_TOTAL,
    TASKS_COMPLETED,
    TASKS_IN_PROGRESS,
    TASKS_SUBMITTED,
    WRITEBACK_ATTEMPTS,
    record_llm_usage,
    record_review,
    record_writeback,
    track_agent_execution,
    track_llm_call,
)


class TestMetricsContextManagers:
    def test_track_agent_execution_success(self):
        before = AGENT_CALLS.labels(agent="researcher", outcome="success")._value.get()
        with track_agent_execution("researcher"):
            pass
        after = AGENT_CALLS.labels(agent="researcher", outcome="success")._value.get()
        assert after == before + 1

    def test_track_agent_execution_error(self):
        before = AGENT_CALLS.labels(agent="writer", outcome="error")._value.get()
        with pytest.raises(ValueError):
            with track_agent_execution("writer"):
                raise ValueError("test")
        after = AGENT_CALLS.labels(agent="writer", outcome="error")._value.get()
        assert after == before + 1

    def test_track_llm_call(self):
        before = LLM_CALLS.labels(provider="ollama", model="llama3.2")._value.get()
        with track_llm_call("ollama", "llama3.2"):
            pass
        after = LLM_CALLS.labels(provider="ollama", model="llama3.2")._value.get()
        assert after == before + 1


class TestMetricsRecorders:
    def test_record_llm_usage(self):
        before_prompt = LLM_TOKENS.labels(provider="anthropic", model="claude-3", type="prompt")._value.get()
        before_comp = LLM_TOKENS.labels(provider="anthropic", model="claude-3", type="completion")._value.get()
        record_llm_usage("anthropic", "claude-3", prompt_tokens=100, completion_tokens=50)
        after_prompt = LLM_TOKENS.labels(provider="anthropic", model="claude-3", type="prompt")._value.get()
        after_comp = LLM_TOKENS.labels(provider="anthropic", model="claude-3", type="completion")._value.get()
        assert after_prompt == before_prompt + 100
        assert after_comp == before_comp + 50

    def test_record_review_approved(self):
        before = REVIEWS_TOTAL.labels(verdict="approved")._value.get()
        record_review(approved=True, confidence=0.9)
        after = REVIEWS_TOTAL.labels(verdict="approved")._value.get()
        assert after == before + 1

    def test_record_review_rejected(self):
        before = REVIEWS_TOTAL.labels(verdict="rejected")._value.get()
        record_review(approved=False, confidence=0.3)
        after = REVIEWS_TOTAL.labels(verdict="rejected")._value.get()
        assert after == before + 1

    def test_record_writeback(self):
        before = WRITEBACK_ATTEMPTS.labels(target="github", outcome="success")._value.get()
        record_writeback("github", "success")
        after = WRITEBACK_ATTEMPTS.labels(target="github", outcome="success")._value.get()
        assert after == before + 1


class TestMetricsExistence:
    """Verify all expected metrics exist and are queryable."""

    def test_task_metrics_exist(self):
        assert TASKS_SUBMITTED is not None
        assert TASKS_COMPLETED is not None
        assert TASKS_IN_PROGRESS is not None

    def test_agent_metrics_exist(self):
        assert AGENT_CALLS is not None
        assert AGENT_LATENCY is not None

    def test_kb_metrics_exist(self):
        assert KB_SEARCHES is not None
