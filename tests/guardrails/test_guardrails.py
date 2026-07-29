"""
Tests for the guardrail framework.
"""

import pytest

from lattice.agents.state import AgentRole, Artifact
from lattice.graph.engine import NetworkXBackend
from lattice.graph.models import Entity
from lattice.guardrails import (
    CitationEnforcementGuardrail,
    ConfidenceGatingGuardrail,
    GuardrailReport,
    GuardrailRunner,
    GuardrailVerdict,
    HallucinationDetectionGuardrail,
)


# --- Citation Enforcement ---


class TestCitationEnforcement:
    @pytest.fixture
    def guardrail(self):
        return CitationEnforcementGuardrail(min_citation_ratio=0.3)

    def test_well_cited_artifact_passes(self, guardrail):
        artifact = Artifact(
            artifact_type="document",
            content=(
                "The turbine requires inspection every 500 hours [ref_1]. "
                "Blade erosion is the primary failure mode [ref_2]. "
                "Maintenance costs have decreased 15% since 2020 [ref_3]. "
                "The team recommends quarterly reviews."
            ),
            source_agent=AgentRole.WRITER,
            citations=["ref_1", "ref_2", "ref_3"],
        )
        result = guardrail.check(artifact)
        assert result.verdict == GuardrailVerdict.PASS
        assert result.score > 0.5

    def test_uncited_factual_claims_flagged(self, guardrail):
        artifact = Artifact(
            artifact_type="document",
            content=(
                "The system processes 10,000 requests per second. "
                "It requires 16GB of RAM minimum. "
                "The database is PostgreSQL version 14. "
                "Performance has improved 300% since the refactor."
            ),
            source_agent=AgentRole.WRITER,
            citations=[],
        )
        result = guardrail.check(artifact)
        # Lots of factual claims with zero citations
        assert result.verdict in (GuardrailVerdict.WARN, GuardrailVerdict.FAIL)
        assert len(result.flagged_claims) > 0

    def test_non_factual_content_passes(self, guardrail):
        artifact = Artifact(
            artifact_type="document",
            content=(
                "This document provides an overview of the system. "
                "We will discuss the architecture in detail below. "
                "The following sections cover each component."
            ),
            source_agent=AgentRole.WRITER,
            citations=[],
        )
        result = guardrail.check(artifact)
        # No factual claims → passes
        assert result.verdict == GuardrailVerdict.PASS

    def test_empty_content_passes(self, guardrail):
        artifact = Artifact(
            artifact_type="document",
            content="",
            source_agent=AgentRole.WRITER,
            citations=[],
        )
        result = guardrail.check(artifact)
        assert result.verdict == GuardrailVerdict.PASS


# --- Confidence Gating ---


class TestConfidenceGating:
    @pytest.fixture
    def guardrail(self):
        return ConfidenceGatingGuardrail(abstention_threshold=0.3, warn_threshold=0.6)

    def test_high_confidence_passes(self, guardrail):
        artifact = Artifact(
            artifact_type="document",
            content="High quality content.",
            source_agent=AgentRole.WRITER,
            confidence=0.9,
            citations=[],
        )
        result = guardrail.check(artifact)
        assert result.verdict == GuardrailVerdict.PASS

    def test_medium_confidence_warns(self, guardrail):
        artifact = Artifact(
            artifact_type="document",
            content="Uncertain content.",
            source_agent=AgentRole.WRITER,
            confidence=0.45,
            citations=[],
        )
        result = guardrail.check(artifact)
        assert result.verdict == GuardrailVerdict.WARN

    def test_very_low_confidence_triggers_abstention(self, guardrail):
        artifact = Artifact(
            artifact_type="document",
            content="Very uncertain content.",
            source_agent=AgentRole.WRITER,
            confidence=0.15,
            citations=[],
        )
        result = guardrail.check(artifact)
        assert result.verdict == GuardrailVerdict.ABSTAIN

    def test_boundary_confidence(self, guardrail):
        artifact = Artifact(
            artifact_type="document",
            content="Boundary.",
            source_agent=AgentRole.WRITER,
            confidence=0.6,
            citations=[],
        )
        result = guardrail.check(artifact)
        assert result.verdict == GuardrailVerdict.PASS


# --- Hallucination Detection ---


class TestHallucinationDetection:
    @pytest.fixture
    def kg(self):
        kg = NetworkXBackend()
        kg.add_entity(Entity(name="TurboFan", entity_type="component", description="A high-bypass turbofan engine"))
        kg.add_entity(Entity(name="Compressor", entity_type="component", description="Air compression stage"))
        kg.add_entity(Entity(name="Blade Erosion", entity_type="failure_mode", description="Wear of fan blade surfaces"))
        return kg

    @pytest.fixture
    def guardrail(self, kg):
        return HallucinationDetectionGuardrail(kg_backend=kg)

    def test_grounded_claims_pass(self, guardrail):
        artifact = Artifact(
            artifact_type="document",
            content=(
                "The TurboFan engine has a compressor stage. "
                "Blade erosion is a known failure mode affecting the TurboFan."
            ),
            source_agent=AgentRole.WRITER,
            citations=[],
        )
        result = guardrail.check(artifact)
        assert result.verdict == GuardrailVerdict.PASS
        assert result.score >= 0.5

    def test_ungrounded_claims_flagged(self, guardrail):
        artifact = Artifact(
            artifact_type="document",
            content=(
                "The quantum flux capacitor operates at 1.21 gigawatts. "
                "The hyperspace motivator requires dilithium crystals. "
                "These components have a 99.9% reliability rating."
            ),
            source_agent=AgentRole.WRITER,
            citations=[],
        )
        result = guardrail.check(artifact)
        # These claims don't match anything in KG
        assert result.verdict in (GuardrailVerdict.WARN, GuardrailVerdict.FAIL)
        assert len(result.flagged_claims) > 0

    def test_no_kg_skips_check(self):
        guardrail = HallucinationDetectionGuardrail(kg_backend=None)
        artifact = Artifact(
            artifact_type="document",
            content="Any content here.",
            source_agent=AgentRole.WRITER,
            citations=[],
        )
        result = guardrail.check(artifact)
        assert result.verdict == GuardrailVerdict.PASS
        assert "skipped" in result.details.lower()


# --- Guardrail Runner ---


class TestGuardrailRunner:
    @pytest.fixture
    def runner(self):
        return GuardrailRunner.default()

    def test_all_pass(self, runner):
        artifact = Artifact(
            artifact_type="document",
            content="This is a general overview of the system architecture.",
            source_agent=AgentRole.WRITER,
            confidence=0.9,
            citations=["src_1"],
        )
        report = runner.run(artifact)
        assert report.passed
        assert report.overall_verdict in (GuardrailVerdict.PASS, GuardrailVerdict.WARN)

    def test_low_confidence_causes_abstention(self, runner):
        artifact = Artifact(
            artifact_type="document",
            content="Very uncertain claim about quantum physics results in 42% efficiency.",
            source_agent=AgentRole.WRITER,
            confidence=0.1,
            citations=[],
        )
        report = runner.run(artifact)
        assert report.overall_verdict == GuardrailVerdict.ABSTAIN
        assert not report.passed

    def test_report_summary(self, runner):
        artifact = Artifact(
            artifact_type="document",
            content="Some content.",
            source_agent=AgentRole.WRITER,
            confidence=0.8,
            citations=[],
        )
        report = runner.run(artifact)
        summary = report.summary()
        assert "artifact_id" in summary
        assert "overall_verdict" in summary
        assert "checks" in summary
        assert len(summary["checks"]) >= 2  # At least citation + confidence

    def test_runner_with_kg(self):
        kg = NetworkXBackend()
        kg.add_entity(Entity(name="AuthService", entity_type="component", description="Authentication service"))

        runner = GuardrailRunner.default(kg_backend=kg)
        artifact = Artifact(
            artifact_type="document",
            content="The AuthService handles all authentication with 99% uptime.",
            source_agent=AgentRole.WRITER,
            confidence=0.85,
            citations=["doc_1"],
        )
        report = runner.run(artifact)
        # Should pass — grounded in KG, has citation, high confidence
        assert report.passed
