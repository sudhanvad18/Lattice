"""
Tests for the citation utility evaluator.
"""

import pytest

from lattice.agents.state import AgentRole, Artifact
from lattice.evaluation.citation_utility import CitationUtilityEvaluator, CitationUtilityScore
from lattice.inference.mock import MockProvider


class TestCitationUtilityScore:
    """Test the score dataclass and composite calculation."""

    def test_composite_score_all_high(self):
        score = CitationUtilityScore(
            artifact_id="test",
            coverage_score=0.9,
            relevance_score=0.9,
            necessity_score=0.9,
            efficiency_score=0.9,
        )
        assert score.composite_score == pytest.approx(0.9, abs=0.01)

    def test_composite_score_mixed(self):
        score = CitationUtilityScore(
            artifact_id="test",
            coverage_score=1.0,
            relevance_score=0.5,
            necessity_score=0.3,
            efficiency_score=0.8,
        )
        # 0.30*1.0 + 0.30*0.5 + 0.25*0.3 + 0.15*0.8 = 0.30 + 0.15 + 0.075 + 0.12 = 0.645
        assert score.composite_score == pytest.approx(0.645, abs=0.01)

    def test_composite_score_all_zero(self):
        score = CitationUtilityScore(artifact_id="test")
        assert score.composite_score == 0.0

    def test_summary_format(self):
        score = CitationUtilityScore(
            artifact_id="abc123",
            total_citations=5,
            unique_citations=3,
            coverage_score=0.8,
            relevance_score=0.7,
            necessity_score=0.6,
            efficiency_score=0.9,
        )
        s = score.summary()
        assert s["artifact_id"] == "abc123"
        assert s["total_citations"] == 5
        assert s["unique_citations"] == 3
        assert "composite_score" in s


class TestCitationUtilityEvaluator:
    """Test the evaluator logic."""

    @pytest.fixture
    def evaluator(self):
        return CitationUtilityEvaluator(provider=None)

    @pytest.fixture
    def artifact_with_citations(self):
        return Artifact(
            artifact_type="document",
            content=(
                "The turbine engine requires maintenance every 500 flight hours. "
                "According to [chunk_1], blade erosion is the primary failure mode. "
                "The inspection protocol [chunk_2] mandates visual inspection of all fan blades. "
                "Thermal fatigue was identified in 30% of removed components."
            ),
            source_agent=AgentRole.WRITER,
            citations=["chunk_1", "chunk_2", "chunk_3"],
        )

    @pytest.fixture
    def source_contents(self):
        return {
            "chunk_1": "Blade erosion accounts for 45% of turbine failures. Primary cause is foreign object damage during takeoff and landing.",
            "chunk_2": "Inspection protocol requires visual check of all fan blades at 500-hour intervals. Use borescope for internal components.",
            "chunk_3": "Thermal fatigue data shows 30% of components removed during maintenance exhibit micro-cracking patterns.",
        }

    @pytest.mark.asyncio
    async def test_evaluate_no_citations(self, evaluator):
        artifact = Artifact(
            artifact_type="document",
            content="A document with no citations at all.",
            source_agent=AgentRole.WRITER,
            citations=[],
        )
        score = await evaluator.evaluate(artifact)
        assert score.total_citations == 0
        assert score.composite_score == 0.0

    @pytest.mark.asyncio
    async def test_evaluate_with_source_contents(self, evaluator, artifact_with_citations, source_contents):
        score = await evaluator.evaluate(artifact_with_citations, source_contents=source_contents)

        assert score.total_citations == 3
        assert score.unique_citations == 3
        # Should have non-zero scores since sources overlap with artifact
        assert score.coverage_score > 0
        assert score.relevance_score > 0
        assert score.necessity_score > 0
        assert score.efficiency_score > 0
        assert score.composite_score > 0

    @pytest.mark.asyncio
    async def test_evaluate_without_source_contents(self, evaluator, artifact_with_citations):
        """Without source text, evaluator uses heuristics."""
        score = await evaluator.evaluate(artifact_with_citations, source_contents=None)
        assert score.total_citations == 3
        # Should still produce scores via heuristics
        assert score.relevance_score > 0
        assert score.necessity_score > 0

    @pytest.mark.asyncio
    async def test_coverage_detects_citation_markers(self, evaluator):
        """Coverage should detect explicit citation markers in text."""
        artifact = Artifact(
            artifact_type="document",
            content=(
                "According to [chunk_1], the system has three components. "
                "The architecture [ref_2] shows a microservice pattern. "
                "Performance testing revealed no issues."
            ),
            source_agent=AgentRole.WRITER,
            citations=["chunk_1", "ref_2"],
        )
        score = await evaluator.evaluate(artifact)
        # 2 out of 3 sentences have citation markers
        assert score.coverage_score > 0.5

    @pytest.mark.asyncio
    async def test_relevance_with_matching_sources(self, evaluator):
        """High relevance when source terms appear in artifact."""
        artifact = Artifact(
            artifact_type="document",
            content="The compressor stage uses titanium alloy blades for maximum durability.",
            source_agent=AgentRole.WRITER,
            citations=["src_1"],
        )
        sources = {
            "src_1": "Compressor blades are manufactured from titanium alloy Ti-6Al-4V for superior strength and durability under high temperature.",
        }
        score = await evaluator.evaluate(artifact, source_contents=sources)
        # High overlap between source and artifact
        assert score.relevance_score > 0.3

    @pytest.mark.asyncio
    async def test_relevance_with_unrelated_sources(self, evaluator):
        """Low relevance when source content doesn't match artifact."""
        artifact = Artifact(
            artifact_type="document",
            content="The user authentication system uses JWT tokens with 24-hour expiry.",
            source_agent=AgentRole.WRITER,
            citations=["src_unrelated"],
        )
        sources = {
            "src_unrelated": "The cafeteria menu for Tuesday includes pasta, salad, and soup options with vegetarian alternatives.",
        }
        score = await evaluator.evaluate(artifact, source_contents=sources)
        # Very low overlap
        assert score.relevance_score < 0.5

    @pytest.mark.asyncio
    async def test_efficiency_penalizes_duplicate_citations(self, evaluator):
        """Efficiency should be lower when citations are duplicated."""
        artifact = Artifact(
            artifact_type="document",
            content="Some content here.",
            source_agent=AgentRole.WRITER,
            citations=["chunk_1", "chunk_1", "chunk_1", "chunk_2"],
        )
        score = await evaluator.evaluate(artifact)
        # 2 unique out of 4 total → dedup ratio 0.5
        assert score.efficiency_score <= 0.6

    @pytest.mark.asyncio
    async def test_necessity_with_llm(self):
        """When provider is available, necessity uses LLM judgment."""
        mock_response = '{"score": 0.85, "reasoning": "Artifact contains specific data from sources"}'
        provider = MockProvider(responses=[mock_response])
        evaluator = CitationUtilityEvaluator(provider=provider)

        artifact = Artifact(
            artifact_type="document",
            content="The failure rate is 2.3% based on 500 inspections over 3 years.",
            source_agent=AgentRole.WRITER,
            citations=["data_src"],
        )
        sources = {"data_src": "Failure rate analysis: 2.3% across 500 inspection events (2021-2024)."}

        score = await evaluator.evaluate(artifact, source_contents=sources)
        assert score.necessity_score == pytest.approx(0.85, abs=0.01)
        assert "specific data" in score.necessity_details

    @pytest.mark.asyncio
    async def test_high_utility_artifact(self, evaluator):
        """A well-cited artifact using source material should score high."""
        artifact = Artifact(
            artifact_type="document",
            content=(
                "According to the maintenance manual [manual_1], turbine inspection must occur every 500 hours. "
                "The inspection protocol [protocol_1] requires borescope examination of all stage-1 blades. "
                "Historical data [data_1] shows that 30% of blades exhibit micro-cracking by 1000 hours. "
                "Therefore, the recommended replacement interval is 800 hours with 95% confidence."
            ),
            source_agent=AgentRole.WRITER,
            citations=["manual_1", "protocol_1", "data_1"],
        )
        sources = {
            "manual_1": "Maintenance manual section 4.2: Turbine inspection interval is 500 flight hours for all engines.",
            "protocol_1": "Borescope protocol: examine all stage-1 fan blades for erosion, cracking, and FOD damage.",
            "data_1": "Analysis of 200 blade samples: 30% micro-cracking at 1000 hours, 5% at 500 hours. 95% confidence interval.",
        }

        score = await evaluator.evaluate(artifact, source_contents=sources)
        # This is a well-grounded artifact — should score high overall
        assert score.composite_score > 0.4
        assert score.coverage_score > 0.5
        # Necessity heuristic is conservative without LLM — checks exact phrase overlap
        assert score.necessity_score > 0.0
