"""
Citation utility evaluation.

Measures how useful citations were in relation to the generated artifact.
This is DIFFERENT from:
- Citation accuracy (are the citations pointing to real sources?) → basic correctness
- Artifact quality (is the document well-written?) → reviewer's job

Citation utility asks: "Did the cited sources actually INFORM and IMPROVE the output?"

Metrics:
1. Citation Coverage — what % of artifact claims are backed by a citation?
2. Citation Relevance — how semantically similar are cited chunks to the artifact content?
3. Citation Necessity — if we removed the cited context, would the artifact be weaker?
4. Citation Efficiency — ratio of useful citations to total citations (no padding)
5. Citation Grounding Score — composite metric combining all above

This provides signal on whether the RAG pipeline is actually working or
if the LLM is just generating from parametric memory and ignoring sources.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import structlog

from lattice.agents.state import Artifact
from lattice.inference.provider import GenerationConfig, LLMProvider, Message

logger = structlog.get_logger()


@dataclass
class CitationUtilityScore:
    """Detailed citation utility metrics for a single artifact."""

    artifact_id: str
    total_citations: int = 0
    unique_citations: int = 0

    # Coverage: what fraction of the artifact's content is supported by citations
    coverage_score: float = 0.0  # 0-1
    coverage_details: str = ""

    # Relevance: how relevant are the cited sources to the artifact content
    relevance_score: float = 0.0  # 0-1
    relevance_details: str = ""

    # Necessity: would the artifact be weaker without the citations
    necessity_score: float = 0.0  # 0-1
    necessity_details: str = ""

    # Efficiency: are there wasted/padding citations
    efficiency_score: float = 0.0  # 0-1
    efficiency_details: str = ""

    @property
    def composite_score(self) -> float:
        """Weighted composite of all utility dimensions."""
        return (
            0.30 * self.coverage_score
            + 0.30 * self.relevance_score
            + 0.25 * self.necessity_score
            + 0.15 * self.efficiency_score
        )

    def summary(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "composite_score": round(self.composite_score, 3),
            "coverage": round(self.coverage_score, 3),
            "relevance": round(self.relevance_score, 3),
            "necessity": round(self.necessity_score, 3),
            "efficiency": round(self.efficiency_score, 3),
            "total_citations": self.total_citations,
            "unique_citations": self.unique_citations,
        }


class CitationUtilityEvaluator:
    """Evaluates how useful citations are to generated artifacts.

    Uses a combination of:
    - Heuristic analysis (citation distribution, density)
    - LLM-as-judge (for semantic relevance and necessity assessment)
    - Text overlap metrics (surface-level grounding)
    """

    def __init__(self, provider: LLMProvider | None = None) -> None:
        self._provider = provider

    async def evaluate(
        self,
        artifact: Artifact,
        source_contents: dict[str, str] | None = None,
    ) -> CitationUtilityScore:
        """Evaluate citation utility for an artifact.

        Args:
            artifact: The generated artifact to evaluate
            source_contents: Mapping of citation_id → source text content.
                If provided, enables deeper relevance/necessity analysis.
        """
        score = CitationUtilityScore(
            artifact_id=artifact.id,
            total_citations=len(artifact.citations),
            unique_citations=len(set(artifact.citations)),
        )

        if not artifact.citations:
            score.coverage_details = "No citations present"
            score.relevance_details = "No citations to evaluate"
            score.necessity_details = "No citations to evaluate"
            score.efficiency_details = "No citations present"
            return score

        # 1. Coverage: analyze how well citations cover the artifact
        score.coverage_score, score.coverage_details = self._evaluate_coverage(
            artifact, source_contents
        )

        # 2. Relevance: semantic overlap between sources and artifact
        score.relevance_score, score.relevance_details = self._evaluate_relevance(
            artifact, source_contents
        )

        # 3. Necessity: would the artifact be weaker without citations
        if self._provider and source_contents:
            nec_score, nec_details = await self._evaluate_necessity_llm(
                artifact, source_contents
            )
            score.necessity_score = nec_score
            score.necessity_details = nec_details
        else:
            score.necessity_score, score.necessity_details = self._evaluate_necessity_heuristic(
                artifact, source_contents
            )

        # 4. Efficiency: ratio of useful citations
        score.efficiency_score, score.efficiency_details = self._evaluate_efficiency(
            artifact, source_contents
        )

        logger.info(
            "citation_utility_evaluated",
            artifact_id=artifact.id[:8],
            composite=round(score.composite_score, 3),
        )
        return score

    def _evaluate_coverage(
        self,
        artifact: Artifact,
        source_contents: dict[str, str] | None,
    ) -> tuple[float, str]:
        """How much of the artifact content is backed by citations?

        Heuristic approach: look for citation markers in text, estimate
        density of cited vs uncited sections.
        """
        content = artifact.content
        sentences = self._split_sentences(content)
        if not sentences:
            return 0.0, "No content to evaluate"

        # Check for explicit citation references in text
        citation_pattern = re.compile(r'\[(?:source|ref|chunk|cite)[_\s]?\w*\]|\[\d+\]|\[[\w-]+\]', re.IGNORECASE)
        cited_sentences = sum(1 for s in sentences if citation_pattern.search(s))

        # Also consider: if sources are provided, check text overlap
        source_backed = 0
        if source_contents:
            source_text_lower = " ".join(source_contents.values()).lower()
            for sentence in sentences:
                # A sentence is "source-backed" if key phrases appear in source material
                words = set(sentence.lower().split())
                significant_words = {w for w in words if len(w) > 4}
                if significant_words:
                    overlap = sum(1 for w in significant_words if w in source_text_lower)
                    if overlap / len(significant_words) > 0.3:
                        source_backed += 1

        # Combine both signals
        explicit_coverage = cited_sentences / len(sentences) if sentences else 0
        implicit_coverage = source_backed / len(sentences) if sentences else 0
        coverage = max(explicit_coverage, implicit_coverage)

        details = f"{cited_sentences}/{len(sentences)} sentences have explicit citations; {source_backed}/{len(sentences)} are source-backed"
        return min(coverage, 1.0), details

    def _evaluate_relevance(
        self,
        artifact: Artifact,
        source_contents: dict[str, str] | None,
    ) -> tuple[float, str]:
        """How relevant are the cited sources to the artifact content?

        Uses term overlap as a proxy for semantic relevance.
        """
        if not source_contents:
            # Without source text, infer from citation count and artifact length
            density = len(artifact.citations) / max(len(artifact.content.split()), 1) * 100
            relevance = min(density / 2.0, 1.0)  # Normalize
            return relevance, f"Estimated from citation density: {density:.1f} citations per 100 words"

        artifact_terms = self._extract_key_terms(artifact.content)
        if not artifact_terms:
            return 0.0, "Could not extract key terms from artifact"

        # For each cited source, check how many artifact terms appear in it
        source_relevance_scores = []
        for cid in set(artifact.citations):
            source_text = source_contents.get(cid, "")
            if not source_text:
                source_relevance_scores.append(0.0)
                continue

            source_terms = set(source_text.lower().split())
            overlap = sum(1 for t in artifact_terms if t in source_terms)
            relevance = overlap / len(artifact_terms) if artifact_terms else 0
            source_relevance_scores.append(relevance)

        avg_relevance = sum(source_relevance_scores) / len(source_relevance_scores) if source_relevance_scores else 0
        details = f"Average term overlap: {avg_relevance:.2f} across {len(source_relevance_scores)} sources"
        return min(avg_relevance * 2, 1.0), details  # Scale up since term overlap is usually low

    async def _evaluate_necessity_llm(
        self,
        artifact: Artifact,
        source_contents: dict[str, str],
    ) -> tuple[float, str]:
        """Use LLM to judge whether citations were necessary for the artifact."""
        sources_summary = "\n".join(
            f"[{cid}]: {text[:300]}" for cid, text in list(source_contents.items())[:5]
        )

        prompt = f"""Evaluate how NECESSARY the cited sources were for generating this artifact.

ARTIFACT:
{artifact.content[:1500]}

CITED SOURCES:
{sources_summary}

Rate the necessity on a scale of 0.0 to 1.0:
- 1.0 = The artifact could NOT have been produced without these sources (contains specific facts/data from them)
- 0.7 = Sources significantly improved the artifact (provided useful context/details)
- 0.4 = Sources were somewhat helpful but the artifact is mostly general knowledge
- 0.1 = Sources were barely used; artifact appears to be generated from parametric memory

Respond with ONLY a JSON object: {{"score": 0.X, "reasoning": "brief explanation"}}"""

        messages = [Message(role="user", content=prompt)]
        config = GenerationConfig(temperature=0.1, max_tokens=200)
        result = await self._provider.generate(messages, config)

        try:
            import json
            start = result.content.find("{")
            end = result.content.rfind("}") + 1
            data = json.loads(result.content[start:end])
            return float(data.get("score", 0.5)), data.get("reasoning", "")
        except (json.JSONDecodeError, ValueError):
            return 0.5, "Could not parse LLM necessity judgment"

    def _evaluate_necessity_heuristic(
        self,
        artifact: Artifact,
        source_contents: dict[str, str] | None,
    ) -> tuple[float, str]:
        """Heuristic necessity evaluation when no LLM is available."""
        if not source_contents:
            if len(artifact.citations) >= 5:
                return 0.7, "High citation count suggests source reliance"
            elif len(artifact.citations) >= 2:
                return 0.5, "Moderate citation count"
            else:
                return 0.3, "Few citations suggest low source reliance"

        # Check for specific data that likely came from sources:
        # 1. Numbers/quantities shared between source and artifact
        # 2. Multi-word phrases shared
        # 3. Rare key terms shared
        artifact_lower = artifact.content.lower()
        specifics = 0

        for source_text in source_contents.values():
            # Check shared numbers (strong signal of data grounding)
            source_numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', source_text))
            artifact_numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', artifact_lower))
            shared_numbers = source_numbers & artifact_numbers
            specifics += len(shared_numbers)

            # Check 2-word phrases
            source_phrases = self._extract_phrases(source_text, min_words=2)
            for phrase in source_phrases[:30]:
                if phrase in artifact_lower:
                    specifics += 1

        if specifics >= 5:
            return 0.9, f"Found {specifics} specific data points/phrases from sources in artifact"
        elif specifics >= 3:
            return 0.7, f"Found {specifics} specific data points/phrases from sources"
        elif specifics >= 1:
            return 0.5, f"Found {specifics} specific data points/phrases from sources"
        else:
            return 0.3, f"Found {specifics} specific phrases — artifact may be largely generated"

    def _evaluate_efficiency(
        self,
        artifact: Artifact,
        source_contents: dict[str, str] | None,
    ) -> tuple[float, str]:
        """Are there wasted/padding citations that don't contribute?"""
        unique = len(set(artifact.citations))
        total = len(artifact.citations)

        if total == 0:
            return 0.0, "No citations"

        # Deduplication ratio
        dedup_ratio = unique / total

        # If we have source contents, check which citations actually overlap with artifact
        if source_contents:
            useful_count = 0
            artifact_lower = artifact.content.lower()
            for cid in set(artifact.citations):
                source = source_contents.get(cid, "")
                if source:
                    # Check if any significant content from this source appears in artifact
                    source_terms = self._extract_key_terms(source)
                    artifact_terms = set(artifact_lower.split())
                    overlap = sum(1 for t in source_terms if t in artifact_terms)
                    if overlap >= 2:
                        useful_count += 1

            efficiency = useful_count / unique if unique > 0 else 0
            details = f"{useful_count}/{unique} citations contributed content; dedup ratio: {dedup_ratio:.2f}"
        else:
            efficiency = dedup_ratio
            details = f"Deduplication ratio: {dedup_ratio:.2f} ({unique} unique / {total} total)"

        return min(efficiency, 1.0), details

    # --- Helpers ---

    def _split_sentences(self, text: str) -> list[str]:
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _extract_key_terms(self, text: str, min_length: int = 5) -> set[str]:
        words = re.findall(r'\b[a-z]+\b', text.lower())
        return {w for w in words if len(w) >= min_length}

    def _extract_phrases(self, text: str, min_words: int = 3) -> list[str]:
        words = text.lower().split()
        phrases = []
        for i in range(len(words) - min_words + 1):
            phrase = " ".join(words[i:i + min_words])
            if len(phrase) > 15:  # Skip very short phrases
                phrases.append(phrase)
        return phrases
