"""
Guardrail framework.

Enforces safety and quality standards on agent outputs BEFORE they
reach the write-back stage. Three main guardrails:

1. Citation Enforcement — every factual claim must be traceable to a source
2. Confidence Gating — outputs below a threshold trigger abstention
3. Hallucination Detection — cross-reference claims against the KG

These run independently of the Reviewer agent. The Reviewer does holistic
quality assessment. Guardrails do mechanical verification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from lattice.agents.state import Artifact

logger = structlog.get_logger()


class GuardrailVerdict(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    ABSTAIN = "abstain"


@dataclass
class GuardrailResult:
    """Result of a guardrail check on an artifact."""

    guardrail_name: str
    verdict: GuardrailVerdict
    score: float = 0.0  # 0-1, higher is better
    details: str = ""
    flagged_claims: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


@dataclass
class GuardrailReport:
    """Combined report from all guardrails on an artifact."""

    artifact_id: str
    results: list[GuardrailResult] = field(default_factory=list)

    @property
    def overall_verdict(self) -> GuardrailVerdict:
        if any(r.verdict == GuardrailVerdict.ABSTAIN for r in self.results):
            return GuardrailVerdict.ABSTAIN
        if any(r.verdict == GuardrailVerdict.FAIL for r in self.results):
            return GuardrailVerdict.FAIL
        if any(r.verdict == GuardrailVerdict.WARN for r in self.results):
            return GuardrailVerdict.WARN
        return GuardrailVerdict.PASS

    @property
    def passed(self) -> bool:
        return self.overall_verdict in (GuardrailVerdict.PASS, GuardrailVerdict.WARN)

    def summary(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "overall_verdict": self.overall_verdict.value,
            "passed": self.passed,
            "checks": [
                {
                    "name": r.guardrail_name,
                    "verdict": r.verdict.value,
                    "score": round(r.score, 3),
                    "details": r.details,
                }
                for r in self.results
            ],
        }


# --- Guardrail Implementations ---


class CitationEnforcementGuardrail:
    """Enforces that factual claims are backed by citations.

    Scans the artifact for factual statements (sentences with specific
    data, numbers, or assertions) and checks whether they reference a source.
    """

    def __init__(self, min_citation_ratio: float = 0.3) -> None:
        self._min_ratio = min_citation_ratio

    def check(self, artifact: Artifact) -> GuardrailResult:
        content = artifact.content
        sentences = self._split_sentences(content)
        if not sentences:
            return GuardrailResult(
                guardrail_name="citation_enforcement",
                verdict=GuardrailVerdict.PASS,
                score=1.0,
                details="No content to check",
            )

        factual_sentences = [s for s in sentences if self._is_factual_claim(s)]
        if not factual_sentences:
            return GuardrailResult(
                guardrail_name="citation_enforcement",
                verdict=GuardrailVerdict.PASS,
                score=1.0,
                details="No factual claims detected",
            )

        # Check citation coverage
        citation_pattern = re.compile(
            r'\[[\w_-]+\]|\[(?:source|ref|chunk|cite)\s*\w*\]|\[\d+\]',
            re.IGNORECASE,
        )
        cited = [s for s in factual_sentences if citation_pattern.search(s)]
        uncited = [s for s in factual_sentences if not citation_pattern.search(s)]

        ratio = len(cited) / len(factual_sentences)
        also_has_citations = len(artifact.citations) > 0

        # Score considers both explicit markers and artifact-level citations
        score = ratio if not also_has_citations else min(ratio + 0.2, 1.0)

        if score >= self._min_ratio:
            verdict = GuardrailVerdict.PASS
        elif score >= self._min_ratio * 0.5:
            verdict = GuardrailVerdict.WARN
        else:
            verdict = GuardrailVerdict.FAIL

        return GuardrailResult(
            guardrail_name="citation_enforcement",
            verdict=verdict,
            score=score,
            details=f"{len(cited)}/{len(factual_sentences)} factual claims are cited",
            flagged_claims=uncited[:5],
            suggestions=["Add source references for uncited claims"] if uncited else [],
        )

    def _split_sentences(self, text: str) -> list[str]:
        parts = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in parts if len(s.strip()) > 15]

    def _is_factual_claim(self, sentence: str) -> bool:
        """Heuristic: a sentence is factual if it contains numbers, specific terms, or assertions."""
        indicators = [
            r'\d+',  # Contains numbers
            r'(?:is|are|was|were|has|have)\s+(?:a|an|the)\s+\w+',  # Definitional
            r'(?:requires?|must|should|always|never)\s',  # Prescriptive
            r'(?:causes?|results?\s+in|leads?\s+to)',  # Causal
            r'(?:according to|based on|shows?\s+that)',  # Evidentiary
        ]
        return any(re.search(p, sentence, re.IGNORECASE) for p in indicators)


class ConfidenceGatingGuardrail:
    """Gates outputs based on confidence thresholds.

    If an artifact's confidence is below the threshold, the guardrail
    triggers abstention — the system should say "I don't know" rather
    than produce low-confidence output.
    """

    def __init__(self, abstention_threshold: float = 0.3, warn_threshold: float = 0.6) -> None:
        self._abstention_threshold = abstention_threshold
        self._warn_threshold = warn_threshold

    def check(self, artifact: Artifact) -> GuardrailResult:
        confidence = artifact.confidence

        if confidence >= self._warn_threshold:
            return GuardrailResult(
                guardrail_name="confidence_gating",
                verdict=GuardrailVerdict.PASS,
                score=confidence,
                details=f"Confidence {confidence:.2f} above threshold {self._warn_threshold:.2f}",
            )
        elif confidence >= self._abstention_threshold:
            return GuardrailResult(
                guardrail_name="confidence_gating",
                verdict=GuardrailVerdict.WARN,
                score=confidence,
                details=f"Low confidence {confidence:.2f} — output may be unreliable",
                suggestions=["Consider gathering more sources", "Flag uncertain sections"],
            )
        else:
            return GuardrailResult(
                guardrail_name="confidence_gating",
                verdict=GuardrailVerdict.ABSTAIN,
                score=confidence,
                details=f"Confidence {confidence:.2f} below abstention threshold {self._abstention_threshold:.2f}",
                suggestions=["System should abstain from producing this output"],
            )


class HallucinationDetectionGuardrail:
    """Detects potential hallucinations by cross-referencing claims against the KG.

    Strategy:
    1. Extract specific claims (entities, relations, numbers) from the artifact
    2. Check if those claims exist in the knowledge graph
    3. Flag claims that contradict KG or have no support

    This is a lightweight, deterministic check (no LLM call). For deeper
    hallucination detection, use the citation utility evaluator.
    """

    def __init__(self, kg_backend=None, max_ungrounded_ratio: float = 0.5) -> None:
        self._kg = kg_backend
        self._max_ungrounded = max_ungrounded_ratio

    def check(self, artifact: Artifact) -> GuardrailResult:
        if not self._kg:
            return GuardrailResult(
                guardrail_name="hallucination_detection",
                verdict=GuardrailVerdict.PASS,
                score=1.0,
                details="No KG available for cross-reference (skipped)",
            )

        claims = self._extract_claims(artifact.content)
        if not claims:
            return GuardrailResult(
                guardrail_name="hallucination_detection",
                verdict=GuardrailVerdict.PASS,
                score=1.0,
                details="No verifiable claims extracted",
            )

        grounded = []
        ungrounded = []

        all_entities = self._kg.get_all_entities()
        entity_names = {e.name.lower() for e in all_entities}
        entity_descriptions = " ".join(
            (e.description or "").lower() for e in all_entities
        )

        for claim in claims:
            claim_lower = claim.lower()
            # A claim is "grounded" if key terms appear in the KG
            terms = [w for w in claim_lower.split() if len(w) > 4]
            matches = sum(1 for t in terms if t in entity_names or t in entity_descriptions)

            if matches >= max(1, len(terms) // 3):
                grounded.append(claim)
            else:
                ungrounded.append(claim)

        total = len(claims)
        grounded_ratio = len(grounded) / total if total else 1.0

        if grounded_ratio >= (1 - self._max_ungrounded):
            verdict = GuardrailVerdict.PASS
        elif grounded_ratio >= 0.3:
            verdict = GuardrailVerdict.WARN
        else:
            verdict = GuardrailVerdict.FAIL

        return GuardrailResult(
            guardrail_name="hallucination_detection",
            verdict=verdict,
            score=grounded_ratio,
            details=f"{len(grounded)}/{total} claims grounded in KG",
            flagged_claims=ungrounded[:5],
            suggestions=["Verify ungrounded claims against authoritative sources"] if ungrounded else [],
        )

    def _extract_claims(self, text: str) -> list[str]:
        """Extract verifiable claims (sentences with specific assertions)."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        claims = []
        for s in sentences:
            s = s.strip()
            if len(s) < 20:
                continue
            # Look for factual patterns
            if re.search(r'\d+|(?:is|are|has|have|was|were)\s+\w+', s, re.IGNORECASE):
                claims.append(s)
        return claims


# --- Guardrail Runner ---


class GuardrailRunner:
    """Runs all configured guardrails on an artifact."""

    def __init__(
        self,
        citation_enforcement: CitationEnforcementGuardrail | None = None,
        confidence_gating: ConfidenceGatingGuardrail | None = None,
        hallucination_detection: HallucinationDetectionGuardrail | None = None,
    ) -> None:
        self._guardrails = []
        if citation_enforcement:
            self._guardrails.append(citation_enforcement)
        if confidence_gating:
            self._guardrails.append(confidence_gating)
        if hallucination_detection:
            self._guardrails.append(hallucination_detection)

    @classmethod
    def default(cls, kg_backend=None) -> "GuardrailRunner":
        """Create a runner with all guardrails at default thresholds."""
        return cls(
            citation_enforcement=CitationEnforcementGuardrail(),
            confidence_gating=ConfidenceGatingGuardrail(),
            hallucination_detection=HallucinationDetectionGuardrail(kg_backend=kg_backend),
        )

    def run(self, artifact: Artifact) -> GuardrailReport:
        """Run all guardrails and return a combined report."""
        report = GuardrailReport(artifact_id=artifact.id)

        for guardrail in self._guardrails:
            result = guardrail.check(artifact)
            report.results.append(result)
            logger.debug(
                "guardrail_checked",
                guardrail=result.guardrail_name,
                verdict=result.verdict.value,
                score=result.score,
            )

        logger.info(
            "guardrails_complete",
            artifact_id=artifact.id[:8],
            verdict=report.overall_verdict.value,
            passed=report.passed,
        )
        return report
