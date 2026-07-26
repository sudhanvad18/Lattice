"""End-to-end tests for the KnowledgeBase.

Proves the full pipeline works:
  Document → Parse → Chunk → Embed → Store → Search (vector + graph)
"""

from pathlib import Path

import pytest

from lattice.knowledge_base import KnowledgeBase

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def kb(tmp_path) -> KnowledgeBase:
    """Create a KnowledgeBase with temp storage."""
    return KnowledgeBase(persist_dir=tmp_path / "data")


class TestKnowledgeBaseE2E:
    def test_ingest_and_vector_search(self, kb: KnowledgeBase):
        chunks_added = kb.ingest_file(FIXTURES / "sample_maintenance.md")
        assert chunks_added > 0

        results = kb.vector_search("What are the erosion limits for fan blades?")
        assert len(results) > 0
        assert any("erosion" in r.content.lower() for r in results)

    def test_ingest_and_graph_search(self, kb: KnowledgeBase):
        kb.ingest_file(FIXTURES / "sample_maintenance.md")

        # Search for spec numbers extracted from the document
        results = kb.graph_search("SB-72-0412")
        assert len(results) > 0
        assert any("SB-72-0412" in e.name for e in results)

    def test_stats_populated_after_ingest(self, kb: KnowledgeBase):
        kb.ingest_file(FIXTURES / "sample_maintenance.md")
        stats = kb.stats
        assert stats["chunks_stored"] > 0
        assert stats["kg_entities"] > 0

    def test_ingest_text_directly(self, kb: KnowledgeBase):
        text = (
            "## Authentication Flow\n\n"
            "The system uses OAuth 2.0 per RFC 6749 for authentication. "
            "Tokens are validated using the JWT specification per RFC 7519. "
            "Failed authentication attempts are logged per NIST 800-53 requirements.\n\n"
            "## Authorization\n\n"
            "Role-based access control (RBAC) determines what authenticated "
            "users can access. Each role maps to a set of permissions defined "
            "in the authorization matrix.\n\n"
        ) * 5  # Repeat to ensure we get enough tokens for chunks

        chunks = kb.ingest_text(text, source="auth-docs")
        assert chunks > 0

        results = kb.vector_search("How does authentication work?")
        assert len(results) > 0
        assert any("OAuth" in r.content or "authentication" in r.content for r in results)

    def test_multiple_ingestions_accumulate(self, kb: KnowledgeBase):
        kb.ingest_file(FIXTURES / "sample_maintenance.md")
        stats_after_first = kb.stats

        extra_text = (
            "## Kubernetes Deployment\n\n"
            "The application is deployed using Helm charts on AWS EKS. "
            "Each microservice runs in its own pod with resource limits defined.\n\n"
        ) * 10

        kb.ingest_text(extra_text, source="deploy-docs")
        stats_after_second = kb.stats

        assert stats_after_second["chunks_stored"] > stats_after_first["chunks_stored"]
        assert stats_after_second["kg_entities"] >= stats_after_first["kg_entities"]

    def test_graph_entity_types_are_diverse(self, kb: KnowledgeBase):
        kb.ingest_file(FIXTURES / "sample_maintenance.md")
        stats = kb.stats
        # Should extract multiple types (Concept, Specification, etc.)
        assert len(stats["kg_entity_types"]) >= 2
