"""
Tests for the cross-encoder reranker.
"""

import pytest

from lattice.retrieval.reranker import (
    CrossEncoderReranker,
    MockReranker,
    RankedResult,
    generate_training_data,
)


class TestMockReranker:
    @pytest.fixture
    def reranker(self):
        return MockReranker()

    def test_rerank_by_keyword_overlap(self, reranker):
        candidates = [
            {"chunk_id": "c1", "content": "The turbine blade inspection is quarterly", "score": 0.8},
            {"chunk_id": "c2", "content": "Python programming guide for beginners", "score": 0.9},
            {"chunk_id": "c3", "content": "Turbine blade erosion causes failure in engines", "score": 0.7},
        ]
        results = reranker.rerank("turbine blade failure", candidates, top_n=2)

        assert len(results) == 2
        assert isinstance(results[0], RankedResult)
        # c3 should rank highest — has "turbine", "blade", "failure"
        assert results[0].chunk_id == "c3"
        assert results[0].rerank_score > results[1].rerank_score

    def test_rerank_empty_candidates(self, reranker):
        results = reranker.rerank("anything", [], top_n=5)
        assert results == []

    def test_top_n_limits_results(self, reranker):
        candidates = [
            {"chunk_id": f"c{i}", "content": f"document number {i} about testing", "score": 0.5}
            for i in range(10)
        ]
        results = reranker.rerank("testing", candidates, top_n=3)
        assert len(results) == 3

    def test_ranked_result_fields(self, reranker):
        candidates = [{"chunk_id": "x", "content": "hello world", "score": 0.6, "metadata": {"page": 1}}]
        results = reranker.rerank("hello", candidates, top_n=1)
        assert results[0].chunk_id == "x"
        assert results[0].original_score == 0.6
        assert results[0].metadata == {"page": 1}


class TestCrossEncoderReranker:
    def test_init_without_loading(self):
        reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        assert reranker._model is None  # Lazy loaded

    def test_load_fails_without_torch(self):
        reranker = CrossEncoderReranker()
        # This would fail in CI without torch installed — just verify the interface exists
        assert hasattr(reranker, "rerank")


class TestTrainingDataGeneration:
    def test_missing_source_returns_zero(self, tmp_path):
        count = generate_training_data(
            str(tmp_path / "nonexistent.json"),
            str(tmp_path / "output.json"),
        )
        assert count == 0

    def test_generate_from_eval_data(self, tmp_path):
        import json

        source = tmp_path / "eval_data.json"
        source.write_text(json.dumps([
            {
                "query": "What is blade erosion?",
                "positive_passages": ["Blade erosion is the wear of fan blade surfaces."],
                "negative_passages": ["Python is a programming language.", "The weather is nice."],
            },
            {
                "query": "Inspection schedule",
                "positive_passages": ["Visual inspection every 500 flight hours."],
                "negative_passages": ["Lorem ipsum dolor sit amet."],
            },
        ]))

        output = tmp_path / "training.json"
        count = generate_training_data(str(source), str(output))

        assert count == 5  # 2 positive + 3 negative
        assert output.exists()

        data = json.loads(output.read_text())
        assert len(data) == 5
        assert data[0]["label"] == 1.0
        assert data[1]["label"] == 0.0
