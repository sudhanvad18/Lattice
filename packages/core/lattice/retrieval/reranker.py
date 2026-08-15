"""
Cross-encoder reranker for retrieval quality improvement.

After initial retrieval (vector store top-K), the reranker re-scores
candidates using a cross-encoder model that sees both query and document
together — dramatically better relevance than bi-encoder similarity alone.

Supports:
- PyTorch cross-encoder (local, fine-tunable)
- Mock reranker for testing/CPU-only environments
- Configurable top-N after reranking

Fine-tuning workflow:
1. Collect (query, passage, relevance_label) triples from eval tasks
2. Train cross-encoder on domain data using `train_reranker()`
3. Export model and load via `CrossEncoderReranker(model_path=...)`
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class RankedResult:
    """A reranked retrieval result."""

    chunk_id: str
    content: str
    original_score: float
    rerank_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class Reranker(ABC):
    """Abstract reranker interface."""

    @abstractmethod
    def rerank(self, query: str, candidates: list[dict[str, Any]], top_n: int = 5) -> list[RankedResult]:
        """Rerank candidates given a query.

        Args:
            query: The search query
            candidates: List of dicts with at minimum 'chunk_id', 'content', 'score'
            top_n: Number of results to return after reranking
        """
        ...


class CrossEncoderReranker(Reranker):
    """PyTorch cross-encoder reranker.

    Uses a pre-trained or fine-tuned cross-encoder model to score
    (query, passage) pairs. Much more accurate than bi-encoder similarity
    but slower — intended for reranking a small candidate set (10-50 docs).
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        model_path: str | None = None,
        device: str = "cpu",
        batch_size: int = 16,
    ) -> None:
        self._model_name = model_name
        self._model_path = model_path
        self._device = device
        self._batch_size = batch_size
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy-load the model on first use."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            path = self._model_path or self._model_name
            logger.info("loading_reranker", model=path, device=self._device)

            self._tokenizer = AutoTokenizer.from_pretrained(path)
            self._model = AutoModelForSequenceClassification.from_pretrained(path)
            self._model.to(self._device)
            self._model.eval()

            logger.info("reranker_loaded", model=path)
        except ImportError:
            raise RuntimeError(
                "PyTorch and transformers are required for CrossEncoderReranker. "
                "Install with: pip install torch transformers"
            )

    def rerank(self, query: str, candidates: list[dict[str, Any]], top_n: int = 5) -> list[RankedResult]:
        self._load_model()
        import torch

        if not candidates:
            return []

        pairs = [(query, c.get("content", "")) for c in candidates]
        scores = []

        for i in range(0, len(pairs), self._batch_size):
            batch = pairs[i : i + self._batch_size]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits.squeeze(-1)
                scores.extend(logits.cpu().tolist())

        results = []
        for candidate, score in zip(candidates, scores):
            results.append(
                RankedResult(
                    chunk_id=candidate.get("chunk_id", ""),
                    content=candidate.get("content", ""),
                    original_score=candidate.get("score", 0.0),
                    rerank_score=float(score),
                    metadata=candidate.get("metadata", {}),
                )
            )

        results.sort(key=lambda r: r.rerank_score, reverse=True)
        return results[:top_n]


class MockReranker(Reranker):
    """Mock reranker for testing — scores by keyword overlap."""

    def rerank(self, query: str, candidates: list[dict[str, Any]], top_n: int = 5) -> list[RankedResult]:
        query_terms = set(query.lower().split())
        results = []

        for c in candidates:
            content = c.get("content", "")
            content_terms = set(content.lower().split())
            overlap = len(query_terms & content_terms)
            score = overlap / max(len(query_terms), 1)

            results.append(
                RankedResult(
                    chunk_id=c.get("chunk_id", ""),
                    content=content,
                    original_score=c.get("score", 0.0),
                    rerank_score=score,
                    metadata=c.get("metadata", {}),
                )
            )

        results.sort(key=lambda r: r.rerank_score, reverse=True)
        return results[:top_n]


# --- Fine-tuning utilities ---


@dataclass
class TrainingSample:
    """A training sample for fine-tuning the cross-encoder."""

    query: str
    passage: str
    label: float  # 0.0 = irrelevant, 1.0 = perfectly relevant


def generate_training_data(
    eval_results_path: str,
    output_path: str,
) -> int:
    """Generate training data from evaluation results.

    Reads eval task results and creates (query, passage, label) triples
    for fine-tuning. High-quality artifacts with citations produce
    positive samples; low-quality or uncited passages produce negatives.
    """
    samples: list[dict] = []
    path = Path(eval_results_path)

    if not path.exists():
        logger.warning("training_data_source_missing", path=eval_results_path)
        return 0

    data = json.loads(path.read_text())

    for item in data:
        query = item.get("query", "")
        for passage in item.get("positive_passages", []):
            samples.append({"query": query, "passage": passage, "label": 1.0})
        for passage in item.get("negative_passages", []):
            samples.append({"query": query, "passage": passage, "label": 0.0})

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(samples, indent=2))

    logger.info("training_data_generated", samples=len(samples), output=output_path)
    return len(samples)


def train_reranker(
    training_data_path: str,
    output_dir: str,
    base_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
) -> str:
    """Fine-tune a cross-encoder on domain-specific data.

    Returns the path to the saved model directory.
    """
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler
    except ImportError:
        raise RuntimeError("PyTorch and transformers required. pip install torch transformers")

    data = json.loads(Path(training_data_path).read_text())
    logger.info("training_started", samples=len(data), base_model=base_model, epochs=epochs)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=1)

    class PairDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            s = self.samples[idx]
            enc = tokenizer(
                s["query"],
                s["passage"],
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "labels": torch.tensor(s["label"], dtype=torch.float),
            }

    dataset = PairDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    num_steps = len(loader) * epochs
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=int(num_steps * 0.1), num_training_steps=num_steps)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"].unsqueeze(-1),
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        logger.info("epoch_complete", epoch=epoch + 1, avg_loss=round(avg_loss, 4))

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))

    logger.info("model_saved", path=str(out_path))
    return str(out_path)
