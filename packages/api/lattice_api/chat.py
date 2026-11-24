"""Retrieval-backed answers for the dashboard Chat tab.

The tab posts one question and renders `response` plus citation ids. This uses
the same knowledge-base searches as the researcher (vector store and graph)
and returns that text directly. It does not call a model, so a missing or
mock provider cannot invent an answer.
"""

from __future__ import annotations

import re
from typing import Any, Protocol

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}")
_STOPWORDS = frozenset({
    "about",
    "and",
    "are",
    "can",
    "does",
    "for",
    "from",
    "have",
    "how",
    "into",
    "our",
    "that",
    "the",
    "this",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "your",
})
_EMPTY_ANSWER = "No matching documents or entities were found in the knowledge base."
_MAX_TERMS = 8
_MAX_ENTITIES = 8


class Retriever(Protocol):
    """The slice of KnowledgeBase that chat needs."""

    def vector_search(self, query: str, top_k: int = 5) -> list[Any]: ...

    def graph_search(self, query: str, entity_type: str | None = None) -> list[Any]: ...


def build_chat_answer(kb: Retriever, message: str, top_k: int = 5) -> tuple[str, list[str]]:
    """Return retrieved passages and matching entities for `message`."""
    chunks = list(kb.vector_search(message, top_k=top_k) or [])
    entities = _graph_hits(kb, message)

    citations: list[str] = []
    sections: list[str] = []

    chunk_lines: list[str] = []
    for hit in chunks:
        content = str(getattr(hit, "content", "") or "").strip()
        if not content:
            continue
        chunk_id = str(getattr(hit, "chunk_id", "") or "")
        if chunk_id and chunk_id not in citations:
            citations.append(chunk_id)
        chunk_lines.append(f"[{chunk_id or 'passage'}] {content}")
    if chunk_lines:
        sections.append("Retrieved passages:\n" + "\n\n".join(chunk_lines))

    entity_lines: list[str] = []
    for entity in entities:
        name = str(getattr(entity, "name", "") or "").strip()
        if not name:
            continue
        entity_id = str(getattr(entity, "id", "") or "")
        if entity_id and entity_id not in citations:
            citations.append(entity_id)
        entity_type = str(getattr(entity, "entity_type", "") or "").strip()
        description = str(getattr(entity, "description", "") or "").strip()
        type_label = f" ({entity_type})" if entity_type else ""
        desc_label = f": {description}" if description else ""
        entity_lines.append(f"- {name}{type_label}{desc_label}")
    if entity_lines:
        sections.append("Knowledge graph:\n" + "\n".join(entity_lines))

    if not sections:
        return _EMPTY_ANSWER, []
    return "\n\n".join(sections), citations


def _graph_hits(kb: Retriever, message: str) -> list[Any]:
    """Search the full question, then salient terms.

    Graph search is a substring match on entity names. A full question rarely
    occurs inside a name, so each content word is tried as well.
    """
    seen: set[str] = set()
    found: list[Any] = []
    for query in (message, *_terms(message)):
        for entity in kb.graph_search(query) or []:
            entity_id = str(getattr(entity, "id", "") or "") or str(id(entity))
            if entity_id in seen:
                continue
            seen.add(entity_id)
            found.append(entity)
            if len(found) >= _MAX_ENTITIES:
                return found
    return found


def _terms(message: str) -> list[str]:
    terms: list[str] = []
    for match in _WORD.findall(message):
        word = match.lower()
        if word in _STOPWORDS or word in terms:
            continue
        terms.append(word)
        if len(terms) >= _MAX_TERMS:
            break
    return terms
