"""Retrieval-backed answers for the dashboard Chat tab."""

from __future__ import annotations

import re
from typing import Any, Protocol

from starlette.concurrency import run_in_threadpool

from lattice.inference.health import friendly_provider_error
from lattice.inference.provider import GenerationConfig, LLMProvider, Message
from lattice_api.models import ChatCitation, ChatResponse, ChatTurn

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
_NO_COVERAGE_ANSWER = (
    "The knowledge base doesn't cover that yet. Try ingesting the relevant files."
)
_MAX_TERMS = 8
_MAX_ENTITIES = 8
_MIN_SIMILARITY = 0.15


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


async def answer_question(
    kb: Any,
    provider: LLMProvider,
    message: str,
    history: list[ChatTurn],
) -> ChatResponse:
    """Generate a grounded answer from the knowledge base using the model."""

    def _retrieve() -> tuple[list[Any], list[Any], dict[str, str]]:
        chunks = list(kb.vector_search(message, top_k=6) or [])
        chunks = [c for c in chunks if getattr(c, "similarity", 1.0) >= _MIN_SIMILARITY]
        entities = _graph_hits(kb, message)

        entity_summaries: dict[str, str] = {}
        for entity in entities:
            entity_id = str(getattr(entity, "id", "") or "")
            entity_name = str(getattr(entity, "name", "") or "")
            summary_parts = [entity_name] if entity_name else []

            if entity_id and hasattr(kb, "get_entity_context"):
                try:
                    ctx = kb.get_entity_context(entity_id, depth=1)
                    for neighbor in (ctx.entities or [])[:5]:
                        n_id = getattr(neighbor, "id", "")
                        n_name = str(getattr(neighbor, "name", "") or "").strip()
                        if n_id != entity_id and n_name:
                            summary_parts.append(n_name)
                except Exception:
                    pass

            entity_summaries[entity_id] = ", ".join(summary_parts)[:200]

        return chunks, entities, entity_summaries

    chunks, entities, entity_summaries = await run_in_threadpool(_retrieve)

    if not chunks and not entities:
        return ChatResponse(
            mode="answer",
            response=_NO_COVERAGE_ANSWER,
            citations=[],
        )

    # Build numbered sources: chunks first, then entities
    sources: list[ChatCitation] = []

    for chunk in chunks:
        metadata = getattr(chunk, "metadata", {}) or {}
        source = str(metadata.get("source", ""))
        content = str(getattr(chunk, "content", ""))
        snippet = " ".join(content.split())[:200]
        sources.append(ChatCitation(source=source, snippet=snippet))

    for entity in entities:
        entity_id = str(getattr(entity, "id", "") or "")
        entity_name = str(getattr(entity, "name", "") or "")
        snippet = entity_summaries.get(entity_id, entity_name)[:200]
        sources.append(ChatCitation(source=entity_name, snippet=snippet))

    # Build prompt
    sources_text = "\n\n".join(
        f"[{i + 1}] Source: {s.source}\n{s.snippet}"
        for i, s in enumerate(sources)
    )
    system_content = (
        "Answer the user's question using only the numbered sources provided below. "
        "Cite each source inline as [n] where n is its number. "
        "If the sources do not contain the answer, say so plainly. "
        "Prefer short paragraphs and use `backticks` for code identifiers."
    )
    user_content = f"Sources:\n{sources_text}\n\nQuestion: {message}"

    messages: list[Message] = [Message(role="system", content=system_content)]
    for turn in history[-6:]:
        messages.append(Message(role=turn.role, content=turn.content))
    messages.append(Message(role="user", content=user_content))

    config = GenerationConfig(temperature=0.2, max_tokens=700)

    try:
        result = await provider.generate(messages, config)
        answer = result.content
    except Exception as exc:
        notice = (
            friendly_provider_error(exc)
            or "The model failed to answer; showing retrieved passages."
        )
        return ChatResponse(
            mode="retrieval_only",
            response=_format_sources_text(sources),
            citations=sources,
            notice=notice,
        )

    # Return only citations actually referenced in the answer
    raw_refs = re.findall(r"\[(\d+)\]", answer)
    cited_indices = {int(r) for r in raw_refs if 1 <= int(r) <= len(sources)}
    cited = [sources[i - 1] for i in sorted(cited_indices)] if cited_indices else sources

    return ChatResponse(mode="answer", response=answer, citations=cited)


def _format_sources_text(sources: list[ChatCitation]) -> str:
    if not sources:
        return _EMPTY_ANSWER
    parts = [f"[{i + 1}] {s.source}: {s.snippet}" for i, s in enumerate(sources)]
    return "Retrieved passages:\n\n" + "\n\n".join(parts)


def _graph_hits(kb: Retriever, message: str) -> list[Any]:
    """Search the full question, then salient terms."""
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
