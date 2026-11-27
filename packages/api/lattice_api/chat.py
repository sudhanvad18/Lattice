"""Retrieval-backed answers for the dashboard Chat tab."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
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


async def _prepare_answer(
    kb: Any,
    message: str,
    history: list[ChatTurn],
) -> tuple[list[ChatCitation], list[Message]]:
    """Retrieve full passages and prepare the common grounded prompt."""

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
        return [], []

    # Build numbered sources: chunks first, then entities
    sources: list[ChatCitation] = []
    prompt_blocks: list[str] = []

    for chunk in chunks:
        metadata = getattr(chunk, "metadata", {}) or {}
        source = str(metadata.get("source", ""))
        content = str(getattr(chunk, "content", ""))
        index = len(sources) + 1
        sources.append(ChatCitation(
            index=index,
            kind="chunk",
            id=str(getattr(chunk, "chunk_id", "") or ""),
            source=source,
            snippet=" ".join(content.split())[:200],
        ))
        prompt_blocks.append(f"[{index}] Source: {source}\n{content.strip()}")

    for entity in entities:
        entity_id = str(getattr(entity, "id", "") or "")
        entity_name = str(getattr(entity, "name", "") or "")
        entity_type = str(getattr(entity, "entity_type", "") or "")
        description = str(getattr(entity, "description", "") or "").strip()
        related = entity_summaries.get(entity_id, entity_name)
        index = len(sources) + 1
        sources.append(ChatCitation(
            index=index, kind="entity", id=entity_id, source=entity_name, snippet=related[:200],
        ))
        detail = f"{entity_type}: {description}" if description else entity_type
        prompt_blocks.append(
            f"[{index}] Knowledge graph entity `{entity_name}` ({detail})\nRelated: {related}"
        )

    sources_text = "\n\n".join(prompt_blocks)
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

    return sources, messages


def _final_answer(
    answer: str, sources: list[ChatCitation], notice: str | None = None,
) -> ChatResponse:
    raw_refs = re.findall(r"\[(\d+)\]", answer)
    cited_indices = {int(r) for r in raw_refs if 1 <= int(r) <= len(sources)}
    cited = [sources[i - 1] for i in sorted(cited_indices)] if cited_indices else sources
    return ChatResponse(mode="answer", response=answer, citations=cited, notice=notice)


def _retrieval_fallback(sources: list[ChatCitation], notice: str) -> ChatResponse:
    return ChatResponse(
        mode="retrieval_only", response=_format_sources_text(sources),
        citations=sources, notice=notice,
    )


async def answer_question(
    kb: Any, provider: LLMProvider, message: str, history: list[ChatTurn],
) -> ChatResponse:
    """Generate the existing, non-streaming grounded answer."""
    sources, messages = await _prepare_answer(kb, message, history)
    if not sources:
        return ChatResponse(response=_NO_COVERAGE_ANSWER)
    try:
        result = await provider.generate(
            messages, GenerationConfig(temperature=0.2, max_tokens=700),
        )
    except Exception as exc:
        return _retrieval_fallback(
            sources, friendly_provider_error(exc)
            or "The model failed to answer; showing retrieved passages.",
        )
    return _final_answer(result.content, sources)


async def stream_answer(
    kb: Any, provider: LLMProvider | None, message: str, history: list[ChatTurn],
    provider_notice: str | None = None,
) -> AsyncIterator[tuple[str, dict]]:
    """Stream progress, source metadata, text deltas, then an authoritative final response."""
    yield "status", {"message": "Searching knowledge base…"}
    try:
        sources, messages = await _prepare_answer(kb, message, history)
    except Exception:
        yield "error", {"message": "Knowledge base retrieval failed. Please try again."}
        return
    if not sources:
        yield "done", ChatResponse(response=_NO_COVERAGE_ANSWER).model_dump()
        return
    yield "sources", {"citations": [s.model_dump() for s in sources]}
    if provider is None:
        yield "done", _retrieval_fallback(
            sources, provider_notice or "The model is unavailable; showing retrieved passages.",
        ).model_dump()
        return
    yield "status", {"message": "Generating answer…"}
    parts: list[str] = []
    try:
        config = GenerationConfig(temperature=0.2, max_tokens=700)
        async for text in provider.stream(messages, config):
            if text:
                parts.append(text)
                yield "delta", {"text": text}
        if not parts:
            raise RuntimeError("Empty model response")
    except Exception as exc:
        if parts:
            result = _final_answer(
                "".join(parts), sources,
                "The answer was interrupted. This is a partial response; please try again.",
            )
        else:
            result = _retrieval_fallback(
                sources, friendly_provider_error(exc)
                or "The model failed to answer; showing retrieved passages.",
            )
        yield "done", result.model_dump()
        return
    yield "done", _final_answer("".join(parts), sources).model_dump()


def _format_sources_text(sources: list[ChatCitation]) -> str:
    if not sources:
        return _EMPTY_ANSWER
    parts = [f"[{s.index}] {s.source}: {s.snippet}" for s in sources]
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
