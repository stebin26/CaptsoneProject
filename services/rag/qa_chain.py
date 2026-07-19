"""QA chain -- turns retrieved chunks into a grounded answer.

Final stage of the RAG query path. The system prompt confines the model to the
supplied excerpts and requires it to say when an answer is not present rather
than reaching for outside knowledge. When nothing relevant is retrieved the
chain refuses outright instead of asking the model to improvise: a truthful
'not found' is more useful than a fluent invention.
"""
# QA chain — turns retrieved chunks into a grounded answer via the LLM. Refuses
# to answer when no relevant context is found (point 12: grounded only). Final
# stage of the RAG query path.

from __future__ import annotations

from dataclasses import dataclass, field

from ops_common.config import settings
from ops_common.logging import get_logger
from retriever import (
    RetrievalResult,
    format_context,
    retrieve,
    sources_from_chunks,
)

logger = get_logger(__name__)


_NO_DOCS_MSG = (
    "No documents have been indexed for this dataset yet. Upload documents on the "
    "Documents page first, then ask again."
)
_NO_MATCH_MSG = (
    "I couldn't find anything relevant to that question in the uploaded documents "
    "for this dataset."
)

_SYSTEM_PROMPT = (
    "You answer questions strictly from the provided document excerpts. "
    "Use only the information in the context. If the answer is not in the context, "
    "say you could not find it in the documents — never use outside knowledge and "
    "never guess. Cite sources inline as [Source N] where relevant. Be concise."
)


@dataclass
class Answer:
    """A generated answer plus the evidence and honesty signals behind it.

    ``grounded`` records whether the answer rests on retrieved chunks, and
    ``llm_used`` whether the model was involved at all, so the interface can show
    the user which kind of answer they are reading.
    """
    answer: str
    grounded: bool
    sources: list[dict] = field(default_factory=list)
    used_chunks: int = 0
    llm_used: bool = False


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def _build_user_prompt(question: str, context: str) -> str:
    return (
        "Answer the question using only the context below.\n\n"
        f"=== CONTEXT ===\n{context}\n=== END CONTEXT ===\n\n"
        f"Question: {question}\n\n"
        "If the context does not contain the answer, reply exactly: "
        '"I could not find that in the documents."'
    )


def _call_llm(question: str, context: str) -> str | None:
    provider = settings.llm_provider.lower()
    if provider == "ollama":
        return _call_ollama(question, context)
    if provider == "anthropic":
        return _call_anthropic(question, context)
    logger.warning("Unknown LLM provider", extra={"provider": provider})
    return None


def _call_ollama(question: str, context: str) -> str | None:
    # Local, on-premise inference — no API key, no external calls.
    import requests

    try:
        resp = requests.post(
            f"{settings.ollama_url.rstrip('/')}/api/chat",
            json={
                "model": settings.ollama_model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_prompt(question, context)},
                ],
                "stream": False,
                "options": {"num_predict": settings.rag_max_answer_tokens},
            },
            timeout=(5, 120),
        )
        resp.raise_for_status()
        data = resp.json()
        return (data.get("message", {}).get("content") or "").strip() or None
    except Exception:  # noqa: BLE001
        logger.exception("Ollama call failed")
        return None


def _call_anthropic(question: str, context: str) -> str | None:
    try:
        from anthropic import Anthropic
    except ImportError:
        logger.warning("anthropic SDK not installed")
        return None
    if not settings.anthropic_api_key:
        logger.warning("No Anthropic API key")
        return None
    try:
        client = Anthropic(api_key=settings.anthropic_api_key)
        resp = client.messages.create(
            model=settings.llm_model,
            max_tokens=settings.rag_max_answer_tokens,
            system=_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": _build_user_prompt(question, context)}
            ],
        )
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return "".join(parts).strip()
    except Exception:  # noqa: BLE001
        logger.exception("RAG LLM call failed")
        return None


# ---------------------------------------------------------------------------
# Offline fallback — extractive answer (no LLM)
# ---------------------------------------------------------------------------


def _extractive_answer(result: RetrievalResult) -> str:
    # When no LLM is available, return the most relevant excerpts verbatim so the
    # user still gets grounded, source-tagged content rather than nothing.
    top = result.chunks[:3]
    lines = ["Here are the most relevant passages from the documents:\n"]
    for i, c in enumerate(top, start=1):
        src = c.filename + (f", p.{c.page_number}" if c.page_number else "")
        snippet = c.content.strip()
        if len(snippet) > 600:
            snippet = snippet[:600].rsplit(" ", 1)[0] + " …"
        lines.append(f"[Source {i}: {src}]\n{snippet}\n")
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def answer_question(
    dataset_id: int,
    question: str,
    top_k: int | None = None,
) -> Answer:
    """Retrieve grounded context for a dataset and answer, or refuse if none found."""
    question = (question or "").strip()
    if not question:
        return Answer(answer="Please enter a question.", grounded=False)

    result = retrieve(dataset_id, question, top_k=top_k)

    if not result.has_documents:
        return Answer(answer=_NO_DOCS_MSG, grounded=False)

    if result.is_empty:
        # Documents exist but nothing cleared the relevance cutoff — refuse rather
        # than answer from outside the documents.
        return Answer(answer=_NO_MATCH_MSG, grounded=False)

    context = format_context(result.chunks)
    sources = sources_from_chunks(result.chunks)

    llm_answer = None
    if settings.llm_enabled:
        llm_answer = _call_llm(question, context)

    if llm_answer:
        return Answer(
            answer=llm_answer,
            grounded=True,
            sources=sources,
            used_chunks=len(result.chunks),
            llm_used=True,
        )

    # LLM off or unavailable: fall back to extractive, still grounded + sourced.
    return Answer(
        answer=_extractive_answer(result),
        grounded=True,
        sources=sources,
        used_chunks=len(result.chunks),
        llm_used=False,
    )
