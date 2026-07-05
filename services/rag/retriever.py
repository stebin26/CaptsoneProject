# Retriever — embeds a query and fetches the most relevant chunks for one dataset.
# Fifth stage of the RAG pipeline; feeds the QA chain. Always dataset_id-scoped.

from __future__ import annotations

from dataclasses import dataclass

from ops_common.config import settings
from ops_common.logging import get_logger

from embedder import embed_query
from vector_store import (
    RetrievedChunk,
    dataset_has_documents,
    similarity_search,
)

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    has_documents: bool
    query: str

    @property
    def is_empty(self) -> bool:
        return not self.chunks


# Cosine distance ranges 0 (identical) .. 2 (opposite). Chunks beyond this are
# too unrelated to be useful context; dropping them keeps answers grounded and
# lets the QA chain honestly say "not found" rather than forcing weak matches.
_MAX_DISTANCE = 0.75


def retrieve(
    dataset_id: int,
    query: str,
    top_k: int | None = None,
    max_distance: float | None = None,
) -> RetrievalResult:
    """Embed the query and return the top matching chunks within one dataset."""
    k = top_k or settings.rag_top_k
    cutoff = _MAX_DISTANCE if max_distance is None else max_distance

    query = (query or "").strip()
    if not query:
        return RetrievalResult(chunks=[], has_documents=False, query=query)

    has_docs = dataset_has_documents(dataset_id)
    if not has_docs:
        return RetrievalResult(chunks=[], has_documents=False, query=query)

    query_vec = embed_query(query)
    if not query_vec:
        return RetrievalResult(chunks=[], has_documents=True, query=query)

    # Over-fetch a little, then filter by distance so the final set is only
    # genuinely relevant chunks (never fewer than what passes the cutoff).
    raw = similarity_search(dataset_id, query_vec, top_k=max(k * 2, k))
    filtered = [c for c in raw if c.distance <= cutoff][:k]

    logger.info(
        "RAG retrieval",
        extra={
            "dataset_id": dataset_id,
            "candidates": len(raw),
            "kept": len(filtered),
            "top_distance": round(raw[0].distance, 4) if raw else None,
        },
    )

    return RetrievalResult(chunks=filtered, has_documents=True, query=query)


def format_context(chunks: list[RetrievedChunk], max_chars: int = 8000) -> str:
    """Concatenate retrieved chunks into a bounded context block with source tags."""
    parts: list[str] = []
    used = 0
    for i, c in enumerate(chunks, start=1):
        src = c.filename
        if c.page_number:
            src += f", p.{c.page_number}"
        block = f"[Source {i}: {src}]\n{c.content.strip()}"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def sources_from_chunks(chunks: list[RetrievedChunk]) -> list[dict]:
    """Compact, de-duplicated source list for display alongside the answer."""
    seen: set[tuple] = set()
    out: list[dict] = []
    for c in chunks:
        key = (c.filename, c.page_number)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "filename": c.filename,
                "page_number": c.page_number,
                "distance": round(c.distance, 4),
            }
        )
    return out