# Chunking — splits extracted text into overlapping chunks for embedding.
# Second stage of the RAG ingest pipeline; reads size/overlap from config.

from __future__ import annotations

from dataclasses import dataclass

from ops_common.config import settings
from extractor import ExtractedDocument  # noqa: F811


@dataclass
class Chunk:
    chunk_index: int
    content: str
    page_number: int | None
    token_estimate: int


# Rough token estimate: ~4 chars/token for English prose. Good enough for sizing
# without pulling in a tokenizer dependency.
def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _split_sentences(text: str) -> list[str]:
    # Lightweight sentence-ish split on paragraph and sentence boundaries, so a
    # chunk boundary rarely lands mid-sentence.
    import re

    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    return [p.strip() for p in parts if p and p.strip()]


def _chunk_text(
    text: str,
    page_number: int | None,
    chunk_size_tokens: int,
    overlap_tokens: int,
    start_index: int,
) -> list[Chunk]:
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    cur: list[str] = []
    cur_tokens = 0
    idx = start_index

    def _flush(carry: list[str]) -> list[str]:
        nonlocal cur, cur_tokens, idx
        if not cur:
            return []
        content = " ".join(cur).strip()
        if content:
            chunks.append(
                Chunk(
                    chunk_index=idx,
                    content=content,
                    page_number=page_number,
                    token_estimate=_estimate_tokens(content),
                )
            )
            idx += 1
        # Build overlap carry from the tail of the current chunk.
        carry_tokens = 0
        carry_rev: list[str] = []
        for s in reversed(cur):
            t = _estimate_tokens(s)
            if carry_tokens + t > overlap_tokens:
                break
            carry_rev.append(s)
            carry_tokens += t
        return list(reversed(carry_rev))

    for sent in sentences:
        t = _estimate_tokens(sent)
        # A single sentence larger than the chunk size becomes its own chunk.
        if t >= chunk_size_tokens:
            carry = _flush(cur)
            cur, cur_tokens = [], 0
            chunks.append(
                Chunk(
                    chunk_index=idx,
                    content=sent,
                    page_number=page_number,
                    token_estimate=t,
                )
            )
            idx += 1
            cur = list(carry)
            cur_tokens = sum(_estimate_tokens(s) for s in cur)
            continue

        if cur_tokens + t > chunk_size_tokens:
            carry = _flush(cur)
            cur = list(carry)
            cur_tokens = sum(_estimate_tokens(s) for s in cur)

        cur.append(sent)
        cur_tokens += t

    _flush(cur)
    return chunks


def chunk_document(doc: ExtractedDocument) -> list[Chunk]:
    """Split an extracted document into overlapping chunks, preserving page numbers."""
    size = settings.rag_chunk_size
    overlap = settings.rag_chunk_overlap

    all_chunks: list[Chunk] = []
    next_index = 0
    for page in doc.pages:
        if not page.text.strip():
            continue
        page_chunks = _chunk_text(
            page.text,
            page_number=page.page_number,
            chunk_size_tokens=size,
            overlap_tokens=overlap,
            start_index=next_index,
        )
        all_chunks.extend(page_chunks)
        next_index += len(page_chunks)

    return all_chunks