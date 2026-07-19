"""Ingest pipeline -- takes one document from raw file to indexed vectors.

Orchestrates extract, chunk, embed, and store, updating the document's status in
``rag.documents`` at each stage so the interface can show progress and surface
any failure against the specific file that caused it. This is what the API's
background task runs per uploaded file.
"""
# Ingest pipeline — orchestrates one document from raw file to indexed vectors:
# extract → chunk → embed → store, updating rag.documents status at each stage.
# This is what the API's background task calls per uploaded file.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from chunker import chunk_document
from embedder import embed_texts, model_name
from extractor import detect_file_type, extract_document
from ops_common.logging import get_logger
from vector_store import (
    create_document,
    set_document_status,
    store_chunks_with_embeddings,
)

logger = get_logger(__name__)


@dataclass
class IngestResult:
    """Outcome of indexing one document, including any failure detail."""
    document_id: int
    filename: str
    status: str
    chunk_count: int
    error: str | None = None


def register_document(
    dataset_id: int,
    business_name: str | None,
    filename: str,
    file_size: int | None,
) -> int:
    """Create the 'pending' document row up front so the UI can show it immediately."""
    file_type = detect_file_type(filename)
    return create_document(
        dataset_id=dataset_id,
        business_name=business_name,
        filename=filename,
        file_type=file_type,
        file_size=file_size,
    )


def index_document(
    dataset_id: int,
    document_id: int,
    stored_path: str | Path,
    filename: str,
) -> IngestResult:
    """Run the full extract→chunk→embed→store flow for one already-registered doc."""
    path = Path(stored_path)
    set_document_status(document_id, "processing")

    try:
        # 1. Extract
        doc = extract_document(path, filename=filename)
        if not doc.full_text.strip():
            set_document_status(
                document_id,
                "failed",
                chunk_count=0,
                error_detail="No extractable text (empty or scanned document).",
            )
            return IngestResult(
                document_id, filename, "failed", 0, "No extractable text."
            )

        # 2. Chunk
        chunks = chunk_document(doc)
        if not chunks:
            set_document_status(
                document_id,
                "failed",
                chunk_count=0,
                error_detail="Extraction produced no chunks.",
            )
            return IngestResult(
                document_id, filename, "failed", 0, "No chunks produced."
            )

        # 3. Embed (batched inside the embedder)
        texts = [c.content for c in chunks]
        vectors = embed_texts(texts)
        if len(vectors) != len(chunks):
            raise RuntimeError(
                f"embedding count {len(vectors)} != chunk count {len(chunks)}"
            )

        # 4. Store chunks + embeddings (single transaction)
        written = store_chunks_with_embeddings(
            dataset_id=dataset_id,
            document_id=document_id,
            chunks=chunks,
            embeddings=vectors,
            model_name=model_name(),
        )

        set_document_status(document_id, "indexed", chunk_count=written)
        logger.info(
            "Document indexed",
            extra={
                "document_id": document_id,
                "chunks": written,
                "dataset_id": dataset_id,
            },
        )
        return IngestResult(document_id, filename, "indexed", written)

    except Exception as exc:  # noqa: BLE001
        logger.exception("Document indexing failed")
        set_document_status(
            document_id,
            "failed",
            error_detail=str(exc)[:1000],
        )
        return IngestResult(document_id, filename, "failed", 0, str(exc))


def ingest_file(
    dataset_id: int,
    business_name: str | None,
    stored_path: str | Path,
    filename: str,
    file_size: int | None = None,
) -> IngestResult:
    """Register + index in one call (used when not pre-registering in the API)."""
    document_id = register_document(dataset_id, business_name, filename, file_size)
    return index_document(dataset_id, document_id, stored_path, filename)
