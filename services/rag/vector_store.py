"""Vector store -- persists documents, chunks, and embeddings in pgvector.

Fourth stage of the RAG pipeline, and the read side of every query. Every
operation is scoped by ``dataset_id``, so retrieval can never cross a company
boundary and one tenant's documents can never surface in another's answer.
Using pgvector rather than a separate vector database keeps the embeddings in
the same Postgres instance as the rest of the platform.
"""
# Vector store — writes documents/chunks/embeddings to pgvector and reads back
# via similarity search. Every operation is dataset_id-scoped so retrieval never
# crosses company boundaries. Fourth stage of the RAG pipeline.

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import psycopg2
import psycopg2.extras
from ops_common.config import settings
from ops_common.logging import get_logger

logger = get_logger(__name__)


def _db_config() -> dict:
    return {
        "host": settings.postgres_host,
        "port": settings.postgres_port,
        "dbname": settings.postgres_db,
        "user": settings.postgres_user,
        "password": settings.postgres_password,
    }


@contextmanager
def _conn():
    config = _db_config()
    try:
        conn = psycopg2.connect(**config)
    except psycopg2.Error:
        logger.exception(
            "Could not connect to Postgres at %s:%s/%s for vector storage",
            config["host"],
            config["port"],
            config["dbname"],
            extra={
                "db_host": config["host"],
                "db_port": config["port"],
                "db_name": config["dbname"],
            },
        )
        raise

    try:
        yield conn
        conn.commit()
    except Exception:
        logger.exception("Vector store transaction failed — rolling back")
        try:
            conn.rollback()
        except psycopg2.Error:
            # Logged, not raised: the original failure is the useful one.
            logger.exception("Rollback failed after a vector store error")
        raise
    finally:
        try:
            conn.close()
        except psycopg2.Error:
            logger.warning(
                "Failed to close the vector store connection cleanly", exc_info=True
            )


def _vector_literal(vec: list[float]) -> str:
    # pgvector accepts a bracketed literal like '[0.1,0.2,...]'.
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


# ---------------------------------------------------------------------------
# Document lifecycle
# ---------------------------------------------------------------------------


@dataclass
class DocumentRecord:
    """A stored document's identity, type, and indexing state."""
    id: int
    dataset_id: int
    filename: str
    file_type: str
    status: str
    chunk_count: int


def create_document(
    dataset_id: int,
    business_name: str | None,
    filename: str,
    file_type: str,
    file_size: int | None,
) -> int:
    """Insert a document row in 'pending' state; returns its id.

    Raises:
        RuntimeError: If the insert returns no id.
    """
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO rag.documents
                (dataset_id, business_name, filename, file_type, file_size, status)
            VALUES (%s, %s, %s, %s, %s, 'pending')
            RETURNING id
            """,
            (dataset_id, business_name, filename, file_type, file_size),
        )
        row = cur.fetchone()
        if row is None:
            # Without an id there is nothing to attach chunks to, so a None must
            # not be allowed to flow downstream as if it were a document.
            logger.error(
                "Document insert returned no id",
                extra={"dataset_id": dataset_id, "document_filename": filename},
            )
            raise RuntimeError(f"Could not create a document row for {filename!r}")
        return int(row[0])


def set_document_status(
    document_id: int,
    status: str,
    chunk_count: int | None = None,
    error_detail: str | None = None,
) -> None:
    """Update a document's processing status (and chunk_count / error when given)."""
    sets = ["status = %s"]
    params: list = [status]
    if chunk_count is not None:
        sets.append("chunk_count = %s")
        params.append(chunk_count)
    if error_detail is not None:
        sets.append("error_detail = %s")
        params.append(error_detail)
    if status == "indexed":
        sets.append("indexed_at = now()")
    params.append(document_id)

    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"UPDATE rag.documents SET {', '.join(sets)} WHERE id = %s",
            tuple(params),
        )


def delete_document(dataset_id: int, document_id: int) -> bool:
    """Delete a document (chunks + embeddings cascade). dataset_id-guarded."""
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM rag.documents WHERE id = %s AND dataset_id = %s",
            (document_id, dataset_id),
        )
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Chunk + embedding writes
# ---------------------------------------------------------------------------


def store_chunks_with_embeddings(
    dataset_id: int,
    document_id: int,
    chunks: list,  # list of chunker.Chunk
    embeddings: list[list[float]],
    model_name: str,
) -> int:
    """Insert all chunks and their embeddings for a document in one transaction."""
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks and embeddings count mismatch: "
            f"{len(chunks)} chunks vs {len(embeddings)} embeddings"
        )
    if not chunks:
        return 0

    try:
        _write_chunks_and_embeddings(
            dataset_id, document_id, chunks, embeddings, model_name
        )
    except Exception:
        logger.exception(
            "Could not store %d chunk(s) for document %s",
            len(chunks),
            document_id,
            extra={
                "dataset_id": dataset_id,
                "document_id": document_id,
                "chunk_count": len(chunks),
                "embedding_model": model_name,
            },
        )
        raise

    return len(chunks)


def _write_chunks_and_embeddings(
    dataset_id: int,
    document_id: int,
    chunks: list,
    embeddings: list[list[float]],
    model_name: str,
) -> None:
    with _conn() as conn:
        with conn.cursor() as cur:
            # Insert chunks, capturing generated ids in order.
            chunk_rows = [
                (
                    dataset_id,
                    document_id,
                    c.chunk_index,
                    c.content,
                    c.page_number,
                    c.token_estimate,
                )
                for c in chunks
            ]

            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO rag.chunks
                    (dataset_id, document_id, chunk_index, content,
                     page_number, token_estimate)
                VALUES %s
                RETURNING id
                """,
                chunk_rows,
                template="(%s,%s,%s,%s,%s,%s)",
                page_size=200,
            )


            # execute_values with RETURNING returns ids for the last page only in
            # some psycopg2 versions; re-fetch deterministically by chunk_index to
            # be safe, mapping index -> id.
            cur.execute(
                """
                SELECT id, chunk_index FROM rag.chunks
                WHERE document_id = %s ORDER BY chunk_index
                """,
                (document_id,),
            )
            id_by_index = {ci: cid for cid, ci in cur.fetchall()}

            emb_rows = []
            for c, vec in zip(chunks, embeddings, strict=False):
                cid = id_by_index.get(c.chunk_index)
                if cid is None:
                    continue
                emb_rows.append(
                    (dataset_id, document_id, cid, _vector_literal(vec), model_name)
                )

            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO rag.embeddings
                    (dataset_id, document_id, chunk_id, embedding, model_name)
                VALUES %s
                """,
                emb_rows,
                template="(%s,%s,%s,%s::vector,%s)",
                page_size=200,
            )


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def list_documents(dataset_id: int) -> list[dict]:
    """List every document belonging to a dataset, newest first.

    Args:
        dataset_id: Dataset whose documents to list.

    Returns:
        One dictionary per document.
    """
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, filename, file_type, status, chunk_count,
                   error_detail, uploaded_at, indexed_at
            FROM rag.documents
            WHERE dataset_id = %s
            ORDER BY uploaded_at DESC
            """,
            (dataset_id,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]


@dataclass
class RetrievedChunk:
    """One chunk returned by a similarity search, with its distance."""
    chunk_id: int
    document_id: int
    filename: str
    page_number: int | None
    content: str
    distance: float


def similarity_search(
    dataset_id: int,
    query_vector: list[float],
    top_k: int,
) -> list[RetrievedChunk]:
    """Cosine nearest chunks within one dataset. Lower distance = more similar."""
    if not query_vector:
        logger.warning(
            "Similarity search called with an empty query vector",
            extra={"dataset_id": dataset_id},
        )
        return []

    vec = _vector_literal(query_vector)
    try:
        rows = _run_similarity_query(vec, dataset_id, top_k)
    except psycopg2.Error:
        logger.exception(
            "Similarity search failed for dataset %s",
            dataset_id,
            extra={"dataset_id": dataset_id, "top_k": top_k},
        )
        raise

    return [
        RetrievedChunk(
            chunk_id=r[0],
            document_id=r[1],
            filename=r[2],
            page_number=r[3],
            content=r[4],
            distance=float(r[5]),
        )
        for r in rows
    ]


def _run_similarity_query(vec: str, dataset_id: int, top_k: int) -> list[tuple]:
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT e.chunk_id, e.document_id, d.filename, c.page_number, c.content,
                   e.embedding <=> %s::vector AS distance
            FROM rag.embeddings e
            JOIN rag.chunks c    ON c.id = e.chunk_id
            JOIN rag.documents d ON d.id = e.document_id
            WHERE e.dataset_id = %s
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s
            """,
            (vec, dataset_id, vec, top_k),
        )
        return cur.fetchall()


def dataset_has_documents(dataset_id: int) -> bool:
    """Report whether a dataset has any indexed embeddings.

    Lets the query path distinguish 'nothing uploaded yet' from 'nothing matched',
    which are very different answers to give a user.

    Args:
        dataset_id: Dataset to check.

    Returns:
        True if at least one embedding exists for the dataset.
    """
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM rag.embeddings WHERE dataset_id = %s LIMIT 1",
            (dataset_id,),
        )
        return cur.fetchone() is not None
