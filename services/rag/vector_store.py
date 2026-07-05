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
    conn = psycopg2.connect(**_db_config())
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _vector_literal(vec: list[float]) -> str:
    # pgvector accepts a bracketed literal like '[0.1,0.2,...]'.
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


# ---------------------------------------------------------------------------
# Document lifecycle
# ---------------------------------------------------------------------------

@dataclass
class DocumentRecord:
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
    """Insert a document row in 'pending' state; returns its id."""
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
        return int(cur.fetchone()[0])


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
    chunks: list,          # list of chunker.Chunk
    embeddings: list[list[float]],
    model_name: str,
) -> int:
    """Insert all chunks and their embeddings for a document in one transaction."""
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings count mismatch")
    if not chunks:
        return 0

    with _conn() as conn:
        with conn.cursor() as cur:
            # Insert chunks, capturing generated ids in order.
            chunk_rows = [
                (dataset_id, document_id, c.chunk_index, c.content,
                 c.page_number, c.token_estimate)
                for c in chunks
            ]
            chunk_ids: list[int] = []
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
            chunk_ids = [r[0] for r in cur.fetchall()]

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
            for c, vec in zip(chunks, embeddings):
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

    return len(chunks)


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def list_documents(dataset_id: int) -> list[dict]:
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
        return [dict(zip(cols, row)) for row in cur.fetchall()]


@dataclass
class RetrievedChunk:
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
    vec = _vector_literal(query_vector)
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
        rows = cur.fetchall()

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


def dataset_has_documents(dataset_id: int) -> bool:
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM rag.embeddings WHERE dataset_id = %s LIMIT 1",
            (dataset_id,),
        )
        return cur.fetchone() is not None