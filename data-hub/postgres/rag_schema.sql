-- ============================================================================
-- rag_schema.sql
-- Phase 3 · Level 3 — RAG document store (pgvector)
--
-- Own schema beside hub/analytics/ml. Every row is dataset_id-scoped so one
-- company's documents never surface in another's retrieval. Vector dimension
-- must match the configured embedding model (default all-MiniLM-L6-v2 = 384).
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS rag;


-- ----------------------------------------------------------------------------
-- 1. rag.documents — one row per uploaded file, per dataset
-- Tracks processing lifecycle so the dashboard can show real status.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS rag.documents (
    id              BIGSERIAL PRIMARY KEY,
    dataset_id      BIGINT           NOT NULL,
    business_name   TEXT,
    filename        TEXT             NOT NULL,
    file_type       TEXT,                            -- 'pdf' | 'docx' | 'txt'
    file_size       BIGINT,
    status          TEXT             NOT NULL DEFAULT 'pending',
                                                     -- 'pending' | 'processing' | 'indexed' | 'failed'
    chunk_count     INTEGER          NOT NULL DEFAULT 0,
    error_detail    TEXT,
    uploaded_at     TIMESTAMPTZ      NOT NULL DEFAULT now(),
    indexed_at      TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_rag_docs_dataset ON rag.documents (dataset_id);
CREATE INDEX IF NOT EXISTS idx_rag_docs_status  ON rag.documents (dataset_id, status);


-- ----------------------------------------------------------------------------
-- 2. rag.chunks — text chunks per document
-- dataset_id duplicated here (denormalized) so retrieval filters without a join.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS rag.chunks (
    id              BIGSERIAL PRIMARY KEY,
    dataset_id      BIGINT           NOT NULL,
    document_id     BIGINT           NOT NULL REFERENCES rag.documents(id) ON DELETE CASCADE,
    chunk_index     INTEGER          NOT NULL,
    content         TEXT             NOT NULL,
    page_number     INTEGER,
    token_estimate  INTEGER,
    created_at      TIMESTAMPTZ      NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_rag_chunks_dataset ON rag.chunks (dataset_id);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_doc     ON rag.chunks (document_id);


-- ----------------------------------------------------------------------------
-- 3. rag.embeddings — one vector per chunk
-- Dimension is fixed at 384 to match the default model. If the embedding model
-- changes to a different dimension (e.g. bge-base = 768), this column type must
-- change too and existing embeddings be regenerated.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS rag.embeddings (
    id              BIGSERIAL PRIMARY KEY,
    dataset_id      BIGINT           NOT NULL,
    document_id     BIGINT           NOT NULL REFERENCES rag.documents(id) ON DELETE CASCADE,
    chunk_id        BIGINT           NOT NULL REFERENCES rag.chunks(id) ON DELETE CASCADE,
    embedding       VECTOR(384)      NOT NULL,
    model_name      TEXT,
    created_at      TIMESTAMPTZ      NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_rag_emb_dataset ON rag.embeddings (dataset_id);
CREATE INDEX IF NOT EXISTS idx_rag_emb_chunk   ON rag.embeddings (chunk_id);

-- Approximate nearest-neighbour index (cosine). Scoped queries still filter
-- dataset_id in the WHERE clause; this index accelerates the vector ordering.
CREATE INDEX IF NOT EXISTS idx_rag_emb_vector
    ON rag.embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);