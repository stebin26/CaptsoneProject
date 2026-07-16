-- data-hub/postgres/agent_schema.sql
-- Phase 4 — Agent conversation memory.
--
-- Two tables, denormalized and no cross-schema foreign keys, following the same
-- robust pattern as the Phase 2 analytics and Phase 3 ml/rag schemas (so writes
-- never fail on ordering/constraint issues and the schema applies idempotently
-- at API startup alongside the others).
--
-- conversation : one row per chat session (the copilot's session_id).
-- message      : one row per turn (user question or assistant answer), ordered.
--                Assistant turns also store the evidence trail + which dataset
--                the turn was scoped to, so a reloaded conversation is complete.

CREATE SCHEMA IF NOT EXISTS agent;

-- ============================================================
-- conversation — one per session_id
-- ============================================================
CREATE TABLE IF NOT EXISTS agent.conversation (
    session_id     TEXT PRIMARY KEY,
    dataset_id     BIGINT,                     -- last dataset scope used (nullable)
    title          TEXT,                       -- first question, for a session list
    message_count  INTEGER      NOT NULL DEFAULT 0,
    created_at     TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_conversation_updated
    ON agent.conversation (updated_at DESC);

-- ============================================================
-- message — one per turn, ordered within a session
-- ============================================================
CREATE TABLE IF NOT EXISTS agent.message (
    id             BIGSERIAL PRIMARY KEY,
    session_id     TEXT         NOT NULL,      -- soft link to conversation
    turn_index     INTEGER      NOT NULL,      -- 0,1,2… order within the session
    role           TEXT         NOT NULL,      -- 'user' | 'assistant'
    content        TEXT         NOT NULL,      -- the question or the answer
    dataset_id     BIGINT,                     -- dataset the turn was scoped to
    tools_used     JSONB,                      -- assistant: list[str] of tools
    evidence       JSONB,                      -- assistant: full evidence trail
    steps          INTEGER,                    -- assistant: reasoning steps
    elapsed_sec    DOUBLE PRECISION,           -- assistant: wall-clock seconds
    created_at     TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- Fast retrieval of a session's turns in order (the core memory read).
CREATE INDEX IF NOT EXISTS idx_message_session_turn
    ON agent.message (session_id, turn_index);

CREATE INDEX IF NOT EXISTS idx_message_session_created
    ON agent.message (session_id, created_at);