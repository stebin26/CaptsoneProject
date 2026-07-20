-- data-hub/postgres/auth_schema.sql
-- Item 6 — Enterprise authentication & RBAC.
--
-- Unlike agent/analytics (append-only, denormalized, no FKs), auth is genuine
-- relational master data: roles, permissions, and their links must stay
-- referentially consistent, so FKs + ON DELETE CASCADE are used *within* this
-- schema. Idempotent — applies at API startup alongside the other schemas.
--
-- Google-ready from day one: password_hash is NULLABLE, auth_provider tracks
-- 'local' vs 'google', google_subject stores the Google 'sub' claim. No schema
-- change needed when OAuth is added later.

CREATE SCHEMA IF NOT EXISTS auth;

-- updated_at auto-touch (shared within this schema)
CREATE OR REPLACE FUNCTION auth.touch_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- roles — fixed set now, extensible later
-- ============================================================
CREATE TABLE IF NOT EXISTS auth.roles (
    id          BIGSERIAL PRIMARY KEY,
    name        TEXT         NOT NULL UNIQUE,
    description TEXT,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- ============================================================
-- permissions — resource:action codes
-- ============================================================
CREATE TABLE IF NOT EXISTS auth.permissions (
    id          BIGSERIAL PRIMARY KEY,
    code        TEXT         NOT NULL UNIQUE,   -- e.g. 'dataset:upload'
    description TEXT,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- ============================================================
-- role_permissions — role <-> permission (M:N)
-- ============================================================
CREATE TABLE IF NOT EXISTS auth.role_permissions (
    role_id       BIGINT NOT NULL REFERENCES auth.roles(id)       ON DELETE CASCADE,
    permission_id BIGINT NOT NULL REFERENCES auth.permissions(id) ON DELETE CASCADE,
    PRIMARY KEY (role_id, permission_id)
);

-- ============================================================
-- users — google-ready (nullable password + provider)
-- ============================================================
CREATE TABLE IF NOT EXISTS auth.users (
    id             BIGSERIAL PRIMARY KEY,
    email          TEXT         NOT NULL UNIQUE,
    full_name      TEXT,
    password_hash  TEXT,                              -- NULL for google-only users
    auth_provider  TEXT         NOT NULL DEFAULT 'local',
    google_subject TEXT         UNIQUE,               -- Google 'sub' claim
    is_active      BOOLEAN      NOT NULL DEFAULT TRUE,
    last_login     TIMESTAMPTZ,
    created_at     TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT auth_provider_chk CHECK (auth_provider IN ('local', 'google')),
    -- a local user must have a password; google users may not
    CONSTRAINT local_needs_password CHECK (
        auth_provider <> 'local' OR password_hash IS NOT NULL
    )
);

DROP TRIGGER IF EXISTS users_touch ON auth.users;
CREATE TRIGGER users_touch
    BEFORE UPDATE ON auth.users
    FOR EACH ROW EXECUTE FUNCTION auth.touch_updated_at();

-- ============================================================
-- user_roles — user <-> role (M:N)
-- ============================================================
CREATE TABLE IF NOT EXISTS auth.user_roles (
    user_id     BIGINT NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    role_id     BIGINT NOT NULL REFERENCES auth.roles(id) ON DELETE CASCADE,
    assigned_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, role_id)
);

-- ============================================================
-- refresh_tokens — DB-stored, revocable sessions
-- ============================================================
CREATE TABLE IF NOT EXISTS auth.refresh_tokens (
    id         BIGSERIAL PRIMARY KEY,
    user_id    BIGINT       NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    token_hash TEXT         NOT NULL UNIQUE,   -- sha256 of raw token, never raw
    user_agent TEXT,
    ip_address TEXT,
    issued_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ  NOT NULL,
    revoked_at TIMESTAMPTZ                     -- NULL = active session
);

CREATE INDEX IF NOT EXISTS idx_refresh_user
    ON auth.refresh_tokens (user_id);
-- fast lookup of a user's live sessions (logout / revoke-all)
CREATE INDEX IF NOT EXISTS idx_refresh_active
    ON auth.refresh_tokens (user_id) WHERE revoked_at IS NULL;

-- ============================================================
-- seed: roles
-- ============================================================
INSERT INTO auth.roles (name, description) VALUES
    ('Admin',   'Full access including user management'),
    ('Analyst', 'Upload data, run analytics and ML, use copilot'),
    ('Viewer',  'Read-only access to dashboards and documents')
ON CONFLICT (name) DO NOTHING;

-- ============================================================
-- seed: permissions
-- ============================================================
INSERT INTO auth.permissions (code, description) VALUES
    ('dataset:read',      'View datasets'),
    ('dataset:upload',    'Upload new datasets'),
    ('dataset:delete',    'Delete datasets'),
    ('mapping:confirm',   'Confirm column-to-domain mappings'),
    ('analytics:read',    'View analytics dashboards'),
    ('ml:read',           'View ML predictions and risk'),
    ('ml:trigger',        'Trigger ML training / DAG runs'),
    ('intelligence:read', 'View cross-domain intelligence'),
    ('documents:read',    'Query documents via RAG'),
    ('documents:upload',  'Upload documents to RAG'),
    ('evaluation:read',   'View model evaluation reports'),
    ('copilot:use',       'Use the agentic copilot'),
    ('user:manage',       'Create, edit, assign roles to users')
ON CONFLICT (code) DO NOTHING;

-- ============================================================
-- seed: role -> permission grants
-- ============================================================

-- Admin: every permission
INSERT INTO auth.role_permissions (role_id, permission_id)
SELECT r.id, p.id
FROM auth.roles r CROSS JOIN auth.permissions p
WHERE r.name = 'Admin'
ON CONFLICT DO NOTHING;

-- Analyst: read everything + upload/confirm/trigger (no delete, no user mgmt)
INSERT INTO auth.role_permissions (role_id, permission_id)
SELECT r.id, p.id
FROM auth.roles r JOIN auth.permissions p ON p.code IN (
    'dataset:read','dataset:upload','mapping:confirm',
    'analytics:read','ml:read','ml:trigger',
    'intelligence:read','documents:read','documents:upload','copilot:use', 'evaluation:read'
)
WHERE r.name = 'Analyst'
ON CONFLICT DO NOTHING;

-- Viewer: read-only + copilot
INSERT INTO auth.role_permissions (role_id, permission_id)
SELECT r.id, p.id
FROM auth.roles r JOIN auth.permissions p ON p.code IN (
    'dataset:read','analytics:read','ml:read',
    'intelligence:read','documents:read','copilot:use'
)
WHERE r.name = 'Viewer'
ON CONFLICT DO NOTHING;