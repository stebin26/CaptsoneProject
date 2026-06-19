CREATE SCHEMA IF NOT EXISTS hub;
CREATE SCHEMA IF NOT EXISTS meta;

-- ============================================================
-- META: Onboarding, profiling, mapping, feature tracking
-- ============================================================

CREATE TABLE IF NOT EXISTS meta.dataset (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    business_name   TEXT NOT NULL,
    industry        TEXT,
    source_filename TEXT NOT NULL,
    row_count       INTEGER,
    uploaded_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS meta.column_profile (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id      BIGINT NOT NULL REFERENCES meta.dataset(id) ON DELETE CASCADE,
    column_name     TEXT NOT NULL,
    data_type       TEXT NOT NULL,
    sample_values   JSONB,
    distinct_count  INTEGER,
    null_count      INTEGER,
    suggested_domain TEXT,
    suggested_metric TEXT,
    mapping_status  TEXT NOT NULL DEFAULT 'suggested'
        CHECK (mapping_status IN ('confirmed', 'skipped', 'suggested', 'added_later')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (dataset_id, column_name)
);

CREATE TABLE IF NOT EXISTS meta.mapping_config (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    business_name   TEXT NOT NULL,
    config          JSONB NOT NULL,
    version         INTEGER NOT NULL DEFAULT 1,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (business_name, version)
);

CREATE TABLE IF NOT EXISTS meta.feature_record (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id      BIGINT NOT NULL REFERENCES meta.dataset(id) ON DELETE CASCADE,
    domain          TEXT NOT NULL,
    feature_name    TEXT NOT NULL,
    source_column   TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'collected'
        CHECK (status IN ('collected', 'skipped', 'added_later')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================
-- HUB: Universal domain tables (industry-agnostic)
-- ============================================================

CREATE TABLE IF NOT EXISTS hub.assets (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id   BIGINT NOT NULL REFERENCES meta.dataset(id) ON DELETE CASCADE,
    entity_ref   TEXT NOT NULL,
    metric_name  TEXT NOT NULL,
    metric_value DOUBLE PRECISION,
    attributes   JSONB,
    recorded_at  TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS hub.operations (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id   BIGINT NOT NULL REFERENCES meta.dataset(id) ON DELETE CASCADE,
    entity_ref   TEXT NOT NULL,
    metric_name  TEXT NOT NULL,
    metric_value DOUBLE PRECISION,
    attributes   JSONB,
    recorded_at  TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS hub.quality (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id   BIGINT NOT NULL REFERENCES meta.dataset(id) ON DELETE CASCADE,
    entity_ref   TEXT NOT NULL,
    metric_name  TEXT NOT NULL,
    metric_value DOUBLE PRECISION,
    attributes   JSONB,
    recorded_at  TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS hub.maintenance (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id   BIGINT NOT NULL REFERENCES meta.dataset(id) ON DELETE CASCADE,
    entity_ref   TEXT NOT NULL,
    metric_name  TEXT NOT NULL,
    metric_value DOUBLE PRECISION,
    attributes   JSONB,
    recorded_at  TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS hub.inventory (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id   BIGINT NOT NULL REFERENCES meta.dataset(id) ON DELETE CASCADE,
    entity_ref   TEXT NOT NULL,
    metric_name  TEXT NOT NULL,
    metric_value DOUBLE PRECISION,
    attributes   JSONB,
    recorded_at  TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS hub.workforce (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id   BIGINT NOT NULL REFERENCES meta.dataset(id) ON DELETE CASCADE,
    entity_ref   TEXT NOT NULL,
    metric_name  TEXT NOT NULL,
    metric_value DOUBLE PRECISION,
    attributes   JSONB,
    recorded_at  TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS hub.finance (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id   BIGINT NOT NULL REFERENCES meta.dataset(id) ON DELETE CASCADE,
    entity_ref   TEXT NOT NULL,
    metric_name  TEXT NOT NULL,
    metric_value DOUBLE PRECISION,
    attributes   JSONB,
    recorded_at  TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS hub.customers (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id   BIGINT NOT NULL REFERENCES meta.dataset(id) ON DELETE CASCADE,
    entity_ref   TEXT NOT NULL,
    metric_name  TEXT NOT NULL,
    metric_value DOUBLE PRECISION,
    attributes   JSONB,
    recorded_at  TIMESTAMPTZ
);

-- ============================================================
-- Indexes for hub reads
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_assets_dataset      ON hub.assets(dataset_id);
CREATE INDEX IF NOT EXISTS idx_operations_dataset  ON hub.operations(dataset_id);
CREATE INDEX IF NOT EXISTS idx_quality_dataset     ON hub.quality(dataset_id);
CREATE INDEX IF NOT EXISTS idx_maintenance_dataset ON hub.maintenance(dataset_id);
CREATE INDEX IF NOT EXISTS idx_inventory_dataset   ON hub.inventory(dataset_id);
CREATE INDEX IF NOT EXISTS idx_workforce_dataset   ON hub.workforce(dataset_id);
CREATE INDEX IF NOT EXISTS idx_finance_dataset     ON hub.finance(dataset_id);
CREATE INDEX IF NOT EXISTS idx_customers_dataset   ON hub.customers(dataset_id);

CREATE INDEX IF NOT EXISTS idx_column_profile_dataset ON meta.column_profile(dataset_id);
CREATE INDEX IF NOT EXISTS idx_feature_record_dataset ON meta.feature_record(dataset_id);