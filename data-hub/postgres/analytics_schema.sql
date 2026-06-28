-- ============================================================
-- Analytics schema — Spark-computed results (separate from raw hub.*)
-- Idempotent: safe to re-run.
-- ============================================================

CREATE SCHEMA IF NOT EXISTS analytics;

-- ------------------------------------------------------------
-- Per-domain, per-metric aggregates (one row per dataset+domain+metric)
-- Written by: domain_analytics.py
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS analytics.domain_metrics (
    id                bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id        bigint NOT NULL,
    business_name     text,
    industry          text,
    domain            text NOT NULL,
    metric_name       text NOT NULL,
    row_count         bigint,
    distinct_entities bigint,
    null_value_count  bigint,
    sum_value         double precision,
    avg_value         double precision,
    min_value         double precision,
    max_value         double precision,
    computed_at       timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_domain_metrics_dataset
    ON analytics.domain_metrics (dataset_id);
CREATE INDEX IF NOT EXISTS idx_domain_metrics_domain
    ON analytics.domain_metrics (domain);

-- ------------------------------------------------------------
-- Daily trend (one row per dataset+domain+metric+day)
-- Written by: domain_analytics.py
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS analytics.daily_trend (
    id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id    bigint NOT NULL,
    business_name text,
    industry      text,
    domain        text NOT NULL,
    metric_name   text NOT NULL,
    day           date NOT NULL,
    row_count     bigint,
    sum_value     double precision,
    avg_value     double precision,
    computed_at   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_daily_trend_dataset
    ON analytics.daily_trend (dataset_id);
CREATE INDEX IF NOT EXISTS idx_daily_trend_domain_day
    ON analytics.daily_trend (domain, day);

-- ------------------------------------------------------------
-- ML feature table (row-level engineered features per entity)
-- Written by: feature_engineering.py
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS analytics.entity_features (
    id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id      bigint NOT NULL,
    business_name   text,
    industry        text,
    domain          text NOT NULL,
    entity_ref      text NOT NULL,
    metric_name     text NOT NULL,
    obs_count       bigint,
    avg_value       double precision,
    std_value       double precision,
    min_value       double precision,
    max_value       double precision,
    last_value      double precision,
    trend_slope     double precision,
    computed_at     timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_entity_features_dataset
    ON analytics.entity_features (dataset_id);
CREATE INDEX IF NOT EXISTS idx_entity_features_entity
    ON analytics.entity_features (domain, entity_ref);