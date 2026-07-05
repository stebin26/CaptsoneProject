-- ============================================================================
-- ml_schema.sql
-- Phase 3 · Level 1 — Machine Learning result store
--
-- Creates the `ml` schema and its 4 result tables. Mirrors the Phase 2
-- analytics_schema.sql pattern: denormalized, no foreign keys, idempotent
-- (IF NOT EXISTS) so it runs safely on every startup and the Python ML jobs
-- can write results without brittle join constraints.
--
-- Which job writes where:
--     forecasting.py        -> ml.forecasts        (the "Future" column)
--     anomaly_detection.py  -> ml.anomalies        (the "Alerts" column)
--     risk_scoring.py       -> ml.risk_scores      (Assets + Maintenance Future)
--     (all jobs)            -> ml.model_registry   (version / lifecycle)
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS ml;


-- ----------------------------------------------------------------------------
-- 1. ml.forecasts  — the "Future" column
-- One row per dataset + domain + metric + future date. Produced by
-- forecasting.py from analytics.daily_trend. lower/upper bound carry the
-- confidence interval so the dashboard can shade uncertainty; both are nullable
-- because the linear-trend fallback may not produce an interval.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ml.forecasts (
    id              BIGSERIAL PRIMARY KEY,
    dataset_id      BIGINT           NOT NULL,
    business_name   TEXT,
    industry        TEXT,
    domain          TEXT             NOT NULL,
    metric_name     TEXT             NOT NULL,
    forecast_date   DATE             NOT NULL,
    forecast_value  DOUBLE PRECISION,
    lower_bound     DOUBLE PRECISION,
    upper_bound     DOUBLE PRECISION,
    model_name      TEXT,                            -- 'holt_winters' | 'linear_trend' | ...
    model_version   TEXT,                            -- soft link to ml.model_registry.version
    generated_at    TIMESTAMPTZ      NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_forecasts_dataset ON ml.forecasts (dataset_id);
CREATE INDEX IF NOT EXISTS idx_forecasts_domain  ON ml.forecasts (dataset_id, domain);


-- ----------------------------------------------------------------------------
-- 2. ml.anomalies  — the "Alerts" column
-- One row per flagged reading. Produced by anomaly_detection.py from
-- analytics.entity_features / daily_trend. Stores observed vs expected plus a
-- severity bucket the dashboard colors on. anomaly_date is nullable because
-- some anomalies are per-entity aggregate, not tied to a single day.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ml.anomalies (
    id              BIGSERIAL PRIMARY KEY,
    dataset_id      BIGINT           NOT NULL,
    business_name   TEXT,
    industry        TEXT,
    domain          TEXT             NOT NULL,
    entity_id       TEXT,
    metric_name     TEXT             NOT NULL,
    anomaly_date    DATE,
    observed_value  DOUBLE PRECISION,
    expected_value  DOUBLE PRECISION,
    deviation       DOUBLE PRECISION,                -- z-score or model anomaly score
    severity        TEXT,                            -- 'low' | 'medium' | 'high'
    method          TEXT,                            -- 'zscore' | 'isolation_forest'
    model_version   TEXT,
    detected_at     TIMESTAMPTZ      NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_anomalies_dataset ON ml.anomalies (dataset_id);
CREATE INDEX IF NOT EXISTS idx_anomalies_domain  ON ml.anomalies (dataset_id, domain);
CREATE INDEX IF NOT EXISTS idx_anomalies_sev     ON ml.anomalies (dataset_id, severity);


-- ----------------------------------------------------------------------------
-- 3. ml.risk_scores  — Assets + Maintenance "Future" (unsupervised)
-- One row per dataset + domain + entity. Produced by risk_scoring.py. No labels
-- exist, so the score is derived from anomaly severity + trend signals and
-- normalized to 0–100. contributing_factors explains the score in plain terms
-- (kept as JSONB so the dashboard/inference engine can read the drivers).
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ml.risk_scores (
    id                    BIGSERIAL PRIMARY KEY,
    dataset_id            BIGINT           NOT NULL,
    business_name         TEXT,
    industry              TEXT,
    domain                TEXT             NOT NULL,
    entity_id             TEXT,
    risk_score            DOUBLE PRECISION,          -- 0–100, higher = more at risk
    risk_level            TEXT,                      -- 'low' | 'medium' | 'high'
    contributing_factors  JSONB,                     -- {"trend_slope": -0.8, "anomaly_count": 3}
    model_name            TEXT,
    model_version         TEXT,
    generated_at          TIMESTAMPTZ      NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_risk_dataset ON ml.risk_scores (dataset_id);
CREATE INDEX IF NOT EXISTS idx_risk_domain  ON ml.risk_scores (dataset_id, domain);
CREATE INDEX IF NOT EXISTS idx_risk_level   ON ml.risk_scores (dataset_id, risk_level);


-- ----------------------------------------------------------------------------
-- 4. ml.model_registry  — version + lifecycle tracking
-- One row each time a job runs and produces results. This is what makes Point 4
-- (the orchestration DAG) honest: every scheduled run records which model and
-- params produced the current outputs, so results stay traceable and versioned
-- even for the statistical / unsupervised models that have no heavy trained
-- artifact. dataset_scope records incremental (a specific id) vs full ('all').
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ml.model_registry (
    id              BIGSERIAL PRIMARY KEY,
    model_name      TEXT             NOT NULL,       -- 'forecasting' | 'anomaly_detection' | 'risk_scoring'
    model_type      TEXT,                            -- 'statistical' | 'unsupervised'
    version         TEXT             NOT NULL,
    dataset_scope   TEXT,                            -- specific dataset_id (incremental) or 'all' (batch)
    params          JSONB,                           -- hyperparameters used
    metrics         JSONB,                           -- evaluation metrics, if any
    row_count       INTEGER,                         -- how many result rows this run wrote
    status          TEXT             NOT NULL DEFAULT 'active',   -- 'active' | 'archived'
    created_at      TIMESTAMPTZ      NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_registry_name ON ml.model_registry (model_name, created_at DESC);