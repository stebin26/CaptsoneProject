-- ============================================================
-- DuckDB analytics layer
-- Attaches the Postgres hub read-only and exposes fast views.
-- Requires: INSTALL postgres; LOAD postgres;
-- Attach call is issued by ops_common.db before running this file:
--   ATTACH 'postgres:dbname=ops host=postgres user=ops password=...' AS pg (TYPE postgres, READ_ONLY);
-- ============================================================

INSTALL postgres;
LOAD postgres;

-- ------------------------------------------------------------
-- 1. Column review: every profiled column with its mapping state.
--    Drives the "what did we collect / skip" dashboard table.
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_column_review AS
SELECT
    d.id                AS dataset_id,
    d.business_name,
    d.industry,
    d.source_filename,
    cp.column_name,
    cp.data_type,
    cp.distinct_count,
    cp.null_count,
    cp.suggested_domain,
    cp.suggested_metric,
    cp.mapping_status,
    CASE cp.mapping_status
        WHEN 'confirmed'    THEN 'collected'
        WHEN 'added_later'  THEN 'collected'
        WHEN 'skipped'      THEN 'missed'
        ELSE 'pending'
    END                 AS review_state
FROM pg.meta.column_profile cp
JOIN pg.meta.dataset d ON d.id = cp.dataset_id;

-- ------------------------------------------------------------
-- 2. Domain coverage: how many features landed in each domain
--    per dataset. Powers the dashboard's domain summary cards.
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_domain_coverage AS
SELECT
    fr.dataset_id,
    d.business_name,
    fr.domain,
    COUNT(*) FILTER (WHERE fr.status IN ('collected', 'added_later')) AS features_collected,
    COUNT(*) FILTER (WHERE fr.status = 'skipped')                     AS features_skipped,
    COUNT(*)                                                          AS features_total
FROM pg.meta.feature_record fr
JOIN pg.meta.dataset d ON d.id = fr.dataset_id
GROUP BY fr.dataset_id, d.business_name, fr.domain;

-- ------------------------------------------------------------
-- 3. Generated features: the actual collected features list
--    shown to the user as "this is what I pulled from your data".
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_generated_features AS
SELECT
    fr.dataset_id,
    d.business_name,
    fr.domain,
    fr.feature_name,
    fr.source_column,
    fr.status,
    fr.created_at
FROM pg.meta.feature_record fr
JOIN pg.meta.dataset d ON d.id = fr.dataset_id
WHERE fr.status IN ('collected', 'added_later');

-- ------------------------------------------------------------
-- 4. Missed columns: skipped columns with profile context so the
--    user can decide whether to pull them in. Drives MissedColumnPanel.
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_missed_columns AS
SELECT
    cp.dataset_id,
    d.business_name,
    cp.column_name,
    cp.data_type,
    cp.distinct_count,
    cp.null_count,
    cp.sample_values,
    cp.suggested_domain
FROM pg.meta.column_profile cp
JOIN pg.meta.dataset d ON d.id = cp.dataset_id
WHERE cp.mapping_status = 'skipped';

-- ------------------------------------------------------------
-- 5. Unified hub metrics: all 8 domain tables stacked into one
--    long view for cross-domain analytics and aggregate reads.
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_hub_metrics AS
SELECT dataset_id, 'assets'      AS domain, entity_ref, metric_name, metric_value, recorded_at FROM pg.hub.assets
UNION ALL
SELECT dataset_id, 'operations'  AS domain, entity_ref, metric_name, metric_value, recorded_at FROM pg.hub.operations
UNION ALL
SELECT dataset_id, 'quality'     AS domain, entity_ref, metric_name, metric_value, recorded_at FROM pg.hub.quality
UNION ALL
SELECT dataset_id, 'maintenance' AS domain, entity_ref, metric_name, metric_value, recorded_at FROM pg.hub.maintenance
UNION ALL
SELECT dataset_id, 'inventory'   AS domain, entity_ref, metric_name, metric_value, recorded_at FROM pg.hub.inventory
UNION ALL
SELECT dataset_id, 'workforce'   AS domain, entity_ref, metric_name, metric_value, recorded_at FROM pg.hub.workforce
UNION ALL
SELECT dataset_id, 'finance'     AS domain, entity_ref, metric_name, metric_value, recorded_at FROM pg.hub.finance
UNION ALL
SELECT dataset_id, 'customers'   AS domain, entity_ref, metric_name, metric_value, recorded_at FROM pg.hub.customers;

-- ------------------------------------------------------------
-- 6. Domain metric summary: per-domain aggregate stats over the
--    unified view. Used by domain KPI/summary endpoints.
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_domain_metric_summary AS
SELECT
    dataset_id,
    domain,
    metric_name,
    COUNT(*)          AS observations,
    SUM(metric_value) AS metric_sum,
    AVG(metric_value) AS metric_avg,
    MIN(metric_value) AS metric_min,
    MAX(metric_value) AS metric_max
FROM v_hub_metrics
WHERE metric_value IS NOT NULL
GROUP BY dataset_id, domain, metric_name;