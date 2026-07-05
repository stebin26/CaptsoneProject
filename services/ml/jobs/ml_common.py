# Shared helpers for all Phase 3 Level 1 ML jobs: DB connections, feature reads,
# ML-table writes, incremental dataset selection, and model-version registration.

from __future__ import annotations

import json
import os
import numpy as np
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd
import psycopg2
import psycopg2.extras


# Connection settings pulled from env, with the same defaults as the rest of the stack.
def _db_config() -> dict:
    return {
        "host": os.getenv("OPS_POSTGRES_HOST", os.getenv("POSTGRES_HOST", "postgres")),
        "port": int(os.getenv("OPS_POSTGRES_PORT", os.getenv("POSTGRES_PORT", "5432"))),
        "dbname": os.getenv("OPS_POSTGRES_DB", os.getenv("POSTGRES_DB", "ops")),
        "user": os.getenv("OPS_POSTGRES_USER", os.getenv("POSTGRES_USER", "ops")),
        "password": os.getenv("OPS_POSTGRES_PASSWORD", os.getenv("POSTGRES_PASSWORD", "ops")),
    }


# A transactional connection scope: commits on success, rolls back on error, always closes.
@contextmanager
def db_conn():
    conn = psycopg2.connect(**_db_config())
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Incremental dataset selection
# ---------------------------------------------------------------------------

# Resolves which dataset to process: CLI arg > env var > None (None = full batch).
def target_dataset_id(argv: list[str] | None = None) -> int | None:
    if argv:
        for a in argv[1:]:
            a = a.strip()
            if a:
                try:
                    return int(a)
                except ValueError:
                    pass
    env = os.getenv("OPS_TARGET_DATASET_ID", "").strip()
    if env:
        try:
            return int(env)
        except ValueError:
            return None
    return None


# Prints and returns the run mode so job logs clearly show incremental vs full.
def announce_mode(dataset_id: int | None) -> str:
    if dataset_id is None:
        print("[MODE] full — processing all datasets")
        return "all"
    print(f"[MODE] incremental — processing dataset_id={dataset_id} only")
    return str(dataset_id)


# ---------------------------------------------------------------------------
# Feature reads (source = Phase 2 analytics tables)
# ---------------------------------------------------------------------------

# Reads analytics.entity_features, optionally filtered to one dataset.
# entity_ref is aliased to entity_id so the ML jobs speak a single name internally.
def read_entity_features(conn, dataset_id: int | None = None) -> pd.DataFrame:
    sql = """
        SELECT dataset_id, business_name, industry, domain,
               entity_ref AS entity_id,
               metric_name, obs_count, avg_value, std_value, min_value,
               max_value, last_value, trend_slope
        FROM analytics.entity_features
    """
    params: tuple = ()
    if dataset_id is not None:
        sql += " WHERE dataset_id = %s"
        params = (dataset_id,)
    return pd.read_sql(sql, conn, params=params)


# Reads analytics.daily_trend (the time series behind forecasts), optionally filtered.
# day is aliased to trend_date so downstream job code can keep using trend_date.
def read_daily_trend(conn, dataset_id: int | None = None) -> pd.DataFrame:
    sql = """
        SELECT dataset_id, business_name, industry, domain, metric_name,
               day AS trend_date, row_count, sum_value, avg_value
        FROM analytics.daily_trend
    """
    params: tuple = ()
    if dataset_id is not None:
        sql += " WHERE dataset_id = %s"
        params = (dataset_id,)
    df = pd.read_sql(sql, conn, params=params)
    if not df.empty:
        df["trend_date"] = pd.to_datetime(df["trend_date"])
    return df


# ---------------------------------------------------------------------------
# Result writes (target = ml schema)
# ---------------------------------------------------------------------------

# Deletes prior rows for the given dataset scope so a re-run replaces, never duplicates.
def _clear_scope(conn, table: str, dataset_id: int | None) -> None:
    with conn.cursor() as cur:
        if dataset_id is None:
            cur.execute(f"TRUNCATE {table}")
        else:
            cur.execute(f"DELETE FROM {table} WHERE dataset_id = %s", (dataset_id,))


# Generic batch insert from a list of dicts; skips silently when there is nothing to write.
def _insert_rows(conn, table: str, columns: list[str], rows: list[dict]) -> int:
    if not rows:
        return 0
    cols = ", ".join(columns)
    template = "(" + ", ".join(["%s"] * len(columns)) + ")"
    values = [
        tuple(_adapt(r.get(c)) for c in columns)
        for r in rows
    ]
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            f"INSERT INTO {table} ({cols}) VALUES %s",
            values,
            template=template,
        )
    return len(rows)


# Serializes dict/list values to JSON so JSONB columns accept them; passes others through.
def _adapt(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    if isinstance(value, np.generic):   # numpy int64 / float64 / bool_ etc.
        return value.item()
    return value


# Replaces ml.forecasts rows for the scope, then inserts the new forecast rows.
def write_forecasts(conn, dataset_id: int | None, rows: list[dict]) -> int:
    _clear_scope(conn, "ml.forecasts", dataset_id)
    cols = [
        "dataset_id", "business_name", "industry", "domain", "metric_name",
        "forecast_date", "forecast_value", "lower_bound", "upper_bound",
        "model_name", "model_version",
    ]
    return _insert_rows(conn, "ml.forecasts", cols, rows)


# Replaces ml.anomalies rows for the scope, then inserts the new anomaly rows.
def write_anomalies(conn, dataset_id: int | None, rows: list[dict]) -> int:
    _clear_scope(conn, "ml.anomalies", dataset_id)
    cols = [
        "dataset_id", "business_name", "industry", "domain", "entity_id",
        "metric_name", "anomaly_date", "observed_value", "expected_value",
        "deviation", "severity", "method", "model_version",
    ]
    return _insert_rows(conn, "ml.anomalies", cols, rows)


# Replaces ml.risk_scores rows for the scope, then inserts the new risk rows.
def write_risk_scores(conn, dataset_id: int | None, rows: list[dict]) -> int:
    _clear_scope(conn, "ml.risk_scores", dataset_id)
    cols = [
        "dataset_id", "business_name", "industry", "domain", "entity_id",
        "risk_score", "risk_level", "contributing_factors",
        "model_name", "model_version",
    ]
    return _insert_rows(conn, "ml.risk_scores", cols, rows)


# ---------------------------------------------------------------------------
# Model versioning
# ---------------------------------------------------------------------------

# Builds a timestamp-based version string unique per model run.
def make_version(model_name: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{model_name}-{ts}"


# Records one run in ml.model_registry so every result set stays traceable and versioned.
def register_model_version(
    conn,
    model_name: str,
    model_type: str,
    version: str,
    dataset_scope: str,
    params: dict | None = None,
    metrics: dict | None = None,
    row_count: int = 0,
    status: str = "active",
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ml.model_registry
                (model_name, model_type, version, dataset_scope,
                 params, metrics, row_count, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                model_name,
                model_type,
                version,
                dataset_scope,
                _adapt(params or {}),
                _adapt(metrics or {}),
                row_count,
                status,
            ),
        )


# Helper for risk/severity bucketing reused across jobs, kept here so thresholds stay consistent.
def bucket_level(score: float, low: float = 33.0, high: float = 66.0) -> str:
    if score >= high:
        return "high"
    if score >= low:
        return "medium"
    return "low"