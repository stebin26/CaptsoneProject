"""Shared helpers for every Level 1 ML job.

Centralises database connections, feature reads from the analytics tables,
writes to the ``ml`` schema, incremental dataset selection, and model-version
registration. Every job replaces rather than appends its scope, so a re-run is
idempotent, and every run is registered so any result set can be traced back to
the code and parameters that produced it.
"""
# Shared helpers for all Phase 3 Level 1 ML jobs: DB connections, feature reads,
# ML-table writes, incremental dataset selection, and model-version registration.

from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

# Format used when a job runs standalone; under Airflow the task handler wins.
_STANDALONE_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"


def configure_job_logging(level: int = logging.INFO) -> None:
    """Configure logging for a job executed outside Airflow.

    Airflow attaches its own handler to the root logger when a task runs, so
    this is a no-op in that case and only takes effect for a direct
    ``python <job>.py`` invocation, where otherwise nothing below WARNING would
    reach the console.

    Args:
        level: Minimum level emitted by the standalone handler.
    """
    if logging.getLogger().handlers:
        return
    logging.basicConfig(level=level, format=_STANDALONE_LOG_FORMAT)


# Connection settings pulled from env, with the same defaults as the rest of the stack.
def _db_config() -> dict:
    raw_port = os.getenv("OPS_POSTGRES_PORT", os.getenv("POSTGRES_PORT", "5432"))
    try:
        port = int(raw_port)
    except (TypeError, ValueError) as exc:
        # A misconfigured port would otherwise surface as a bare int() failure
        # with no hint about which variable is wrong.
        logger.error(
            "Invalid database port %r — set OPS_POSTGRES_PORT to an integer",
            raw_port,
            extra={"raw_port": raw_port},
        )
        raise ValueError(
            f"OPS_POSTGRES_PORT must be an integer, got {raw_port!r}"
        ) from exc

    return {
        "host": os.getenv("OPS_POSTGRES_HOST", os.getenv("POSTGRES_HOST", "postgres")),
        "port": port,
        "dbname": os.getenv("OPS_POSTGRES_DB", os.getenv("POSTGRES_DB", "ops")),
        "user": os.getenv("OPS_POSTGRES_USER", os.getenv("POSTGRES_USER", "ops")),
        "password": os.getenv(
            "OPS_POSTGRES_PASSWORD", os.getenv("POSTGRES_PASSWORD", "ops")
        ),
    }


# A transactional connection scope: commits on success, rolls back on error, always closes.
@contextmanager
def db_conn():
    """Provide a transactional Postgres connection.

    Commits on success, rolls back on error, and always closes. A failure to
    roll back or to close is logged but never allowed to mask the error that
    caused it.

    Yields:
        An open psycopg2 connection.

    Raises:
        psycopg2.Error: If the connection cannot be established.
    """
    config = _db_config()
    try:
        conn = psycopg2.connect(**config)
    except psycopg2.Error:
        logger.exception(
            "Could not connect to Postgres at %s:%s/%s as user %s",
            config["host"],
            config["port"],
            config["dbname"],
            config["user"],
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
        logger.exception("Database transaction failed — rolling back")
        try:
            conn.rollback()
        except psycopg2.Error:
            # Logged, not raised: the original failure is the useful one.
            logger.exception("Rollback failed after a transaction error")
        raise
    finally:
        try:
            conn.close()
        except psycopg2.Error:
            logger.warning(
                "Failed to close the database connection cleanly", exc_info=True
            )


# ---------------------------------------------------------------------------
# Incremental dataset selection
# ---------------------------------------------------------------------------


# Resolves which dataset to process: CLI arg > env var > None (None = full batch).
def _own_argv() -> list[str] | None:
    """Return the process arguments only when a job script is the entry point.

    Under Airflow the jobs are imported and run in-process by the Celery worker,
    so ``sys.argv`` belongs to the worker command line -- ``["airflow", "celery",
    "worker"]`` -- and has nothing to do with this job. Reading it there is not
    merely noisy: a numeric argument anywhere on the worker's command line would
    silently scope the run to the wrong dataset.

    Returns:
        The process arguments when this was launched as ``python <job>.py``, and
        None when the job was imported by something else.
    """
    main_file = getattr(sys.modules.get("__main__"), "__file__", None)
    if not main_file:
        # No entry-point file at all: an interactive session, ``python -c``, or
        # an embedded interpreter. None of those pass a dataset id on argv.
        return None
    try:
        entry_point = Path(main_file).resolve()
        # is_file() matters: a placeholder such as "<stdin>" resolves against the
        # working directory and would otherwise look like a job script.
        launched_as_job = (
            entry_point.is_file()
            and entry_point.parent == Path(__file__).resolve().parent
        )
    except OSError:
        return None
    return sys.argv if launched_as_job else None


def target_dataset_id(argv: list[str] | None = None) -> int | None:
    """Resolve which dataset to process.

    A command-line argument wins over the environment variable; if neither is set
    the job runs the full batch. Command-line arguments are only consulted when
    this process was started as a job script -- see ``_own_argv`` -- so an
    orchestrator's own arguments can never be mistaken for a dataset id.

    A value that is not an integer is ignored rather than failing the job, but
    it is logged, because silently running the full batch when an incremental
    run was intended is the kind of thing that goes unnoticed for weeks.

    Args:
        argv: Argument list to read. Defaults to the process arguments when the
            caller is a job script, and to nothing otherwise.

    Returns:
        The dataset id to scope to, or None for a full batch run.
    """
    if argv is None:
        argv = _own_argv()

    if argv:
        for a in argv[1:]:
            a = a.strip()
            if a:
                try:
                    return int(a)
                except ValueError:
                    logger.warning(
                        "Ignoring non-integer dataset id argument %r",
                        a,
                        extra={"argument": a},
                    )
    env = os.getenv("OPS_TARGET_DATASET_ID", "").strip()
    if env:
        try:
            return int(env)
        except ValueError:
            logger.warning(
                "Ignoring non-integer OPS_TARGET_DATASET_ID %r — running full batch",
                env,
                extra={"env_value": env},
            )
            return None
    return None


# Logs and returns the run mode so job logs clearly show incremental vs full.
def announce_mode(dataset_id: int | None) -> str:
    """Log and return the run mode so job output states its scope.

    Args:
        dataset_id: The dataset being processed, or None for a full batch.

    Returns:
        A human-readable description of the scope.
    """
    if dataset_id is None:
        logger.info("Run mode: full — processing all datasets")
        return "all"
    logger.info(
        "Run mode: incremental — processing dataset_id=%s only",
        dataset_id,
        extra={"dataset_id": dataset_id},
    )
    return str(dataset_id)


# ---------------------------------------------------------------------------
# Feature reads (source = Phase 2 analytics tables)
# ---------------------------------------------------------------------------


# Reads analytics.entity_features, optionally filtered to one dataset.
# entity_ref is aliased to entity_id so the ML jobs speak a single name internally.
def read_entity_features(conn, dataset_id: int | None = None) -> pd.DataFrame:
    """Read entity-level features, optionally scoped to one dataset.

    ``entity_ref`` is aliased to ``entity_id`` so every ML job uses one internal
    name.

    Args:
        conn: An open database connection.
        dataset_id: Dataset to scope to, or None for all datasets.

    Returns:
        The feature rows.

    Raises:
        Exception: If the query fails, for example when the Phase 2 analytics
            tables have not been built yet.
    """
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
    try:
        return pd.read_sql(sql, conn, params=params)
    except Exception:
        logger.exception(
            "Could not read analytics.entity_features (dataset_id=%s) — has the "
            "feature engineering job run?",
            dataset_id,
            extra={"table": "analytics.entity_features", "dataset_id": dataset_id},
        )
        raise


# Reads analytics.daily_trend (the time series behind forecasts), optionally filtered.
# day is aliased to trend_date so downstream job code can keep using trend_date.
def read_daily_trend(conn, dataset_id: int | None = None) -> pd.DataFrame:
    """Read the daily trend series, optionally scoped to one dataset.

    ``day`` is aliased to ``trend_date`` to keep the downstream job code stable.

    Args:
        conn: An open database connection.
        dataset_id: Dataset to scope to, or None for all datasets.

    Returns:
        The daily trend rows.

    Raises:
        Exception: If the query fails or the date column cannot be parsed.
    """
    sql = """
        SELECT dataset_id, business_name, industry, domain, metric_name,
               day AS trend_date, row_count, sum_value, avg_value
        FROM analytics.daily_trend
    """
    params: tuple = ()
    if dataset_id is not None:
        sql += " WHERE dataset_id = %s"
        params = (dataset_id,)
    try:
        df = pd.read_sql(sql, conn, params=params)
    except Exception:
        logger.exception(
            "Could not read analytics.daily_trend (dataset_id=%s) — has the "
            "domain analytics job run?",
            dataset_id,
            extra={"table": "analytics.daily_trend", "dataset_id": dataset_id},
        )
        raise

    if not df.empty:
        try:
            df["trend_date"] = pd.to_datetime(df["trend_date"])
        except (ValueError, TypeError):
            logger.exception(
                "Could not parse trend_date as a datetime for %d row(s)",
                len(df),
                extra={"row_count": len(df)},
            )
            raise
    return df


# ---------------------------------------------------------------------------
# Result writes (target = ml schema)
# ---------------------------------------------------------------------------


# Deletes prior rows for the given dataset scope so a re-run replaces, never duplicates.
def _clear_scope(conn, table: str, dataset_id: int | None) -> None:
    try:
        with conn.cursor() as cur:
            if dataset_id is None:
                cur.execute(f"TRUNCATE {table}")
            else:
                cur.execute(f"DELETE FROM {table} WHERE dataset_id = %s", (dataset_id,))
    except psycopg2.Error:
        logger.exception(
            "Could not clear previous rows from %s (dataset_id=%s) — the write "
            "is abandoned so the table is not left half replaced",
            table,
            dataset_id,
            extra={"table": table, "dataset_id": dataset_id},
        )
        raise


# Generic batch insert from a list of dicts; skips silently when there is nothing to write.
def _insert_rows(conn, table: str, columns: list[str], rows: list[dict]) -> int:
    if not rows:
        return 0
    cols = ", ".join(columns)
    template = "(" + ", ".join(["%s"] * len(columns)) + ")"
    values = [tuple(_adapt(r.get(c)) for c in columns) for r in rows]
    try:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                f"INSERT INTO {table} ({cols}) VALUES %s",
                values,
                template=template,
            )
    except psycopg2.Error:
        logger.exception(
            "Failed to insert %d row(s) into %s",
            len(rows),
            table,
            extra={"table": table, "row_count": len(rows), "columns": columns},
        )
        raise
    return len(rows)


# Serializes dict/list values to JSON so JSONB columns accept them; passes others through.
def _adapt(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    if isinstance(value, np.generic):  # numpy int64 / float64 / bool_ etc.
        return value.item()
    return value


# Replaces ml.forecasts rows for the scope, then inserts the new forecast rows.
def write_forecasts(conn, dataset_id: int | None, rows: list[dict]) -> int:
    """Replace and rewrite the forecast rows for a scope.

    Args:
        conn: An open database connection.
        dataset_id: Dataset scope to clear, or None for all datasets.
        rows: The forecast rows to insert.

    Returns:
        The number of rows written.
    """
    _clear_scope(conn, "ml.forecasts", dataset_id)
    cols = [
        "dataset_id",
        "business_name",
        "industry",
        "domain",
        "metric_name",
        "forecast_date",
        "forecast_value",
        "lower_bound",
        "upper_bound",
        "model_name",
        "model_version",
    ]
    return _insert_rows(conn, "ml.forecasts", cols, rows)


# Replaces ml.anomalies rows for the scope, then inserts the new anomaly rows.
def write_anomalies(conn, dataset_id: int | None, rows: list[dict]) -> int:
    """Replace and rewrite the anomaly rows for a scope.

    Args:
        conn: An open database connection.
        dataset_id: Dataset scope to clear, or None for all datasets.
        rows: The anomaly rows to insert.

    Returns:
        The number of rows written.
    """
    _clear_scope(conn, "ml.anomalies", dataset_id)
    cols = [
        "dataset_id",
        "business_name",
        "industry",
        "domain",
        "entity_id",
        "metric_name",
        "anomaly_date",
        "observed_value",
        "expected_value",
        "deviation",
        "severity",
        "method",
        "model_version",
    ]
    return _insert_rows(conn, "ml.anomalies", cols, rows)


# Replaces ml.risk_scores rows for the scope, then inserts the new risk rows.
def write_risk_scores(conn, dataset_id: int | None, rows: list[dict]) -> int:
    """Replace and rewrite the risk-score rows for a scope.

    Args:
        conn: An open database connection.
        dataset_id: Dataset scope to clear, or None for all datasets.
        rows: The risk-score rows to insert.

    Returns:
        The number of rows written.
    """
    _clear_scope(conn, "ml.risk_scores", dataset_id)
    cols = [
        "dataset_id",
        "business_name",
        "industry",
        "domain",
        "entity_id",
        "risk_score",
        "risk_level",
        "contributing_factors",
        "model_name",
        "model_version",
    ]
    return _insert_rows(conn, "ml.risk_scores", cols, rows)


# ---------------------------------------------------------------------------
# Model versioning
# ---------------------------------------------------------------------------


# Builds a timestamp-based version string unique per model run.
def make_version(model_name: str) -> str:
    """Build a timestamped version string unique to one model run.

    Args:
        model_name: Name of the model being versioned.

    Returns:
        The version string.
    """
    ts = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
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
    """Record one model run in the registry so its results stay traceable.

    Args:
        conn: An open database connection.
        model_name: Name of the model that ran.
        model_type: Family of the model.
        version: Version string for this run.
        dataset_scope: Human-readable description of the processed scope.
        params: Parameters the run used.
        metrics: Summary metrics the run produced.
        row_count: Number of result rows written.
        status: Lifecycle status recorded for this version.

    Raises:
        psycopg2.Error: If the registry row cannot be written.
    """
    try:
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
    except psycopg2.Error:
        logger.exception(
            "Could not register model version %s for %s — results would be "
            "untraceable, so the run is failed rather than left unrecorded",
            version,
            model_name,
            extra={"model_name": model_name, "version": version},
        )
        raise


# Helper for risk/severity bucketing reused across jobs, kept here so thresholds stay consistent.
def bucket_level(score: float, low: float = 33.0, high: float = 66.0) -> str:
    """Bucket a 0-100 score into a low, medium, or high band.

    Kept here so every job applies the same thresholds.

    Args:
        score: The score to bucket.
        low: Upper bound of the low band.
        high: Lower bound of the high band.

    Returns:
        The band name.
    """
    if score >= high:
        return "high"
    if score >= low:
        return "medium"
    return "low"
