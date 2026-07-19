"""Spark session construction and JDBC access to the hub.

Centralises how every Spark job reaches Postgres: one place builds the session,
one place builds the JDBC URL and credentials, and one place performs reads and
writes. Jobs therefore never hardcode connection details, and switching hosts is
a single environment change.
"""
from __future__ import annotations

import logging
import os

from pyspark.sql import DataFrame, SparkSession

logger = logging.getLogger(__name__)

# Spark submits these jobs as plain scripts, so nothing configures the Python
# logging root for us the way an application server would.
_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"


def configure_job_logging(level: int = logging.INFO) -> None:
    """Configure Python logging for a Spark job entry point.

    Spark controls its own JVM log4j output; this only sets up the Python-side
    logger so job progress messages reach the driver console with a level and a
    timestamp. Existing handlers are left untouched so an outer runner can own
    the configuration instead.

    Args:
        level: Minimum level emitted by the handler.
    """
    if logging.getLogger().handlers:
        return
    logging.basicConfig(level=level, format=_LOG_FORMAT)


PG_JDBC_VERSION = os.environ.get("PG_JDBC_VERSION", "42.7.4")

DOMAIN_TABLES = [
    "hub.assets",
    "hub.operations",
    "hub.quality",
    "hub.maintenance",
    "hub.inventory",
    "hub.workforce",
    "hub.finance",
    "hub.customers",
]


def _jdbc_url() -> str:
    host = os.environ.get("OPS_POSTGRES_HOST", "postgres")
    port = os.environ.get("OPS_POSTGRES_PORT", "5432")
    db = os.environ.get("OPS_POSTGRES_DB", "ops")
    return f"jdbc:postgresql://{host}:{port}/{db}"


def _jdbc_properties() -> dict[str, str]:
    return {
        "user": os.environ.get("OPS_POSTGRES_USER", "ops"),
        "password": os.environ.get("OPS_POSTGRES_PASSWORD", "ops"),
        "driver": "org.postgresql.Driver",
    }


def build_spark(app_name: str = "ops-analytics") -> SparkSession:
    """Build or return the Spark session for a job.

    Args:
        app_name: Application name shown in the Spark UI.

    Returns:
        The active Spark session.

    Raises:
        Exception: If the session cannot be created, for example when the
            configured master is unreachable.
    """
    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "8")
    )
    master = os.environ.get("SPARK_MASTER_URL")
    if master:
        builder = builder.master(master)
    try:
        return builder.getOrCreate()
    except Exception:
        logger.exception(
            "Could not create Spark session %r (master=%s)",
            app_name,
            master or "local default",
            extra={"app_name": app_name, "spark_master": master},
        )
        raise


def read_table(spark: SparkSession, table: str) -> DataFrame:
    """Read a whole table from Postgres over JDBC.

    Args:
        spark: The active Spark session.
        table: Fully qualified table name.

    Returns:
        The table as a DataFrame.

    Raises:
        Exception: If the table cannot be reached over JDBC.
    """
    try:
        return spark.read.jdbc(
            url=_jdbc_url(),
            table=table,
            properties=_jdbc_properties(),
        )
    except Exception:
        logger.exception(
            "JDBC read failed for table %s at %s",
            table,
            _jdbc_url(),
            extra={"table": table, "jdbc_url": _jdbc_url()},
        )
        raise


def read_query(spark: SparkSession, query: str, alias: str = "subq") -> DataFrame:
    """Read the result of a SQL query from Postgres over JDBC.

    Args:
        spark: The active Spark session.
        query: The SQL query to push down.
        alias: Alias for the generated subquery.

    Returns:
        The query result as a DataFrame.

    Raises:
        Exception: If the query cannot be executed over JDBC.
    """
    subquery = f"({query}) AS {alias}"
    try:
        return spark.read.jdbc(
            url=_jdbc_url(),
            table=subquery,
            properties=_jdbc_properties(),
        )
    except Exception:
        # Logged at debug because table_exists() calls this expecting failures.
        logger.debug(
            "JDBC query failed: %s",
            query,
            extra={"query": query, "jdbc_url": _jdbc_url()},
            exc_info=True,
        )
        raise


def table_exists(spark: SparkSession, table: str) -> bool:
    """Check whether a table can be read.

    Used to skip optional tables rather than fail a job when one is absent.

    Args:
        spark: The active Spark session.
        table: Fully qualified table name.

    Returns:
        True if the table could be queried, False otherwise.
    """
    try:
        read_query(spark, f"SELECT 1 FROM {table} LIMIT 1")
        return True
    except Exception:
        logger.debug(
            "Table %s is not readable and will be treated as absent",
            table,
            extra={"table": table},
            exc_info=True,
        )
        return False


def write_table(df: DataFrame, table: str, mode: str = "append") -> None:
    """Write a DataFrame to Postgres over JDBC.

    Args:
        df: The rows to write.
        table: Fully qualified target table.
        mode: Spark save mode.

    Raises:
        Exception: If the write cannot be completed over JDBC.
    """
    try:
        df.write.jdbc(
            url=_jdbc_url(),
            table=table,
            mode=mode,
            properties=_jdbc_properties(),
        )
    except Exception:
        logger.exception(
            "JDBC write failed for table %s (mode=%s)",
            table,
            mode,
            extra={"table": table, "mode": mode, "jdbc_url": _jdbc_url()},
        )
        raise


def replace_dataset_rows(
    df: DataFrame,
    table: str,
    dataset_ids: list[int],
    domain: str | None = None,
) -> None:
    """Replace a dataset's rows in a table, then append the new ones.

    Deleting the previous scope first makes a re-run idempotent: results are
    replaced rather than duplicated.

    Args:
        df: The rows to write.
        table: Fully qualified target table.
        dataset_ids: Datasets whose existing rows should be cleared.
        domain: Optional domain to narrow the delete to.

    Raises:
        Exception: If the delete or the write fails. A write that fails after a
            successful delete is logged at critical level, because the table is
            then missing rows until the job is re-run.
    """
    cleared = bool(dataset_ids)
    if cleared:
        _delete_existing(table, dataset_ids, domain)
    try:
        write_table(df, table, mode="append")
    except Exception:
        if cleared:
            logger.critical(
                "Cleared %d dataset(s) from %s but the replacement write "
                "failed — those rows are missing until this job is re-run",
                len(dataset_ids),
                table,
                extra={
                    "table": table,
                    "dataset_ids": dataset_ids,
                    "domain": domain,
                },
                exc_info=True,
            )
        raise


def _delete_existing(
    table: str,
    dataset_ids: list[int],
    domain: str | None = None,
) -> None:
    import psycopg2

    host = os.environ.get("OPS_POSTGRES_HOST", "postgres")
    port = os.environ.get("OPS_POSTGRES_PORT", "5432")
    db = os.environ.get("OPS_POSTGRES_DB", "ops")
    user = os.environ.get("OPS_POSTGRES_USER", "ops")
    password = os.environ.get("OPS_POSTGRES_PASSWORD", "ops")

    try:
        # Cast every id so a non-numeric value cannot reach the SQL string.
        ids = ",".join(str(int(i)) for i in dataset_ids)
    except (TypeError, ValueError) as exc:
        logger.error(
            "Refusing to delete from %s: dataset ids are not all integers (%r)",
            table,
            dataset_ids,
            extra={"table": table, "dataset_ids": dataset_ids},
        )
        raise ValueError(
            f"dataset_ids must all be integers, got {dataset_ids!r}"
        ) from exc

    where = f"dataset_id IN ({ids})"
    params: list[str] = []
    if domain is not None:
        where += " AND domain = %s"
        params.append(domain)

    try:
        conn = psycopg2.connect(
            host=host, port=port, dbname=db, user=user, password=password
        )
    except psycopg2.Error:
        logger.exception(
            "Could not connect to Postgres at %s:%s/%s to clear %s",
            host,
            port,
            db,
            table,
            extra={"db_host": host, "db_port": port, "db_name": db, "table": table},
        )
        raise

    try:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {table} WHERE {where}", params)
        conn.commit()
    except psycopg2.Error:
        logger.exception(
            "Failed to clear previous rows from %s for dataset(s) %s",
            table,
            dataset_ids,
            extra={"table": table, "dataset_ids": dataset_ids, "domain": domain},
        )
        try:
            conn.rollback()
        except psycopg2.Error:
            logger.exception("Rollback failed after a failed delete on %s", table)
        raise
    finally:
        try:
            conn.close()
        except psycopg2.Error:
            logger.warning(
                "Failed to close the delete connection cleanly", exc_info=True
            )
