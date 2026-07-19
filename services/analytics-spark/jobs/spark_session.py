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
    """
    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "8")
    )
    master = os.environ.get("SPARK_MASTER_URL")
    if master:
        builder = builder.master(master)
    return builder.getOrCreate()


def read_table(spark: SparkSession, table: str) -> DataFrame:
    """Read a whole table from Postgres over JDBC.

    Args:
        spark: The active Spark session.
        table: Fully qualified table name.

    Returns:
        The table as a DataFrame.
    """
    return spark.read.jdbc(
        url=_jdbc_url(),
        table=table,
        properties=_jdbc_properties(),
    )


def read_query(spark: SparkSession, query: str, alias: str = "subq") -> DataFrame:
    """Read the result of a SQL query from Postgres over JDBC.

    Args:
        spark: The active Spark session.
        query: The SQL query to push down.
        alias: Alias for the generated subquery.

    Returns:
        The query result as a DataFrame.
    """
    subquery = f"({query}) AS {alias}"
    return spark.read.jdbc(
        url=_jdbc_url(),
        table=subquery,
        properties=_jdbc_properties(),
    )


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
    """
    df.write.jdbc(
        url=_jdbc_url(),
        table=table,
        mode=mode,
        properties=_jdbc_properties(),
    )


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
    """
    if dataset_ids:
        _delete_existing(table, dataset_ids, domain)
    write_table(df, table, mode="append")


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

    ids = ",".join(str(int(i)) for i in dataset_ids)
    where = f"dataset_id IN ({ids})"
    params: list[str] = []
    if domain is not None:
        where += " AND domain = %s"
        params.append(domain)

    conn = psycopg2.connect(
        host=host, port=port, dbname=db, user=user, password=password
    )
    try:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {table} WHERE {where}", params)
        conn.commit()
    finally:
        conn.close()
