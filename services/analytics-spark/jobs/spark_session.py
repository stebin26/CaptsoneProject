from __future__ import annotations

import os

from pyspark.sql import DataFrame, SparkSession

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
    return spark.read.jdbc(
        url=_jdbc_url(),
        table=table,
        properties=_jdbc_properties(),
    )


def read_query(spark: SparkSession, query: str, alias: str = "subq") -> DataFrame:
    subquery = f"({query}) AS {alias}"
    return spark.read.jdbc(
        url=_jdbc_url(),
        table=subquery,
        properties=_jdbc_properties(),
    )


def table_exists(spark: SparkSession, table: str) -> bool:
    try:
        read_query(spark, f"SELECT 1 FROM {table} LIMIT 1")
        return True
    except Exception:
        return False


def write_table(df: DataFrame, table: str, mode: str = "append") -> None:
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