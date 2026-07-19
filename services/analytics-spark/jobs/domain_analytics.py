"""Spark job computing per-domain metric summaries and daily trends.

Reads all eight hub tables, aggregates each domain's metrics into summary and
daily-trend shapes, labels them with the owning business, and writes the result
to ``analytics.domain_metrics`` and ``analytics.daily_trend``. Running with a
dataset id processes only that dataset; running without one reprocesses every
dataset.
"""
from __future__ import annotations

import logging
import os
import sys

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from spark_session import (
    DOMAIN_TABLES,
    build_spark,
    configure_job_logging,
    read_table,
    replace_dataset_rows,
)

logger = logging.getLogger(__name__)


def _target_dataset_id() -> int | None:
    if len(sys.argv) > 1 and sys.argv[1].strip():
        try:
            return int(sys.argv[1].strip())
        except ValueError:
            pass
    env = os.environ.get("OPS_TARGET_DATASET_ID", "").strip()
    if env:
        try:
            return int(env)
        except ValueError:
            pass
    return None


def _domain_name(table: str) -> str:
    return table.split(".")[-1]


def _is_empty(df: DataFrame) -> bool:
    return df.limit(1).count() == 0


def _dataset_ids(df: DataFrame) -> list[int]:
    rows = df.select("dataset_id").distinct().collect()
    return [int(r["dataset_id"]) for r in rows]


def compute_metric_aggregates(df: DataFrame, domain: str) -> DataFrame:
    """Aggregate one domain's readings into per-metric summary rows.

    Args:
        df: The domain's hub readings.
        domain: Name of the domain being aggregated.

    Returns:
        One summary row per dataset and metric.
    """
    return (
        df.groupBy("dataset_id", "metric_name")
        .agg(
            F.count("*").alias("row_count"),
            F.countDistinct("entity_ref").alias("distinct_entities"),
            F.sum(F.when(F.col("metric_value").isNull(), 1).otherwise(0)).alias(
                "null_value_count"
            ),
            F.sum("metric_value").alias("sum_value"),
            F.avg("metric_value").alias("avg_value"),
            F.min("metric_value").alias("min_value"),
            F.max("metric_value").alias("max_value"),
        )
        .withColumn("domain", F.lit(domain))
    )


def compute_daily_trend(df: DataFrame, domain: str) -> DataFrame:
    """Aggregate one domain's readings into daily trend points.

    Rows without a timestamp are excluded, since they cannot be placed on a
    timeline.

    Args:
        df: The domain's hub readings.
        domain: Name of the domain being aggregated.

    Returns:
        One trend row per dataset, metric, and day.
    """
    return (
        df.filter(F.col("recorded_at").isNotNull())
        .withColumn("day", F.to_date("recorded_at"))
        .groupBy("dataset_id", "metric_name", "day")
        .agg(
            F.count("*").alias("row_count"),
            F.sum("metric_value").alias("sum_value"),
            F.avg("metric_value").alias("avg_value"),
        )
        .withColumn("domain", F.lit(domain))
    )


def label_with_business(spark: SparkSession, df: DataFrame) -> DataFrame:
    """Attach business name and industry to analytics rows.

    Args:
        spark: The active Spark session.
        df: The analytics rows to label.

    Returns:
        The rows joined to their dataset's business details.
    """
    datasets = read_table(spark, "meta.dataset").select(
        F.col("id").alias("dataset_id"),
        F.col("business_name"),
        F.col("industry"),
    )
    return df.join(datasets, on="dataset_id", how="left")


def _select_metric_columns(df: DataFrame) -> DataFrame:
    return df.select(
        "dataset_id",
        "business_name",
        "industry",
        "domain",
        "metric_name",
        "row_count",
        "distinct_entities",
        "null_value_count",
        "sum_value",
        "avg_value",
        "min_value",
        "max_value",
    )


def _select_trend_columns(df: DataFrame) -> DataFrame:
    return df.select(
        "dataset_id",
        "business_name",
        "industry",
        "domain",
        "metric_name",
        "day",
        "row_count",
        "sum_value",
        "avg_value",
    )


def run() -> None:
    """Run the domain analytics job across every hub domain.

    Computes summaries and daily trends per domain and replaces the previous
    results for the processed scope.
    """
    target_id = _target_dataset_id()
    spark = build_spark("domain-analytics")
    spark.sparkContext.setLogLevel("WARN")

    if target_id is not None:
        logger.info(
            "Run mode: incremental — processing dataset_id=%s only",
            target_id,
            extra={"dataset_id": target_id, "mode": "incremental"},
        )
    else:
        logger.info("Run mode: full — processing all datasets", extra={"mode": "full"})

    processed = 0
    skipped: list[str] = []
    total_metric_rows = 0
    total_trend_rows = 0

    for table in DOMAIN_TABLES:
        domain = _domain_name(table)
        try:
            df = read_table(spark, table)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Skipping domain %s: table %s could not be read (%s)",
                domain,
                table,
                exc,
                extra={"domain": domain, "table": table},
                exc_info=True,
            )
            skipped.append(domain)
            continue

        if target_id is not None:
            df = df.filter(F.col("dataset_id") == target_id)

        if _is_empty(df):
            logger.info(
                "Skipping domain %s: no rows in scope for this run",
                domain,
                extra={"domain": domain},
            )
            skipped.append(domain)
            continue

        df = df.cache()
        ds_ids = _dataset_ids(df)

        aggregates = _select_metric_columns(
            label_with_business(spark, compute_metric_aggregates(df, domain))
        )
        trend = _select_trend_columns(
            label_with_business(spark, compute_daily_trend(df, domain))
        )

        agg_count = aggregates.count()
        trend_count = trend.count()

        replace_dataset_rows(aggregates, "analytics.domain_metrics", ds_ids, domain)
        replace_dataset_rows(trend, "analytics.daily_trend", ds_ids, domain)

        total_metric_rows += agg_count
        total_trend_rows += trend_count
        processed += 1

        logger.info(
            "Domain %s: wrote %d metric rows and %d trend rows",
            domain,
            agg_count,
            trend_count,
            extra={
                "domain": domain,
                "metric_rows": agg_count,
                "trend_rows": trend_count,
                "dataset_ids": ds_ids,
            },
        )

        df.unpersist()

    logger.info(
        "Domain analytics complete: %d domain(s) processed, %d skipped, "
        "%d metric rows and %d trend rows written",
        processed,
        len(skipped),
        total_metric_rows,
        total_trend_rows,
        extra={
            "domains_processed": processed,
            "domains_skipped": skipped,
            "metric_rows_written": total_metric_rows,
            "trend_rows_written": total_trend_rows,
        },
    )

    spark.stop()


if __name__ == "__main__":
    configure_job_logging()
    try:
        run()
    except Exception:  # noqa: BLE001
        logger.exception("Domain analytics job failed")
        raise
