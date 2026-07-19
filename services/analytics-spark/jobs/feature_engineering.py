"""Spark job computing entity-level features from the hub.

For every entity and metric this derives the observation count, average,
spread, extremes, latest value, and trend slope, and writes the result to
``analytics.entity_features``. These features are what the ML layer consumes,
so this job is the boundary between raw hub readings and modelling.
"""
from __future__ import annotations

import logging
import os
import sys

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from spark_session import (
    DOMAIN_TABLES,
    build_spark,
    configure_job_logging,
    read_table,
    replace_dataset_rows,
)

logger = logging.getLogger(__name__)


def _target_dataset_id() -> int | None:
    # A bad value is ignored rather than failing the job, but it is logged:
    # silently running the full batch when an incremental run was intended is
    # the kind of thing that goes unnoticed for weeks.
    if len(sys.argv) > 1 and sys.argv[1].strip():
        raw = sys.argv[1].strip()
        try:
            return int(raw)
        except ValueError:
            logger.warning(
                "Ignoring non-integer dataset id argument %r",
                raw,
                extra={"argument": raw},
            )
    env = os.environ.get("OPS_TARGET_DATASET_ID", "").strip()
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


def _domain_name(table: str) -> str:
    return table.split(".")[-1]


def _is_empty(df: DataFrame) -> bool:
    return df.limit(1).count() == 0


def _dataset_ids(df: DataFrame) -> list[int]:
    rows = df.select("dataset_id").distinct().collect()
    ids: list[int] = []
    for r in rows:
        value = r["dataset_id"]
        try:
            ids.append(int(value))
        except (TypeError, ValueError):
            # A null or malformed id cannot scope a delete, so it is dropped
            # rather than allowed to widen the replace to the whole table.
            logger.warning(
                "Skipping unusable dataset_id %r found in the hub rows",
                value,
                extra={"dataset_id": value},
            )
    return ids


def compute_entity_features(df: DataFrame, domain: str) -> DataFrame:
    """Derive per-entity, per-metric features for one domain.

    Rows without a value are excluded, and the trend slope is computed over the
    time-ordered readings for each entity and metric.

    Args:
        df: The domain's hub readings.
        domain: Name of the domain being processed.

    Returns:
        One feature row per dataset, entity, and metric.
    """
    base = df.filter(F.col("metric_value").isNotNull())

    ordered = Window.partitionBy("dataset_id", "entity_ref", "metric_name").orderBy(
        "recorded_at"
    )

    with_index = base.withColumn("rn", F.row_number().over(ordered)).withColumn(
        "last_value",
        F.last("metric_value").over(
            ordered.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
        ),
    )

    aggregated = with_index.groupBy("dataset_id", "entity_ref", "metric_name").agg(
        F.count("*").alias("obs_count"),
        F.avg("metric_value").alias("avg_value"),
        F.stddev("metric_value").alias("std_value"),
        F.min("metric_value").alias("min_value"),
        F.max("metric_value").alias("max_value"),
        F.first("last_value").alias("last_value"),
        F.avg("rn").alias("_avg_x"),
        F.avg("metric_value").alias("_avg_y"),
        F.sum(F.col("rn") * F.col("metric_value")).alias("_sum_xy"),
        F.sum(F.col("rn") * F.col("rn")).alias("_sum_xx"),
        F.sum("rn").alias("_sum_x"),
        F.sum("metric_value").alias("_sum_y"),
    )

    slope = (
        F.col("_sum_xy") - F.col("obs_count") * F.col("_avg_x") * F.col("_avg_y")
    ) / F.nullif(
        F.col("_sum_xx") - F.col("obs_count") * F.col("_avg_x") * F.col("_avg_x"),
        F.lit(0.0),
    )

    return (
        aggregated.withColumn("trend_slope", slope)
        .withColumn("domain", F.lit(domain))
        .drop("_avg_x", "_avg_y", "_sum_xy", "_sum_xx", "_sum_x", "_sum_y")
    )


def label_with_business(spark: SparkSession, df: DataFrame) -> DataFrame:
    """Attach business name and industry to feature rows.

    Args:
        spark: The active Spark session.
        df: The feature rows to label.

    Returns:
        The rows joined to their dataset's business details.
    """
    datasets = read_table(spark, "meta.dataset").select(
        F.col("id").alias("dataset_id"),
        F.col("business_name"),
        F.col("industry"),
    )
    return df.join(datasets, on="dataset_id", how="left")


def _select_feature_columns(df: DataFrame) -> DataFrame:
    return df.select(
        "dataset_id",
        "business_name",
        "industry",
        "domain",
        "entity_ref",
        "metric_name",
        "obs_count",
        "avg_value",
        "std_value",
        "min_value",
        "max_value",
        "last_value",
        "trend_slope",
    )


def run() -> None:
    """Run the feature engineering job across every hub domain.

    Computes entity-level features per domain and replaces the previous results for
    the processed scope.
    """
    target_id = _target_dataset_id()
    spark = build_spark("feature-engineering")
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
    total_rows = 0

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
        # Computation is guarded so one malformed domain is skipped rather than
        # aborting the whole run. The write below is deliberately NOT guarded:
        # a half-completed replace is a data-integrity problem, not something to
        # skip past silently.
        try:
            ds_ids = _dataset_ids(df)

            features = _select_feature_columns(
                label_with_business(spark, compute_entity_features(df, domain))
            )
            row_count = features.count()
        except Exception:
            logger.exception(
                "Skipping domain %s: feature computation failed",
                domain,
                extra={"domain": domain, "table": table},
            )
            skipped.append(domain)
            df.unpersist()
            continue

        replace_dataset_rows(features, "analytics.entity_features", ds_ids, domain)

        total_rows += row_count
        processed += 1
        logger.info(
            "Domain %s: wrote %d feature rows",
            domain,
            row_count,
            extra={
                "domain": domain,
                "feature_rows": row_count,
                "dataset_ids": ds_ids,
            },
        )

        df.unpersist()

    logger.info(
        "Feature engineering complete: %d domain(s) processed, %d skipped, "
        "%d feature rows written",
        processed,
        len(skipped),
        total_rows,
        extra={
            "domains_processed": processed,
            "domains_skipped": skipped,
            "feature_rows_written": total_rows,
        },
    )

    spark.stop()


if __name__ == "__main__":
    configure_job_logging()
    try:
        run()
    except Exception:  # noqa: BLE001
        logger.exception("Feature engineering job failed")
        raise
