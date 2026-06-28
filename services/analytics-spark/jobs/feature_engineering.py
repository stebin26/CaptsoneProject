from __future__ import annotations

import sys

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from spark_session import (
    DOMAIN_TABLES,
    build_spark,
    read_table,
    replace_dataset_rows,
)


def _domain_name(table: str) -> str:
    return table.split(".")[-1]


def _is_empty(df: DataFrame) -> bool:
    return df.limit(1).count() == 0


def _dataset_ids(df: DataFrame) -> list[int]:
    rows = df.select("dataset_id").distinct().collect()
    return [int(r["dataset_id"]) for r in rows]


def compute_entity_features(df: DataFrame, domain: str) -> DataFrame:
    base = df.filter(F.col("metric_value").isNotNull())

    ordered = Window.partitionBy(
        "dataset_id", "entity_ref", "metric_name"
    ).orderBy("recorded_at")

    with_index = (
        base.withColumn("rn", F.row_number().over(ordered))
        .withColumn(
            "last_value",
            F.last("metric_value").over(
                ordered.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
            ),
        )
    )

    aggregated = with_index.groupBy(
        "dataset_id", "entity_ref", "metric_name"
    ).agg(
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
        (F.col("_sum_xy") - F.col("obs_count") * F.col("_avg_x") * F.col("_avg_y"))
        / F.nullif(
            F.col("_sum_xx") - F.col("obs_count") * F.col("_avg_x") * F.col("_avg_x"),
            F.lit(0.0),
        )
    )

    return (
        aggregated.withColumn("trend_slope", slope)
        .withColumn("domain", F.lit(domain))
        .drop("_avg_x", "_avg_y", "_sum_xy", "_sum_xx", "_sum_x", "_sum_y")
    )


def label_with_business(spark: SparkSession, df: DataFrame) -> DataFrame:
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
    spark = build_spark("feature-engineering")
    spark.sparkContext.setLogLevel("WARN")

    processed = 0
    skipped: list[str] = []
    total_rows = 0

    for table in DOMAIN_TABLES:
        domain = _domain_name(table)
        try:
            df = read_table(spark, table)
        except Exception as exc:  # noqa: BLE001
            print(f"[SKIP] {domain}: could not read table ({exc})")
            skipped.append(domain)
            continue

        if _is_empty(df):
            print(f"[SKIP] {domain}: table is empty")
            skipped.append(domain)
            continue

        df = df.cache()
        ds_ids = _dataset_ids(df)

        features = _select_feature_columns(
            label_with_business(spark, compute_entity_features(df, domain))
        )
        row_count = features.count()

        replace_dataset_rows(
            features, "analytics.entity_features", ds_ids, domain
        )

        total_rows += row_count
        processed += 1
        print(f"[OK] {domain}: wrote {row_count} feature rows (datasets={ds_ids})")

        df.unpersist()

    print("\n========== SUMMARY ==========")
    print(f"domains processed:  {processed}")
    print(f"domains skipped:    {len(skipped)} {skipped if skipped else ''}")
    print(f"feature rows:       {total_rows}")

    spark.stop()


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:  # noqa: BLE001
        print(f"[FATAL] feature engineering job failed: {exc}", file=sys.stderr)
        raise