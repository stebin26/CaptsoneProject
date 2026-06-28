from __future__ import annotations

import sys

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

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


def compute_metric_aggregates(df: DataFrame, domain: str) -> DataFrame:
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
    spark = build_spark("domain-analytics")
    spark.sparkContext.setLogLevel("WARN")

    processed = 0
    skipped: list[str] = []
    total_metric_rows = 0
    total_trend_rows = 0

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

        print(
            f"[OK] {domain}: wrote {agg_count} metric rows, "
            f"{trend_count} trend rows (datasets={ds_ids})"
        )

        df.unpersist()

    print("\n========== SUMMARY ==========")
    print(f"domains processed:    {processed}")
    print(f"domains skipped:      {len(skipped)} {skipped if skipped else ''}")
    print(f"metric rows written:  {total_metric_rows}")
    print(f"trend rows written:   {total_trend_rows}")

    spark.stop()


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:  # noqa: BLE001
        print(f"[FATAL] analytics job failed: {exc}", file=sys.stderr)
        raise