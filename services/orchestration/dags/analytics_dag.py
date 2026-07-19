"""Airflow DAG that runs the Spark analytics jobs over the hub.

Submits domain analytics and feature engineering to the Spark cluster. Triggered
by the API after an onboarding completes, or run manually as a batch. A
``dataset_id`` in the trigger conf scopes the run to that one dataset; without
it every dataset is reprocessed.
"""
from __future__ import annotations

import os
from datetime import timedelta

import pendulum
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.sdk import dag

DEFAULT_ARGS = {
    "owner": "ops-platform",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

JOBS_DIR = os.environ.get("SPARK_JOBS_DIR", "/opt/spark/jobs")
SPARK_CONN_ID = "spark_default"

COMMON_ENV = {
    "OPS_POSTGRES_HOST": os.environ.get("OPS_POSTGRES_HOST", "postgres"),
    "OPS_POSTGRES_PORT": os.environ.get("OPS_POSTGRES_PORT", "5432"),
    "OPS_POSTGRES_DB": os.environ.get("OPS_POSTGRES_DB", "ops"),
    "OPS_POSTGRES_USER": os.environ.get("OPS_POSTGRES_USER", "ops"),
    "OPS_POSTGRES_PASSWORD": os.environ.get("OPS_POSTGRES_PASSWORD", "ops"),
}

SPARK_CONF = {
    "spark.executor.memory": "512m",
    "spark.executor.cores": "1",
    "spark.cores.max": "2",
    "spark.driver.memory": "512m",
}

# If a dataset_id is passed in the trigger conf, only that dataset is processed.
# If not (e.g. scheduled batch run), the job processes all datasets.
DATASET_ARG = "{{ dag_run.conf.get('dataset_id', '') if dag_run else '' }}"


@dag(
    dag_id="analytics_pipeline",
    description="Runs Spark analytics and feature-engineering jobs on the hub.",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["analytics", "spark", "phase-2"],
)
def analytics_pipeline():
    """Define the analytics DAG: domain analytics, then feature engineering."""
    domain_analytics = SparkSubmitOperator(
        task_id="domain_analytics",
        application=f"{JOBS_DIR}/domain_analytics.py",
        conn_id=SPARK_CONN_ID,
        py_files=f"{JOBS_DIR}/spark_session.py",
        jars="/opt/spark/jars/postgresql-42.7.4.jar",
        application_args=[DATASET_ARG],
        name="domain-analytics",
        deploy_mode="client",
        conf=SPARK_CONF,
        env_vars=COMMON_ENV,
        verbose=False,
    )

    feature_engineering = SparkSubmitOperator(
        task_id="feature_engineering",
        application=f"{JOBS_DIR}/feature_engineering.py",
        conn_id=SPARK_CONN_ID,
        py_files=f"{JOBS_DIR}/spark_session.py",
        jars="/opt/spark/jars/postgresql-42.7.4.jar",
        application_args=[DATASET_ARG],
        name="feature-engineering",
        deploy_mode="client",
        conf=SPARK_CONF,
        env_vars=COMMON_ENV,
        verbose=False,
    )

    domain_analytics >> feature_engineering


analytics_pipeline()
