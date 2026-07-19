"""Airflow DAG that recomputes the Level 1 ML outputs.

Runs forecasting, anomaly detection, and risk scoring in that order (risk
scoring consumes the anomalies the previous task wrote) and refreshes
``ml.forecasts``, ``ml.anomalies``, and ``ml.risk_scores``. The jobs are
imported and run in-process by PythonOperator -- no Spark and no extra
container. A ``dataset_id`` in the trigger conf scopes the run; without it the
full batch is recomputed.
"""
# ML Orchestration DAG — scheduled recompute of Phase 3 Level 1 outputs.
# Runs forecasting, anomaly detection, and risk scoring (in that order) on the
# latest features and refreshes ml.forecasts / ml.anomalies / ml.risk_scores.

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# The ML jobs are mounted here (see docker-compose ml jobs volume) and imported
# directly, so PythonOperator runs them in-process — no Spark, no extra container.
ML_JOBS_PATH = os.getenv("OPS_ML_JOBS_PATH", "/opt/ml/jobs")
if ML_JOBS_PATH not in sys.path:
    sys.path.insert(0, ML_JOBS_PATH)


# Resolves the dataset_id from the trigger conf; empty/absent means full batch.
def _resolve_dataset_id(**context) -> str:
    conf = (context.get("dag_run").conf or {}) if context.get("dag_run") else {}
    ds = str(conf.get("dataset_id", "") or "").strip()
    return ds


# Sets OPS_TARGET_DATASET_ID for the job, then calls its run(); shared by all 3 tasks.
def _run_job(module_name: str, **context) -> int:
    dataset_id = _resolve_dataset_id(**context)
    if dataset_id:
        os.environ["OPS_TARGET_DATASET_ID"] = dataset_id
    else:
        os.environ.pop("OPS_TARGET_DATASET_ID", None)

    # Imported inside the task so a syntax error in one job can't break DAG parsing.
    import importlib

    module = importlib.import_module(module_name)
    importlib.reload(module)
    return module.run()


default_args = {
    "owner": "ops-platform",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "depends_on_past": False,
}

with DAG(
    dag_id="ml_orchestration_pipeline",
    description="Phase 3 Level 1 — forecasting, anomaly detection, risk scoring",
    default_args=default_args,
    schedule=None,  # triggered on upload (incremental) or manually
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,  # a stuck full run must not overlap incremental runs
    is_paused_upon_creation=True,  # kept paused by default, per Phase 2 lesson
    tags=["phase3", "ml", "level1"],
) as dag:
    # Task 1 — Future column. Fits per-series forecasts from analytics.daily_trend.
    forecasting = PythonOperator(
        task_id="forecasting",
        python_callable=_run_job,
        op_kwargs={"module_name": "forecasting"},
    )

    # Task 2 — Alerts column. Flags anomalies; must precede risk scoring, which reads them.
    anomaly_detection = PythonOperator(
        task_id="anomaly_detection",
        python_callable=_run_job,
        op_kwargs={"module_name": "anomaly_detection"},
    )

    # Task 3 — Assets + Maintenance Future. Unsupervised risk from trend + anomalies.
    risk_scoring = PythonOperator(
        task_id="risk_scoring",
        python_callable=_run_job,
        op_kwargs={"module_name": "risk_scoring"},
    )

    # forecasting is independent; anomaly_detection must run before risk_scoring
    # because risk_scoring reads ml.anomalies for its anomaly component.
    forecasting >> anomaly_detection >> risk_scoring
