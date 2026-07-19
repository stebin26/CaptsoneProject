"""Airflow DAG that auto-onboards CSV files dropped into the upload directory.

Runs on a schedule, discovers unprocessed CSVs, and takes each through the full
onboarding pipeline using the suggested mapping auto-confirmed. This is the
unattended counterpart to the dashboard's interactive upload flow.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

import pendulum
from airflow.sdk import dag, task

DEFAULT_ARGS = {
    "owner": "ops-platform",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}

UPLOAD_DIR = os.environ.get("OPS_UPLOAD_DIR", "/data/uploads")
PROCESSED_DIR = os.path.join(UPLOAD_DIR, "_processed")
SUPPORTED_SUFFIXES = (".csv",)


@dag(
    dag_id="ingestion_pipeline",
    description="Scheduled auto-onboarding ingestion of CSV drops into the universal hub.",
    schedule="*/15 * * * *",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["ingestion", "phase-2"],
)
def ingestion_pipeline():
    """Define the ingestion DAG: discover CSV drops, then onboard each one."""

    @task
    def discover_files() -> list[str]:
        if not os.path.isdir(UPLOAD_DIR):
            return []
        found: list[str] = []
        for name in sorted(os.listdir(UPLOAD_DIR)):
            path = os.path.join(UPLOAD_DIR, name)
            if not os.path.isfile(path):
                continue
            if not name.lower().endswith(SUPPORTED_SUFFIXES):
                continue
            found.append(path)
        return found

    @task
    def ingest_file(path: str) -> dict:
        from app.pipeline import complete_onboarding, start_onboarding
        from ops_common.db import session_scope

        business_name = _derive_business_name(path)

        with session_scope() as session:
            start = start_onboarding(
                session=session,
                csv_path=path,
                business_name=business_name,
            )

            confirmed = _auto_confirm(start.suggestions)

            result = complete_onboarding(
                session=session,
                dataset_id=start.dataset_id,
                csv_path=path,
                confirmed=confirmed,
            )

        return {
            "file": os.path.basename(path),
            "business": business_name,
            "dataset_id": start.dataset_id,
            "hub_rows_written": result.hub_rows_written,
            "features_collected": result.features_collected,
            "features_skipped": result.features_skipped,
            "validation_ok": result.validation.get("ok", True),
        }

    @task(trigger_rule="all_done", retries=0)
    def archive_file(report: dict) -> dict:
        import shutil

        archive_root = os.environ.get("OPS_ARCHIVE_DIR", "/opt/airflow/processed")
        try:
            os.makedirs(archive_root, exist_ok=True)
            src = os.path.join(UPLOAD_DIR, report["file"])
            if os.path.isfile(src):
                stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
                dst = os.path.join(archive_root, f"{stamp}__{report['file']}")
                shutil.copy2(src, dst)
                os.remove(src)
                report["archived_to"] = dst
            else:
                report["archived_to"] = None
                report["archive_note"] = "source not found"
        except Exception as exc:
            report["archived_to"] = None
            report["archive_note"] = f"archive skipped: {exc}"
        return report

    @task
    def summarize(reports: list[dict]) -> dict:
        total_rows = sum((r.get("hub_rows_written") or 0) for r in reports)
        total_collected = sum((r.get("features_collected") or 0) for r in reports)
        return {
            "files_processed": len(reports),
            "total_hub_rows_written": total_rows,
            "total_features_collected": total_collected,
            "businesses": sorted({r["business"] for r in reports}),
        }

    files = discover_files()
    reports = ingest_file.expand(path=files)
    archived = archive_file.expand(report=reports)
    summarize(archived)


def _derive_business_name(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    cleaned = stem.replace("_", " ").replace("-", " ").strip()
    return cleaned.title() or "Unknown Business"


def _auto_confirm(suggestions: list[dict]) -> list[dict]:
    confirmed: list[dict] = []
    for s in suggestions:
        column = s["column_name"]
        domain = s.get("suggested_domain")
        role = s.get("role", "skip")

        is_time = any(
            tok in column.lower() for tok in ("date", "time", "timestamp", "_at", "_on")
        )

        if is_time:
            confirmed.append(
                {
                    "column_name": column,
                    "domain": None,
                    "metric_name": None,
                    "role": "skip",
                }
            )
            continue

        if role == "skip" or not domain:
            continue

        confirmed.append(
            {
                "column_name": column,
                "domain": domain,
                "metric_name": s.get("suggested_metric"),
                "role": role,
            }
        )
    return confirmed


ingestion_pipeline()
