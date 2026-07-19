"""Airflow DAG that auto-onboards CSV files dropped into the upload directory.

Runs on a schedule, discovers unprocessed CSVs, and takes each through the full
onboarding pipeline using the suggested mapping auto-confirmed. This is the
unattended counterpart to the dashboard's interactive upload flow.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

import pendulum
from airflow.sdk import dag, task

# Airflow captures the standard logging module into each task's log, which is
# the only place an unattended run can be inspected after the fact.
logger = logging.getLogger(__name__)

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
            logger.warning(
                "Upload directory %s does not exist; nothing to ingest",
                UPLOAD_DIR,
                extra={"upload_dir": UPLOAD_DIR},
            )
            return []

        try:
            names = sorted(os.listdir(UPLOAD_DIR))
        except OSError:
            # A permission or mount problem would otherwise look identical to an
            # empty directory, and the DAG would quietly do nothing every run.
            logger.exception(
                "Could not list the upload directory %s",
                UPLOAD_DIR,
                extra={"upload_dir": UPLOAD_DIR},
            )
            raise

        found: list[str] = []
        for name in names:
            path = os.path.join(UPLOAD_DIR, name)
            if not os.path.isfile(path):
                continue
            if not name.lower().endswith(SUPPORTED_SUFFIXES):
                continue
            found.append(path)

        logger.info(
            "Discovered %d CSV file(s) to ingest",
            len(found),
            extra={"upload_dir": UPLOAD_DIR, "file_count": len(found)},
        )
        return found

    @task
    def ingest_file(path: str) -> dict:
        from app.pipeline import complete_onboarding, start_onboarding
        from ops_common.db import session_scope

        business_name = _derive_business_name(path)
        logger.info(
            "Onboarding %s as business %r",
            os.path.basename(path),
            business_name,
            extra={"source_file": os.path.basename(path), "business": business_name},
        )

        try:
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
        except Exception:
            # This task is mapped over every discovered file, so one bad CSV
            # fails only its own instance. Naming the file is what makes that
            # failure actionable in the Airflow UI.
            logger.exception(
                "Onboarding failed for %s",
                os.path.basename(path),
                extra={
                    "source_file": os.path.basename(path),
                    "business": business_name,
                },
            )
            raise

        logger.info(
            "Onboarded %s: %d hub rows, %d features collected",
            os.path.basename(path),
            result.hub_rows_written,
            result.features_collected,
            extra={
                "source_file": os.path.basename(path),
                "dataset_id": start.dataset_id,
                "hub_rows_written": result.hub_rows_written,
            },
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
        except OSError as exc:
            # Archiving is housekeeping, so a failure must not fail the run --
            # but an un-archived file is re-ingested on the next schedule, so
            # this is recorded rather than swallowed.
            logger.warning(
                "Could not archive %s; it will be picked up again next run",
                report.get("file"),
                extra={"source_file": report.get("file")},
                exc_info=True,
            )
            report["archived_to"] = None
            report["archive_note"] = f"archive skipped: {exc}"
        return report

    @task
    def summarize(reports: list[dict]) -> dict:
        # Upstream runs with trigger_rule="all_done", so a failed ingest can send
        # a partial or empty report through. Reading defensively keeps the run
        # summary available instead of failing on the last task.
        usable = [r for r in reports if isinstance(r, dict)]
        total_rows = sum((r.get("hub_rows_written") or 0) for r in usable)
        total_collected = sum((r.get("features_collected") or 0) for r in usable)
        businesses = sorted({r["business"] for r in usable if r.get("business")})

        summary = {
            "files_processed": len(usable),
            "total_hub_rows_written": total_rows,
            "total_features_collected": total_collected,
            "businesses": businesses,
        }
        logger.info(
            "Ingestion run complete: %d file(s), %d hub rows",
            len(usable),
            total_rows,
            extra=summary,
        )
        return summary

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
    for index, s in enumerate(suggestions):
        try:
            column = s["column_name"]
        except (KeyError, TypeError):
            # A suggestion without a column name cannot be mapped to anything;
            # it is dropped so one malformed entry does not fail the whole file.
            logger.warning(
                "Skipping suggestion %d with no column_name: %r",
                index,
                s,
                extra={"suggestion_index": index},
            )
            continue

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
