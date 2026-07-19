"""Fire-and-forget trigger for the Airflow analytics DAG.

After a dataset finishes onboarding the analytics pipeline must run, but the
HTTP request must not wait for it. This module authenticates against the Airflow
API and posts a DAG run on a background thread, logging every failure rather
than raising, so a problem in orchestration can never fail the user's upload.
"""
from __future__ import annotations

import datetime
import json
import threading
import urllib.request

from ops_common.logging import get_logger

logger = get_logger(__name__)

_AIRFLOW_BASE = "http://airflow-apiserver:8080"
_TOKEN_URL = f"{_AIRFLOW_BASE}/auth/token"
_DAG_ID = "analytics_pipeline"
_TRIGGER_URL = f"{_AIRFLOW_BASE}/api/v2/dags/{_DAG_ID}/dagRuns"
_USERNAME = "admin"
_PASSWORD = "admin"


def _get_token() -> str | None:
    """Authenticate against Airflow and return an access token.

    Returns:
        The access token, or None if authentication failed.
    """
    data = json.dumps({"username": _USERNAME, "password": _PASSWORD}).encode()
    req = urllib.request.Request(
        _TOKEN_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read())
            return payload.get("access_token")
    except Exception:  # noqa: BLE001
        logger.exception("Failed to get Airflow token")
        return None


def _trigger_dag(dataset_id: int | None) -> None:
    """Post a run of the analytics DAG to the Airflow API.

    Runs on a background thread. Every failure is logged rather than raised, since
    there is no caller left to handle it.

    Args:
        dataset_id: Dataset to scope the run to, or None to run unscoped.
    """
    token = _get_token()
    if not token:
        logger.error("No Airflow token; cannot trigger analytics")
        return

    now = datetime.datetime.now(datetime.UTC).isoformat()
    run_id = f"api__{now}"
    conf: dict[str, int] = {}
    if dataset_id is not None:
        conf["dataset_id"] = int(dataset_id)

    body = json.dumps(
        {
            "dag_run_id": run_id,
            "logical_date": now,
            "conf": conf,
        }
    ).encode()
    req = urllib.request.Request(
        _TRIGGER_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            logger.info(
                "Triggered analytics DAG",
                extra={"status": resp.status, "dataset_id": dataset_id},
            )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to trigger analytics DAG")


def trigger_analytics_async(dataset_id: int | None = None) -> None:
    """Request an analytics DAG run without blocking the caller.

    Starts the trigger on a daemon thread and returns immediately, so an onboarding
    request never waits on Airflow.

    Args:
        dataset_id: Dataset to scope the run to, or None to run unscoped.
    """
    thread = threading.Thread(target=_trigger_dag, args=(dataset_id,), daemon=True)
    thread.start()
    logger.info(
        "Analytics DAG trigger requested in background",
        extra={"dataset_id": dataset_id},
    )
