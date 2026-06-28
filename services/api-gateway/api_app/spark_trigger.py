from __future__ import annotations

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


def _trigger_dag() -> None:
    token = _get_token()
    if not token:
        logger.error("No Airflow token; cannot trigger analytics")
        return

    import datetime
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    run_id = f"api__{now}"
    body = json.dumps({
        "dag_run_id": run_id,
        "logical_date": now,
    }).encode()
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
            logger.info("Triggered analytics DAG", extra={"status": resp.status})
    except Exception:  # noqa: BLE001
        logger.exception("Failed to trigger analytics DAG")


def trigger_analytics_async() -> None:
    thread = threading.Thread(target=_trigger_dag, daemon=True)
    thread.start()
    logger.info("Analytics DAG trigger requested in background")