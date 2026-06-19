from __future__ import annotations

import os
from typing import Any

import requests

API_BASE_URL = os.environ.get("OPS_API_BASE_URL", "http://api:8000/api/v1")
_TIMEOUT = (5, 60)  # (connect, read) seconds


class APIError(Exception):
    def __init__(self, message: str, status_code: int | None = None, detail: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


def _handle(response: requests.Response) -> Any:
    try:
        payload = response.json()
    except ValueError:
        payload = None

    if not response.ok:
        detail = None
        if isinstance(payload, dict):
            detail = payload.get("detail")
        raise APIError(
            message=detail or f"Request failed ({response.status_code})",
            status_code=response.status_code,
            detail=detail,
        )
    return payload


def _url(path: str) -> str:
    return f"{API_BASE_URL.rstrip('/')}/{path.lstrip('/')}"


# ============================================================
# Health
# ============================================================

def health() -> dict[str, Any]:
    try:
        resp = requests.get(
            _url("/../../health"),  # /health is outside /api/v1
            timeout=(3, 5),
        )
        return _handle(resp)
    except requests.RequestException as exc:
        raise APIError(f"API unreachable: {exc}") from exc


# ============================================================
# Onboarding
# ============================================================

def start_onboarding(
    file_bytes: bytes,
    filename: str,
    business_name: str,
    industry: str | None = None,
) -> dict[str, Any]:
    files = {"file": (filename, file_bytes, "text/csv")}
    data = {"business_name": business_name}
    if industry:
        data["industry"] = industry

    try:
        resp = requests.post(
            _url("/onboard/start"),
            files=files,
            data=data,
            timeout=_TIMEOUT,
        )
        return _handle(resp)
    except requests.RequestException as exc:
        raise APIError(f"Failed to start onboarding: {exc}") from exc


def confirm_onboarding(
    dataset_id: int,
    stored_path: str,
    columns: list[dict[str, Any]],
) -> dict[str, Any]:
    body = {
        "dataset_id": dataset_id,
        "stored_path": stored_path,
        "columns": columns,
    }
    try:
        resp = requests.post(
            _url("/onboard/confirm"),
            json=body,
            timeout=_TIMEOUT,
        )
        return _handle(resp)
    except requests.RequestException as exc:
        raise APIError(f"Failed to confirm onboarding: {exc}") from exc


# ============================================================
# Feature review
# ============================================================

def feature_review(dataset_id: int) -> dict[str, Any]:
    try:
        resp = requests.get(
            _url(f"/features/{dataset_id}/review"),
            timeout=_TIMEOUT,
        )
        return _handle(resp)
    except requests.RequestException as exc:
        raise APIError(f"Failed to load feature review: {exc}") from exc


def add_feature(
    dataset_id: int,
    column_name: str,
    domain: str,
    metric_name: str,
) -> dict[str, Any]:
    body = {
        "dataset_id": dataset_id,
        "column_name": column_name,
        "domain": domain,
        "metric_name": metric_name,
    }
    try:
        resp = requests.post(
            _url("/features/add"),
            json=body,
            timeout=_TIMEOUT,
        )
        return _handle(resp)
    except requests.RequestException as exc:
        raise APIError(f"Failed to add feature: {exc}") from exc


# ============================================================
# Domains + hub data (for charts)
# ============================================================

def list_domains() -> list[dict[str, Any]]:
    try:
        resp = requests.get(_url("/domains"), timeout=_TIMEOUT)
        return _handle(resp)
    except requests.RequestException as exc:
        raise APIError(f"Failed to list domains: {exc}") from exc


def dataset_summary(dataset_id: int) -> dict[str, Any]:
    try:
        resp = requests.get(
            _url(f"/datasets/{dataset_id}/summary"),
            timeout=_TIMEOUT,
        )
        return _handle(resp)
    except requests.RequestException as exc:
        raise APIError(f"Failed to load dataset summary: {exc}") from exc


def domain_data(dataset_id: int, domain: str, limit: int = 200) -> dict[str, Any]:
    try:
        resp = requests.get(
            _url(f"/datasets/{dataset_id}/domains/{domain}"),
            params={"limit": limit},
            timeout=_TIMEOUT,
        )
        return _handle(resp)
    except requests.RequestException as exc:
        raise APIError(f"Failed to load domain data: {exc}") from exc

def list_datasets() -> list[dict[str, Any]]:
    try:
        resp = requests.get(_url("/datasets"), timeout=_TIMEOUT)
        return _handle(resp)
    except requests.RequestException as exc:
        raise APIError(f"Failed to list datasets: {exc}") from exc
