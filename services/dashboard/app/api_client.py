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


def _auth(token: str | None) -> dict[str, str]:
    # Every protected endpoint needs a bearer token. Callbacks read it from the
    # ACCESS_TOKEN store and pass it in; this turns it into a header.
    return {"Authorization": f"Bearer {token}"} if token else {}


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
        resp = requests.get(_url("/../../health"), timeout=(3, 5))
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
    token: str | None = None,
) -> dict[str, Any]:
    files = {"file": (filename, file_bytes, "text/csv")}
    data = {"business_name": business_name}
    if industry:
        data["industry"] = industry
    try:
        resp = requests.post(_url("/onboard/start"), files=files, data=data,
                             headers=_auth(token), timeout=_TIMEOUT)
        return _handle(resp)
    except requests.RequestException as exc:
        raise APIError(f"Failed to start onboarding: {exc}") from exc


def confirm_onboarding(
    dataset_id: int,
    stored_path: str,
    columns: list[dict[str, Any]],
    token: str | None = None,
) -> dict[str, Any]:
    body = {"dataset_id": dataset_id, "stored_path": stored_path, "columns": columns}
    try:
        resp = requests.post(_url("/onboard/confirm"), json=body,
                             headers=_auth(token), timeout=_TIMEOUT)
        return _handle(resp)
    except requests.RequestException as exc:
        raise APIError(f"Failed to confirm onboarding: {exc}") from exc


# ============================================================
# Feature review
# ============================================================

def feature_review(dataset_id: int, token: str | None = None) -> dict[str, Any]:
    try:
        resp = requests.get(_url(f"/features/{dataset_id}/review"),
                            headers=_auth(token), timeout=_TIMEOUT)
        return _handle(resp)
    except requests.RequestException as exc:
        raise APIError(f"Failed to load feature review: {exc}") from exc


def add_feature(
    dataset_id: int,
    column_name: str,
    domain: str,
    metric_name: str,
    token: str | None = None,
) -> dict[str, Any]:
    body = {"dataset_id": dataset_id, "column_name": column_name,
            "domain": domain, "metric_name": metric_name}
    try:
        resp = requests.post(_url("/features/add"), json=body,
                             headers=_auth(token), timeout=_TIMEOUT)
        return _handle(resp)
    except requests.RequestException as exc:
        raise APIError(f"Failed to add feature: {exc}") from exc


# ============================================================
# Domains + hub data (for charts)
# ============================================================

def list_domains(token: str | None = None) -> list[dict[str, Any]]:
    try:
        resp = requests.get(_url("/domains"), headers=_auth(token), timeout=_TIMEOUT)
        return _handle(resp)
    except requests.RequestException as exc:
        raise APIError(f"Failed to list domains: {exc}") from exc


def dataset_summary(dataset_id: int, token: str | None = None) -> dict[str, Any]:
    try:
        resp = requests.get(_url(f"/datasets/{dataset_id}/summary"),
                            headers=_auth(token), timeout=_TIMEOUT)
        return _handle(resp)
    except requests.RequestException as exc:
        raise APIError(f"Failed to load dataset summary: {exc}") from exc


def domain_data(dataset_id: int, domain: str, limit: int = 200,
                token: str | None = None) -> dict[str, Any]:
    try:
        resp = requests.get(_url(f"/datasets/{dataset_id}/domains/{domain}"),
                            params={"limit": limit}, headers=_auth(token),
                            timeout=_TIMEOUT)
        return _handle(resp)
    except requests.RequestException as exc:
        raise APIError(f"Failed to load domain data: {exc}") from exc


def list_datasets(token: str | None = None) -> list[dict[str, Any]]:
    try:
        resp = requests.get(_url("/datasets"), headers=_auth(token), timeout=_TIMEOUT)
        return _handle(resp)
    except requests.RequestException as exc:
        raise APIError(f"Failed to list datasets: {exc}") from exc


# ============================================================
# Analytics (Spark-computed results)
# ============================================================

def analytics_overview(dataset_id: int, token: str | None = None) -> dict[str, Any]:
    try:
        resp = requests.get(_url(f"/analytics/{dataset_id}/overview"),
                            headers=_auth(token), timeout=_TIMEOUT)
        return _handle(resp)
    except requests.RequestException as exc:
        raise APIError(f"Failed to load analytics overview: {exc}") from exc


def analytics_metrics(dataset_id: int, token: str | None = None) -> list[dict[str, Any]]:
    try:
        resp = requests.get(_url(f"/analytics/{dataset_id}/metrics"),
                            headers=_auth(token), timeout=_TIMEOUT)
        return _handle(resp)
    except requests.RequestException as exc:
        raise APIError(f"Failed to load analytics metrics: {exc}") from exc


def analytics_trend(
    dataset_id: int,
    domain: str | None = None,
    metric_name: str | None = None,
    token: str | None = None,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {}
    if domain:
        params["domain"] = domain
    if metric_name:
        params["metric_name"] = metric_name
    try:
        resp = requests.get(_url(f"/analytics/{dataset_id}/trend"), params=params,
                            headers=_auth(token), timeout=_TIMEOUT)
        return _handle(resp)
    except requests.RequestException as exc:
        raise APIError(f"Failed to load analytics trend: {exc}") from exc


def analytics_features(
    dataset_id: int,
    domain: str | None = None,
    limit: int = 200,
    token: str | None = None,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"limit": limit}
    if domain:
        params["domain"] = domain
    try:
        resp = requests.get(_url(f"/analytics/{dataset_id}/features"), params=params,
                            headers=_auth(token), timeout=_TIMEOUT)
        return _handle(resp)
    except requests.RequestException as exc:
        raise APIError(f"Failed to load analytics features: {exc}") from exc


# ============================================================
# ML (Phase 3 — forecasts, anomalies, risk scores)
# ============================================================

def ml_overview(dataset_id: int, token: str | None = None) -> dict[str, Any]:
    try:
        resp = requests.get(_url(f"/ml/{dataset_id}/overview"),
                            headers=_auth(token), timeout=_TIMEOUT)
        return _handle(resp)
    except requests.RequestException as exc:
        raise APIError(f"Failed to load ML overview: {exc}") from exc


def ml_forecasts(
    dataset_id: int,
    domain: str | None = None,
    metric_name: str | None = None,
    token: str | None = None,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {}
    if domain:
        params["domain"] = domain
    if metric_name:
        params["metric_name"] = metric_name
    try:
        resp = requests.get(_url(f"/ml/{dataset_id}/forecasts"), params=params,
                            headers=_auth(token), timeout=_TIMEOUT)
        return _handle(resp)
    except requests.RequestException as exc:
        raise APIError(f"Failed to load forecasts: {exc}") from exc


def ml_anomalies(
    dataset_id: int,
    domain: str | None = None,
    severity: str | None = None,
    limit: int = 500,
    token: str | None = None,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"limit": limit}
    if domain:
        params["domain"] = domain
    if severity:
        params["severity"] = severity
    try:
        resp = requests.get(_url(f"/ml/{dataset_id}/anomalies"), params=params,
                            headers=_auth(token), timeout=_TIMEOUT)
        return _handle(resp)
    except requests.RequestException as exc:
        raise APIError(f"Failed to load anomalies: {exc}") from exc


def ml_risk_scores(
    dataset_id: int,
    domain: str | None = None,
    risk_level: str | None = None,
    token: str | None = None,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {}
    if domain:
        params["domain"] = domain
    if risk_level:
        params["risk_level"] = risk_level
    try:
        resp = requests.get(_url(f"/ml/{dataset_id}/risk-scores"), params=params,
                            headers=_auth(token), timeout=_TIMEOUT)
        return _handle(resp)
    except requests.RequestException as exc:
        raise APIError(f"Failed to load risk scores: {exc}") from exc


def ml_domain_intelligence(dataset_id: int, domain: str,
                           token: str | None = None) -> dict[str, Any]:
    try:
        resp = requests.get(_url(f"/ml/{dataset_id}/domain/{domain}"),
                            headers=_auth(token), timeout=_TIMEOUT)
        return _handle(resp)
    except requests.RequestException as exc:
        raise APIError(f"Failed to load domain intelligence: {exc}") from exc


# ============================================================
# Intelligence (Phase 3 Level 2 — cross-domain insights)
# ============================================================

def intelligence(dataset_id: int, token: str | None = None) -> dict[str, Any]:
    try:
        resp = requests.get(_url(f"/intelligence/{dataset_id}"),
                            headers=_auth(token), timeout=(5, 180))
        return _handle(resp)
    except requests.RequestException as exc:
        raise APIError(f"Failed to load intelligence: {exc}") from exc


# ============================================================
# RAG (Phase 3 Level 3 — document assistant)
# ============================================================

def rag_upload(dataset_id: int, files: list[tuple[str, bytes]],
               business_name: str | None = None,
               token: str | None = None) -> dict[str, Any]:
    multipart = [("files", (name, data, "application/octet-stream")) for name, data in files]
    form: dict[str, Any] = {}
    if business_name:
        form["business_name"] = business_name
    try:
        resp = requests.post(_url(f"/rag/{dataset_id}/upload"), files=multipart,
                             data=form, headers=_auth(token), timeout=(5, 300))
        return _handle(resp)
    except requests.RequestException as exc:
        raise APIError(f"Failed to upload documents: {exc}") from exc


def rag_documents(dataset_id: int, token: str | None = None) -> list[dict[str, Any]]:
    try:
        resp = requests.get(_url(f"/rag/{dataset_id}/documents"),
                            headers=_auth(token), timeout=_TIMEOUT)
        return _handle(resp)
    except requests.RequestException as exc:
        raise APIError(f"Failed to load documents: {exc}") from exc


def rag_query(dataset_id: int, question: str, top_k: int | None = None,
              token: str | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"question": question}
    if top_k is not None:
        body["top_k"] = top_k
    try:
        resp = requests.post(_url(f"/rag/{dataset_id}/query"), json=body,
                             headers=_auth(token), timeout=(5, 120))
        return _handle(resp)
    except requests.RequestException as exc:
        raise APIError(f"Failed to query documents: {exc}") from exc


def rag_delete_document(dataset_id: int, document_id: int,
                        token: str | None = None) -> dict[str, Any]:
    try:
        resp = requests.delete(_url(f"/rag/{dataset_id}/documents/{document_id}"),
                               headers=_auth(token), timeout=_TIMEOUT)
        return _handle(resp)
    except requests.RequestException as exc:
        raise APIError(f"Failed to delete document: {exc}") from exc


# ============================================================
# Agent (Phase 4 — AI copilot)
# ============================================================

def agent_ask(
    question: str,
    dataset_id: int | None = None,
    session_id: str | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"question": question}
    if dataset_id is not None:
        body["dataset_id"] = dataset_id
    if session_id is not None:
        body["session_id"] = session_id
    try:
        resp = requests.post(_url("/agent/ask"), json=body,
                             headers=_auth(token), timeout=(5, 240))
        return _handle(resp)
    except requests.RequestException as exc:
        raise APIError(f"Agent request failed: {exc}") from exc


def agent_health(token: str | None = None) -> dict[str, Any]:
    try:
        resp = requests.get(_url("/agent/health"), headers=_auth(token),
                            timeout=(3, 10))
        return _handle(resp)
    except requests.RequestException as exc:
        raise APIError(f"Agent health check failed: {exc}") from exc


# ============================================================
# Executive summary (single aggregating endpoint)
# ============================================================

def executive_summary(dataset_id: int, token: str | None = None) -> dict[str, Any]:
    try:
        resp = requests.get(_url(f"/executive/{dataset_id}/summary"),
                            headers=_auth(token), timeout=_TIMEOUT)
        return _handle(resp)
    except requests.RequestException as exc:
        raise APIError(f"Failed to load executive summary: {exc}") from exc


# ============================================================
# Auth (Item 6 — login, identity, refresh, logout)
# ============================================================

def login(email: str, password: str) -> dict[str, Any]:
    try:
        resp = requests.post(_url("/auth/login"),
                             json={"email": email, "password": password},
                             timeout=_TIMEOUT)
        return _handle(resp)
    except requests.RequestException as exc:
        raise APIError(f"Login failed: {exc}") from exc


def auth_me(access_token: str) -> dict[str, Any]:
    try:
        resp = requests.get(_url("/auth/me"),
                            headers={"Authorization": f"Bearer {access_token}"},
                            timeout=_TIMEOUT)
        return _handle(resp)
    except requests.RequestException as exc:
        raise APIError(f"Failed to load identity: {exc}") from exc


def refresh_access(refresh_token: str) -> dict[str, Any]:
    try:
        resp = requests.post(_url("/auth/refresh"),
                             json={"refresh_token": refresh_token}, timeout=_TIMEOUT)
        return _handle(resp)
    except requests.RequestException as exc:
        raise APIError(f"Token refresh failed: {exc}") from exc


def logout(access_token: str, refresh_token: str) -> None:
    try:
        resp = requests.post(_url("/auth/logout"),
                             headers={"Authorization": f"Bearer {access_token}"},
                             json={"refresh_token": refresh_token}, timeout=_TIMEOUT)
        _handle(resp)
    except requests.RequestException as exc:
        raise APIError(f"Logout failed: {exc}") from exc