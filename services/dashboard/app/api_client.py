"""HTTP client for the API gateway.

Every call the dashboard makes to the backend goes through this module, so the
base URL, timeouts, bearer-token header, and error translation live in exactly
one place. Failed responses are raised as ``APIError`` carrying the API's own
detail message, which lets callbacks show the user what actually went wrong
instead of a generic failure.
"""
from __future__ import annotations

import os
from typing import Any

import requests

from app.logging_setup import get_logger

logger = get_logger(__name__)

API_BASE_URL = os.environ.get("OPS_API_BASE_URL", "http://api:8000/api/v1")
_TIMEOUT = (5, 60)  # (connect, read) seconds


class APIError(Exception):
    """Raised when an API call fails, carrying the status and detail."""
    def __init__(
        self, message: str, status_code: int | None = None, detail: Any = None
    ) -> None:
        """Store the failure message alongside the status code and detail.

        Args:
            message: Human-readable description of the failure.
            status_code: HTTP status returned, when there was one.
            detail: The API's own error detail, when provided.
        """
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
        # A non-JSON body from a proxy or an error page is not fatal on its own;
        # the status code below still decides whether this is a failure.
        payload = None
        if response.ok:
            logger.warning(
                "Backend returned a non-JSON body for %s",
                response.url,
                extra={"endpoint": response.url, "status_code": response.status_code},
            )

    if not response.ok:
        detail = None
        if isinstance(payload, dict):
            detail = payload.get("detail")
        # Logged here, in the one place every response passes through, so a
        # failing endpoint is visible even when the callback shows only a
        # friendly message to the user.
        logger.warning(
            "Backend call failed: %s %s -- %s",
            response.status_code,
            response.url,
            detail or "no detail returned",
            extra={
                "endpoint": response.url,
                "status_code": response.status_code,
                "detail": detail,
            },
        )
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
    """Check that the API gateway is reachable.

    Returns:
        The gateway's health payload.

    Raises:
        APIError: If the request fails or the API is unreachable.
    """
    try:
        resp = requests.get(_url("/../../health"), timeout=(3, 5))
        return _handle(resp)
    except requests.RequestException as exc:
        logger.warning(
            "Could not reach the API gateway at %s: %s",
            API_BASE_URL,
            exc,
            extra={"api_base_url": API_BASE_URL},
        )
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
    """Upload a CSV and start onboarding.

    Args:
        file_bytes: Raw contents of the uploaded file.
        filename: Original filename.
        business_name: Business the dataset belongs to.
        industry: Optional industry label.
        token: Caller's access token.

    Returns:
        The registered dataset and its suggested column mapping.

    Raises:
        APIError: If the request fails or the API is unreachable.
    """
    files = {"file": (filename, file_bytes, "text/csv")}
    data = {"business_name": business_name}
    if industry:
        data["industry"] = industry
    try:
        resp = requests.post(
            _url("/onboard/start"),
            files=files,
            data=data,
            headers=_auth(token),
            timeout=_TIMEOUT,
        )
        return _handle(resp)
    except requests.RequestException as exc:
        logger.warning(
            "Could not reach the API gateway at %s: %s",
            API_BASE_URL,
            exc,
            extra={"api_base_url": API_BASE_URL},
        )
        raise APIError(f"Failed to start onboarding: {exc}") from exc


def confirm_onboarding(
    dataset_id: int,
    stored_path: str,
    columns: list[dict[str, Any]],
    token: str | None = None,
) -> dict[str, Any]:
    """Submit the confirmed column mapping and load the dataset.

    Args:
        dataset_id: Dataset being confirmed.
        stored_path: Path of the stored upload.
        columns: The confirmed column decisions.
        token: Caller's access token.

    Returns:
        Load counts and the validation report.

    Raises:
        APIError: If the request fails or the API is unreachable.
    """
    body = {"dataset_id": dataset_id, "stored_path": stored_path, "columns": columns}
    try:
        resp = requests.post(
            _url("/onboard/confirm"), json=body, headers=_auth(token), timeout=_TIMEOUT
        )
        return _handle(resp)
    except requests.RequestException as exc:
        logger.warning(
            "Could not reach the API gateway at %s: %s",
            API_BASE_URL,
            exc,
            extra={"api_base_url": API_BASE_URL},
        )
        raise APIError(f"Failed to confirm onboarding: {exc}") from exc


# ============================================================
# Feature review
# ============================================================


def feature_review(dataset_id: int, token: str | None = None) -> dict[str, Any]:
    """Fetch a dataset's collected, skipped, and coverage breakdown.

    Args:
        dataset_id: Dataset to review.
        token: Caller's access token.

    Returns:
        The feature-review payload.

    Raises:
        APIError: If the request fails or the API is unreachable.
    """
    try:
        resp = requests.get(
            _url(f"/features/{dataset_id}/review"),
            headers=_auth(token),
            timeout=_TIMEOUT,
        )
        return _handle(resp)
    except requests.RequestException as exc:
        logger.warning(
            "Could not reach the API gateway at %s: %s",
            API_BASE_URL,
            exc,
            extra={"api_base_url": API_BASE_URL},
        )
        raise APIError(f"Failed to load feature review: {exc}") from exc


def add_feature(
    dataset_id: int,
    column_name: str,
    domain: str,
    metric_name: str,
    token: str | None = None,
) -> dict[str, Any]:
    """Add a previously skipped column as a new feature.

    Args:
        dataset_id: Dataset to update.
        column_name: Source column to add.
        domain: Domain to map it into.
        metric_name: Metric name to record it as.
        token: Caller's access token.

    Returns:
        The number of features added.

    Raises:
        APIError: If the request fails or the API is unreachable.
    """
    body = {
        "dataset_id": dataset_id,
        "column_name": column_name,
        "domain": domain,
        "metric_name": metric_name,
    }
    try:
        resp = requests.post(
            _url("/features/add"), json=body, headers=_auth(token), timeout=_TIMEOUT
        )
        return _handle(resp)
    except requests.RequestException as exc:
        logger.warning(
            "Could not reach the API gateway at %s: %s",
            API_BASE_URL,
            exc,
            extra={"api_base_url": API_BASE_URL},
        )
        raise APIError(f"Failed to add feature: {exc}") from exc


# ============================================================
# Domains + hub data (for charts)
# ============================================================


def list_domains(token: str | None = None) -> list[dict[str, Any]]:
    """Fetch the catalog of universal domains and their features.

    Args:
        token: Caller's access token.

    Returns:
        One entry per domain.

    Raises:
        APIError: If the request fails or the API is unreachable.
    """
    try:
        resp = requests.get(_url("/domains"), headers=_auth(token), timeout=_TIMEOUT)
        return _handle(resp)
    except requests.RequestException as exc:
        logger.warning(
            "Could not reach the API gateway at %s: %s",
            API_BASE_URL,
            exc,
            extra={"api_base_url": API_BASE_URL},
        )
        raise APIError(f"Failed to list domains: {exc}") from exc


def dataset_summary(dataset_id: int, token: str | None = None) -> dict[str, Any]:
    """Fetch a dataset's per-domain metric summary.

    Args:
        dataset_id: Dataset to summarize.
        token: Caller's access token.

    Returns:
        The summary payload.

    Raises:
        APIError: If the request fails or the API is unreachable.
    """
    try:
        resp = requests.get(
            _url(f"/datasets/{dataset_id}/summary"),
            headers=_auth(token),
            timeout=_TIMEOUT,
        )
        return _handle(resp)
    except requests.RequestException as exc:
        logger.warning(
            "Could not reach the API gateway at %s: %s",
            API_BASE_URL,
            exc,
            extra={"api_base_url": API_BASE_URL},
        )
        raise APIError(f"Failed to load dataset summary: {exc}") from exc


def domain_data(
    dataset_id: int, domain: str, limit: int = 200, token: str | None = None
) -> dict[str, Any]:
    """Fetch entity-level readings for one domain of a dataset.

    Args:
        dataset_id: Dataset to read.
        domain: Domain to read from.
        limit: Maximum readings to return.
        token: Caller's access token.

    Returns:
        The readings for that domain.

    Raises:
        APIError: If the request fails or the API is unreachable.
    """
    try:
        resp = requests.get(
            _url(f"/datasets/{dataset_id}/domains/{domain}"),
            params={"limit": limit},
            headers=_auth(token),
            timeout=_TIMEOUT,
        )
        return _handle(resp)
    except requests.RequestException as exc:
        logger.warning(
            "Could not reach the API gateway at %s: %s",
            API_BASE_URL,
            exc,
            extra={"api_base_url": API_BASE_URL},
        )
        raise APIError(f"Failed to load domain data: {exc}") from exc


def list_datasets(token: str | None = None) -> list[dict[str, Any]]:
    """Fetch every dataset available to the caller.

    Args:
        token: Caller's access token.

    Returns:
        One entry per dataset.

    Raises:
        APIError: If the request fails or the API is unreachable.
    """
    try:
        resp = requests.get(_url("/datasets"), headers=_auth(token), timeout=_TIMEOUT)
        return _handle(resp)
    except requests.RequestException as exc:
        logger.warning(
            "Could not reach the API gateway at %s: %s",
            API_BASE_URL,
            exc,
            extra={"api_base_url": API_BASE_URL},
        )
        raise APIError(f"Failed to list datasets: {exc}") from exc


# ============================================================
# Analytics (Spark-computed results)
# ============================================================


def analytics_overview(dataset_id: int, token: str | None = None) -> dict[str, Any]:
    """Fetch a dataset's analytics overview.

    Args:
        dataset_id: Dataset to summarize.
        token: Caller's access token.

    Returns:
        The overview payload.

    Raises:
        APIError: If the request fails or the API is unreachable.
    """
    try:
        resp = requests.get(
            _url(f"/analytics/{dataset_id}/overview"),
            headers=_auth(token),
            timeout=_TIMEOUT,
        )
        return _handle(resp)
    except requests.RequestException as exc:
        logger.warning(
            "Could not reach the API gateway at %s: %s",
            API_BASE_URL,
            exc,
            extra={"api_base_url": API_BASE_URL},
        )
        raise APIError(f"Failed to load analytics overview: {exc}") from exc


def analytics_metrics(
    dataset_id: int, token: str | None = None
) -> list[dict[str, Any]]:
    """Fetch a dataset's aggregate domain metrics.

    Args:
        dataset_id: Dataset to read.
        token: Caller's access token.

    Returns:
        One summary row per domain metric.

    Raises:
        APIError: If the request fails or the API is unreachable.
    """
    try:
        resp = requests.get(
            _url(f"/analytics/{dataset_id}/metrics"),
            headers=_auth(token),
            timeout=_TIMEOUT,
        )
        return _handle(resp)
    except requests.RequestException as exc:
        logger.warning(
            "Could not reach the API gateway at %s: %s",
            API_BASE_URL,
            exc,
            extra={"api_base_url": API_BASE_URL},
        )
        raise APIError(f"Failed to load analytics metrics: {exc}") from exc


def analytics_trend(
    dataset_id: int,
    domain: str | None = None,
    metric_name: str | None = None,
    token: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch daily trend points, optionally filtered.

    Args:
        dataset_id: Dataset to read.
        domain: Optional domain filter.
        metric_name: Optional metric filter.
        token: Caller's access token.

    Returns:
        The daily trend points.

    Raises:
        APIError: If the request fails or the API is unreachable.
    """
    params: dict[str, Any] = {}
    if domain:
        params["domain"] = domain
    if metric_name:
        params["metric_name"] = metric_name
    try:
        resp = requests.get(
            _url(f"/analytics/{dataset_id}/trend"),
            params=params,
            headers=_auth(token),
            timeout=_TIMEOUT,
        )
        return _handle(resp)
    except requests.RequestException as exc:
        logger.warning(
            "Could not reach the API gateway at %s: %s",
            API_BASE_URL,
            exc,
            extra={"api_base_url": API_BASE_URL},
        )
        raise APIError(f"Failed to load analytics trend: {exc}") from exc


def analytics_features(
    dataset_id: int,
    domain: str | None = None,
    limit: int = 200,
    token: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch entity-level features, optionally filtered by domain.

    Args:
        dataset_id: Dataset to read.
        domain: Optional domain filter.
        limit: Maximum feature rows to return.
        token: Caller's access token.

    Returns:
        The entity feature rows.

    Raises:
        APIError: If the request fails or the API is unreachable.
    """
    params: dict[str, Any] = {"limit": limit}
    if domain:
        params["domain"] = domain
    try:
        resp = requests.get(
            _url(f"/analytics/{dataset_id}/features"),
            params=params,
            headers=_auth(token),
            timeout=_TIMEOUT,
        )
        return _handle(resp)
    except requests.RequestException as exc:
        logger.warning(
            "Could not reach the API gateway at %s: %s",
            API_BASE_URL,
            exc,
            extra={"api_base_url": API_BASE_URL},
        )
        raise APIError(f"Failed to load analytics features: {exc}") from exc


# ============================================================
# ML (Phase 3 — forecasts, anomalies, risk scores)
# ============================================================


def ml_overview(dataset_id: int, token: str | None = None) -> dict[str, Any]:
    """Fetch a dataset's forecast, alert, and risk counts.

    Args:
        dataset_id: Dataset to summarize.
        token: Caller's access token.

    Returns:
        The ML overview payload.

    Raises:
        APIError: If the request fails or the API is unreachable.
    """
    try:
        resp = requests.get(
            _url(f"/ml/{dataset_id}/overview"), headers=_auth(token), timeout=_TIMEOUT
        )
        return _handle(resp)
    except requests.RequestException as exc:
        logger.warning(
            "Could not reach the API gateway at %s: %s",
            API_BASE_URL,
            exc,
            extra={"api_base_url": API_BASE_URL},
        )
        raise APIError(f"Failed to load ML overview: {exc}") from exc


def ml_forecasts(
    dataset_id: int,
    domain: str | None = None,
    metric_name: str | None = None,
    token: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch forecasts, optionally filtered.

    Args:
        dataset_id: Dataset to read.
        domain: Optional domain filter.
        metric_name: Optional metric filter.
        token: Caller's access token.

    Returns:
        The forecast points.

    Raises:
        APIError: If the request fails or the API is unreachable.
    """
    params: dict[str, Any] = {}
    if domain:
        params["domain"] = domain
    if metric_name:
        params["metric_name"] = metric_name
    try:
        resp = requests.get(
            _url(f"/ml/{dataset_id}/forecasts"),
            params=params,
            headers=_auth(token),
            timeout=_TIMEOUT,
        )
        return _handle(resp)
    except requests.RequestException as exc:
        logger.warning(
            "Could not reach the API gateway at %s: %s",
            API_BASE_URL,
            exc,
            extra={"api_base_url": API_BASE_URL},
        )
        raise APIError(f"Failed to load forecasts: {exc}") from exc


def ml_anomalies(
    dataset_id: int,
    domain: str | None = None,
    severity: str | None = None,
    limit: int = 500,
    token: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch detected anomalies, optionally filtered.

    Args:
        dataset_id: Dataset to read.
        domain: Optional domain filter.
        severity: Optional severity filter.
        limit: Maximum anomalies to return.
        token: Caller's access token.

    Returns:
        The anomalies, most severe first.

    Raises:
        APIError: If the request fails or the API is unreachable.
    """
    params: dict[str, Any] = {"limit": limit}
    if domain:
        params["domain"] = domain
    if severity:
        params["severity"] = severity
    try:
        resp = requests.get(
            _url(f"/ml/{dataset_id}/anomalies"),
            params=params,
            headers=_auth(token),
            timeout=_TIMEOUT,
        )
        return _handle(resp)
    except requests.RequestException as exc:
        logger.warning(
            "Could not reach the API gateway at %s: %s",
            API_BASE_URL,
            exc,
            extra={"api_base_url": API_BASE_URL},
        )
        raise APIError(f"Failed to load anomalies: {exc}") from exc


def ml_risk_scores(
    dataset_id: int,
    domain: str | None = None,
    risk_level: str | None = None,
    token: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch entity risk scores, optionally filtered.

    Args:
        dataset_id: Dataset to read.
        domain: Optional domain filter.
        risk_level: Optional risk-level filter.
        token: Caller's access token.

    Returns:
        The risk scores, highest first.

    Raises:
        APIError: If the request fails or the API is unreachable.
    """
    params: dict[str, Any] = {}
    if domain:
        params["domain"] = domain
    if risk_level:
        params["risk_level"] = risk_level
    try:
        resp = requests.get(
            _url(f"/ml/{dataset_id}/risk-scores"),
            params=params,
            headers=_auth(token),
            timeout=_TIMEOUT,
        )
        return _handle(resp)
    except requests.RequestException as exc:
        logger.warning(
            "Could not reach the API gateway at %s: %s",
            API_BASE_URL,
            exc,
            extra={"api_base_url": API_BASE_URL},
        )
        raise APIError(f"Failed to load risk scores: {exc}") from exc


def ml_domain_intelligence(
    dataset_id: int, domain: str, token: str | None = None
) -> dict[str, Any]:
    """Fetch one domain's current metrics, forecasts, alerts, and risks.

    Args:
        dataset_id: Dataset to read.
        domain: Domain to roll up.
        token: Caller's access token.

    Returns:
        The per-domain roll-up.

    Raises:
        APIError: If the request fails or the API is unreachable.
    """
    try:
        resp = requests.get(
            _url(f"/ml/{dataset_id}/domain/{domain}"),
            headers=_auth(token),
            timeout=_TIMEOUT,
        )
        return _handle(resp)
    except requests.RequestException as exc:
        logger.warning(
            "Could not reach the API gateway at %s: %s",
            API_BASE_URL,
            exc,
            extra={"api_base_url": API_BASE_URL},
        )
        raise APIError(f"Failed to load domain intelligence: {exc}") from exc


# ============================================================
# Intelligence (Phase 3 Level 2 — cross-domain insights)
# ============================================================


def intelligence(dataset_id: int, token: str | None = None) -> dict[str, Any]:
    """Fetch the cross-domain insights for a dataset.

    Args:
        dataset_id: Dataset to analyze.
        token: Caller's access token.

    Returns:
        The translated insights.

    Raises:
        APIError: If the request fails or the API is unreachable.
    """
    try:
        resp = requests.get(
            _url(f"/intelligence/{dataset_id}"), headers=_auth(token), timeout=(5, 180)
        )
        return _handle(resp)
    except requests.RequestException as exc:
        logger.warning(
            "Could not reach the API gateway at %s: %s",
            API_BASE_URL,
            exc,
            extra={"api_base_url": API_BASE_URL},
        )
        raise APIError(f"Failed to load intelligence: {exc}") from exc


# ============================================================
# RAG (Phase 3 Level 3 — document assistant)
# ============================================================


def rag_upload(
    dataset_id: int,
    files: list[tuple[str, bytes]],
    business_name: str | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Upload documents for indexing against a dataset.

    Args:
        dataset_id: Dataset the documents belong to.
        files: The files to upload, as name and content pairs.
        business_name: Optional business name to record.
        token: Caller's access token.

    Returns:
        The accepted and rejected documents.

    Raises:
        APIError: If the request fails or the API is unreachable.
    """
    multipart = [
        ("files", (name, data, "application/octet-stream")) for name, data in files
    ]
    form: dict[str, Any] = {}
    if business_name:
        form["business_name"] = business_name
    try:
        resp = requests.post(
            _url(f"/rag/{dataset_id}/upload"),
            files=multipart,
            data=form,
            headers=_auth(token),
            timeout=(5, 300),
        )
        return _handle(resp)
    except requests.RequestException as exc:
        logger.warning(
            "Could not reach the API gateway at %s: %s",
            API_BASE_URL,
            exc,
            extra={"api_base_url": API_BASE_URL},
        )
        raise APIError(f"Failed to upload documents: {exc}") from exc


def rag_documents(dataset_id: int, token: str | None = None) -> list[dict[str, Any]]:
    """Fetch the documents indexed for a dataset.

    Args:
        dataset_id: Dataset to read.
        token: Caller's access token.

    Returns:
        One entry per document.

    Raises:
        APIError: If the request fails or the API is unreachable.
    """
    try:
        resp = requests.get(
            _url(f"/rag/{dataset_id}/documents"), headers=_auth(token), timeout=_TIMEOUT
        )
        return _handle(resp)
    except requests.RequestException as exc:
        logger.warning(
            "Could not reach the API gateway at %s: %s",
            API_BASE_URL,
            exc,
            extra={"api_base_url": API_BASE_URL},
        )
        raise APIError(f"Failed to load documents: {exc}") from exc


def rag_query(
    dataset_id: int, question: str, top_k: int | None = None, token: str | None = None
) -> dict[str, Any]:
    """Ask a question grounded in a dataset's documents.

    Args:
        dataset_id: Dataset to query.
        question: The question to answer.
        top_k: Optional override for how many chunks to retrieve.
        token: Caller's access token.

    Returns:
        The grounded answer with its sources.

    Raises:
        APIError: If the request fails or the API is unreachable.
    """
    body: dict[str, Any] = {"question": question}
    if top_k is not None:
        body["top_k"] = top_k
    try:
        resp = requests.post(
            _url(f"/rag/{dataset_id}/query"),
            json=body,
            headers=_auth(token),
            timeout=(5, 120),
        )
        return _handle(resp)
    except requests.RequestException as exc:
        logger.warning(
            "Could not reach the API gateway at %s: %s",
            API_BASE_URL,
            exc,
            extra={"api_base_url": API_BASE_URL},
        )
        raise APIError(f"Failed to query documents: {exc}") from exc


def rag_delete_document(
    dataset_id: int, document_id: int, token: str | None = None
) -> dict[str, Any]:
    """Delete one indexed document.

    Args:
        dataset_id: Dataset the document belongs to.
        document_id: Document to delete.
        token: Caller's access token.

    Returns:
        The deletion confirmation.

    Raises:
        APIError: If the request fails or the API is unreachable.
    """
    try:
        resp = requests.delete(
            _url(f"/rag/{dataset_id}/documents/{document_id}"),
            headers=_auth(token),
            timeout=_TIMEOUT,
        )
        return _handle(resp)
    except requests.RequestException as exc:
        logger.warning(
            "Could not reach the API gateway at %s: %s",
            API_BASE_URL,
            exc,
            extra={"api_base_url": API_BASE_URL},
        )
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
    """Ask the copilot agent a natural-language question.

    Args:
        question: The question to answer.
        dataset_id: Optional dataset to scope the question to.
        session_id: Conversation id, so the agent keeps context across turns.
        token: Caller's access token.

    Returns:
        The grounded answer and its evidence trail.

    Raises:
        APIError: If the request fails or the API is unreachable.
    """
    body: dict[str, Any] = {"question": question}
    if dataset_id is not None:
        body["dataset_id"] = dataset_id
    if session_id is not None:
        body["session_id"] = session_id
    try:
        resp = requests.post(
            _url("/agent/ask"), json=body, headers=_auth(token), timeout=(5, 240)
        )
        return _handle(resp)
    except requests.RequestException as exc:
        logger.warning(
            "Could not reach the API gateway at %s: %s",
            API_BASE_URL,
            exc,
            extra={"api_base_url": API_BASE_URL},
        )
        raise APIError(f"Agent request failed: {exc}") from exc


def agent_health(token: str | None = None) -> dict[str, Any]:
    """Check whether the agent and its model are ready.

    Args:
        token: Caller's access token.

    Returns:
        The agent readiness snapshot.

    Raises:
        APIError: If the request fails or the API is unreachable.
    """
    try:
        resp = requests.get(
            _url("/agent/health"), headers=_auth(token), timeout=(3, 10)
        )
        return _handle(resp)
    except requests.RequestException as exc:
        logger.warning(
            "Could not reach the API gateway at %s: %s",
            API_BASE_URL,
            exc,
            extra={"api_base_url": API_BASE_URL},
        )
        raise APIError(f"Agent health check failed: {exc}") from exc


# ============================================================
# Executive summary (single aggregating endpoint)
# ============================================================


def executive_summary(dataset_id: int, token: str | None = None) -> dict[str, Any]:
    """Fetch the full executive dashboard payload for a dataset.

    Args:
        dataset_id: Dataset to summarize.
        token: Caller's access token.

    Returns:
        The assembled executive summary.

    Raises:
        APIError: If the request fails or the API is unreachable.
    """
    try:
        resp = requests.get(
            _url(f"/executive/{dataset_id}/summary"),
            headers=_auth(token),
            timeout=_TIMEOUT,
        )
        return _handle(resp)
    except requests.RequestException as exc:
        logger.warning(
            "Could not reach the API gateway at %s: %s",
            API_BASE_URL,
            exc,
            extra={"api_base_url": API_BASE_URL},
        )
        raise APIError(f"Failed to load executive summary: {exc}") from exc


# ============================================================
# Auth (Item 6 — login, identity, refresh, logout)
# ============================================================


def login(email: str, password: str) -> dict[str, Any]:
    """Exchange email and password for access and refresh tokens.

    Args:
        email: The user's email address.
        password: The user's password.

    Returns:
        The issued token pair.

    Raises:
        APIError: If the request fails or the API is unreachable.
    """
    try:
        resp = requests.post(
            _url("/auth/login"),
            json={"email": email, "password": password},
            timeout=_TIMEOUT,
        )
        return _handle(resp)
    except requests.RequestException as exc:
        logger.warning(
            "Could not reach the API gateway at %s: %s",
            API_BASE_URL,
            exc,
            extra={"api_base_url": API_BASE_URL},
        )
        raise APIError(f"Login failed: {exc}") from exc


def auth_me(access_token: str) -> dict[str, Any]:
    """Fetch the caller's identity, roles, and permissions.

    Args:
        access_token: The caller's access token.

    Returns:
        The caller's profile.

    Raises:
        APIError: If the request fails or the API is unreachable.
    """
    try:
        resp = requests.get(
            _url("/auth/me"),
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=_TIMEOUT,
        )
        return _handle(resp)
    except requests.RequestException as exc:
        logger.warning(
            "Could not reach the API gateway at %s: %s",
            API_BASE_URL,
            exc,
            extra={"api_base_url": API_BASE_URL},
        )
        raise APIError(f"Failed to load identity: {exc}") from exc


def refresh_access(refresh_token: str) -> dict[str, Any]:
    """Exchange a refresh token for a new access token.

    Args:
        refresh_token: The refresh token to redeem.

    Returns:
        The new access token.

    Raises:
        APIError: If the request fails or the API is unreachable.
    """
    try:
        resp = requests.post(
            _url("/auth/refresh"),
            json={"refresh_token": refresh_token},
            timeout=_TIMEOUT,
        )
        return _handle(resp)
    except requests.RequestException as exc:
        logger.warning(
            "Could not reach the API gateway at %s: %s",
            API_BASE_URL,
            exc,
            extra={"api_base_url": API_BASE_URL},
        )
        raise APIError(f"Token refresh failed: {exc}") from exc


def logout(access_token: str, refresh_token: str) -> None:
    """Revoke the caller's refresh token, ending the session.

    Args:
        access_token: The caller's access token.
        refresh_token: The refresh token to revoke.

    Returns:
        The API's response payload.

    Raises:
        APIError: If the request fails or the API is unreachable.
    """
    try:
        resp = requests.post(
            _url("/auth/logout"),
            headers={"Authorization": f"Bearer {access_token}"},
            json={"refresh_token": refresh_token},
            timeout=_TIMEOUT,
        )
        _handle(resp)
    except requests.RequestException as exc:
        logger.warning(
            "Could not reach the API gateway at %s: %s",
            API_BASE_URL,
            exc,
            extra={"api_base_url": API_BASE_URL},
        )
        raise APIError(f"Logout failed: {exc}") from exc
