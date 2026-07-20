"""Evaluation API endpoints backed by the committed ``evaluation/results`` files.

The evaluation scripts write their reports to JSON on disk rather than to a
database table, because they are run occasionally and by hand rather than on
every upload. This router exposes those reports read-only so the dashboard can
render them, guarded by the ``evaluation:read`` permission, which is granted to
Admin and Analyst but not to Viewer.

Only a fixed set of known filenames is served. The report name from the URL is
never used to build a path directly; it is looked up in a whitelist first, so a
crafted name such as ``../../etc/passwd`` cannot escape the results directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from api_app.auth.dependencies import require_permission
from fastapi import APIRouter, Depends, HTTPException
from ops_common.logging import get_logger
from pydantic import BaseModel

logger = get_logger(__name__)

router = APIRouter()

# Directory the evaluation scripts write to, mounted into the api container.
_RESULTS_DIR = Path("/app/evaluation/results")

# The reports this API is willing to serve, keyed by a short stable slug the
# dashboard uses. The slug is what the client sends; the filename is never taken
# from client input, which is what makes path traversal impossible here.
_REPORTS: dict[str, dict[str, str]] = {
    "retrieval": {
        "filename": "retrieval_600.json",
        "title": "RAG retrieval",
        "group": "rag",
    },
    "threshold": {
        "filename": "threshold_analysis_600.json",
        "title": "RAG distance threshold",
        "group": "rag",
    },
    "generation": {
        "filename": "generation_600_t0.json",
        "title": "RAG generation",
        "group": "rag",
    },
    "forecast": {
        "filename": "forecast_backtest.json",
        "title": "Forecasting backtest",
        "group": "ml",
    },
    "anomaly": {
        "filename": "anomaly_eval.json",
        "title": "Anomaly detection",
        "group": "ml",
    },
    "risk": {
        "filename": "risk_eval.json",
        "title": "Risk scoring",
        "group": "ml",
    },
}


class ReportInfo(BaseModel):
    """Metadata describing one evaluation report and whether it is on disk."""

    slug: str
    title: str
    group: str
    available: bool


class ReportListOut(BaseModel):
    """The catalogue of evaluation reports the API knows about."""

    reports: list[ReportInfo]


def _report_path(slug: str) -> Path:
    """Resolve a report slug to its file path through the whitelist.

    Args:
        slug: The short report identifier from the request.

    Returns:
        The path to the report's JSON file.

    Raises:
        HTTPException: 404 if the slug is not a known report.

    """
    entry = _REPORTS.get(slug)
    if entry is None:
        logger.info("Unknown evaluation report requested", extra={"slug": slug})
        raise HTTPException(status_code=404, detail=f"Unknown report: {slug}")
    return _RESULTS_DIR / entry["filename"]


@router.get("/evaluation/reports", response_model=ReportListOut)
def list_reports(
    _user=Depends(require_permission("evaluation:read")),
) -> ReportListOut:
    """List the evaluation reports and whether each one has been generated.

    The catalogue is always returned in full so the dashboard can show a report
    as present or not-yet-run, rather than silently omitting the ones that have
    not been produced.

    Args:
        _user: The authenticated caller, required to hold ``evaluation:read``.

    Returns:
        The catalogue of known reports with an availability flag on each.

    """
    reports: list[ReportInfo] = []
    for slug, entry in _REPORTS.items():
        path = _RESULTS_DIR / entry["filename"]
        reports.append(
            ReportInfo(
                slug=slug,
                title=entry["title"],
                group=entry["group"],
                available=path.is_file(),
            )
        )
    logger.info(
        "Listed evaluation reports",
        extra={"available": sum(1 for r in reports if r.available)},
    )
    return ReportListOut(reports=reports)


@router.get("/evaluation/reports/{slug}")
def get_report(
    slug: str,
    _user=Depends(require_permission("evaluation:read")),
) -> dict[str, Any]:
    """Return the parsed contents of one evaluation report.

    Args:
        slug: The short report identifier, resolved through the whitelist.
        _user: The authenticated caller, required to hold ``evaluation:read``.

    Returns:
        The report's JSON content as a dictionary.

    Raises:
        HTTPException: 404 if the report is unknown or has not been generated;
            500 if the file exists but cannot be read or parsed.

    """
    path = _report_path(slug)

    if not path.is_file():
        logger.info(
            "Evaluation report not yet generated",
            extra={"slug": slug, "path": str(path)},
        )
        raise HTTPException(
            status_code=404,
            detail=(
                f"Report '{slug}' has not been generated yet. Run its evaluation "
                "script to produce it."
            ),
        )

    try:
        with path.open(encoding="utf-8") as handle:
            content = json.load(handle)
    except json.JSONDecodeError as exc:
        # The file is present but corrupt. This is a server-side data problem,
        # not a bad request, so it is logged with the location and surfaced as a
        # 500 rather than silently returning an empty body.
        logger.exception(
            "Evaluation report is not valid JSON",
            extra={"slug": slug, "path": str(path)},
        )
        raise HTTPException(
            status_code=500,
            detail=f"Report '{slug}' is present but could not be parsed.",
        ) from exc
    except OSError as exc:
        logger.exception(
            "Could not read evaluation report from disk",
            extra={"slug": slug, "path": str(path)},
        )
        raise HTTPException(
            status_code=500,
            detail=f"Report '{slug}' could not be read.",
        ) from exc

    logger.info("Served evaluation report", extra={"slug": slug})
    return content