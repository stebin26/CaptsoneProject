"""Cross-domain intelligence API endpoint.

Reads the Level 1 ML outputs (forecasts, anomalies, risk scores) for a dataset,
runs the Level 2 inference engine over the active domain subgraph, and
translates the result into business language.
"""
from __future__ import annotations

import os
import sys
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from ops_common.db import get_db
from ops_common.logging import get_logger
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from api_app.auth.dependencies import require_permission

logger = get_logger(__name__)

router = APIRouter()

# The Level 2 engine + translator live in services/intelligence; make importable.
_INTEL_PATH = os.getenv("OPS_INTELLIGENCE_PATH", "/app/services/intelligence")
if _INTEL_PATH not in sys.path:
    sys.path.insert(0, _INTEL_PATH)

from inference_engine import run_inference  # noqa: E402
from translator import translate_all  # noqa: E402

# ============================================================
# Response models
# ============================================================


class ImpactedOut(BaseModel):
    """One domain impacted by a root cause."""
    domain: str
    term: str
    strength: str
    effect: str
    label: str


class InsightOut(BaseModel):
    """One translated cross-domain insight with its impacts."""
    root: str
    root_term: str
    direction: str
    score: float
    narrative: str
    recommendation: str
    impacted: list[ImpactedOut]


class IntelligenceOut(BaseModel):
    """Cross-domain intelligence payload for a dataset."""
    dataset_id: int
    business_name: str | None
    industry: str | None
    active_domains: list[str]
    insight_count: int
    insights: list[InsightOut]


# ============================================================
# Helpers — pull Level 1 outputs + analytics terms from the DB
# ============================================================


def _fetch_forecasts(session: Session, dataset_id: int) -> list[dict[str, Any]]:
    rows = session.execute(
        text(
            """
            SELECT domain, metric_name, forecast_date, forecast_value
            FROM ml.forecasts WHERE dataset_id = :d
            ORDER BY domain, metric_name, forecast_date
            """
        ),
        {"d": dataset_id},
    ).fetchall()
    return [
        {
            "domain": r[0],
            "metric_name": r[1],
            "forecast_date": str(r[2]),
            "forecast_value": r[3],
        }
        for r in rows
    ]


def _fetch_anomalies(session: Session, dataset_id: int) -> list[dict[str, Any]]:
    rows = session.execute(
        text(
            """
            SELECT domain, metric_name, severity
            FROM ml.anomalies WHERE dataset_id = :d
            """
        ),
        {"d": dataset_id},
    ).fetchall()
    return [{"domain": r[0], "metric_name": r[1], "severity": r[2]} for r in rows]


def _fetch_risks(session: Session, dataset_id: int) -> list[dict[str, Any]]:
    rows = session.execute(
        text(
            """
            SELECT domain, entity_id, risk_score, risk_level
            FROM ml.risk_scores WHERE dataset_id = :d
            """
        ),
        {"d": dataset_id},
    ).fetchall()
    return [
        {"domain": r[0], "entity_ref": r[1], "risk_score": r[2], "risk_level": r[3]}
        for r in rows
    ]


def _fetch_metrics(session: Session, dataset_id: int) -> list[dict[str, Any]]:
    rows = session.execute(
        text(
            """
            SELECT domain, metric_name, business_name, industry
            FROM analytics.domain_metrics WHERE dataset_id = :d
            ORDER BY domain, metric_name
            """
        ),
        {"d": dataset_id},
    ).fetchall()
    return [
        {"domain": r[0], "metric_name": r[1], "business_name": r[2], "industry": r[3]}
        for r in rows
    ]


def _fetch_mapping(session: Session, dataset_id: int) -> dict[str, Any]:
    # Optional: a saved per-domain display label from the onboarding mapping config.
    try:
        rows = session.execute(
            text(
                """
                SELECT domain, display_label
                FROM meta.mapping_config
                WHERE dataset_id = :d AND display_label IS NOT NULL
                """
            ),
            {"d": dataset_id},
        ).fetchall()
        return {str(r[0]).lower(): r[1] for r in rows if r[1]}
    except Exception:  # noqa: BLE001 — mapping labels are optional, never fatal
        return {}


# ============================================================
# Endpoint
# ============================================================


@router.get("/intelligence/{dataset_id}", response_model=IntelligenceOut)
def cross_domain_intelligence(
    dataset_id: int,
    session: Session = Depends(get_db),
    _user=Depends(require_permission("intelligence:read")),
) -> IntelligenceOut:
    """Return cross-domain insights for a dataset.

    Reads Level 1 ML outputs, runs the Level 2 inference engine over the active
    domains, and translates the result into business language.

    Args:
        dataset_id: Id of the dataset to analyze.
        session: Active database session.
        _user: Authenticated caller, injected to enforce ``intelligence:read``.

    Returns:
        The translated cross-domain insights for the dataset.

    Raises:
        HTTPException: 503 if the ML/analytics layer cannot be read.
    """
    try:
        forecasts = _fetch_forecasts(session, dataset_id)
        anomalies = _fetch_anomalies(session, dataset_id)
        risks = _fetch_risks(session, dataset_id)
        metrics = _fetch_metrics(session, dataset_id)
        mapping = _fetch_mapping(session, dataset_id)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to read Level 1 outputs for intelligence")
        raise HTTPException(status_code=503, detail="ML/analytics layer unavailable.") from None

    business_name = metrics[0]["business_name"] if metrics else None
    industry = metrics[0]["industry"] if metrics else None

    # Active domains = anything that produced any ML signal for this dataset.
    active = sorted(
        {str(r["domain"]).lower() for r in forecasts}
        | {str(r["domain"]).lower() for r in anomalies}
        | {str(r["domain"]).lower() for r in risks}
    )

    if not active:
        return IntelligenceOut(
            dataset_id=dataset_id,
            business_name=business_name,
            industry=industry,
            active_domains=[],
            insight_count=0,
            insights=[],
        )

    # Level 2: traverse the active subgraph, then translate to business terms.
    raw_insights = run_inference(
        forecasts, anomalies, risks, active_domains=set(active)
    )
    translated = translate_all(raw_insights, metrics=metrics, mapping=mapping)

    insights = [
        InsightOut(
            root=t["root"],
            root_term=t["root_term"],
            direction=t["direction"],
            score=t["score"],
            narrative=t["narrative"],
            recommendation=t["recommendation"],
            impacted=[ImpactedOut(**imp) for imp in t["impacted"]],
        )
        for t in translated
    ]

    return IntelligenceOut(
        dataset_id=dataset_id,
        business_name=business_name,
        industry=industry,
        active_domains=active,
        insight_count=len(insights),
        insights=insights,
    )
