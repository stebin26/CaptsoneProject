"""The Executive Dashboard's single endpoint.

This is the opposite of the copilot: it answers "how is the operation doing?"
in one round trip, with no LLM anywhere in the path. Everything it returns
already exists in analytics.*, ml.*, and the intelligence engine -- this is a
read-and-assemble layer, not a compute layer.

One endpoint, not seven. The dashboard makes a single call; the fan-out across
tables happens here, server-side, close to the database. Seven round trips from
the browser against a stack with a cold-attach cost is how a first screen ends
up slow.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from ops_common.db import get_db
from ops_common.logging import get_logger
from api_app.auth.dependencies import require_permission

# The risk-index formula lives in the ML layer as a pure function, so it is the
# single source of truth and can be unit-tested without a database.

logger = get_logger(__name__)

router = APIRouter()

# The eight universal domains, canonical order. Duplicated deliberately: the
# API must not import from the dashboard, and this list rarely changes.
DOMAIN_ORDER = [
    "assets", "operations", "quality", "maintenance",
    "inventory", "workforce", "finance", "customers",
]

# Same 70/30 blend at both levels: entity -> domain, and domain -> overall.
# One philosophy, applied recursively.
MEAN_WEIGHT = 0.7
MAX_WEIGHT = 0.3


# ============================================================
# Response models
# ============================================================

class DomainHealth(BaseModel):
    domain: str
    active: bool
    score: float | None          # 0-100 relative risk, None when absent
    band: str                    # low | elevated | high | absent
    open_alerts: int


class RiskEntry(BaseModel):
    entity_ref: str | None
    domain: str
    metric_name: str | None
    score: float | None
    band: str | None
    trend_slope: float | None    # direction of travel for this entity


class AlertEntry(BaseModel):
    domain: str
    entity_ref: str | None
    metric_name: str
    severity: str | None
    when: str | None
    observed: float | None
    expected: float | None


class ForecastEntry(BaseModel):
    domain: str
    metric_name: str
    last_value: float | None
    next_value: float | None
    pct_change: float | None
    history: list[float]         # daily_trend tail, for the sparkline
    band_low: list[float]        # forecast lower bound, aligned to forecast pts
    band_high: list[float]


class InsightEntry(BaseModel):
    root: str
    root_term: str
    direction: str
    narrative: str
    score: float


class RiskIndexOut(BaseModel):
    value: int
    band: str
    label: str
    domain_count: int
    mean: float
    peak: float
    peak_domain: str | None


class ExecutiveSummary(BaseModel):
    dataset_id: int
    business_name: str | None
    industry: str | None
    date_start: str | None
    date_end: str | None
    active_domain_count: int

    risk_index: RiskIndexOut
    open_alert_count: int
    high_alert_count: int
    entities_at_risk: int
    insight_count: int

    domain_health: list[DomainHealth]
    top_risks: list[RiskEntry]
    active_alerts: list[AlertEntry]
    forecasts: list[ForecastEntry]
    insights: list[InsightEntry]


# ============================================================
# Helpers
# ============================================================

def _f(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _blend(scores: list[float]) -> float:
    """0.7*mean + 0.3*max -- the same formula compute_index uses for domains."""
    clean = [s for s in scores if s is not None]
    if not clean:
        return 0.0
    total = MEAN_WEIGHT + MAX_WEIGHT
    mean = sum(clean) / len(clean)
    return (MEAN_WEIGHT / total) * mean + (MAX_WEIGHT / total) * max(clean)


def _band(value: float | None) -> str:
    if value is None:
        return "absent"
    if value >= 67:
        return "high"
    if value >= 34:
        return "elevated"
    return "low"

def _compute_index(domain_score: dict[str, float], mean_w: float, max_w: float):
    """Import the risk-index formula lazily, same sys.path trick the
    intelligence engine uses -- keeps 'ml' out of the top-level import graph."""
    import os
    import sys

    ml_path = os.getenv("OPS_ML_PATH", "/app/services/ml")
    if ml_path not in sys.path:
        sys.path.insert(0, ml_path)
    from risk_index import compute  # noqa: E402

    return compute(domain_score, mean_w, max_w)
# ============================================================
# The endpoint
# ============================================================

@router.get("/executive/{dataset_id}/summary", response_model=ExecutiveSummary)
def executive_summary(
    dataset_id: int,
    session: Session = Depends(get_db),
    _user=Depends(require_permission("analytics:read")),
) -> ExecutiveSummary:
    meta = _meta(session, dataset_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

    risks = _risk_rows(session, dataset_id)
    alerts = _alert_rows(session, dataset_id)
    features = _feature_rows(session, dataset_id)
    forecasts = _forecast_entries(session, dataset_id)
    insights = _insight_entries(session, dataset_id)

    # --- collapse entities into a per-domain score (level one of the blend) ---
    by_domain: dict[str, list[float]] = {}
    for r in risks:
        score = _f(r["risk_score"])
        if score is not None:
            by_domain.setdefault(r["domain"], []).append(score)

    domain_score = {d: _blend(scores) for d, scores in by_domain.items()}

    # --- alert counts per domain ---
    alert_by_domain: dict[str, int] = {}
    for a in alerts:
        alert_by_domain[a["domain"]] = alert_by_domain.get(a["domain"], 0) + 1

    active = set(by_domain) | {a["domain"] for a in alerts} | {
        f["domain"] for f in features
    }

    domain_health = [
        DomainHealth(
            domain=d,
            active=d in active,
            score=round(domain_score[d], 1) if d in domain_score else None,
            band=_band(domain_score.get(d)) if d in active else "absent",
            open_alerts=alert_by_domain.get(d, 0),
        )
        for d in DOMAIN_ORDER
    ]

    # --- overall index (level two: same blend across domain scores) ---
    index = _compute_index(domain_score, MEAN_WEIGHT, MAX_WEIGHT)

    # --- top five risks, with each entity's trend slope for the arrow ---
    slope_lookup = {
        (f["domain"], f["entity_ref"], f["metric_name"]): _f(f["trend_slope"])
        for f in features
    }
    top_risks = _top_risks(risks, slope_lookup)

    high_alerts = sum(1 for a in alerts if a["severity"] == "high")

    return ExecutiveSummary(
        dataset_id=dataset_id,
        business_name=meta["business_name"],
        industry=meta["industry"],
        date_start=meta["date_start"],
        date_end=meta["date_end"],
        active_domain_count=len(active),
        risk_index=RiskIndexOut(
            value=index.value,
            band=index.band,
            label=index.label,
            domain_count=index.domain_count,
            mean=index.mean,
            peak=index.peak,
            peak_domain=index.peak_domain,
        ),
        open_alert_count=len(alerts),
        high_alert_count=high_alerts,
        entities_at_risk=sum(
            1 for r in risks if (r["risk_level"] in ("high", "medium"))
        ),
        insight_count=len(insights),
        domain_health=domain_health,
        top_risks=top_risks,
        active_alerts=[
            AlertEntry(
                domain=a["domain"],
                entity_ref=a["entity_ref"],
                metric_name=a["metric_name"],
                severity=a["severity"],
                when=a["anomaly_date"],
                observed=_f(a["observed_value"]),
                expected=_f(a["expected_value"]),
            )
            for a in alerts[:8]
        ],
        forecasts=forecasts,
        insights=insights,
    )


# ============================================================
# Data access -- one query each, all scoped by dataset_id
# ============================================================

def _meta(session: Session, dataset_id: int) -> dict[str, Any] | None:
    row = session.execute(
        text(
            """
            SELECT d.business_name, d.industry,
                   MIN(t.day)::text AS date_start,
                   MAX(t.day)::text AS date_end
            FROM meta.dataset d
            LEFT JOIN analytics.daily_trend t ON t.dataset_id = d.id
            WHERE d.id = :d
            GROUP BY d.business_name, d.industry
            """
        ),
        {"d": dataset_id},
    ).fetchone()

    if row is None:
        return None
    return {
        "business_name": row[0],
        "industry": row[1],
        "date_start": row[2],
        "date_end": row[3],
    }

def _risk_rows(session: Session, dataset_id: int) -> list[dict[str, Any]]:
    rows = session.execute(
        text(
            """
            SELECT domain, entity_id, risk_score, risk_level
            FROM ml.risk_scores
            WHERE dataset_id = :d
            ORDER BY risk_score DESC NULLS LAST
            """
        ),
        {"d": dataset_id},
    ).fetchall()
    return [
        {
            "domain": r[0],
            "entity_ref": None if r[1] is None else str(r[1]),
            "risk_score": r[2],
            "risk_level": r[3],
        }
        for r in rows
    ]


def _alert_rows(session: Session, dataset_id: int) -> list[dict[str, Any]]:
    rows = session.execute(
        text(
            """
            SELECT domain, entity_id, metric_name, anomaly_date,
                   observed_value, expected_value, severity
            FROM ml.anomalies
            WHERE dataset_id = :d
            ORDER BY
              CASE severity WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
              anomaly_date DESC NULLS LAST
            """
        ),
        {"d": dataset_id},
    ).fetchall()
    return [
        {
            "domain": r[0],
            "entity_ref": None if r[1] is None else str(r[1]),
            "metric_name": str(r[2]),
            "anomaly_date": None if r[3] is None else str(r[3]),
            "observed_value": r[4],
            "expected_value": r[5],
            "severity": r[6],
        }
        for r in rows
    ]


def _feature_rows(session: Session, dataset_id: int) -> list[dict[str, Any]]:
    rows = session.execute(
        text(
            """
            SELECT domain, entity_ref, metric_name, trend_slope
            FROM analytics.entity_features
            WHERE dataset_id = :d
            """
        ),
        {"d": dataset_id},
    ).fetchall()
    return [
        {
            "domain": r[0],
            "entity_ref": None if r[1] is None else str(r[1]),
            "metric_name": r[2],
            "trend_slope": r[3],
        }
        for r in rows
    ]


def _top_risks(
    risks: list[dict[str, Any]],
    slopes: dict[tuple, float | None],
) -> list[RiskEntry]:
    top = []
    for r in risks[:5]:
        # entity_features is keyed by (domain, entity, metric); the risk row has
        # no metric, so take the first slope that matches domain+entity.
        slope = next(
            (
                v
                for (dom, ent, _metric), v in slopes.items()
                if dom == r["domain"] and ent == r["entity_ref"]
            ),
            None,
        )
        top.append(
            RiskEntry(
                entity_ref=r["entity_ref"],
                domain=r["domain"],
                metric_name=None,
                score=_f(r["risk_score"]),
                band=r["risk_level"],
                trend_slope=slope,
            )
        )
    return top


def _forecast_entries(
    session: Session,
    dataset_id: int,
    max_metrics: int = 5,
) -> list[ForecastEntry]:
    """Headline forecasts with a history tail and a forecast band per metric.

    History comes from analytics.daily_trend (what happened); the band and next
    value come from ml.forecasts (what is projected). The sparkline stitches the
    two: solid history, then the projected point inside its confidence band.
    """
    forecast_rows = session.execute(
        text(
            """
            SELECT domain, metric_name, forecast_date,
                   forecast_value, lower_bound, upper_bound
            FROM ml.forecasts
            WHERE dataset_id = :d
            ORDER BY domain, metric_name, forecast_date
            """
        ),
        {"d": dataset_id},
    ).fetchall()

    grouped: dict[tuple[str, str], list[Any]] = {}
    for r in forecast_rows:
        grouped.setdefault((r[0], r[1]), []).append(r)

    entries: list[ForecastEntry] = []
    for (domain, metric), rows in list(grouped.items())[:max_metrics]:
        rows.sort(key=lambda x: x[2])
        values = [_f(x[3]) for x in rows if x[3] is not None]
        lows = [_f(x[4]) for x in rows]
        highs = [_f(x[5]) for x in rows]

        history = _trend_tail(session, dataset_id, domain, metric)
        last = history[-1] if history else (values[0] if values else None)
        nxt = values[-1] if values else None

        pct = None
        if last not in (None, 0) and nxt is not None:
            pct = (nxt - last) / abs(last) * 100

        entries.append(
            ForecastEntry(
                domain=domain,
                metric_name=metric,
                last_value=last,
                next_value=nxt,
                pct_change=round(pct, 1) if pct is not None else None,
                history=history,
                band_low=[v for v in lows if v is not None],
                band_high=[v for v in highs if v is not None],
            )
        )
    return entries


def _trend_tail(
    session: Session,
    dataset_id: int,
    domain: str,
    metric: str,
    points: int = 14,
) -> list[float]:
    rows = session.execute(
        text(
            """
            SELECT avg_value
            FROM analytics.daily_trend
            WHERE dataset_id = :d AND domain = :dom AND metric_name = :m
            ORDER BY day DESC
            LIMIT :n
            """
        ),
        {"d": dataset_id, "dom": domain, "m": metric, "n": points},
    ).fetchall()
    return [float(r[0]) for r in reversed(rows) if r[0] is not None]


def _insight_entries(
    session: Session,
    dataset_id: int,
) -> list[InsightEntry]:
    """Top cross-domain insights, template-rendered, no LLM.

    Runs the same inference engine the /intelligence page uses, but takes only
    the strongest few and never calls the model -- an exec page must not be able
    to take two minutes.
    """
    import os
    import sys

    intel_path = os.getenv("OPS_INTELLIGENCE_PATH", "/app/services/intelligence")
    if intel_path not in sys.path:
        sys.path.insert(0, intel_path)
    from inference_engine import run_inference  # noqa: E402

    forecasts = _mini_forecasts(session, dataset_id)
    anomalies = _mini_anomalies(session, dataset_id)
    risks = [
        {"domain": r["domain"], "entity_ref": r["entity_ref"],
         "risk_score": r["risk_score"], "risk_level": r["risk_level"]}
        for r in _risk_rows(session, dataset_id)
    ]

    active = {r["domain"] for r in risks}
    if not active:
        return []

    try:
        raw = run_inference(forecasts, anomalies, risks, active_domains=active)
    except Exception:  # noqa: BLE001
        logger.exception("Executive insight inference failed")
        return []

    return [
        InsightEntry(
            root=t.get("root", ""),
            root_term=t.get("root_term") or t.get("root", ""),
            direction=t.get("direction", ""),
            narrative=t.get("narrative", ""),
            score=t.get("score", 0.0),
        )
        for t in raw[:3]
    ]


def _mini_forecasts(session: Session, dataset_id: int) -> list[dict[str, Any]]:
    rows = session.execute(
        text(
            """
            SELECT domain, metric_name, forecast_date, forecast_value
            FROM ml.forecasts WHERE dataset_id = :d
            """
        ),
        {"d": dataset_id},
    ).fetchall()
    return [
        {"domain": r[0], "metric_name": r[1],
         "forecast_date": str(r[2]), "forecast_value": r[3]}
        for r in rows
    ]


def _mini_anomalies(session: Session, dataset_id: int) -> list[dict[str, Any]]:
    rows = session.execute(
        text(
            """
            SELECT domain, metric_name, severity
            FROM ml.anomalies WHERE dataset_id = :d
            """
        ),
        {"d": dataset_id},
    ).fetchall()
    return [
        {"domain": r[0], "metric_name": r[1], "severity": r[2]}
        for r in rows
    ]