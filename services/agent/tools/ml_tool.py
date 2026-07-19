"""ML tools -- the agent's access to forecasts, alerts, and risk.

These turn the Level 1 model outputs into short statements the agent can reason
over: what is projected to happen, what has already been flagged as unusual, and
which entities carry the most degradation risk. As elsewhere, the router's own
query functions are reused so the tools and the API can never disagree.
"""
from __future__ import annotations

from typing import Any

# Reuse the router's own query functions so there is ONE source of truth for
# ML reads. These take a SQLAlchemy session, which we supply from session_scope.
from api_app.routers.v1.ml import (
    dataset_anomalies,
    dataset_forecasts,
    dataset_risk_scores,
    ml_overview,
)
from ops_common.db import session_scope
from ops_common.logging import get_logger

from .base import (
    ToolResult,
    domain_hint_for_schema,
    normalize_domain,
    tool_error,
    tool_ok,
)

logger = get_logger(__name__)


# ============================================================
# Tool 1 — ML overview (how much signal exists, and where)
# ============================================================


def ml_overview_tool(dataset_id: int) -> ToolResult:
    # The agent's first ML look: does this dataset even have predictions, and
    # which domains carry forecast/anomaly/risk signal worth drilling into.
    """Summarize a dataset's forecast, alert, and risk counts.

    Args:
        dataset_id: Dataset to summarize.

    Returns:
        A summary of the ML signals available.
    """
    try:
        with session_scope() as session:
            overview = ml_overview(dataset_id, session)
    except Exception as exc:  # noqa: BLE001
        logger.exception("ml_overview_tool failed for dataset_id=%s", dataset_id)
        return tool_error(f"Could not read ML overview for dataset {dataset_id}: {exc}")

    total_signal = (
        overview.forecast_count + overview.anomaly_count + overview.high_risk_count
    )
    if total_signal == 0:
        return tool_error(
            f"No ML results for dataset {dataset_id}. The ML pipeline may not "
            "have run for it yet."
        )

    business = overview.business_name or "Unknown"
    industry = overview.industry or "Unknown"

    summary = (
        f"Dataset {dataset_id} — {business} ({industry}). "
        f"ML signal across domains {', '.join(overview.domains) or 'none'}: "
        f"{overview.forecast_count} forecasts, "
        f"{overview.anomaly_count} anomalies, "
        f"{overview.high_risk_count} high-risk entities."
    )
    return tool_ok(
        summary=summary,
        data={
            "dataset_id": dataset_id,
            "business_name": business,
            "industry": industry,
            "domains": overview.domains,
            "forecast_count": overview.forecast_count,
            "anomaly_count": overview.anomaly_count,
            "high_risk_count": overview.high_risk_count,
        },
    )


# ============================================================
# Tool 2 — ML alerts (anomalies + risk: the "what is wrong" view)
# ============================================================


def ml_alerts(dataset_id: int, domain: str | None = None) -> ToolResult:
    # Combine anomalies and risk scores into one "problems" picture — this is
    # what the agent needs for "what's going wrong / what should I worry about"
    # and for corroborating a production/quality drop with a concrete cause.
    #
    # The model's domain word is normalized (downtime -> maintenance). An
    # unrecognized word becomes None, dropping the filter so we scan every
    # domain rather than returning nothing.
    """Describe the anomalies flagged for a dataset.

    Args:
        dataset_id: Dataset to read.
        domain: Optional domain to focus on.

    Returns:
        A summary of the flagged anomalies and their severities.
    """
    resolved = normalize_domain(domain)
    widened = domain is not None and resolved is None

    try:
        with session_scope() as session:
            anomalies = dataset_anomalies(
                dataset_id, domain=resolved, severity=None, limit=500, session=session
            )
            risks = dataset_risk_scores(
                dataset_id, domain=resolved, risk_level=None, session=session
            )
    except Exception as exc:  # noqa: BLE001
        logger.exception("ml_alerts failed for dataset_id=%s", dataset_id)
        return tool_error(f"Could not read ML alerts for dataset {dataset_id}: {exc}")

    if not anomalies and not risks:
        scope = f" in domain '{resolved}'" if resolved else ""
        return tool_error(
            f"No anomalies or risk scores for dataset {dataset_id}{scope}."
        )

    # Anomalies are already severity-sorted by the router (high first). Take the
    # top few and describe them compactly.
    top_anomalies = [
        {
            "domain": a.domain,
            "entity": a.entity_ref,
            "metric": a.metric_name,
            "severity": a.severity,
            "observed": _round(a.observed_value),
            "expected": _round(a.expected_value),
            "deviation": _round(a.deviation),
        }
        for a in anomalies[:8]
    ]

    # Risk scores come sorted highest-first from the router. Surface the top
    # entities and count how many are high level.
    high_risks = [r for r in risks if (r.risk_level or "").lower() == "high"]
    top_risks = [
        {
            "domain": r.domain,
            "entity": r.entity_ref,
            "risk_score": _round(r.risk_score),
            "risk_level": r.risk_level,
        }
        for r in risks[:8]
    ]

    sev_counts = _severity_counts(anomalies)
    summary_bits: list[str] = []
    if anomalies:
        summary_bits.append(
            f"{len(anomalies)} anomalies "
            f"(high={sev_counts.get('high', 0)}, "
            f"med={sev_counts.get('medium', 0)}, "
            f"low={sev_counts.get('low', 0)})"
        )
    if risks:
        worst = top_risks[0] if top_risks else None
        worst_txt = (
            f"; worst risk {worst['entity']} in {worst['domain']} ({worst['risk_score']})"
            if worst
            else ""
        )
        summary_bits.append(f"{len(high_risks)} high-risk entities{worst_txt}")

    if widened:
        prefix = (
            f"Dataset {dataset_id} — '{domain}' is not a known domain, so I "
            f"checked all domains. Alerts: "
        )
    else:
        scope = f" in {resolved}" if resolved else ""
        prefix = f"Dataset {dataset_id} alerts{scope} — "

    summary = prefix + "; ".join(summary_bits)
    return tool_ok(
        summary=summary,
        data={
            "dataset_id": dataset_id,
            "domain_requested": domain,
            "domain_used": resolved,
            "searched_all_domains": widened,
            "anomaly_count": len(anomalies),
            "severity_counts": sev_counts,
            "top_anomalies": top_anomalies,
            "high_risk_count": len(high_risks),
            "top_risks": top_risks,
        },
    )


# ============================================================
# Tool 3 — ML forecast (the "where it is heading" view)
# ============================================================


def ml_forecast(
    dataset_id: int,
    domain: str | None = None,
    metric_name: str | None = None,
) -> ToolResult:
    # Forward projection per metric. For the agent we describe the direction and
    # end-point of each forecast series rather than every dated point. Domain
    # word normalized; unknown word widens the search instead of emptying it.
    """Describe where each forecast series is heading.

    Reports direction and endpoint per series rather than every dated point. An
    unrecognised domain word widens the search instead of emptying it.

    Args:
        dataset_id: Dataset to read.
        domain: Optional domain to focus on.
        metric_name: Optional metric to focus on.

    Returns:
        A summary of the projected movement.
    """
    resolved = normalize_domain(domain)
    widened = domain is not None and resolved is None

    # Metric names differ per business and cannot be mapped, but the WORD the
    # model used usually resolves to a domain. Infer it so the fallback stays
    # scoped to one domain instead of sweeping all eight.
    inferred_domain = False
    if resolved is None and not widened and metric_name:
        guess = normalize_domain(metric_name)
        if guess:
            resolved = guess
            inferred_domain = True
            logger.info("Domain inferred from metric %r -> %r", metric_name, guess)

    metric_filter = metric_name
    metric_dropped = False

    try:
        with session_scope() as session:
            forecasts = dataset_forecasts(
                dataset_id, domain=resolved, metric_name=metric_filter, session=session
            )

            # Metric fail-open: the real column may be 'units_produced' while the
            # model asked for 'production'. Drop the metric filter, keep the
            # domain, return every forecast in it — the model picks the right one.
            if not forecasts and metric_filter:
                metric_filter = None
                metric_dropped = True
                logger.info(
                    "Metric %r matched no forecasts; retrying without metric "
                    "filter (domain=%s)",
                    metric_name,
                    resolved,
                )
                forecasts = dataset_forecasts(
                    dataset_id, domain=resolved, metric_name=None, session=session
                )
    except Exception as exc:  # noqa: BLE001
        logger.exception("ml_forecast failed for dataset_id=%s", dataset_id)
        return tool_error(f"Could not read forecasts for dataset {dataset_id}: {exc}")

    if not forecasts:
        scope = _describe_scope(resolved, metric_filter)
        return tool_error(f"No forecasts for dataset {dataset_id}{scope}.")

    # Group by (domain, metric) into series, then describe first->last movement.
    series: dict[tuple[str, str], list[Any]] = {}
    for f in forecasts:
        series.setdefault((f.domain, f.metric_name), []).append(f)

    described: list[dict[str, Any]] = []
    for (dom, metric), pts in series.items():
        pts_sorted = sorted(pts, key=lambda x: x.forecast_date)
        start = _first_forecast(pts_sorted)
        end = _last_forecast(pts_sorted)
        described.append(
            {
                "domain": dom,
                "metric": metric,
                "horizon_days": len(pts_sorted),
                "next_value": _round(start),
                "end_value": _round(end),
                "direction": _direction(start, end),
                "movement": _movement(start, end),
            }
        )

    # Widened search can return every domain's forecast. Rank by projected
    # movement and keep only the strongest, so the 3B model sees signal.
    if widened and len(described) > _WIDE_CAP:
        described.sort(key=lambda d: d["movement"], reverse=True)
        described = described[:_WIDE_CAP]

    summary_bits = [
        f"{d['domain']}/{d['metric']}: {d['direction']} "
        f"(next {d['next_value']} → end {d['end_value']})"
        for d in described[:_WIDE_CAP]
    ]
    # Tell the model plainly what happened to its filters. Left unsaid, it guesses.
    if widened:
        prefix = (
            f"Dataset {dataset_id} — '{domain}' is not a known domain, so I forecast "
            f"across all domains. Biggest projected moves: "
        )
    elif metric_dropped:
        prefix = (
            f"Dataset {dataset_id} — there is no metric named '{metric_name}' in "
            f"this dataset. Here are ALL forecasts in the "
            f"{'`' + resolved + '`' if resolved else 'dataset'} domain — identify "
            f"which one corresponds to '{metric_name}' and answer using it: "
        )
    elif inferred_domain:
        prefix = f"Dataset {dataset_id} forecast for {resolved} — "
    else:
        prefix = f"Dataset {dataset_id} forecast — "

    summary = prefix + "; ".join(summary_bits)
    return tool_ok(
        summary=summary,
        data={
            "dataset_id": dataset_id,
            "domain_requested": domain,
            "domain_used": resolved,
            "domain_inferred_from_metric": inferred_domain,
            "metric_requested": metric_name,
            "metric_filter_dropped": metric_dropped,
            "searched_all_domains": widened,
            "series": described,
        },
    )


# ============================================================
# Tool schemas — what the LLM reads to decide the call
# ============================================================

ML_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "ml_overview_tool",
            "description": (
                "Check what machine-learning predictions exist for a dataset: how "
                "many forecasts, anomalies, and high-risk entities, and in which "
                "domains. Use this to see whether ML signal is worth drilling into "
                "before calling ml_alerts or ml_forecast."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "integer", "description": "The dataset id."}
                },
                "required": ["dataset_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ml_alerts",
            "description": (
                "Get what is going WRONG in a dataset: detected anomalies (unusual "
                "spikes/drops with severity) and entity risk scores (which "
                "machines/assets are most at risk). Use for questions about "
                "problems, warnings, what to worry about, root cause, e.g. 'what "
                "anomalies are there', 'which asset is most at risk', or as "
                "corroborating evidence for why a metric dropped. Optionally filter "
                "by domain."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "integer", "description": "The dataset id."},
                    "domain": {
                        "type": "string",
                        "description": "Optional domain filter. "
                        + domain_hint_for_schema(),
                    },
                },
                "required": ["dataset_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ml_forecast",
            "description": (
                "See where a dataset's metrics are HEADING in the near future "
                "(predicted values and direction). Use for questions about the "
                "future, projections, what will happen next, e.g. 'what is the "
                "production forecast', 'will downtime get worse'. Optionally narrow "
                "to one domain and/or one metric."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "integer", "description": "The dataset id."},
                    "domain": {
                        "type": "string",
                        "description": "Optional domain filter. "
                        + domain_hint_for_schema(),
                    },
                    "metric_name": {
                        "type": "string",
                        "description": (
                            "Optional. The EXACT metric column name, e.g. "
                            "'units_produced'. If you do not know the exact name, "
                            "OMIT this and pass only the domain — every metric in "
                            "that domain will be returned and you can pick the "
                            "right one."
                        ),
                    },
                },
                "required": ["dataset_id"],
            },
        },
    },
]


# Maps tool name -> callable, so the graph can dispatch by the name the LLM returns.
ML_TOOL_FUNCTIONS = {
    "ml_overview_tool": ml_overview_tool,
    "ml_alerts": ml_alerts,
    "ml_forecast": ml_forecast,
}


# ============================================================
# Small formatting helpers
# ============================================================


def _round(value: float | None, places: int = 2) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), places)
    except (TypeError, ValueError):
        return None


def _direction(start: float | None, end: float | None) -> str:
    if start is None or end is None:
        return "unknown"
    if end > start * 1.02:
        return "rising"
    if end < start * 0.98:
        return "falling"
    return "flat"


def _severity_counts(anomalies: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for a in anomalies:
        key = (a.severity or "unknown").lower()
        counts[key] = counts.get(key, 0) + 1
    return counts


def _first_forecast(points: list[Any]) -> float | None:
    for p in points:
        if p.forecast_value is not None:
            return p.forecast_value
    return None


def _last_forecast(points: list[Any]) -> float | None:
    for p in reversed(points):
        if p.forecast_value is not None:
            return p.forecast_value
    return None


def _describe_scope(domain: str | None, metric_name: str | None) -> str:
    bits = []
    if domain:
        bits.append(f" in domain '{domain}'")
    if metric_name:
        bits.append(f" for metric '{metric_name}'")
    return "".join(bits)


# When an unknown domain widens the search to every domain, cap how many series
# reach the model — the same signal-over-noise guard used in analytics_tool.
_WIDE_CAP = 6


def _movement(start: float | None, end: float | None) -> float:
    # Magnitude-of-change score, used only to rank series when the search was
    # widened. Relative, so a large metric doesn't automatically outrank a small one.
    if start is None or end is None:
        return 0.0
    if start == 0:
        return abs(end)
    return abs((end - start) / start)
