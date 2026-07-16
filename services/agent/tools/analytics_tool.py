from __future__ import annotations

from typing import Any

from ops_common.db import session_scope
from ops_common.logging import get_logger

from .base import ToolResult, domain_hint_for_schema, normalize_domain, tool_error, tool_ok

# Reuse the router's own query functions so there is ONE source of truth for
# analytics reads. If these signatures ever change, the tool changes with them.
from api_app.routers.v1.analytics import (
    dataset_features,
    dataset_metrics,
    dataset_trend,
)

logger = get_logger(__name__)


# ============================================================
# Tool 1 — analytics overview (the "current state" of a dataset)
# ============================================================

def analytics_overview(dataset_id: int) -> ToolResult:
    # Give the model a domain-by-domain snapshot: which domains exist and, per
    # metric, the headline aggregates. This is the agent's default first look.
    try:
        with session_scope() as session:
            metrics = dataset_metrics(dataset_id, session)
    except Exception as exc:  # noqa: BLE001
        logger.exception("analytics_overview failed for dataset_id=%s", dataset_id)
        return tool_error(f"Could not read analytics for dataset {dataset_id}: {exc}")

    if not metrics:
        return tool_error(
            f"No analytics found for dataset {dataset_id}. "
            "It may not be processed yet, or the id is wrong."
        )

    business = metrics[0].business_name or "Unknown"
    industry = metrics[0].industry or "Unknown"
    domains = sorted({m.domain for m in metrics})

    # Condense to one line per metric so the whole state fits in context.
    per_metric: list[dict[str, Any]] = [
        {
            "domain": m.domain,
            "metric": m.metric_name,
            "avg": _round(m.avg_value),
            "min": _round(m.min_value),
            "max": _round(m.max_value),
            "rows": m.row_count,
            "entities": m.distinct_entities,
        }
        for m in metrics
    ]

    summary = (
        f"Dataset {dataset_id} — {business} ({industry}). "
        f"Active domains: {', '.join(domains)}. "
        f"{len(metrics)} metrics across {len(domains)} domains."
    )
    return tool_ok(
        summary=summary,
        data={
            "dataset_id": dataset_id,
            "business_name": business,
            "industry": industry,
            "domains": domains,
            "metrics": per_metric,
        },
    )


# ============================================================
# Tool 2 — domain trend (is a metric rising or falling over time)
# ============================================================

def analytics_trend(
    dataset_id: int,
    domain: str | None = None,
    metric_name: str | None = None,
    include_context: bool = False,
) -> ToolResult:
    # The time dimension: used to answer "why is X changing" questions. Returns
    # first/last/direction per series rather than every daily point.
    #
    # Two fail-opens live here, and they run in this order:
    #   1. domain word unknown  -> drop the domain filter, search ALL domains
    #   2. metric word unknown  -> drop the METRIC filter, KEEP the domain
    # An unrecognized input must widen the search, never empty it.
    resolved = normalize_domain(domain)
    widened = domain is not None and resolved is None

    # The model often names a metric in user vocabulary ("production") with no
    # domain at all. Metric names differ per business and cannot be mapped, but
    # the WORD usually resolves to a domain — so infer it and stay scoped to one
    # domain instead of sweeping all eight.
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
            points = dataset_trend(
                dataset_id, domain=resolved, metric_name=metric_filter, session=session
            )

            # The metric fail-open. A business's real column may be
            # 'units_produced' while the model asked for 'production'. Rather
            # than reporting "no data" on a data-rich dataset, drop the metric
            # filter and return every metric in the domain — the model reads
            # them and works out which one the user meant.
            if not points and metric_filter:
                metric_filter = None
                metric_dropped = True
                logger.info(
                    "Metric %r matched no rows; retrying without metric filter "
                    "(domain=%s)", metric_name, resolved
                )
                points = dataset_trend(
                    dataset_id, domain=resolved, metric_name=None, session=session
                )
    except Exception as exc:  # noqa: BLE001
        logger.exception("analytics_trend failed for dataset_id=%s", dataset_id)
        return tool_error(f"Could not read trend for dataset {dataset_id}: {exc}")

    if not points:
        scope = _describe_scope(resolved, metric_filter)
        return tool_error(f"No trend data for dataset {dataset_id}{scope}.")

    # A cause never lives in the metric that changed — it lives in another part
    # of the business. Asking a small model to fetch that separately does not
    # work: it either stops early, or drowns once four tool results are in its
    # context. So when the question is causal, this ONE call also brings back the
    # biggest movers everywhere else. The model then reads a single ranked list
    # and spots the pattern, instead of having to plan and synthesise a
    # multi-tool investigation it cannot hold in its head.
    context_points: list[Any] = []
    if include_context and resolved:
        try:
            with session_scope() as session:
                context_points = dataset_trend(
                    dataset_id, domain=None, metric_name=None, session=session
                )
        except Exception as exc:  # noqa: BLE001
            # Best-effort. The primary answer already exists; losing the context
            # sweep must never lose the answer with it.
            logger.warning("Context sweep failed for dataset_id=%s: %s", dataset_id, exc)
            context_points = []

    # Group points into series and describe each series' movement compactly.
    series: dict[tuple[str, str], list[Any]] = {}
    for p in points:
        series.setdefault((p.domain, p.metric_name), []).append(p)

    described: list[dict[str, Any]] = []
    for (dom, metric), pts in series.items():
        pts_sorted = sorted(pts, key=lambda x: x.day)
        first = _first_valid(pts_sorted)
        last = _last_valid(pts_sorted)
        described.append(
            {
                "domain": dom,
                "metric": metric,
                "from_day": pts_sorted[0].day,
                "to_day": pts_sorted[-1].day,
                "start_avg": _round(first),
                "end_avg": _round(last),
                "direction": _direction(first, last),
                "movement": _movement(first, last),
            }
        )

    # Only a WIDENED search (unknown domain) can return all eight domains' series.
    # Rank by how much each moved and keep the strongest, so the 3B model gets
    # signal, not a wall of flat lines. A domain-scoped metric drop stays intact.
    if widened and len(described) > _WIDE_CAP:
        described.sort(key=lambda d: d["movement"], reverse=True)
        described = described[:_WIDE_CAP]

    # Organize the elsewhere-in-the-business series into two buckets by DIRECTION
    # relative to the series being asked about.
    #
    # What code can prove: which metrics moved the opposite way, and by how much.
    # What code must NOT claim: that any of them caused anything. Correlation is
    # not causation, and a tool that declares causes would be lying with a
    # confident voice. So the buckets are named for what they factually are —
    # 'opposite' and 'together' — and the LLM is left to reason and hedge.
    opposite: list[dict[str, Any]] = []
    together: list[dict[str, Any]] = []

    if context_points and described:
        # The direction of the series the question is actually about. With a
        # metric filter this is one series; otherwise take the strongest mover.
        anchor = max(described, key=lambda d: d["movement"])
        anchor_dir = anchor["direction"]

        ctx_series: dict[tuple[str, str], list[Any]] = {}
        for p in context_points:
            if p.domain == resolved:
                continue  # already reported above; do not repeat it
            ctx_series.setdefault((p.domain, p.metric_name), []).append(p)

        for (dom, metric), pts in ctx_series.items():
            pts_sorted = sorted(pts, key=lambda x: x.day)
            first = _first_valid(pts_sorted)
            last = _last_valid(pts_sorted)
            direction = _direction(first, last)
            if direction in ("unknown", "flat"):
                continue  # a metric that did not move is not evidence of anything

            row = {
                "domain": dom,
                "metric": metric,
                "start_avg": _round(first),
                "end_avg": _round(last),
                "direction": direction,
                "movement": _movement(first, last),
            }
            if direction != anchor_dir:
                opposite.append(row)
            else:
                together.append(row)

        opposite.sort(key=lambda d: d["movement"], reverse=True)
        together.sort(key=lambda d: d["movement"], reverse=True)
        opposite = opposite[:_CONTEXT_CAP]
        together = together[:_CONTEXT_CAP]

    summary_bits = [
        f"{d['domain']}/{d['metric']}: {d['direction']} "
        f"({d['start_avg']} → {d['end_avg']})"
        for d in described
    ]

    # Tell the model plainly what happened to its filters. Left unsaid, it will
    # guess — and a guess here is the fabrication we are trying to prevent.
    if widened:
        prefix = (
            f"Dataset {dataset_id} — '{domain}' is not a known domain, so I searched "
            f"all domains. Biggest movers: "
        )
    elif metric_dropped:
        prefix = (
            f"Dataset {dataset_id} — there is no metric named '{metric_name}' in "
            f"this dataset. Here are ALL metrics in the "
            f"{'`' + resolved + '`' if resolved else 'dataset'} domain — identify "
            f"which one corresponds to '{metric_name}' and answer using it: "
        )
    elif inferred_domain:
        prefix = f"Dataset {dataset_id} trends in {resolved} — "
    else:
        prefix = f"Dataset {dataset_id} trends — "

    summary = prefix + "; ".join(summary_bits)

    # Present the two buckets as observations, ranked, with the strongest first.
    # The instruction to the model is to WEIGH them — not to be told the answer.
    if opposite or together:
        summary += "\n\nWHAT MOVED ELSEWHERE over the same period:"

        if opposite:
            summary += "\n\nMOVED IN THE OPPOSITE DIRECTION (strongest first):\n" + "\n".join(
                f"  - {c['domain']}/{c['metric']}: {c['direction']} "
                f"({c['start_avg']} → {c['end_avg']})"
                for c in opposite
            )
        if together:
            summary += "\n\nMOVED IN THE SAME DIRECTION (strongest first):\n" + "\n".join(
                f"  - {c['domain']}/{c['metric']}: {c['direction']} "
                f"({c['start_avg']} → {c['end_avg']})"
                for c in together
            )

        summary += (
            "\n\nHOW TO READ THIS: metrics moving the OPPOSITE way are the "
            "strongest candidates for what is driving the change — the first one "
            "listed moved the most. Metrics moving the SAME way are more likely "
            "consequences of it than drivers of it. These are correlations, not "
            "proven causes: name the strongest one, say what it suggests, and be "
            "clear that it is an association the data shows, not a proven cause. "
            "Do NOT list every metric back — pick the strongest and explain it."
        )

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
            "context_included": bool(opposite or together),
            "series": described,
            "moved_opposite": opposite,
            "moved_together": together,
        },
    )


# ============================================================
# Tool 3 — entity features (which specific entities stand out)
# ============================================================

def analytics_features(
    dataset_id: int,
    domain: str | None = None,
    limit: int = 200,
) -> ToolResult:
    # Per-entity engineered features. For the agent we surface the notable ones:
    # strongest positive and negative trend slopes, which is what "which machine
    # is degrading" type questions need. Domain word normalized as above.
    resolved = normalize_domain(domain)
    widened = domain is not None and resolved is None

    try:
        with session_scope() as session:
            features = dataset_features(
                dataset_id, domain=resolved, limit=limit, session=session
            )
    except Exception as exc:  # noqa: BLE001
        logger.exception("analytics_features failed for dataset_id=%s", dataset_id)
        return tool_error(f"Could not read features for dataset {dataset_id}: {exc}")

    if not features:
        scope = _describe_scope(resolved, None)
        return tool_error(f"No features for dataset {dataset_id}{scope}.")

    # Rank by absolute trend slope so the most-moving entities surface first.
    ranked = sorted(
        features,
        key=lambda f: abs(f.trend_slope) if f.trend_slope is not None else 0.0,
        reverse=True,
    )
    top = ranked[:10]

    notable: list[dict[str, Any]] = [
        {
            "domain": f.domain,
            "entity": f.entity_ref,
            "metric": f.metric_name,
            "avg": _round(f.avg_value),
            "last": _round(f.last_value),
            "trend_slope": _round(f.trend_slope),
            "direction": _slope_direction(f.trend_slope),
        }
        for f in top
    ]

    prefix = (
        f"Dataset {dataset_id} — '{domain}' is not a known domain, so I searched "
        f"all domains. "
        if widened
        else f"Dataset {dataset_id} — "
    )
    summary = (
        prefix
        + f"{len(features)} entity-metric features"
        + ("" if widened else _describe_scope(resolved, None))
        + ". Top movers: "
        + "; ".join(
            f"{n['entity']}/{n['metric']} ({n['domain']}) {n['direction']}"
            for n in notable[:5]
        )
    )
    return tool_ok(
        summary=summary,
        data={
            "dataset_id": dataset_id,
            "domain_requested": domain,
            "domain_used": resolved,
            "searched_all_domains": widened,
            "total_features": len(features),
            "top_movers": notable,
        },
    )


# ============================================================
# Tool schemas — what the LLM reads to decide the call
# ============================================================
# These follow the OpenAI/Ollama function-calling schema. Descriptions are
# written FOR the model: they say plainly when to reach for each tool, because
# a small model leans heavily on these hints to route correctly.

ANALYTICS_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "analytics_overview",
            "description": (
                "Get the current state of a dataset: which business domains are "
                "present and the average/min/max of every metric. Use this FIRST "
                "for any question about what a dataset contains or its overall "
                "health, e.g. 'how is dataset 542 doing', 'what domains does it have'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "integer",
                        "description": "The numeric id of the dataset to inspect.",
                    }
                },
                "required": ["dataset_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analytics_trend",
            "description": (
                "See whether metrics are rising or falling over time for a dataset. "
                "Use this for questions about change, direction, increase or "
                "decrease, e.g. 'why is production dropping', 'is downtime rising'. "
                "Optionally narrow to one domain and/or one metric."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "integer", "description": "The dataset id."},
                    "domain": {
                        "type": "string",
                        "description": "Optional domain filter. " + domain_hint_for_schema(),
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
                    "include_context": {
                        "type": "boolean",
                        "description": (
                            "Set TRUE whenever the question asks WHY something is "
                            "changing, dropping, rising, or getting worse. This "
                            "also returns what moved everywhere else in the "
                            "business over the same period — which is where the "
                            "cause actually is. Leave FALSE for plain 'what is the "
                            "trend' questions."
                        ),
                    },
                },
                "required": ["dataset_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analytics_features",
            "description": (
                "Find which specific entities (machines, towers, vehicles, etc.) "
                "stand out, ranked by how strongly their values are trending. Use "
                "for questions about individual entities, e.g. 'which machine is "
                "degrading', 'what asset is the worst'. Optionally filter by domain."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "integer", "description": "The dataset id."},
                    "domain": {
                        "type": "string",
                        "description": "Optional domain filter. " + domain_hint_for_schema(),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max feature rows to scan (default 200).",
                    },
                },
                "required": ["dataset_id"],
            },
        },
    },
]


# Maps tool name -> callable, so the graph can dispatch by the name the LLM returns.
ANALYTICS_TOOL_FUNCTIONS = {
    "analytics_overview": analytics_overview,
    "analytics_trend": analytics_trend,
    "analytics_features": analytics_features,
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


def _slope_direction(slope: float | None) -> str:
    if slope is None:
        return "unknown"
    if slope > 0:
        return "rising"
    if slope < 0:
        return "falling"
    return "flat"


# How many movers ride along in EACH bucket on a causal question. Four, not six:
# the last run echoed all twelve metrics back at the user instead of picking one.
# A short ranked list forces a choice; a long one invites a recital.
_CONTEXT_CAP = 4


def _movement(start: float | None, end: float | None) -> float:
    # A magnitude-of-change score used only to rank series when the search was
    # widened. Relative, so a big metric doesn't always outrank a small one.
    if start is None or end is None:
        return 0.0
    if start == 0:
        return abs(end)
    return abs((end - start) / start)


def _first_valid(points: list[Any]) -> float | None:
    for p in points:
        if p.avg_value is not None:
            return p.avg_value
    return None


def _last_valid(points: list[Any]) -> float | None:
    for p in reversed(points):
        if p.avg_value is not None:
            return p.avg_value
    return None


def _describe_scope(domain: str | None, metric_name: str | None) -> str:
    bits = []
    if domain:
        bits.append(f" in domain '{domain}'")
    if metric_name:
        bits.append(f" for metric '{metric_name}'")
    return "".join(bits)