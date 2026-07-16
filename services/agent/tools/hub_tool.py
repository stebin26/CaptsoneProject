from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from ops_common.db import session_scope
from ops_common.logging import get_logger

from .base import ToolResult, domain_hint_for_schema, normalize_domain, tool_error, tool_ok

# Reuse the router's own functions so there is ONE source of truth for hub reads.
from api_app.routers.v1.domains import (
    dataset_summary,
    domain_data,
    list_datasets,
)

logger = get_logger(__name__)


# ============================================================
# Tool 1 — list datasets (discovery)
# ============================================================

def list_datasets_tool() -> ToolResult:
    # No arguments: the agent uses this to discover what data exists, e.g. when
    # the user asks about "the factory data" without giving an id, or to confirm
    # a dataset_id is valid before drilling in.
    try:
        with session_scope() as session:
            datasets = list_datasets(session)
    except Exception as exc:  # noqa: BLE001
        logger.exception("list_datasets_tool failed")
        return tool_error(f"Could not list datasets: {exc}")

    if not datasets:
        return tool_error("No datasets have been onboarded yet.")

    listed = [
        {
            "dataset_id": d.dataset_id,
            "business_name": d.business_name,
            "industry": d.industry,
            "rows": d.row_count,
            "features_collected": d.features_collected,
        }
        for d in datasets
    ]

    summary = (
        f"{len(datasets)} dataset(s) available: "
        + "; ".join(
            f"id {d['dataset_id']} = {d['business_name']}"
            + (f" ({d['industry']})" if d["industry"] else "")
            for d in listed[:12]
        )
    )
    return tool_ok(
        summary=summary,
        data={"dataset_count": len(datasets), "datasets": listed},
    )


# ============================================================
# Tool 2 — hub summary (quick per-metric shape of a dataset)
# ============================================================

def hub_summary(dataset_id: int) -> ToolResult:
    # A compact per-metric summary straight from the hub (observations + basic
    # aggregates). Lighter than analytics_overview and works even before the
    # analytics pipeline runs, since it reads the hub views directly.
    try:
        with session_scope() as session:
            result = _call_with_dataset_guard(dataset_summary, session, dataset_id)
    except _DatasetMissing:
        return tool_error(f"Dataset {dataset_id} not found.")
    except Exception as exc:  # noqa: BLE001
        logger.exception("hub_summary failed for dataset_id=%s", dataset_id)
        return tool_error(f"Could not read hub summary for dataset {dataset_id}: {exc}")

    if not result.metrics:
        return tool_error(
            f"Dataset {dataset_id} ({result.business_name}) has no metric data in the hub yet."
        )

    per_metric = [
        {
            "domain": m.domain,
            "metric": m.metric_name,
            "obs": m.observations,
            "avg": _round(m.metric_avg),
            "min": _round(m.metric_min),
            "max": _round(m.metric_max),
        }
        for m in result.metrics
    ]
    domains = sorted({m.domain for m in result.metrics})

    summary = (
        f"Dataset {dataset_id} — {result.business_name}. "
        f"Domains in hub: {', '.join(domains)}. "
        f"{len(result.metrics)} metrics."
    )
    return tool_ok(
        summary=summary,
        data={
            "dataset_id": dataset_id,
            "business_name": result.business_name,
            "domains": domains,
            "metrics": per_metric,
        },
    )


# ============================================================
# Tool 3 — hub domain data (raw entity readings — ground truth)
# ============================================================

def hub_domain_data(dataset_id: int, domain: str, limit: int = 200) -> ToolResult:
    # Raw per-entity readings for one domain. The agent reaches here when derived
    # layers are not enough and it needs the actual values. Capped + summarized
    # so a large table never floods the model's context.
    #
    # Unlike the analytics/ML tools, `domain` is REQUIRED by the underlying hub
    # view (it reads one domain at a time). So an unrecognized word cannot simply
    # drop the filter — instead we sweep every domain and return the ones that
    # actually hold data, which keeps the fail-open promise: an unknown word
    # widens the search, it never returns nothing.
    resolved = normalize_domain(domain)

    if resolved is None:
        return _hub_sweep_all_domains(dataset_id, requested=domain, limit=limit)

    try:
        with session_scope() as session:
            result = _call_with_dataset_guard(
                domain_data, session, dataset_id, domain=resolved, limit=limit
            )
    except _DatasetMissing:
        return tool_error(f"Dataset {dataset_id} not found.")
    except HTTPException as exc:
        return tool_error(
            f"Cannot read domain '{resolved}' for dataset {dataset_id}: {exc.detail}"
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("hub_domain_data failed for dataset_id=%s domain=%s", dataset_id, resolved)
        return tool_error(f"Could not read hub data for dataset {dataset_id}: {exc}")

    if not result.points:
        return tool_error(
            f"No raw readings for dataset {dataset_id} in domain '{resolved}'."
        )

    entities, sample = _summarize_points(result.points)
    summary = (
        f"Dataset {dataset_id} domain '{resolved}': {len(result.points)} raw readings "
        f"across {len(entities)} entities. "
        f"Entities: {', '.join(e['entity'] for e in entities[:10])}."
    )
    return tool_ok(
        summary=summary,
        data={
            "dataset_id": dataset_id,
            "domain_requested": domain,
            "domain_used": resolved,
            "searched_all_domains": False,
            "reading_count": len(result.points),
            "entities": entities,
            "sample": sample,
        },
    )


def _hub_sweep_all_domains(
    dataset_id: int,
    requested: str,
    limit: int,
) -> ToolResult:
    # Fallback for an unrecognized domain word: probe every canonical domain and
    # report the ones that hold data. We deliberately request a SMALL slice per
    # domain — the goal is to tell the model where the data lives, not to dump
    # eight domains of raw rows into a 3B context window.
    per_domain_cap = max(20, min(limit, 60))
    found: list[dict[str, Any]] = []

    try:
        with session_scope() as session:
            for dom in CANONICAL_DOMAINS:
                try:
                    result = _call_with_dataset_guard(
                        domain_data, session, dataset_id, domain=dom, limit=per_domain_cap
                    )
                except _DatasetMissing:
                    return tool_error(f"Dataset {dataset_id} not found.")
                except Exception:  # noqa: BLE001
                    # A single domain failing must not sink the whole sweep.
                    continue

                if not result.points:
                    continue

                entities, _ = _summarize_points(result.points)
                metrics = sorted({p.metric_name for p in result.points})
                found.append(
                    {
                        "domain": dom,
                        "readings_sampled": len(result.points),
                        "entities": [e["entity"] for e in entities][:6],
                        "metrics": metrics[:6],
                    }
                )
    except Exception as exc:  # noqa: BLE001
        logger.exception("hub sweep failed for dataset_id=%s", dataset_id)
        return tool_error(f"Could not read hub data for dataset {dataset_id}: {exc}")

    if not found:
        return tool_error(f"Dataset {dataset_id} has no raw hub readings in any domain.")

    summary = (
        f"Dataset {dataset_id} — '{requested}' is not a known domain, so I checked "
        f"all of them. Data exists in: "
        + "; ".join(
            f"{f['domain']} ({', '.join(f['metrics'])})" for f in found
        )
        + ". Ask again naming one of these domains for the raw readings."
    )
    return tool_ok(
        summary=summary,
        data={
            "dataset_id": dataset_id,
            "domain_requested": requested,
            "domain_used": None,
            "searched_all_domains": True,
            "domains_with_data": found,
        },
    )

# ============================================================
# Tool schemas — what the LLM reads to decide the call
# ============================================================

HUB_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_datasets_tool",
            "description": (
                "List all available datasets with their ids, business names, and "
                "industries. Use this FIRST when the user refers to data without "
                "giving a dataset id (e.g. 'the factory data', 'the telecom "
                "dataset'), so you can find the right dataset_id to use, or to "
                "confirm what data exists. Takes no arguments."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hub_summary",
            "description": (
                "Get a quick per-metric summary of one dataset straight from the "
                "data hub (which domains, how many observations, basic averages). "
                "Use for a fast shape-of-the-data look, especially if analytics has "
                "not been computed. For richer aggregates prefer analytics_overview."
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
            "name": "hub_domain_data",
            "description": (
                "Get the raw entity-level readings for one domain of a dataset "
                "(the actual values per machine/asset/entity). Use when summaries "
                "are not enough and you need ground-truth readings, e.g. 'show the "
                "raw downtime readings', 'what are the actual values for machine "
                "M-12'. Results are capped and summarized. Domain must be one of: "
                "assets, operations, quality, maintenance, inventory, workforce, "
                "finance, customers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "integer", "description": "The dataset id."},
                    "domain": {
                        "type": "string",
                        "description": "The domain to read raw data from. " + domain_hint_for_schema(),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max raw rows to scan (default 200).",
                    },
                },
                "required": ["dataset_id", "domain"],
            },
        },
    },
]


# Maps tool name -> callable, so the graph can dispatch by the name the LLM returns.
HUB_TOOL_FUNCTIONS = {
    "list_datasets_tool": list_datasets_tool,
    "hub_summary": hub_summary,
    "hub_domain_data": hub_domain_data,
}


# ============================================================
# Dataset-guard plumbing
# ============================================================
# domains.py raises HTTPException(404) via _require_dataset when a dataset is
# missing. In-process we translate that into a clean tool error rather than
# letting a 404 bubble up as a raw exception.

class _DatasetMissing(Exception):
    pass


def _call_with_dataset_guard(fn: Any, session: Any, dataset_id: int, **kwargs: Any) -> Any:
    try:
        return fn(dataset_id, session=session, **kwargs)
    except HTTPException as exc:
        if exc.status_code == 404 and "not found" in str(exc.detail).lower():
            raise _DatasetMissing() from exc
        raise

def _summarize_points(points: list[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Collapse raw readings into per-entity counts + a tiny concrete sample.
    # Shared by the single-domain read and the all-domain sweep so both compress
    # the same way.
    per_entity: dict[str, dict[str, Any]] = {}
    for p in points:
        bucket = per_entity.setdefault(
            p.entity_ref, {"entity": p.entity_ref, "readings": 0, "metrics": set()}
        )
        bucket["readings"] += 1
        bucket["metrics"].add(p.metric_name)

    entities = [
        {"entity": b["entity"], "readings": b["readings"], "metrics": sorted(b["metrics"])}
        for b in per_entity.values()
    ]
    sample = [
        {
            "entity": p.entity_ref,
            "metric": p.metric_name,
            "value": _round(p.metric_value),
            "at": p.recorded_at,
        }
        for p in points[:5]
    ]
    return entities, sample

# ============================================================
# Small formatting helper
# ============================================================

def _round(value: float | None, places: int = 2) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), places)
    except (TypeError, ValueError):
        return None