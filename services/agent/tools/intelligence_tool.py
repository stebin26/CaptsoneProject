# services/agent/tools/intelligence_tool.py
"""Intelligence tool — the agent's window into the Phase 3 cross-domain engine.

This is the tool that makes the agent more than a dashboard reader. The other
tools report on ONE domain at a time; this one reports on how domains AFFECT
each other — the knowledge-graph inference over ML signals (e.g. "maintenance
downtime is driving the production drop, and it loops back into asset wear").

It wraps the existing intelligence endpoint (intelligence.py), which already
runs the inference engine + translator and returns business-term insights. The
tool's job is to compress those insight objects into a short, ranked, LLM-facing
summary so a 3B model can weave them into a root-cause answer without drowning
in the full insight payload.

One tool exposed:
  cross_domain_insights — the ranked cross-domain story for a dataset
"""

from __future__ import annotations

from typing import Any

# Reuse the router's own function so there is ONE source of truth for how
# insights are produced (inference engine + translator run inside it).
from api_app.routers.v1.intelligence import cross_domain_intelligence
from ops_common.db import session_scope
from ops_common.logging import get_logger

from .base import ToolResult, tool_error, tool_ok

logger = get_logger(__name__)


# ============================================================
# Tool — cross-domain insights (the "how it connects" view)
# ============================================================


def cross_domain_insights(dataset_id: int) -> ToolResult:
    # The agent reaches for this on "why" and "root cause" questions, where a
    # single-domain look is not enough and the answer lies in domain-to-domain
    # influence. Returns the ranked insights already translated to the business's
    # own terms by the engine.
    """Report how a dataset's domains are affecting each other.

    Runs the cross-domain inference and compresses the resulting insights into a
    short ranked summary, so the model can weave a root-cause story without
    drowning in the full insight payload.

    Args:
        dataset_id: Dataset to analyze.

    Returns:
        A ranked summary of the cross-domain insights.
    """
    try:
        with session_scope() as session:
            result = cross_domain_intelligence(dataset_id, session)
    except Exception as exc:  # noqa: BLE001
        logger.exception("cross_domain_insights failed for dataset_id=%s", dataset_id)
        return tool_error(
            f"Could not read cross-domain intelligence for dataset {dataset_id}: {exc}"
        )

    if result.insight_count == 0:
        active = ", ".join(result.active_domains) or "none"
        return tool_error(
            f"No cross-domain insights for dataset {dataset_id}. "
            f"Active domains with ML signal: {active}. There may not be enough "
            "corroborating signal across domains to connect them."
        )

    business = result.business_name or "Unknown"
    industry = result.industry or "Unknown"

    # Insights arrive ranked (strongest first). Describe the top few compactly:
    # each is a root domain influencing one or more impacted domains, with the
    # relationship strength and effect direction.
    described: list[dict[str, Any]] = []
    for insight in result.insights[:5]:
        impacted = [
            {
                "domain": imp.domain,
                "term": imp.term,
                "strength": imp.strength,
                "effect": imp.effect,
            }
            for imp in insight.impacted
        ]
        described.append(
            {
                "root": insight.root,
                "root_term": insight.root_term,
                "direction": insight.direction,
                "score": _round(insight.score),
                "narrative": insight.narrative,
                "recommendation": insight.recommendation,
                "impacted": impacted,
            }
        )

    # Build a one-line-per-insight summary the model can reason and quote from.
    summary_lines: list[str] = []
    for d in described:
        targets = (
            ", ".join(
                f"{imp['term']} ({imp['strength']}, {imp['effect']})"
                for imp in d["impacted"]
            )
            or "no clear downstream"
        )
        summary_lines.append(f"{d['root_term']} {d['direction']} → {targets}")

    summary = (
        f"Dataset {dataset_id} — {business} ({industry}). "
        f"{result.insight_count} cross-domain insight(s) over domains "
        f"{', '.join(result.active_domains)}. Top: " + " | ".join(summary_lines)
    )
    return tool_ok(
        summary=summary,
        data={
            "dataset_id": dataset_id,
            "business_name": business,
            "industry": industry,
            "active_domains": result.active_domains,
            "insight_count": result.insight_count,
            "insights": described,
        },
    )


# ============================================================
# Tool schema — what the LLM reads to decide the call
# ============================================================

INTELLIGENCE_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "cross_domain_insights",
            "description": (
                "Get the cross-domain story for a dataset: how one business area "
                "is affecting others (e.g. maintenance downtime driving a "
                "production drop, which loops back into asset wear). This connects "
                "the domains instead of looking at one in isolation. Use for 'why' "
                "and 'root cause' questions where the cause likely lies in one "
                "domain influencing another, e.g. 'why is production down', 'what "
                "is causing the quality issues', 'what is the biggest problem and "
                "why'. Returns ranked insights already phrased in the business's "
                "own terms."
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
]


# Maps tool name -> callable, so the graph can dispatch by the name the LLM returns.
INTELLIGENCE_TOOL_FUNCTIONS = {
    "cross_domain_insights": cross_domain_insights,
}


# ============================================================
# Small formatting helper
# ============================================================


def _round(value: float | None, places: int = 3) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), places)
    except (TypeError, ValueError):
        return None
