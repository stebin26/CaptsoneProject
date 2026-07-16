from __future__ import annotations

import re
from typing import Any

from ops_common.logging import get_logger

from .analytics_tool import ANALYTICS_TOOL_FUNCTIONS, ANALYTICS_TOOL_SCHEMAS
from .hub_tool import HUB_TOOL_FUNCTIONS, HUB_TOOL_SCHEMAS
from .intelligence_tool import (
    INTELLIGENCE_TOOL_FUNCTIONS,
    INTELLIGENCE_TOOL_SCHEMAS,
)
from .ml_tool import ML_TOOL_FUNCTIONS, ML_TOOL_SCHEMAS
from .rag_tool import RAG_TOOL_FUNCTIONS, RAG_TOOL_SCHEMAS

logger = get_logger(__name__)


# ============================================================
# The groups
# ============================================================

_GROUPS: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {
    "hub": (HUB_TOOL_SCHEMAS, HUB_TOOL_FUNCTIONS),
    "analytics": (ANALYTICS_TOOL_SCHEMAS, ANALYTICS_TOOL_FUNCTIONS),
    "ml": (ML_TOOL_SCHEMAS, ML_TOOL_FUNCTIONS),
    "intelligence": (INTELLIGENCE_TOOL_SCHEMAS, INTELLIGENCE_TOOL_FUNCTIONS),
    "rag": (RAG_TOOL_SCHEMAS, RAG_TOOL_FUNCTIONS),
}

# Always on. Discovery + current state underpin nearly every question, and they
# are the agent's fallback when a specialised tool returns nothing.
_ALWAYS: tuple[str, ...] = ("hub", "analytics")


# ============================================================
# Intent vocabularies
# ============================================================
# Each group lists the words that signal a manager wants that capability. These
# are intents, not exact tool names — the vocabulary a person actually uses.

_INTENT_WORDS: dict[str, set[str]] = {
    # Looking FORWARD, or asking what is wrong / at risk right now.
    "ml": {
        "forecast", "forecasts", "predict", "prediction", "predicted",
        "future", "next", "upcoming", "projection", "project", "expect",
        "will", "heading", "outlook", "tomorrow", "week", "month",
        "risk", "risks", "risky", "at-risk", "anomaly", "anomalies",
        "abnormal", "unusual", "outlier", "outliers", "alert", "alerts",
        "warning", "warnings", "fail", "failure", "breakdown", "danger",
    },
    # Asking WHY — causation, connection between areas.
    "intelligence": {
        "why", "cause", "causes", "causing", "reason", "reasons",
        "root", "root-cause", "rootcause", "driver", "driving", "drive",
        "because", "impact", "impacts", "affect", "affects", "affecting",
        "influence", "connection", "connected", "relationship", "related",
        "cross-domain", "crossdomain", "across", "link", "linked",
        "explain", "behind", "underlying", "loop", "knock-on",
    },
    # Asking what a DOCUMENT says — procedures, policy, definitions.
    "rag": {
        "document", "documents", "manual", "manuals", "sop", "sops",
        "procedure", "procedures", "policy", "policies", "guideline",
        "guidelines", "standard", "standards", "spec", "specification",
        "handbook", "instructions", "protocol", "rule", "rules",
        "escalation", "escalate", "code", "codes", "error-code",
        "says", "say", "state", "stated", "written", "documented",
        "compliance", "requirement", "requirements", "threshold",
    },
    # Current state / history. (analytics is always on, but these words also
    # confirm the intent and are kept for clarity + future tuning.)
    "analytics": {
        "trend", "trends", "trending", "history", "historical", "over",
        "current", "status", "now", "today", "recent", "average",
        "compare", "comparison", "performance", "degrading", "declining",
        "rising", "falling", "increase", "decrease", "worst", "best",
        "drop", "drops", "dropping", "dropped", "down", "up", "change",
        "which", "rank", "ranking", "top",
    },
    # Discovery / raw values. (also always on.)
    "hub": {
        "dataset", "datasets", "data", "available", "list", "show",
        "raw", "readings", "records", "entities", "machine", "machines",
        "asset", "assets", "line", "lines", "what",
    },
}

# A "why" question can never be answered from the metric that dropped — the
# cause lives in a DIFFERENT domain (a maintenance breakdown behind an
# operations drop). So any causal word force-adds the evidence groups the agent
# needs to actually find that cause, whether or not their own vocabulary hit.
_CAUSAL_WORDS: set[str] = {
    "why", "cause", "causes", "causing", "reason", "reasons",
    "root", "rootcause", "driver", "driving", "because", "explain",
    "behind", "underlying", "blame", "responsible",
    "drop", "drops", "dropping", "dropped",
    "falling", "fell", "declining", "decline", "degrading", "worse",
}

# Groups a causal question always needs, on top of whatever else matched.
_CAUSAL_COMPANIONS: tuple[str, ...] = ("ml", "intelligence")



# ============================================================
# Selection
# ============================================================

def select_tools(question: str) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    # Returns (schemas, functions, group_names) — exactly what run_once needs.
    # group_names is returned so the caller can log/debug what was exposed.
    tokens = _tokenize(question)

    matched: set[str] = set(_ALWAYS)

    # Test EVERY group's vocabulary, including the always-on ones. We need to
    # know whether the question showed *any* recognizable intent — if hub or
    # analytics words are present, the baseline is a deliberate, correct choice,
    # not an absence of signal.
    any_intent = False
    for group, vocabulary in _INTENT_WORDS.items():
        if tokens & vocabulary:
            matched.add(group)
            any_intent = True

    # Causal questions need corroborating evidence (anomalies, risk, cross-domain
    # links), not just the trend of the metric being asked about. Force those
    # groups in — the vocabulary alone would have missed them.
    if tokens & _CAUSAL_WORDS:
        matched.update(_CAUSAL_COMPANIONS)
        any_intent = True
        logger.info("Tool selector: causal question — adding %s", list(_CAUSAL_COMPANIONS))

    # Fail-open: only widen to everything when the question showed NO recognizable
    # intent at all AND is substantive. A question that clearly wants the baseline
    # ("what does this dataset contain?") keeps the baseline — that is the whole
    # point of selecting.
    if not any_intent and _looks_substantive(tokens):
        matched = set(_GROUPS)
        logger.info("Tool selector: no recognizable intent; exposing all groups.")
        
    schemas: list[dict[str, Any]] = []
    functions: dict[str, Any] = {}
    # Stable order: discovery, state, then the specialised groups.
    for group in ("hub", "analytics", "ml", "intelligence", "rag"):
        if group not in matched:
            continue
        group_schemas, group_functions = _GROUPS[group]
        schemas.extend(group_schemas)
        functions.update(group_functions)

    group_names = sorted(matched)
    logger.info(
        "Tool selector: %d tools from groups %s",
        len(functions), group_names,
    )
    return schemas, functions, group_names


def _tokenize(question: str) -> set[str]:
    # Lowercase word tokens; hyphens kept as separators so "root-cause" also
    # yields "root" and "cause".
    lowered = (question or "").lower()
    words = re.findall(r"[a-z][a-z\-]*", lowered)
    tokens: set[str] = set()
    for w in words:
        tokens.add(w)
        if "-" in w:
            tokens.update(part for part in w.split("-") if part)
    return tokens


def _looks_substantive(tokens: set[str]) -> bool:
    # A real question (not a greeting) with no clear intent should get every
    # tool. Very short/trivial input can stay on the baseline.
    return len(tokens) >= 3


# ============================================================
# Escape hatch
# ============================================================

def all_tools() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Every tool, unfiltered. Used when selection is disabled via config, and by
    # tests that need the full surface.
    schemas: list[dict[str, Any]] = []
    functions: dict[str, Any] = {}
    for group in ("hub", "analytics", "ml", "intelligence", "rag"):
        group_schemas, group_functions = _GROUPS[group]
        schemas.extend(group_schemas)
        functions.update(group_functions)
    return schemas, functions