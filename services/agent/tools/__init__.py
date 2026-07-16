from __future__ import annotations

from typing import Any

from ops_common.logging import get_logger

from .base import ToolFn, ToolResult, dispatch, tool_error, tool_ok

# Import each tool module's schemas + functions.
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
# Aggregate every tool
# ============================================================
# Order matters only for how the schemas are presented to the model; we lead
# with discovery/analytics (the usual first moves) and end with document search.

ALL_TOOL_SCHEMAS: list[dict[str, Any]] = [
    *HUB_TOOL_SCHEMAS,           # discovery (list datasets) + raw ground truth
    *ANALYTICS_TOOL_SCHEMAS,     # current state (overview, trend, features)
    *ML_TOOL_SCHEMAS,            # future + alerts (forecast, anomalies, risk)
    *INTELLIGENCE_TOOL_SCHEMAS,  # cross-domain "why" / root cause
    *RAG_TOOL_SCHEMAS,           # document-grounded knowledge
]

ALL_TOOL_FUNCTIONS: dict[str, ToolFn] = {
    **HUB_TOOL_FUNCTIONS,
    **ANALYTICS_TOOL_FUNCTIONS,
    **ML_TOOL_FUNCTIONS,
    **INTELLIGENCE_TOOL_FUNCTIONS,
    **RAG_TOOL_FUNCTIONS,
}


# ============================================================
# Consistency check — fail loud at import, not at inference
# ============================================================

def _schema_names(schemas: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for s in schemas:
        fn = (s.get("function") or {})
        name = fn.get("name")
        if name:
            names.append(name)
    return names


def _validate_registry() -> None:
    schema_names = _schema_names(ALL_TOOL_SCHEMAS)
    fn_names = set(ALL_TOOL_FUNCTIONS)

    # Duplicate schema names would make the model's choice ambiguous.
    seen: set[str] = set()
    duplicates = {n for n in schema_names if n in seen or seen.add(n)}
    if duplicates:
        raise RuntimeError(f"Duplicate tool schema names: {sorted(duplicates)}")

    schema_set = set(schema_names)

    # Every advertised schema must have a callable behind it.
    missing_fn = schema_set - fn_names
    if missing_fn:
        raise RuntimeError(
            f"Tool schemas without matching functions: {sorted(missing_fn)}"
        )

    # Every callable should be advertised, or the model can never reach it.
    missing_schema = fn_names - schema_set
    if missing_schema:
        raise RuntimeError(
            f"Tool functions without matching schemas: {sorted(missing_schema)}"
        )

    logger.info("Tool registry OK — %d tools registered.", len(fn_names))


# Run the check at import so a mismatch surfaces the moment the app loads.
_validate_registry()


# ============================================================
# Convenience helpers
# ============================================================

def get_all_schemas() -> list[dict[str, Any]]:
    return ALL_TOOL_SCHEMAS


def get_all_functions() -> dict[str, ToolFn]:
    return ALL_TOOL_FUNCTIONS


def tool_names() -> list[str]:
    return sorted(ALL_TOOL_FUNCTIONS)


# Re-export the tool-contract essentials so callers can `from .tools import ...`.
__all__ = [
    "ALL_TOOL_SCHEMAS",
    "ALL_TOOL_FUNCTIONS",
    "get_all_schemas",
    "get_all_functions",
    "tool_names",
    "dispatch",
    "ToolResult",
    "ToolFn",
    "tool_ok",
    "tool_error",
]