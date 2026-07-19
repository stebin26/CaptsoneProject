# services/agent/tools/base.py
"""Common tool contract for every agent tool.

Two responsibilities:
  1. ToolResult — the single shape EVERY tool returns, so the agent graph can
     handle success and failure uniformly without knowing which tool ran.
  2. Dispatch helpers — safely run a tool by name with the arguments the LLM
     produced, catching bad names, bad arguments, and tool crashes so a
     misbehaving 3B model can never take the whole loop down.

Why a uniform result shape: the LLM needs a short natural-language 'summary' to
reason over, while the graph/debugger benefits from structured 'data'. Bundling
both, plus an explicit ok/error flag, means the loop logic is the same for every
tool: run it, feed summary back to the model, continue.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ops_common.logging import get_logger

logger = get_logger(__name__)


# ============================================================
# The uniform tool result
# ============================================================


@dataclass
class ToolResult:
    """The single shape every tool returns.

    Bundles a short natural-language summary for the model to reason over with the
    structured detail the graph and debugger need, plus an explicit success flag --
    so the loop handles every tool identically whether it succeeded or failed.
    """
    ok: bool
    summary: str  # short, LLM-facing natural language
    data: dict[str, Any] = field(default_factory=dict)  # structured detail

    def to_model_text(self) -> str:
        # What gets fed back to the LLM as the tool's observation. Kept as the
        # summary only — the model reasons on plain language, not raw dicts.
        """Render this result as the observation fed back to the model.

        Only the summary is returned, since the model reasons over plain language
        rather than raw structures.

        Returns:
            The summary, prefixed to mark an error.
        """
        prefix = "" if self.ok else "ERROR: "
        return f"{prefix}{self.summary}"

    def to_dict(self) -> dict[str, Any]:
        # Full structured form for the API/debug layer (not shown to the model).
        """Return the full structured result for the API and debug layers.

        Returns:
            The success flag, summary, and structured data.
        """
        return {"ok": self.ok, "summary": self.summary, "data": self.data}


def tool_ok(summary: str, data: dict[str, Any] | None = None) -> ToolResult:
    """Build a successful tool result.

    Args:
        summary: Short natural-language summary for the model.
        data: Optional structured detail.

    Returns:
        A successful result.
    """
    return ToolResult(ok=True, summary=summary, data=data or {})


def tool_error(summary: str, data: dict[str, Any] | None = None) -> ToolResult:
    """Build a failed tool result.

    Args:
        summary: Short natural-language description of the failure.
        data: Optional structured detail.

    Returns:
        A failed result.
    """
    return ToolResult(ok=False, summary=summary, data=data or {})


# ============================================================
# Safe dispatch
# ============================================================

# A tool is just a callable returning a ToolResult. The registry maps the name
# the LLM emits to the function to run.
ToolFn = Callable[..., ToolResult]


def dispatch(
    name: str,
    arguments: dict[str, Any],
    registry: dict[str, ToolFn],
) -> ToolResult:
    # Turn an LLM's requested (name, arguments) into an actual call — defensively.
    # Every failure mode returns a ToolResult(ok=False) rather than raising, so
    # the agent can hand the error back to the model and let it recover.
    """Run the tool the model asked for, defensively.

    An unknown name, bad arguments, a crash inside the tool, or a tool that breaks
    the result contract all come back as a failed result rather than an exception,
    so a misbehaving small model can never take the whole loop down -- the error is
    handed back to the model to recover from instead.

    Args:
        name: Tool name the model emitted.
        arguments: Arguments the model produced.
        registry: Mapping of tool name to implementation.

    Returns:
        The tool's result, or a failed result describing what went wrong.
    """
    fn = registry.get(name)
    if fn is None:
        available = ", ".join(sorted(registry)) or "none"
        logger.warning("LLM requested unknown tool '%s'", name)
        return tool_error(f"Unknown tool '{name}'. Available tools: {available}.")

    cleaned = _coerce_arguments(fn, arguments)
    if isinstance(cleaned, ToolResult):
        return cleaned  # coercion already produced an error result

    try:
        result = fn(**cleaned)
    except TypeError as exc:
        # Wrong/missing arguments that slipped past coercion.
        logger.warning("Bad arguments for tool '%s': %s", name, exc)
        return tool_error(f"Tool '{name}' called with invalid arguments: {exc}")
    except Exception as exc:  # noqa: BLE001
        # Any tool-internal crash — never propagate into the loop.
        logger.exception("Tool '%s' raised", name)
        return tool_error(f"Tool '{name}' failed while running: {exc}")

    if not isinstance(result, ToolResult):
        # A tool that forgot the contract — wrap so downstream stays uniform.
        logger.error("Tool '%s' returned %s, expected ToolResult", name, type(result))
        return tool_error(f"Tool '{name}' returned an unexpected result type.")

    return result


def _coerce_arguments(
    fn: ToolFn,
    arguments: dict[str, Any],
) -> dict[str, Any] | ToolResult:
    # Small models often send arguments as strings ("542" not 542), send extra
    # junk keys, or omit optionals. We filter to the function's real parameters
    # and coerce simple types from their annotations. Anything unrecoverable
    # (a required arg missing) comes back as a ToolResult error.
    if not isinstance(arguments, dict):
        return tool_error(
            f"Tool arguments must be an object, got {type(arguments).__name__}."
        )

    sig = inspect.signature(fn)
    params = sig.parameters
    accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in params.values())

    cleaned: dict[str, Any] = {}
    for pname, param in params.items():
        if pname not in arguments:
            continue  # optional or will be flagged as missing below
        cleaned[pname] = _coerce_one(arguments[pname], param.annotation)

    # Keep unexpected keys only if the function explicitly accepts **kwargs.
    if accepts_kwargs:
        for key, val in arguments.items():
            if key not in cleaned:
                cleaned[key] = val

    # Verify every required (no-default) parameter is present.
    missing = [
        pname
        for pname, param in params.items()
        if param.default is inspect.Parameter.empty
        and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
        and pname not in cleaned
    ]
    if missing:
        return tool_error(f"Missing required argument(s): {', '.join(missing)}.")

    return cleaned


# Strings a small model sends when it means "no value". Left as-is they reach the
# database as literal text and crash the query.
_NULL_WORDS = {"null", "none", "nil", "nan", "undefined", "n/a", ""}


def _coerce_one(value: Any, annotation: Any) -> Any:
    # Best-effort type coercion for the common scalar cases a small model garbles.
    # Unknown/complex annotations pass through untouched.
    if annotation is inspect.Parameter.empty or value is None:
        return value

    # Small models frequently emit the STRING "null"/"none"/"nil" instead of a
    # real JSON null. Passed through, that string reaches SQL and blows up
    # ("invalid input syntax for type bigint: 'null'"). Treat these as None so
    # the parameter is simply absent, which every tool already handles.
    if isinstance(value, str) and value.strip().lower() in _NULL_WORDS:
        return None

    ann = str(annotation)

    if annotation is int or "int" in ann:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            try:
                return int(stripped)
            except ValueError:
                try:
                    return int(float(stripped))
                except ValueError:
                    return value
        return value

    if annotation is float or "float" in ann:
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return value
        return value

    if annotation is str or "str" in ann:
        # Don't stringify dicts/lists; only tidy scalars the model over-typed.
        if isinstance(value, (int, float)):
            return str(value)
        return value

    return value


# ============================================================
# Domain normalization — user language -> the platform's schema
# ============================================================

CANONICAL_DOMAINS: tuple[str, ...] = (
    "assets",
    "operations",
    "quality",
    "maintenance",
    "inventory",
    "workforce",
    "finance",
    "customers",
)

# Synonym -> canonical domain. Keys are lowercase, singularized where sensible.
# Deliberately biased toward the words a plant manager actually says.
_DOMAIN_SYNONYMS: dict[str, str] = {
    # --- assets: the things being operated ---
    "asset": "assets",
    "machine": "assets",
    "machines": "assets",
    "machinery": "assets",
    "equipment": "assets",
    "line": "assets",
    "lines": "assets",
    "device": "assets",
    "devices": "assets",
    "tower": "assets",
    "towers": "assets",
    "vehicle": "assets",
    "vehicles": "assets",
    "plant": "assets",
    # --- operations: the core work / production ---
    "operation": "operations",
    "production": "operations",
    "produce": "operations",
    "produced": "operations",
    "output": "operations",
    "throughput": "operations",
    "manufacturing": "operations",
    "units": "operations",
    "volume": "operations",
    "energy": "operations",
    "power": "operations",
    "consumption": "operations",
    "efficiency": "operations",
    "productivity": "operations",
    # --- quality: what goes wrong with the product ---
    "defect": "quality",
    "defects": "quality",
    "reject": "quality",
    "rejects": "quality",
    "rejection": "quality",
    "scrap": "quality",
    "failure": "quality",
    "failures": "quality",
    "fault": "quality",
    "faults": "quality",
    "sla": "quality",
    # --- maintenance: repairs and downtime ---
    "downtime": "maintenance",
    "down": "maintenance",
    "outage": "maintenance",
    "outages": "maintenance",
    "repair": "maintenance",
    "repairs": "maintenance",
    "breakdown": "maintenance",
    "breakdowns": "maintenance",
    "servicing": "maintenance",
    "service": "maintenance",
    "uptime": "maintenance",
    "reliability": "maintenance",
    "wear": "maintenance",
    # --- inventory: stock and materials ---
    "stock": "inventory",
    "material": "inventory",
    "materials": "inventory",
    "supply": "inventory",
    "supplies": "inventory",
    "raw": "inventory",
    "parts": "inventory",
    "spares": "inventory",
    "warehouse": "inventory",
    # --- workforce: the people ---
    "staff": "workforce",
    "employee": "workforce",
    "employees": "workforce",
    "worker": "workforce",
    "workers": "workforce",
    "labor": "workforce",
    "labour": "workforce",
    "operator": "workforce",
    "operators": "workforce",
    "personnel": "workforce",
    "overtime": "workforce",
    "shift": "workforce",
    "shifts": "workforce",
    "hr": "workforce",
    "headcount": "workforce",
    # --- finance: money in and out ---
    "revenue": "finance",
    "sales": "finance",
    "income": "finance",
    "cost": "finance",
    "costs": "finance",
    "expense": "finance",
    "expenses": "finance",
    "spend": "finance",
    "spending": "finance",
    "profit": "finance",
    "margin": "finance",
    "budget": "finance",
    "money": "finance",
    "financial": "finance",
    # --- customers: who we serve ---
    "customer": "customers",
    "client": "customers",
    "clients": "customers",
    "order": "customers",
    "orders": "customers",
    "subscriber": "customers",
    "subscribers": "customers",
    "student": "customers",
    "students": "customers",
    "complaint": "customers",
    "complaints": "customers",
    "churn": "customers",
    "satisfaction": "customers",
    "demand": "customers",
}


def normalize_domain(value: str | None) -> str | None:
    # Resolve whatever the model passed into a canonical domain, or None.
    #
    # None  -> caller drops the domain filter and searches ALL domains.
    # This is the deliberate fail-open: an unknown word must widen the search,
    # never narrow it to nothing.
    """Resolve whatever the model passed into a canonical domain name.

    Tries the canonical names, then exact synonyms, then individual words of a
    phrase. An unrecognised value returns None, which deliberately fails open: the
    caller drops the domain filter and searches everything, because an unknown word
    must widen the search rather than narrow it to nothing.

    Args:
        value: The domain word or phrase the model supplied.

    Returns:
        The canonical domain name, or None to search all domains.
    """
    if value is None:
        return None

    raw = str(value).strip().lower()
    if not raw:
        return None

    # 1) Already canonical — the common, happy path.
    if raw in CANONICAL_DOMAINS:
        return raw

    # 2) Exact synonym hit.
    hit = _DOMAIN_SYNONYMS.get(raw)
    if hit:
        logger.info("Domain normalized: %r -> %r", value, hit)
        return hit

    # 3) The model often sends a phrase, not a word ("downtime minutes",
    #    "production output", "maintenance domain"). Scan its tokens for any
    #    word we recognize, canonical first so an explicit domain name wins.
    tokens = _tokenize(raw)
    for tok in tokens:
        if tok in CANONICAL_DOMAINS:
            logger.info("Domain normalized from phrase: %r -> %r", value, tok)
            return tok
    for tok in tokens:
        hit = _DOMAIN_SYNONYMS.get(tok)
        if hit:
            logger.info("Domain normalized from phrase: %r -> %r", value, hit)
            return hit

    # 4) Unknown. Fail OPEN: no filter, search everything.
    logger.info("Domain %r not recognized; searching all domains.", value)
    return None


def _tokenize(text: str) -> list[str]:
    # Split a phrase into comparable word tokens (underscores/hyphens too).
    cleaned = text.replace("_", " ").replace("-", " ").replace(".", " ")
    return [t for t in cleaned.split() if t]


def domain_hint_for_schema() -> str:
    # A single line the tool SCHEMAS can embed so the model is nudged toward
    # canonical names up front. The normalizer is the real guarantee; this just
    # improves the odds the model gets it right the first time.
    """Return the domain guidance line embedded in the tool schemas.

    Nudges the model toward canonical names up front; the normalizer remains the
    actual guarantee.

    Returns:
        The guidance line.
    """
    return (
        "One of: assets, operations, quality, maintenance, inventory, "
        "workforce, finance, customers. Common mappings: downtime/repairs/"
        "breakdowns -> maintenance; production/output/throughput/energy -> "
        "operations; revenue/cost/profit -> finance; defects/rejects -> "
        "quality; stock/materials -> inventory; staff/overtime -> workforce; "
        "orders/complaints -> customers; machines/lines -> assets. "
        "Omit this to search all domains."
    )
