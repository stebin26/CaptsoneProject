"""The copilot agent's public entry point.

Takes one manager-level question and returns one grounded answer with the
evidence trail behind it. Two context-shaping steps run before the reasoning
loop -- prior conversation turns are loaded from memory, and the tool surface is
narrowed to the groups the question actually implies -- because a 3B local model
stays fast and accurate only when it is given a small, relevant decision space.
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from ops_common.logging import get_logger

from . import memory
from .graph import run_once
from .tools import ALL_TOOL_FUNCTIONS, ALL_TOOL_SCHEMAS
from .tools.selector import all_tools, select_tools

logger = get_logger(__name__)


# ============================================================
# Public result shape
# ============================================================


@dataclass
class EvidenceStep:
    """One tool execution recorded in the answer's evidence trail."""
    tool: str
    arguments: dict[str, Any]
    ok: bool
    summary: str


@dataclass
class AgentAnswer:
    """A completed agent run: the answer plus how it was reached.

    Carries the evidence trail, which tools ran, and the elapsed time, so the
    interface can show the user what the answer is based on rather than asking them
    to trust it.
    """
    question: str
    answer: str
    dataset_id: int | None
    steps: int
    tools_used: list[str]
    evidence: list[EvidenceStep] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the answer as a plain dictionary for serialization."""
        d = asdict(self)
        return d


# ============================================================
# Public entry point
# ============================================================


def run_agent(
    question: str,
    session_id: str | None = None,
    dataset_id: int | None = None,
) -> AgentAnswer:
    # One question in, one grounded answer out.
    #
    # Two context-shaping steps happen before the loop, and both exist to keep a
    # 3B model fast and accurate:
    #   - TOOL SELECTION: expose only the tool groups the question could need.
    #     The schema block is re-sent on every reasoning step, so trimming it is
    #     the single biggest latency win available.
    #   - MEMORY: fold in the last few turns so follow-ups ("why is that?") work.
    # Both are best-effort: a failure in either degrades gracefully rather than
    # blocking the answer.
    """Answer one question and return the grounded result.

    Loads any prior turns for the session, narrows the tool surface to the groups
    the question implies, runs the reasoning loop, and records the exchange back to
    memory.

    Args:
        question: The manager's natural-language question.
        session_id: Conversation id used to load and save memory.
        dataset_id: Dataset to scope the question to; resolved by the agent when
            omitted.

    Returns:
        The answer together with its evidence trail.
    """
    q = (question or "").strip()
    if not q:
        return AgentAnswer(
            question="",
            answer="Please ask a question about your operational data.",
            dataset_id=dataset_id,
            steps=0,
            tools_used=[],
            error="empty_question",
        )

    resolved_dataset = _resolve_dataset(dataset_id)

    # Pick the tools this question could plausibly need (fail-open: unclear
    # questions get everything). Disable via OPS_AGENT_TOOL_SELECT=false.
    schemas, functions, groups = _tools_for(q)

    # Load prior conversation turns (capped) and turn them into prompt context.
    prior_turns = memory.load_history(session_id)
    history_messages = memory.as_context_messages(prior_turns)

    t0 = time.time()
    try:
        raw = run_once(
            question=q,
            tool_schemas=schemas,
            tool_registry=functions,
            dataset_hint=resolved_dataset,
            history=history_messages,
        )
    except Exception as exc:  # noqa: BLE001
        # The graph has its own safety nets; this is the last backstop so the API
        # always gets a well-formed answer, never a 500 from the agent core.
        elapsed = time.time() - t0
        logger.exception("run_agent crashed for question=%r", q)
        return AgentAnswer(
            question=q,
            answer=(
                "I hit an unexpected problem while investigating this. "
                "Please try rephrasing, or check that the data services are up."
            ),
            dataset_id=resolved_dataset,
            steps=0,
            tools_used=[],
            elapsed_seconds=round(elapsed, 2),
            error=f"{type(exc).__name__}: {exc}",
        )

    elapsed = time.time() - t0
    evidence = _shape_evidence(raw.get("evidence", []))
    tools_used = _dedupe_preserving_order([e.tool for e in evidence])

    answer = AgentAnswer(
        question=q,
        answer=raw.get("answer", "No answer produced."),
        dataset_id=resolved_dataset,
        steps=raw.get("steps", 0),
        tools_used=tools_used,
        evidence=evidence,
        elapsed_seconds=round(elapsed, 2),
    )

    logger.info(
        "run_agent done — groups=%s exposed=%d steps=%s tools=%s elapsed=%.1fs",
        groups,
        len(functions),
        answer.steps,
        tools_used,
        elapsed,
    )

    # Persist this exchange for future turns (best-effort; won't raise).
    memory.save_exchange(session_id, q, answer.to_dict())

    return answer


def _tools_for(question: str) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    # Tool selection with an explicit off-switch and a hard safety net: if the
    # selector fails for any reason, fall back to the full tool surface rather
    # than leaving the agent under-equipped.
    if os.getenv("OPS_AGENT_TOOL_SELECT", "true").lower() in ("false", "0", "no"):
        schemas, functions = all_tools()
        return schemas, functions, ["all (selection disabled)"]

    try:
        return select_tools(question)
    except Exception:  # noqa: BLE001
        logger.exception("Tool selection failed; exposing all tools.")
        return ALL_TOOL_SCHEMAS, ALL_TOOL_FUNCTIONS, ["all (selector error)"]


# ============================================================
# Dataset resolution
# ============================================================


def _resolve_dataset(dataset_id: int | None) -> int | None:
    # If the caller gave an explicit dataset, trust it. Otherwise, try to pick a
    # sensible default so single-dataset demos "just work" without the user
    # typing an id. If several exist, we do NOT guess — we leave it to the agent,
    # which can call list_datasets_tool and ask/choose within the conversation.
    if dataset_id is not None:
        return dataset_id

    try:
        from .tools.hub_tool import list_datasets_tool

        result = list_datasets_tool()
        if not result.ok:
            return None
        datasets = result.data.get("datasets", [])
        if len(datasets) == 1:
            only = datasets[0].get("dataset_id")
            logger.info("Auto-resolved single dataset -> %s", only)
            return only
        # Multiple datasets: don't hard-pick. The hint stays None and the agent's
        # own tool call can disambiguate.
        return None
    except Exception:  # noqa: BLE001
        # Discovery is best-effort; never let it block a run. The traceback is
        # kept because a permanent failure here quietly degrades every answer.
        logger.warning(
            "Dataset auto-resolution failed; proceeding without a hint.",
            exc_info=True,
        )
        return None


# ============================================================
# Shaping helpers
# ============================================================


def _shape_evidence(raw_evidence: list[dict[str, Any]]) -> list[EvidenceStep]:
    steps: list[EvidenceStep] = []
    for e in raw_evidence:
        steps.append(
            EvidenceStep(
                tool=str(e.get("tool", "")),
                arguments=e.get("arguments", {}) or {},
                ok=bool(e.get("ok", False)),
                summary=str(e.get("summary", "")),
            )
        )
    return steps


def _dedupe_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it and it not in seen:
            seen.add(it)
            out.append(it)
    return out


# ============================================================
# Lightweight self-check (used by the API health path / demo)
# ============================================================


def agent_health() -> dict[str, Any]:
    # Confirms the agent's brain is reachable and reports how many tools are
    # registered. Used by the copilot page / API to show readiness before a user
    # sends a question that would otherwise hang on a cold model.
    """Report whether the agent's model is reachable and how many tools are live.

    Used by the copilot page before a question is sent, so a cold model shows as a
    warning rather than an apparent hang.

    Returns:
        A readiness snapshot of the model and tool registry.
    """
    from .llm import get_llm

    try:
        client = get_llm()
        health = client.health_check()
        model = client.config.model
    except Exception as exc:  # noqa: BLE001
        # This is the readiness probe the copilot page calls before letting a
        # user ask anything, so it must always answer rather than raise.
        logger.exception("Agent health check could not reach the model client")
        return {
            "llm_reachable": False,
            "model_present": False,
            "model": None,
            "tool_count": len(ALL_TOOL_FUNCTIONS),
            "tools": sorted(ALL_TOOL_FUNCTIONS),
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "llm_reachable": health.get("reachable", False),
        "model_present": health.get("model_present", False),
        "model": model,
        "tool_count": len(ALL_TOOL_FUNCTIONS),
        "tools": sorted(ALL_TOOL_FUNCTIONS),
    }
