"""Copilot (agent) API endpoints.

Exposes the natural-language question endpoint plus readiness and tool-listing
endpoints for the ReAct agent. The reasoning loop itself lives in
``services/agent``; this router is a thin, permission-guarded HTTP surface.
"""
from __future__ import annotations

import os
import sys
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from ops_common.logging import get_logger
from pydantic import BaseModel, Field

from api_app.auth.dependencies import require_permission

logger = get_logger(__name__)

router = APIRouter()

_SERVICES_PATH = os.getenv("OPS_SERVICES_PATH", "/app/services")
if _SERVICES_PATH not in sys.path:
    sys.path.insert(0, _SERVICES_PATH)

from agent.agent import AgentAnswer, agent_health, run_agent  # noqa: E402

# ============================================================
# Request / response models
# ============================================================


class AskIn(BaseModel):
    """Request body for a copilot question."""
    question: str = Field(..., description="The manager's natural-language question.")
    dataset_id: int | None = Field(
        default=None,
        description="Optional dataset to scope the question to. If omitted the "
        "agent resolves it (auto-picks a lone dataset, else discovers via tools).",
    )
    session_id: str | None = Field(
        default=None,
        description="Optional conversation id. Reserved for memory; accepted now "
        "so the client contract is stable before memory ships.",
    )


class EvidenceStepOut(BaseModel):
    """One tool execution in the agent's evidence trail."""
    tool: str
    arguments: dict[str, Any]
    ok: bool
    summary: str


class AskOut(BaseModel):
    """Copilot response: the grounded answer plus its evidence trail."""
    question: str
    answer: str
    dataset_id: int | None
    steps: int
    tools_used: list[str]
    evidence: list[EvidenceStepOut]
    elapsed_seconds: float
    error: str | None = None


class AgentHealthOut(BaseModel):
    """Readiness snapshot of the agent and its LLM backend."""
    llm_reachable: bool
    model_present: bool
    model: str
    tool_count: int
    tools: list[str]


# ============================================================
# Endpoints
# ============================================================


@router.post("/agent/ask", response_model=AskOut)
def agent_ask(
    body: AskIn,
    _user=Depends(require_permission("copilot:use")),
) -> AskOut:
    # The main copilot endpoint. Validates the question, delegates everything to
    # run_agent, and returns the grounded answer plus the evidence trail so the
    # UI can show which tools were consulted.
    """Answer a manager's natural-language question via the copilot agent.

    Validates the question, delegates the reasoning to the agent, and returns the
    grounded answer together with the evidence trail of tools consulted.

    Args:
        body: The question and optional dataset/session scoping.
        _user: Authenticated caller, injected to enforce ``copilot:use``.

    Returns:
        The grounded answer and its evidence trail.

    Raises:
        HTTPException: 400 if the question is empty.
    """
    question = (body.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="A question is required.")

    result: AgentAnswer = run_agent(
        question=question,
        session_id=body.session_id,
        dataset_id=body.dataset_id,
    )

    return AskOut(
        question=result.question,
        answer=result.answer,
        dataset_id=result.dataset_id,
        steps=result.steps,
        tools_used=result.tools_used,
        evidence=[
            EvidenceStepOut(
                tool=e.tool,
                arguments=e.arguments,
                ok=e.ok,
                summary=e.summary,
            )
            for e in result.evidence
        ],
        elapsed_seconds=result.elapsed_seconds,
        error=result.error,
    )


@router.get("/agent/health", response_model=AgentHealthOut)
def agent_health_endpoint(
    _user=Depends(require_permission("copilot:use")),
) -> AgentHealthOut:
    # Lets the copilot page show readiness (and warn about a cold model) before
    # the user sends a question that would otherwise hang on first load.
    """Report agent and LLM readiness for the copilot page.

    Lets the UI warn about a cold model before a question is sent that would
    otherwise hang on first load.

    Args:
        _user: Authenticated caller, injected to enforce ``copilot:use``.

    Returns:
        A readiness snapshot of the agent and its model.

    Raises:
        HTTPException: 503 if the health probe fails.
    """
    try:
        h = agent_health()
    except Exception as exc:  # noqa: BLE001
        logger.exception("agent_health failed")
        raise HTTPException(
            status_code=503, detail=f"Agent health check failed: {exc}"
        ) from exc

    return AgentHealthOut(
        llm_reachable=h.get("llm_reachable", False),
        model_present=h.get("model_present", False),
        model=h.get("model", "unknown"),
        tool_count=h.get("tool_count", 0),
        tools=h.get("tools", []),
    )


@router.get("/agent/tools", response_model=list[str])
def agent_tools(
    _user=Depends(require_permission("copilot:use")),
) -> list[str]:
    # Simple list of tool names the agent can call — handy for the UI to display
    # "this copilot can query analytics, ML, intelligence, documents, and the hub."
    """List the tool names the agent can call.

    Args:
        _user: Authenticated caller, injected to enforce ``copilot:use``.

    Returns:
        The tool names available to the agent.

    Raises:
        HTTPException: 503 if the tool list cannot be read.
    """
    try:
        h = agent_health()
        return h.get("tools", [])
    except Exception as exc:  # noqa: BLE001
        logger.exception("agent_tools failed")
        raise HTTPException(
            status_code=503, detail=f"Could not list agent tools: {exc}"
        ) from exc
