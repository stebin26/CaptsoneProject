"""Callbacks for /copilot.

The submit flow is deliberately split across two callbacks. The first is fast:
it renders the user's message and a thinking bubble, then stashes the question
in a store. Writing to that store triggers the second callback, which makes the
slow agent call. Without the split the UI would freeze for the 60-200s the
local model takes, with nothing on screen to show the question was received.
"""

from __future__ import annotations

from typing import Any

import dash
from dash import Input, Output, State, callback, dcc, html, no_update

from app import feedback, ids
from app.api_client import APIError, agent_ask, agent_health, list_datasets
from app.constants import SUGGESTED_PROMPTS
from app.logging_setup import get_logger

logger = get_logger(__name__)


@callback(
    Output(ids.COPILOT_STATUS, "children"),
    Output(ids.COPILOT_DATASET, "options"),
    Input(ids.COPILOT_INIT, "n_intervals"),
    State(ids.ACCESS_TOKEN, "data"),
)
def init_page(_init: int | None, token: str | None) -> tuple[Any, list[dict[str, Any]]]:
    """Check the agent is reachable and load datasets for the scope dropdown."""
    status = _render_status(token)

    try:
        options = [
            {"label": d["business_name"], "value": d["dataset_id"]}
            for d in list_datasets(token=token)
        ]
    except APIError:
        logger.warning("Callback copilot.init_page failed", exc_info=True)
        options = []

    return status, options


@callback(
    Output(ids.COPILOT_INPUT, "value"),
    Input(ids.copilot_suggestion(dash.ALL), "n_clicks"),
    State(ids.COPILOT_INPUT, "value"),
    prevent_initial_call=True,
)
def fill_suggestion(clicks: list[int], _current: str | None) -> Any:
    """Put a clicked suggestion chip into the question box.

    Args:
        clicks: Click counts for each suggestion chip.
        _current: The current question text.

    Returns:
        The chosen suggestion as the new question text.
    """
    if not any(clicks):
        return no_update

    triggered = dash.callback_context.triggered_id
    if not triggered or "index" not in triggered:
        return no_update

    index = triggered["index"]
    if 0 <= index < len(SUGGESTED_PROMPTS):
        return SUGGESTED_PROMPTS[index]
    return no_update


@callback(
    Output(ids.COPILOT_TRANSCRIPT, "children"),
    Output(ids.COPILOT_HISTORY, "data"),
    Output(ids.COPILOT_PENDING, "data"),
    Output(ids.COPILOT_INPUT, "value", allow_duplicate=True),
    Output(ids.COPILOT_SEND, "disabled"),
    Input(ids.COPILOT_SEND, "n_clicks"),
    State(ids.COPILOT_INPUT, "value"),
    State(ids.COPILOT_DATASET, "value"),
    State(ids.COPILOT_HISTORY, "data"),
    prevent_initial_call=True,
)
def submit_question(
    _clicks: int,
    question: str | None,
    dataset_id: int | None,
    history: list[dict[str, Any]] | None,
) -> tuple[Any, list[dict[str, Any]], Any, Any, bool]:
    """Phase 1 -- fast. Show the question, show a spinner, stash the request."""
    history = history or []
    text = (question or "").strip()

    if not text:
        return _render_transcript(history), history, no_update, no_update, False

    history = history + [{"role": "user", "content": text}]
    pending = {"question": text, "dataset_id": dataset_id}

    # Clear the box and disable Send while the agent works.
    return _render_transcript(history, thinking=True), history, pending, "", True


@callback(
    Output(ids.COPILOT_TRANSCRIPT, "children", allow_duplicate=True),
    Output(ids.COPILOT_HISTORY, "data", allow_duplicate=True),
    Output(ids.COPILOT_PENDING, "data", allow_duplicate=True),
    Output(ids.COPILOT_SEND, "disabled", allow_duplicate=True),
    Input(ids.COPILOT_PENDING, "data"),
    State(ids.COPILOT_HISTORY, "data"),
    State(ids.COPILOT_SESSION, "data"),
    State(ids.ACCESS_TOKEN, "data"),
    prevent_initial_call=True,
)
def run_agent_call(
    pending: dict[str, Any] | None,
    history: list[dict[str, Any]] | None,
    session_id: str | None,
    token: str | None,
) -> tuple[Any, list[dict[str, Any]], Any, bool]:
    """Phase 2 -- slow. The actual agent call, 60-200s on a local CPU model."""
    if not pending:
        return no_update, no_update, no_update, no_update

    history = history or []

    try:
        result = agent_ask(
            question=pending["question"],
            dataset_id=pending.get("dataset_id"),
            session_id=session_id,
            token=token,
        )
        turn = {
            "role": "assistant",
            "content": result.get("answer", "No answer produced."),
            "tools_used": result.get("tools_used", []),
            "evidence": result.get("evidence", []),
            "steps": result.get("steps", 0),
            "elapsed": result.get("elapsed_seconds", 0),
            "dataset_id": result.get("dataset_id"),
        }
    except APIError as exc:
        # A failed agent call is shown as a turn, not swallowed. The user must
        # see that the question was received and that it failed.
        logger.warning("Callback copilot.run_agent_call failed", exc_info=True)
        turn = {
            "role": "assistant",
            "content": (
                f"I could not complete that: {exc}. The model may be busy or "
                "still loading. Try again."
            ),
            "tools_used": [],
            "evidence": [],
            "error": True,
        }

    history = history + [turn]

    # Clearing pending stops this callback re-firing; re-enable Send.
    return _render_transcript(history), history, None, False


# ============================================================
# Readiness banner
# ============================================================


def _render_status(token: str | None = None) -> Any:
    try:
        health = agent_health(token=token)
    except APIError as exc:
        logger.warning("Callback copilot._render_status failed", exc_info=True)
        return feedback.error(f"The copilot is unavailable: {exc}")

    if not health.get("llm_reachable"):
        return feedback.error(
            f"The model is not reachable (looked for {health.get('model')}). "
            "Start Ollama on the host, then reload."
        )

    if not health.get("model_present"):
        return feedback.error(
            f"Ollama is running but '{health.get('model')}' is not pulled. "
            f"Run: ollama pull {health.get('model')}"
        )

    return feedback.success(
        f"Ready \u00b7 {health.get('model')} \u00b7 "
        f"{health.get('tool_count')} tools available."
    )


# ============================================================
# Transcript
# ============================================================


def _render_transcript(
    history: list[dict[str, Any]],
    thinking: bool = False,
) -> Any:
    if not history and not thinking:
        return html.Div(
            "Ask a question to begin, or pick one of the suggestions below.",
            className="msg-empty",
            style={"textAlign": "center"},
        )

    turns = [
        _user_turn(t["content"]) if t["role"] == "user" else _agent_turn(t)
        for t in history
    ]
    if thinking:
        turns.append(_thinking_turn())

    return html.Div(turns)


def _user_turn(content: str) -> html.Div:
    return html.Div(
        html.Div(content, className="bubble bubble-user"),
        className="turn turn-user",
    )


def _agent_turn(turn: dict[str, Any]) -> html.Div:
    classes = "bubble bubble-agent"
    if turn.get("error"):
        classes += " bubble-error"

    children: list[Any] = [html.Div(turn["content"])]

    evidence = turn.get("evidence") or []
    if evidence:
        children.append(_evidence(evidence, turn))

    return html.Div(
        html.Div(children, className=classes),
        className="turn turn-agent",
    )


def _evidence(evidence: list[dict[str, Any]], turn: dict[str, Any]) -> html.Div:
    """One chip per tool that actually ran, with a tick or a cross.

    This is the most important element on the page. It is what makes an AI
    answer auditable rather than merely confident. A tool the agent did not
    call never appears here.
    """
    chips = []
    for e in evidence:
        ok = e.get("ok")
        chips.append(
            html.Span(
                [
                    e.get("tool", "?"),
                    html.Span(
                        " \u2713" if ok else " \u2715",
                        className="evidence-ok" if ok else "evidence-fail",
                    ),
                ],
                title=e.get("summary", ""),  # hover shows what the tool returned
                className="evidence-chip",
            )
        )

    steps = turn.get("steps")
    elapsed = turn.get("elapsed")
    meta = ""
    if steps:
        meta = f"{steps} steps"
        if isinstance(elapsed, (int, float)):
            meta += f" \u00b7 {elapsed:.0f}s"

    return html.Div(
        [html.Span("Evidence: "), *chips, html.Span(meta)],
        className="evidence",
    )


def _thinking_turn() -> html.Div:
    return html.Div(
        html.Div(
            [
                html.Span("Investigating"),
                dcc.Loading(
                    type="dot",
                    children=html.Div(style={"width": "20px", "height": "8px"}),
                ),
            ],
            className="bubble bubble-agent bubble-thinking",
            style={"display": "flex", "alignItems": "center", "gap": "0.5rem"},
        ),
        className="turn turn-agent",
    )
