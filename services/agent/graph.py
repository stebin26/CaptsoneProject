"""The ReAct reasoning loop, built as a LangGraph state machine.

The model alternates between reasoning and acting: it either calls a tool or
returns a final answer, and every tool result is fed back as an observation.
Two guarded exits keep the loop honest and bounded -- a step limit that gives up
cleanly rather than looping forever, and a nudge that fires once when the model
tries to answer a causal question without having gathered any evidence. The
nudge never fires twice, so a stubborn model cannot be trapped in a cycle.
"""
from __future__ import annotations

import os
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from ops_common.logging import get_logger

from .llm import LLMResponse, OllamaToolClient, ToolCall, get_llm
from .tools.base import ToolFn, ToolResult, dispatch

logger = get_logger(__name__)


# ============================================================
# Configuration
# ============================================================


_DEFAULT_MAX_STEPS = 8


def _max_steps() -> int:
    # Total agent turns (reason cycles) before we force a stop. A 3B model can
    # loop or re-call the same tool; this bounds the cost and latency hard.
    # 8, not 6: a causal question spends one step confirming WHAT, one on the
    # rejected answer, then needs room to actually investigate WHY and write up.
    raw = os.getenv("OPS_AGENT_MAX_STEPS", str(_DEFAULT_MAX_STEPS))
    try:
        return int(raw)
    except (TypeError, ValueError):
        # Falling back keeps the hard step cap in force; dropping it would let a
        # looping model run without a bound.
        logger.warning(
            "Invalid OPS_AGENT_MAX_STEPS %r — using %d steps instead",
            raw,
            _DEFAULT_MAX_STEPS,
            extra={"env_value": raw},
        )
        return _DEFAULT_MAX_STEPS


# Words that mark a question as causal — it asks WHY something happened, not
# merely WHAT the value is. Causal questions are the one class where code can
# prove an answer is impossible before the model even speaks.
_CAUSAL_WORDS: frozenset[str] = frozenset(
    {
        "why",
        "cause",
        "causes",
        "causing",
        "reason",
        "reasons",
        "root",
        "rootcause",
        "driver",
        "driving",
        "because",
        "explain",
        "behind",
        "underlying",
        "blame",
        "responsible",
        "drop",
        "drops",
        "dropping",
        "dropped",
        "falling",
        "fell",
        "declining",
        "decline",
        "degrading",
        "worse",
    }
)


def _is_causal(question: str) -> bool:
    lowered = (question or "").lower()
    words = {w.strip(".,!?;:'\"") for w in lowered.replace("-", " ").split()}
    return bool(words & _CAUSAL_WORDS)


def _metrics_seen(evidence: list[dict[str, Any]]) -> set[str]:
    # Every distinct metric the model has actually been shown, across all tools.
    # Tools nest their rows under different keys, so we read whichever they used.
    #
    # This counts EVIDENCE, not tool calls. One tool returning five metrics is
    # richer than three tools returning the same one — the guard below must not
    # confuse activity with information.
    metrics: set[str] = set()
    for entry in evidence:
        if not entry.get("ok"):
            continue
        data = entry.get("data") or {}
        for key in ("series", "top_movers", "top_anomalies", "metrics"):
            for row in data.get(key) or []:
                if isinstance(row, dict) and row.get("metric"):
                    metrics.add(str(row["metric"]))
    return metrics


# The system prompt frames the whole ReAct contract for the model. It is
# deliberately explicit because small models need firm, concrete instruction
# about the loop, the dataset_id, and when to stop.
SYSTEM_PROMPT = (
    "You are an operations intelligence analyst agent. You answer questions "
    "about a business's operational data by calling tools to gather evidence, "
    "then reasoning over what they return.\n\n"
    "Rules:\n"
    "1. Use tools to get real data. Never invent numbers, entities, or trends.\n"
    "2. Call one tool at a time. Read its result before deciding the next step.\n"
    "3. Most questions start with analytics_overview to see what the dataset "
    "contains, then drill into trend or features as needed.\n"
    "4. When you have enough evidence, stop calling tools and write a clear, "
    "direct final answer in plain language for a business manager.\n"
    "5. If a tool returns an error, adjust and try a different approach; do not "
    "repeat the same failing call.\n"
    "6. Ground every claim in tool results. If the data cannot answer the "
    "question, say so honestly.\n"
    "7. CRITICAL: If a tool reports that no data/documents were found, you MUST "
    "tell the user that directly. NEVER invent facts, numbers, procedures, or "
    "document contents to fill the gap. For document questions with no documents, "
    "say plainly that no relevant documents have been uploaded — do not make up "
    "what a manual 'might' say. A truthful 'not found' is always better than a "
    "fabricated answer.\n"
    "8. ALWAYS pass a real numeric dataset_id to tools that need one — use the "
    "dataset_id given in the question. Never pass the word 'null', an empty "
    "value, or a guess. If you truly do not know it, call list_datasets_tool "
    "first to find it.\n"
    "9. CRITICAL: NEVER claim you called a tool that you did not actually call, "
    "and never invent what a tool returned. Only the tool results shown to you "
    "in this conversation are real. If you did not call a tool, you have no "
    "information from it.\n"
    "10. CRITICAL: NEVER state a cause, conclusion, or explanation that no tool "
    "result supports. Do not speculate with phrases like 'this might be due to' "
    "or 'this could be caused by'. If the tools returned no usable data, say "
    "exactly that and stop — do not fill the gap with a plausible-sounding guess.\n"
    "11. CRITICAL — 'WHY' QUESTIONS: call analytics_trend with "
    "include_context=true. Never explain a change using the metric that changed "
    "— 'production dropped because units_produced fell' is circular. The tool "
    "returns what moved elsewhere, ranked. Name the SINGLE strongest metric that "
    "moved the opposite way, state what it suggests, and be honest that it is an "
    "association in the data rather than a proven cause. Do not list every "
    "metric back to the user, and do not ask the user to interpret the list "
    "themselves — that is YOUR job."
)


# ============================================================
# Graph state
# ============================================================
# messages     : the running Ollama-format chat history (system/user/assistant/tool)
# steps        : how many agent (reason) turns have run — enforces the cap
# evidence     : structured record of every tool call + result, for the API/debug
#                layer and for building the final answer's "sources"
# answer       : the final natural-language answer once the loop ends
# is_causal    : the question asks WHY — enables the impossibility guard
# nudged       : the guard has fired once already; it never fires twice
# tool_schemas / tool_registry : passed through so nodes can reach them


class AgentState(TypedDict, total=False):
    """The state carried through the reasoning loop.

    Holds the conversation, the step count, the accumulated evidence, the tool
    surface exposed for this question, and whether the question is causal (which
    arms the evidence guard).
    """
    messages: list[dict[str, Any]]
    steps: int
    evidence: Annotated[list[dict[str, Any]], _append]
    answer: str
    tool_schemas: list[dict[str, Any]]
    tool_registry: dict[str, ToolFn]
    is_causal: bool
    nudged: bool


def _append(existing: list[Any] | None, new: list[Any]) -> list[Any]:
    # Reducer so each node's evidence accumulates instead of overwriting.
    return (existing or []) + (new or [])


# ============================================================
# Nodes
# ============================================================


def _agent_node(state: AgentState) -> dict[str, Any]:
    # REASON: ask the LLM for the next action given the full history + tools.
    client = _client_from_state(state)
    messages = state["messages"]
    schemas = state.get("tool_schemas", [])

    response: LLMResponse = client.chat(messages, tools=schemas)
    steps = state.get("steps", 0) + 1

    if response.is_final:
        # Model chose to answer. Record it and let routing decide whether that
        # answer is acceptable — it may still be sent back for more evidence.
        answer = (
            response.content
            or "I could not determine an answer from the available data."
        )
        assistant_msg = {"role": "assistant", "content": answer}
        return {
            "messages": messages + [assistant_msg],
            "steps": steps,
            "answer": answer,
        }

    # Model requested tool(s). Persist the assistant turn (with tool_calls) so
    # the following tool observations attach to a valid conversation.
    tool_call = response.tool_calls[0]  # we honor one call per turn (rule #2)
    assistant_msg = _assistant_toolcall_message(response, tool_call)
    return {
        "messages": messages + [assistant_msg],
        "steps": steps,
    }


def _tools_node(state: AgentState) -> dict[str, Any]:
    # ACT: execute the tool the model just requested, feed the result back.
    registry = state["tool_registry"]
    last = state["messages"][-1]
    requested: list[dict[str, Any]] = last.get("tool_calls", [])

    new_messages: list[dict[str, Any]] = []
    new_evidence: list[dict[str, Any]] = []

    for call in requested:
        fn = call.get("function") or {}
        name = fn.get("name", "")
        args = fn.get("arguments", {}) or {}

        result: ToolResult = dispatch(name, args, registry)

        # The observation the model reads next turn — plain-language summary only.
        new_messages.append(
            {
                "role": "tool",
                "content": result.to_model_text(),
                "name": name,
            }
        )
        # The structured trace kept for the API response / debugging.
        new_evidence.append(
            {
                "tool": name,
                "arguments": args,
                "ok": result.ok,
                "summary": result.summary,
                "data": result.data,
            }
        )
        logger.info("Executed tool '%s' ok=%s", name, result.ok)

    return {
        "messages": state["messages"] + new_messages,
        "evidence": new_evidence,
    }


def _nudge_node(state: AgentState) -> dict[str, Any]:
    # The model has already read rule 11 and ignored it — it even announced
    # "I will call another tool" and then stopped. A system prompt is read once,
    # at the top of a long context. This arrives as the most recent message,
    # immediately before the model's next turn, which a small model weights far
    # more heavily. Fires at most once per question.
    seen = sorted(_metrics_seen(state.get("evidence", [])))
    only = seen[0] if seen else None

    if only:
        head = (
            f"STOP. The only metric you have looked at is '{only}'. You cannot "
            f"explain why '{only}' changed by pointing at '{only}' — that is "
            f"circular reasoning, not an answer."
        )
    else:
        head = (
            "STOP. No tool has returned any usable data yet. You have nothing "
            "to base an answer on."
        )

    instruction = (
        f"{head}\n\n"
        "The cause is a DIFFERENT metric, in a different part of the business. "
        "Call a tool NOW to find it. Do not describe what you are going to do — "
        "DO IT. Your next output must be a tool call, not text.\n\n"
        "Choose one:\n"
        "- ml_alerts — anomalies and at-risk entities\n"
        "- analytics_trend with domain='maintenance' — downtime, breakdowns\n"
        "- analytics_trend with domain='quality' — defects, rejects\n"
        "- analytics_features — the worst-performing entity\n\n"
        "Do NOT call analytics_trend on the same domain again."
    )

    # Drop the rejected answer so it is not mistaken for a real assistant turn.
    messages = list(state["messages"])
    if (
        messages
        and messages[-1].get("role") == "assistant"
        and not messages[-1].get("tool_calls")
    ):
        messages.pop()
    messages.append({"role": "user", "content": instruction})

    return {"messages": messages, "answer": "", "nudged": True}


def _giveup_node(state: AgentState) -> dict[str, Any]:
    # Graceful termination when the step cap is reached without a final answer.
    # We summarize what we DID gather so the user gets partial value, honestly.
    gathered = state.get("evidence", [])
    if gathered:
        lines = (
            "; ".join(e["summary"] for e in gathered if e.get("ok"))
            or "no conclusive data"
        )
        answer = (
            "I reached my investigation limit before fully resolving this. "
            f"Here is what I found: {lines}."
        )
    else:
        answer = (
            "I was unable to gather enough data to answer this within my step "
            "limit. Please try a more specific question or check the dataset id."
        )
    return {
        "answer": answer,
        "messages": state["messages"] + [{"role": "assistant", "content": answer}],
    }


# ============================================================
# Routing
# ============================================================


def _route_after_agent(state: AgentState) -> str:
    # A produced answer is checked BEFORE the step cap. The cap exists to stop
    # the model looping through tools forever — it must never discard a finished
    # answer the model has already paid for. Hitting the cap ON the answering
    # turn was silently throwing away good work.
    at_cap = state.get("steps", 0) >= _max_steps()

    if state.get("answer"):
        # Only send it back for more evidence if there is budget left to gather
        # any. At the cap, an impossible answer is still the best we have — the
        # give_up node will report the evidence honestly instead.
        if _answer_is_impossible(state) and not at_cap:
            return "nudge"
        return "end"

    if at_cap:
        logger.warning("Agent hit max steps (%d); forcing stop.", _max_steps())
        return "give_up"

    last = state["messages"][-1]
    if last.get("tool_calls"):
        return "tools"  # a tool was requested
    return "end"  # assistant spoke with no tool call and no recorded answer


def _answer_is_impossible(state: AgentState) -> bool:
    # This is NOT a sufficiency check. Code cannot judge whether evidence is
    # "enough" to answer a question — that is a semantic call, and the model
    # already makes it (rule 4). This checks the one thing code CAN prove:
    # that the answer being offered is logically impossible.
    #
    # A cause is never the metric that changed. If the model has seen exactly
    # one metric and is claiming to explain WHY that metric moved, the only
    # explanation available to it is itself — circular by construction.
    #
    # Deliberately NOT gated on tool count or domain count:
    #   - one tool returning five metrics  -> passes, no nudge
    #   - two metrics in the same domain   -> passes, no nudge
    #   - four tools across four domains   -> passes, no nudge
    # The model decides when it has enough. This only blocks the answer that
    # cannot exist. It is a floor, not a ceiling.
    if not state.get("is_causal") or state.get("nudged"):
        return False

    seen = _metrics_seen(state.get("evidence", []))
    if len(seen) >= 2:
        return False

    logger.warning(
        "Causal question with only %d metric(s) seen (%s) — a cause is "
        "impossible from this alone. Sending the agent back for evidence.",
        len(seen),
        sorted(seen) or ["none"],
    )
    return True


# ============================================================
# Message helpers
# ============================================================


def _assistant_toolcall_message(
    response: LLMResponse, primary: ToolCall
) -> dict[str, Any]:
    # Rebuild an Ollama-format assistant message carrying the tool call, so the
    # subsequent 'tool' role message is a valid continuation of the history.
    return {
        "role": "assistant",
        "content": response.content or "",
        "tool_calls": [
            {"function": {"name": primary.name, "arguments": primary.arguments}}
        ],
    }


def _client_from_state(state: AgentState) -> OllamaToolClient:
    # The client is not stored in state (not serializable-friendly); use the
    # shared singleton. Kept behind a function so tests can monkeypatch it.
    return get_llm()


# ============================================================
# Build
# ============================================================


def build_agent_graph():
    # The ReAct loop: agent <-> tools, plus two exits — give_up (out of steps)
    # and nudge (the answer was impossible; go get real evidence).
    """Build the ReAct graph: reason, act, and the two guarded exits.

    Returns:
        The compiled LangGraph state machine.
    """
    graph = StateGraph(AgentState)

    graph.add_node("agent", _agent_node)
    graph.add_node("tools", _tools_node)
    graph.add_node("nudge", _nudge_node)
    graph.add_node("give_up", _giveup_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        _route_after_agent,
        {"tools": "tools", "nudge": "nudge", "end": END, "give_up": "give_up"},
    )
    graph.add_edge("tools", "agent")  # after acting, reason again
    graph.add_edge("nudge", "agent")  # after being blocked, reason again
    graph.add_edge("give_up", END)

    return graph.compile()


# A module-level compiled graph so we don't rebuild it on every call.
_compiled = None


def get_agent_graph():
    """Return the compiled graph, building it once per process.

    Returns:
        The shared compiled graph.
    """
    global _compiled
    if _compiled is None:
        _compiled = build_agent_graph()
    return _compiled


# ============================================================
# Entry point
# ============================================================


def run_once(
    question: str,
    tool_schemas: list[dict[str, Any]],
    tool_registry: dict[str, ToolFn],
    dataset_hint: int | None = None,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    # One full question -> answer run. Seeds the conversation, invokes the graph,
    # and returns the answer plus the evidence trail (what tools ran, results).
    # `history` is prior-turn context (from memory), inserted between the system
    # prompt and the current question so the model sees the conversation so far.
    """Run one question through the reasoning loop to completion.

    Seeds the conversation with the system prompt and any prior turns, invokes the
    graph, and returns the answer with the record of which tools ran and what they
    returned.

    Args:
        question: The question to answer.
        tool_schemas: Tool schemas to expose to the model.
        tool_registry: Callable implementations behind those schemas.
        dataset_hint: Dataset to scope to, when the caller already knows it.
        history: Prior conversation turns loaded from memory.

    Returns:
        The answer and the evidence trail.
    """
    user_content = question
    if dataset_hint is not None:
        # Nudge the model toward the right dataset without hard-coding routing.
        user_content = (
            f"{question}\n\n(Use dataset_id {dataset_hint} unless told otherwise.)"
        )

    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Fold in prior conversation turns (already capped by the memory layer).
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_content})

    initial: AgentState = {
        "messages": messages,
        "steps": 0,
        "evidence": [],
        "tool_schemas": tool_schemas,
        "tool_registry": tool_registry,
        "is_causal": _is_causal(question),
        "nudged": False,
    }

    # recursion_limit is LangGraph's own guard; set above our step cap so OUR
    # cap (with its graceful give_up) fires first, not LangGraph's hard error.
    try:
        final_state = get_agent_graph().invoke(
            initial,
            config={"recursion_limit": _max_steps() * 2 + 4},
        )
    except Exception:
        # The caller has its own backstop; this log is what names the question
        # and the exposed tool surface, which the caller cannot see.
        logger.exception(
            "Reasoning loop failed for question=%r",
            question,
            extra={
                "tools_exposed": len(tool_schemas),
                "dataset_hint": dataset_hint,
                "history_turns": len(history or []),
            },
        )
        raise

    return {
        "question": question,
        "answer": final_state.get("answer", "No answer produced."),
        "steps": final_state.get("steps", 0),
        "evidence": final_state.get("evidence", []),
    }
