# services/agent/memory.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text

from ops_common.db import session_scope
from ops_common.logging import get_logger

logger = get_logger(__name__)


def _history_limit() -> int:
    return int(os.getenv("OPS_AGENT_MEMORY_TURNS", "6"))


# ============================================================
# Public shapes
# ============================================================

@dataclass
class Turn:
    role: str
    content: str
    dataset_id: int | None = None
    tools_used: list[str] | None = None
    evidence: list[dict[str, Any]] | None = None
    steps: int | None = None
    elapsed_sec: float | None = None


# ============================================================
# READ — load prior turns for a session
# ============================================================

def load_history(session_id: str | None, limit: int | None = None) -> list[Turn]:
    if not session_id:
        return []

    cap = limit if limit is not None else _history_limit()
    try:
        with session_scope() as session:
            rows = session.execute(
                text(
                    """
                    SELECT role, content, dataset_id, tools_used, evidence,
                           steps, elapsed_sec
                    FROM agent.message
                    WHERE session_id = :sid
                    ORDER BY turn_index DESC
                    LIMIT :lim
                    """
                ),
                {"sid": session_id, "lim": cap},
            ).fetchall()
    except Exception:  # noqa: BLE001
        logger.warning("Memory load failed for session %s; continuing stateless.", session_id)
        return []

    turns: list[Turn] = []
    for r in reversed(rows):
        turns.append(
            Turn(
                role=r[0],
                content=r[1],
                dataset_id=r[2],
                tools_used=_loads(r[3]),
                evidence=_loads(r[4]),
                steps=r[5],
                elapsed_sec=r[6],
            )
        )
    return turns


def load_full_conversation(session_id: str | None) -> list[Turn]:
    if not session_id:
        return []
    try:
        with session_scope() as session:
            rows = session.execute(
                text(
                    """
                    SELECT role, content, dataset_id, tools_used, evidence,
                           steps, elapsed_sec
                    FROM agent.message
                    WHERE session_id = :sid
                    ORDER BY turn_index ASC
                    """
                ),
                {"sid": session_id},
            ).fetchall()
    except Exception:  # noqa: BLE001
        logger.warning("Full conversation load failed for session %s.", session_id)
        return []

    return [
        Turn(
            role=r[0], content=r[1], dataset_id=r[2],
            tools_used=_loads(r[3]), evidence=_loads(r[4]),
            steps=r[5], elapsed_sec=r[6],
        )
        for r in rows
    ]


# ============================================================
# WRITE — persist a completed turn pair
# ============================================================

def save_exchange(
    session_id: str | None,
    question: str,
    answer_turn: dict[str, Any],
) -> None:
    if not session_id:
        return

    dataset_id = answer_turn.get("dataset_id")
    try:
        with session_scope() as session:
            next_index = _next_turn_index(session, session_id)

            session.execute(
                text(
                    """
                    INSERT INTO agent.message
                        (session_id, turn_index, role, content, dataset_id)
                    VALUES (:sid, :idx, 'user', :content, :dsid)
                    """
                ),
                {
                    "sid": session_id,
                    "idx": next_index,
                    "content": question,
                    "dsid": dataset_id,
                },
            )

            session.execute(
                text(
                    """
                    INSERT INTO agent.message
                        (session_id, turn_index, role, content, dataset_id,
                         tools_used, evidence, steps, elapsed_sec)
                    VALUES
                        (:sid, :idx, 'assistant', :content, :dsid,
                         :tools, :evidence, :steps, :elapsed)
                    """
                ),
                {
                    "sid": session_id,
                    "idx": next_index + 1,
                    "content": answer_turn.get("answer", ""),
                    "dsid": dataset_id,
                    "tools": _dumps(answer_turn.get("tools_used", [])),
                    "evidence": _dumps(answer_turn.get("evidence", [])),
                    "steps": answer_turn.get("steps"),
                    "elapsed": answer_turn.get("elapsed_seconds"),
                },
            )

            session.execute(
                text(
                    """
                    INSERT INTO agent.conversation
                        (session_id, dataset_id, title, message_count, updated_at)
                    VALUES (:sid, :dsid, :title, 2, now())
                    ON CONFLICT (session_id) DO UPDATE SET
                        dataset_id    = EXCLUDED.dataset_id,
                        message_count = agent.conversation.message_count + 2,
                        updated_at    = now()
                    """
                ),
                {
                    "sid": session_id,
                    "dsid": dataset_id,
                    "title": _title_from(question),
                },
            )
    except Exception:  # noqa: BLE001
        logger.warning("Memory save failed for session %s; answer was still returned.", session_id)


# ============================================================
# Prompt assembly
# ============================================================

def as_context_messages(turns: list[Turn]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for t in turns:
        if t.role == "user":
            messages.append({"role": "user", "content": t.content})
        elif t.role == "assistant":
            messages.append({"role": "assistant", "content": t.content})
    return messages


# ============================================================
# Internals
# ============================================================

def _next_turn_index(session: Any, session_id: str) -> int:
    row = session.execute(
        text(
            "SELECT COALESCE(MAX(turn_index), -1) FROM agent.message WHERE session_id = :sid"
        ),
        {"sid": session_id},
    ).fetchone()
    return int(row[0]) + 1 if row and row[0] is not None else 0


def _title_from(question: str, max_len: int = 80) -> str:
    q = " ".join((question or "").split())
    return q if len(q) <= max_len else q[: max_len - 1] + "…"


def _dumps(value: Any) -> str:
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return json.dumps(None)


def _loads(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value