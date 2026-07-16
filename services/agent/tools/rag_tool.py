from __future__ import annotations

from typing import Any

from ops_common.db import session_scope
from ops_common.logging import get_logger

from .base import ToolResult, tool_error, tool_ok

# Reuse the router's own functions so there is ONE source of truth for RAG
# querying and document listing.
from api_app.routers.v1.rag import (
    QueryIn,
    get_documents,
    query_documents,
)

logger = get_logger(__name__)


# ============================================================
# Tool 1 — search documents (grounded Q&A over uploaded docs)
# ============================================================

def search_documents(dataset_id: int, question: str) -> ToolResult:
    # The agent uses this for knowledge questions whose answer lives in text, not
    # numbers: procedures, definitions, codes, policies. The RAG layer stays the
    # authority on grounding — we never soften a "not found" into an invented answer.
    q = (question or "").strip()
    if not q:
        return tool_error("A question is required to search the documents.")

    try:
        with session_scope() as session:
            # query_documents takes (dataset_id, body, _session). The session is
            # a no-op dependency in-process; we pass it for signature parity.
            result = query_documents(dataset_id, QueryIn(question=q), _session=session)
    except Exception as exc:  # noqa: BLE001
        logger.exception("search_documents failed for dataset_id=%s", dataset_id)
        return tool_error(
            f"Could not search documents for dataset {dataset_id}: {exc}"
        )

    # RAG explicitly reports whether it could ground the answer. If it could not
    # (no relevant chunks), pass that through honestly rather than as a real answer.
    if not result.grounded or result.used_chunks == 0:
        return tool_error(
            f"The uploaded documents do not contain an answer to: '{q}'. "
            "There may be no relevant document, or none has been uploaded/indexed."
        )

    sources = [
        {
            "filename": s.filename,
            "page": s.page_number,
            "distance": _round(s.distance, 4),
        }
        for s in result.sources
    ]
    source_txt = "; ".join(
        f"{s['filename']}" + (f" p.{s['page']}" if s["page"] is not None else "")
        for s in sources
    ) or "unspecified source"

    # The summary the model reads IS the grounded answer plus where it came from,
    # so the agent can fold it into its final response with a citation.
    summary = f"{result.answer}  [sources: {source_txt}]"
    return tool_ok(
        summary=summary,
        data={
            "dataset_id": dataset_id,
            "question": q,
            "answer": result.answer,
            "grounded": result.grounded,
            "llm_used": result.llm_used,
            "used_chunks": result.used_chunks,
            "sources": sources,
        },
    )


# ============================================================
# Tool 2 — list documents (what is available to search)
# ============================================================

def list_documents_tool(dataset_id: int) -> ToolResult:
    # Lets the agent check whether any documents exist / are indexed before
    # promising a document-grounded answer. Useful when search returns nothing:
    # is it "no relevant passage" or "no documents at all".
    try:
        with session_scope() as session:
            docs = get_documents(dataset_id, _session=session)
    except Exception as exc:  # noqa: BLE001
        logger.exception("list_documents_tool failed for dataset_id=%s", dataset_id)
        return tool_error(
            f"Could not list documents for dataset {dataset_id}: {exc}"
        )

    if not docs:
        return tool_error(
            f"No documents have been uploaded for dataset {dataset_id}."
        )

    indexed = [d for d in docs if d.status == "indexed"]
    listed = [
        {
            "filename": d.filename,
            "status": d.status,
            "chunks": d.chunk_count,
        }
        for d in docs
    ]

    summary = (
        f"Dataset {dataset_id} has {len(docs)} document(s), "
        f"{len(indexed)} indexed and searchable: "
        + ", ".join(f"{d['filename']} ({d['status']})" for d in listed[:8])
    )
    return tool_ok(
        summary=summary,
        data={
            "dataset_id": dataset_id,
            "document_count": len(docs),
            "indexed_count": len(indexed),
            "documents": listed,
        },
    )


# ============================================================
# Tool schemas — what the LLM reads to decide the call
# ============================================================

RAG_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Answer a question using the business's uploaded documents "
                "(manuals, SOPs, maintenance guides, reports). Use this for "
                "knowledge/procedure/definition questions whose answer is written "
                "text, not numbers, e.g. 'what does error code E-45 mean', 'what "
                "is the shutdown procedure', 'what does the manual say about "
                "bearing replacement'. The answer is grounded only in the "
                "documents and includes its sources."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "integer", "description": "The dataset id."},
                    "question": {
                        "type": "string",
                        "description": "The natural-language question to look up in the documents.",
                    },
                },
                "required": ["dataset_id", "question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents_tool",
            "description": (
                "List which documents have been uploaded for a dataset and whether "
                "they are indexed/searchable. Use to check document availability "
                "before relying on search_documents, or when a document search "
                "returns nothing and you need to know if any documents exist at all."
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
RAG_TOOL_FUNCTIONS = {
    "search_documents": search_documents,
    "list_documents_tool": list_documents_tool,
}


# ============================================================
# Small formatting helper
# ============================================================

def _round(value: float | None, places: int = 4) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), places)
    except (TypeError, ValueError):
        return None