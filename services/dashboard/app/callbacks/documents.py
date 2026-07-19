"""Callbacks for /documents (RAG).

The numbers can tell you downtime rose to 147 minutes. They cannot tell you
that 147 crosses the RED threshold, that E-27 means a seal failure, or that
the SOP requires escalation within 30 minutes. That knowledge lives only in
text, and this is where it gets asked.
"""

from __future__ import annotations

import base64
from typing import Any

from dash import Input, Output, State, callback, html, no_update

from app import feedback, ids
from app.api_client import (
    APIError,
    list_datasets,
    rag_documents,
    rag_query,
    rag_upload,
)
from app.components import ui

_ACTIVE_STATUSES = ("pending", "processing")

_STATUS_TONE: dict[str, str] = {
    "indexed": "trend-down",
    "processing": "trend-flat",
    "pending": "trend-flat",
    "failed": "trend-up",
}


@callback(
    Output(ids.DOC_DATASET, "options"),
    Output(ids.DOC_DATASET, "value"),
    Input(ids.DOC_INIT, "n_intervals"),
    State(ids.ACCESS_TOKEN, "data"),
)
def populate_datasets(
    _init: int | None, token: str | None
) -> tuple[list[dict[str, Any]], Any]:
    """Fill the dataset selector and preselect the first entry.

    Args:
        _init: Interval tick that triggers the initial load.
        token: Caller's access token.

    Returns:
        The selector options and the initially selected dataset.
    """
    try:
        datasets = list_datasets(token=token)
    except APIError:
        return [], None

    options = [
        {"label": d["business_name"], "value": d["dataset_id"]} for d in datasets
    ]
    return options, (options[0]["value"] if options else None)


@callback(
    Output(ids.DOC_UPLOAD_STATUS, "children"),
    Output(ids.DOC_LIST, "children", allow_duplicate=True),
    Output(ids.DOC_POLL, "disabled", allow_duplicate=True),
    Input(ids.DOC_UPLOAD, "contents"),
    State(ids.DOC_UPLOAD, "filename"),
    State(ids.DOC_DATASET, "value"),
    State(ids.ACCESS_TOKEN, "data"),
    prevent_initial_call=True,
)
def handle_upload(
    contents: Any,
    filenames: Any,
    dataset_id: int | None,
    token: str | None,
) -> tuple[Any, Any, Any]:
    """Upload the chosen documents and start polling for indexing.

    Files that cannot be read are skipped rather than failing the whole batch, so
    one bad file does not block the rest.

    Args:
        contents: Base64 contents of the uploaded files.
        filenames: Names of the uploaded files.
        dataset_id: Dataset the documents belong to.
        token: Caller's access token.

    Returns:
        The upload status, refreshed document list, and polling state.
    """
    hold = (no_update, no_update, no_update)

    if not contents or dataset_id is None:
        return hold

    if not isinstance(contents, list):
        contents = [contents]
        filenames = [filenames]

    files: list[tuple[str, bytes]] = []
    for content, name in zip(contents, filenames, strict=False):
        try:
            _header, _, payload = content.partition(",")
            files.append((name, base64.b64decode(payload)))
        except Exception:  # noqa: BLE001
            continue

    if not files:
        return feedback.error("Could not read those files."), *hold[1:]

    try:
        response = rag_upload(dataset_id, files, token=token)
    except APIError as exc:
        return feedback.error(f"Upload failed: {exc}"), *hold[1:]

    accepted = response.get("accepted", [])
    rejected = response.get("rejected", [])

    parts = []
    if accepted:
        parts.append(f"{len(accepted)} file(s) uploaded. Indexing now.")
    if rejected:
        names = ", ".join(r.get("filename", "?") for r in rejected)
        parts.append(f"Rejected: {names}.")

    message = " ".join(parts)
    box = feedback.success(message) if accepted else feedback.error(message)

    # Enable polling so the list updates as indexing progresses.
    # Enable polling so the list updates as indexing progresses.
    return box, _render_list(_fetch(dataset_id, token)), False


@callback(
    Output(ids.DOC_LIST, "children"),
    Output(ids.DOC_POLL, "disabled"),
    Input(ids.DOC_DATASET, "value"),
    Input(ids.DOC_POLL, "n_intervals"),
    State(ids.ACCESS_TOKEN, "data"),
)
def refresh_list(
    dataset_id: int | None, _tick: int | None, token: str | None
) -> tuple[Any, bool]:
    """Refresh the document list, polling only while indexing is in progress.

    Args:
        dataset_id: Selected dataset.
        _tick: Polling interval tick.
        token: Caller's access token.

    Returns:
        The rendered document list and whether polling should continue.
    """
    if dataset_id is None:
        return "", True

    docs = _fetch(dataset_id, token)
    if isinstance(docs, str):
        return feedback.error(docs), True

    # Keep polling only while something is still being indexed.
    still_working = any(d.get("status") in _ACTIVE_STATUSES for d in docs)
    return _render_list(docs), not still_working


@callback(
    Output(ids.DOC_ANSWER, "children"),
    Input(ids.DOC_ASK, "n_clicks"),
    Input(ids.DOC_QUESTION, "n_submit"),
    State(ids.DOC_QUESTION, "value"),
    State(ids.DOC_DATASET, "value"),
    State(ids.ACCESS_TOKEN, "data"),
    prevent_initial_call=True,
)
def ask(
    _clicks: int | None,
    _submit: int | None,
    question: str | None,
    dataset_id: int | None,
    token: str | None,
) -> Any:
    """Answer a question from the selected dataset's documents.

    Args:
        _clicks: Ask button clicks.
        _submit: Enter presses in the question box.
        question: The question to answer.
        dataset_id: Dataset to query.
        token: Caller's access token.

    Returns:
        The rendered answer, or a message when it could not be produced.
    """
    if not question or not question.strip():
        return no_update
    if dataset_id is None:
        return feedback.error("Select a dataset first.")

    try:
        response = rag_query(dataset_id, question.strip(), token=token)
    except APIError as exc:
        return feedback.error(f"Could not answer that: {exc}")

    return _render_answer(response)


# ============================================================
# Data access
# ============================================================


def _fetch(dataset_id: int, token: str | None = None) -> list[dict[str, Any]] | str:
    """Fetch the document list once. Returns an error string, never empty-on-error."""
    try:
        return rag_documents(dataset_id, token=token)
    except APIError as exc:
        return f"Could not load documents: {exc}"


# ============================================================
# Render helpers
# ============================================================


def _render_list(docs: list[dict[str, Any]] | str) -> Any:
    if isinstance(docs, str):
        return feedback.error(docs)

    if not docs:
        return feedback.empty("No documents for this dataset yet.")

    rows = []
    for d in docs:
        status = d.get("status", "")
        rows.append(
            [
                d.get("filename", ""),
                (d.get("file_type") or "").upper(),
                html.Span(status, className=_STATUS_TONE.get(status, "trend-flat")),
                f"{d.get('chunk_count', 0):,}",
                d.get("error_detail") or "\u2014",
            ]
        )

    return ui.card(ui.table(["Document", "Type", "Status", "Chunks", "Detail"], rows))


def _render_answer(response: dict[str, Any]) -> html.Div:
    """Render a grounded answer.

    The left edge carries the honesty signal: green when the answer is grounded
    in retrieved chunks, amber when nothing matched. A truthful "not found"
    beats a fabricated answer, and the UI says which one this is.
    """
    grounded = response.get("grounded", False)
    sources = response.get("sources", [])

    children: list[Any] = [
        html.Div(
            response.get("answer", ""),
            style={
                "whiteSpace": "pre-wrap",
                "lineHeight": "1.55",
            },
        )
    ]

    if sources:
        children.append(_sources(sources))

    if grounded and not response.get("llm_used", False):
        children.append(
            html.Div(
                "Assembled from document excerpts. The language model is off.",
                className="kpi-note",
                style={"marginTop": "0.5rem"},
            )
        )

    return html.Div(
        children,
        className="card",
        style={
            "borderLeft": f"3px solid var(--{'ok' if grounded else 'warn'})",
        },
    )


def _sources(sources: list[dict[str, Any]]) -> html.Div:
    chips = []
    for s in sources:
        label = s.get("filename", "")
        if s.get("page_number"):
            label += f" \u00b7 p.{s['page_number']}"
        chips.append(html.Span(label, className="evidence-chip"))

    return html.Div(
        [html.Span("Sources: "), *chips],
        className="evidence",
    )
