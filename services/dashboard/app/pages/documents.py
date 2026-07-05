from __future__ import annotations

import base64
from typing import Any

import dash
from dash import Input, Output, State, callback, dcc, html, no_update

from app.api_client import (
    APIError,
    list_datasets,
    rag_documents,
    rag_query,
    rag_upload,
)
from app import theme

dash.register_page(__name__, path="/documents", name="Documents")


_STATUS_COLOR = {
    "indexed": theme.COLORS["collected"],
    "processing": theme.COLORS["brand"],
    "pending": theme.COLORS["skipped"],
    "failed": theme.COLORS["danger"],
}


def layout() -> html.Div:
    return html.Div(
        style=theme.PAGE_STYLE,
        children=[
            html.H1("Documents", style=theme.HEADING_STYLE),
            html.P(
                "Upload manuals, SOPs, logs and reports for a dataset, then ask "
                "questions answered strictly from those documents. Each dataset's "
                "documents are kept separate.",
                style=theme.SUBHEADING_STYLE,
            ),
            html.Label("Dataset", style=theme.LABEL_STYLE),
            dcc.Dropdown(
                id="doc-dataset",
                options=[],
                placeholder="Select a dataset",
                clearable=False,
                style={"marginBottom": "1.5rem"},
            ),
            dcc.Interval(id="doc-init", interval=300, max_intervals=1),

            # ---- Upload ----
            dcc.Upload(
                id="doc-upload",
                multiple=True,
                children=html.Div(
                    [
                        "Drag & drop documents here, or ",
                        html.Span("browse", style={"color": theme.COLORS["brand"],
                                                   "fontWeight": 600}),
                        html.Div("PDF, DOCX, TXT — multiple files supported",
                                 style={"fontSize": "0.78rem",
                                        "color": theme.COLORS["text_muted"],
                                        "marginTop": "0.3rem"}),
                    ],
                    style={"textAlign": "center"},
                ),
                style={
                    "border": f"2px dashed {theme.COLORS['border']}",
                    "borderRadius": "12px",
                    "padding": "1.75rem 1rem",
                    "cursor": "pointer",
                    "backgroundColor": theme.COLORS["surface"],
                    "marginBottom": "0.75rem",
                },
            ),
            html.Div(id="doc-upload-status", style={"marginBottom": "1rem"}),

            # ---- Document list (auto-refreshing while anything is processing) ----
            dcc.Interval(id="doc-poll", interval=3000, disabled=True),
            html.Div(id="doc-list", style={"marginBottom": "2rem"}),

            # ---- Chat ----
            html.H3("Ask the documents", style={"fontSize": "1.2rem",
                                                "marginBottom": "0.25rem"}),
            html.P("Answers come only from the uploaded documents for this dataset.",
                   style=theme.SUBHEADING_STYLE),
            html.Div(
                style={"display": "flex", "gap": "0.5rem", "marginBottom": "1rem"},
                children=[
                    dcc.Input(
                        id="doc-question",
                        type="text",
                        placeholder="e.g. What is the maintenance interval for the pumps?",
                        style={**theme.INPUT_STYLE, "marginBottom": 0, "flex": 1},
                        debounce=True,
                    ),
                    html.Button("Ask", id="doc-ask", n_clicks=0,
                                style=theme.PRIMARY_BUTTON_STYLE),
                ],
            ),
            dcc.Loading(html.Div(id="doc-answer"), type="dot"),
        ],
    )


# ============================================================
# Dataset dropdown
# ============================================================

@callback(
    Output("doc-dataset", "options"),
    Output("doc-dataset", "value"),
    Input("doc-init", "n_intervals"),
)
def populate_datasets(_init: int | None) -> tuple[list[dict[str, Any]], Any]:
    try:
        datasets = list_datasets()
    except APIError:
        return [], None
    options = [
        {"label": f"{d['business_name']} ({d['source_filename']})", "value": d["dataset_id"]}
        for d in datasets
    ]
    return options, (options[0]["value"] if options else None)


# ============================================================
# Upload handler
# ============================================================

@callback(
    Output("doc-upload-status", "children"),
    Output("doc-list", "children", allow_duplicate=True),
    Output("doc-poll", "disabled", allow_duplicate=True),
    Input("doc-upload", "contents"),
    State("doc-upload", "filename"),
    State("doc-dataset", "value"),
    prevent_initial_call=True,
)
def handle_upload(contents, filenames, dataset_id):
    if not contents or dataset_id is None:
        return no_update, no_update, no_update

    if not isinstance(contents, list):
        contents = [contents]
        filenames = [filenames]

    files: list[tuple[str, bytes]] = []
    for content, name in zip(contents, filenames):
        try:
            _header, b64 = content.split(",", 1)
            raw = base64.b64decode(b64)
            files.append((name, raw))
        except Exception:
            continue

    if not files:
        return _msg("Could not read the selected files.", theme.COLORS["danger"]), no_update, no_update

    try:
        resp = rag_upload(dataset_id, files)
    except APIError as exc:
        return _msg(f"Upload failed: {exc}", theme.COLORS["danger"]), no_update, no_update

    accepted = resp.get("accepted", [])
    rejected = resp.get("rejected", [])
    parts = []
    if accepted:
        parts.append(f"{len(accepted)} file(s) uploaded and queued for indexing.")
    if rejected:
        names = ", ".join(r.get("filename", "?") for r in rejected)
        parts.append(f"Rejected: {names}.")
    color = theme.COLORS["collected"] if accepted else theme.COLORS["danger"]

    # Enable polling so the list refreshes as background indexing progresses.
    return _msg(" ".join(parts), color), _render_list(dataset_id), False


# ============================================================
# Document list (poll-driven)
# ============================================================

@callback(
    Output("doc-list", "children"),
    Output("doc-poll", "disabled"),
    Input("doc-dataset", "value"),
    Input("doc-poll", "n_intervals"),
)
def refresh_list(dataset_id, _tick):
    if dataset_id is None:
        return "", True
    children = _render_list(dataset_id)
    # Stop polling once nothing is pending/processing.
    still_working = _has_active(dataset_id)
    return children, (not still_working)


def _has_active(dataset_id: int) -> bool:
    try:
        docs = rag_documents(dataset_id)
    except APIError:
        return False
    return any(d.get("status") in ("pending", "processing") for d in docs)


def _render_list(dataset_id: int):
    try:
        docs = rag_documents(dataset_id)
    except APIError as exc:
        return _error(f"Could not load documents: {exc}")

    if not docs:
        return html.Div("No documents uploaded for this dataset yet.",
                        style={"color": theme.COLORS["text_muted"],
                               "fontSize": "0.9rem"})

    header = html.Tr([
        html.Th(h, style=_TH) for h in
        ["Document", "Type", "Status", "Chunks", "Detail"]
    ])
    rows = []
    for d in docs:
        status = d.get("status", "")
        color = _STATUS_COLOR.get(status, theme.COLORS["text_muted"])
        detail = d.get("error_detail") or ("—" if status == "indexed" else "")
        rows.append(
            html.Tr([
                html.Td(d.get("filename", ""), style=_TD),
                html.Td((d.get("file_type") or "").upper(), style=_TD),
                html.Td(
                    html.Span(status, style={"color": color, "fontWeight": 600}),
                    style=_TD,
                ),
                html.Td(str(d.get("chunk_count", 0)), style=_TD),
                html.Td(detail, style={**_TD, "color": theme.COLORS["text_muted"],
                                       "fontSize": "0.8rem"}),
            ])
        )
    table = html.Table([html.Thead(header), html.Tbody(rows)],
                       style={"width": "100%", "borderCollapse": "collapse",
                              "fontSize": "0.85rem"})
    return html.Div(table, style=theme.CARD_STYLE)


# ============================================================
# Chat
# ============================================================

@callback(
    Output("doc-answer", "children"),
    Input("doc-ask", "n_clicks"),
    Input("doc-question", "n_submit"),
    State("doc-question", "value"),
    State("doc-dataset", "value"),
    prevent_initial_call=True,
)
def ask(_clicks, _submit, question, dataset_id):
    if not question or not question.strip():
        return no_update
    if dataset_id is None:
        return _error("Select a dataset first.")

    try:
        resp = rag_query(dataset_id, question.strip())
    except APIError as exc:
        return _error(f"Query failed: {exc}")

    grounded = resp.get("grounded", False)
    answer = resp.get("answer", "")
    sources = resp.get("sources", [])

    answer_block = html.Div(
        answer,
        style={
            "fontSize": "0.95rem", "lineHeight": "1.55",
            "color": theme.COLORS["text"], "whiteSpace": "pre-wrap",
            "padding": "0.9rem 1rem",
            "backgroundColor": theme.COLORS["surface"],
            "border": f"1px solid {theme.COLORS['border']}",
            "borderRadius": "10px",
            "borderLeft": f"4px solid "
                          f"{theme.COLORS['collected'] if grounded else theme.COLORS['skipped']}",
        },
    )

    children = [answer_block]
    if sources:
        chips = []
        for s in sources:
            label = s.get("filename", "")
            if s.get("page_number"):
                label += f" · p.{s['page_number']}"
            chips.append(
                html.Span(
                    label,
                    style={
                        "fontSize": "0.78rem", "padding": "0.2rem 0.6rem",
                        "borderRadius": "999px", "marginRight": "0.4rem",
                        "marginTop": "0.5rem", "display": "inline-block",
                        "border": f"1px solid {theme.COLORS['border']}",
                        "color": theme.COLORS["text_muted"],
                    },
                )
            )
        children.append(
            html.Div(
                [html.Div("Sources", style={"fontSize": "0.7rem", "fontWeight": 700,
                                            "textTransform": "uppercase",
                                            "letterSpacing": "0.04em",
                                            "color": theme.COLORS["text_muted"],
                                            "marginTop": "0.75rem"}),
                 html.Div(chips)],
            )
        )
    if not resp.get("llm_used", False) and grounded:
        children.append(
            html.Div("Answer assembled from document excerpts (LLM off).",
                     style={"fontSize": "0.75rem", "color": theme.COLORS["text_muted"],
                            "marginTop": "0.5rem"})
        )

    return html.Div(children)


# ============================================================
# Helpers
# ============================================================

def _msg(text: str, color: str):
    return html.Div(text, style={"fontSize": "0.85rem", "color": color})


_TH = {
    "textAlign": "left",
    "padding": "0.5rem 0.6rem",
    "borderBottom": f"2px solid {theme.COLORS['border']}",
    "color": theme.COLORS["text_muted"],
    "fontWeight": 600,
}
_TD = {
    "padding": "0.4rem 0.6rem",
    "borderBottom": f"1px solid {theme.COLORS['border']}",
}


def _error(message: str) -> html.Div:
    return html.Div(
        message,
        style={
            "color": theme.COLORS["danger"],
            "backgroundColor": "#fef2f2",
            "border": f"1px solid {theme.COLORS['danger']}",
            "borderRadius": "8px",
            "padding": "0.6rem 0.9rem",
            "fontSize": "0.9rem",
        },
    )