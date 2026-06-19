from __future__ import annotations

import base64
from typing import Any

import dash
from dash import Input, Output, State, callback, dcc, html

from app.api_client import APIError, start_onboarding
from app import theme

dash.register_page(__name__, path="/", name="Upload")


def layout() -> html.Div:
    return html.Div(
        style=theme.PAGE_STYLE,
        children=[
            html.H1("Onboard your data", style=theme.HEADING_STYLE),
            html.P(
                "Upload any business's CSV. The platform profiles every column, "
                "suggests how each maps to a universal domain, and lets you confirm.",
                style=theme.SUBHEADING_STYLE,
            ),
            html.Div(
                style=theme.CARD_STYLE,
                children=[
                    html.Label("Business name", style=theme.LABEL_STYLE),
                    dcc.Input(
                        id="upload-business-name",
                        type="text",
                        placeholder="e.g. NorthStar Telecom",
                        style=theme.INPUT_STYLE,
                    ),
                    html.Label("Industry (optional)", style=theme.LABEL_STYLE),
                    dcc.Input(
                        id="upload-industry",
                        type="text",
                        placeholder="e.g. telecom, manufacturing, aerospace",
                        style=theme.INPUT_STYLE,
                    ),
                    html.Label("Data file (CSV)", style=theme.LABEL_STYLE),
                    dcc.Upload(
                        id="upload-data",
                        children=html.Div(
                            ["Drag and drop or ", html.B("select a CSV file")]
                        ),
                        multiple=False,
                        accept=".csv",
                        style={
                            "width": "100%",
                            "height": "120px",
                            "lineHeight": "120px",
                            "borderWidth": "2px",
                            "borderStyle": "dashed",
                            "borderColor": theme.COLORS["border"],
                            "borderRadius": "10px",
                            "textAlign": "center",
                            "color": theme.COLORS["text_muted"],
                            "marginBottom": "1rem",
                            "cursor": "pointer",
                        },
                    ),
                    html.Div(id="upload-filename", style={"marginBottom": "1rem"}),
                    html.Button(
                        "Profile and suggest mappings",
                        id="upload-submit",
                        n_clicks=0,
                        style=theme.PRIMARY_BUTTON_STYLE,
                    ),
                    html.Div(id="upload-status", style={"marginTop": "1rem"}),
                ],
            ),
        ],
    )


@callback(
    Output("upload-filename", "children"),
    Input("upload-data", "filename"),
    prevent_initial_call=True,
)
def show_filename(filename: str | None) -> Any:
    if not filename:
        return ""
    return html.Span(
        f"Selected: {filename}",
        style={"color": theme.COLORS["collected"], "fontSize": "0.9rem"},
    )


@callback(
    Output("upload-status", "children"),
    Output("onboarding-store", "data"),
    Output("url", "pathname"),
    Input("upload-submit", "n_clicks"),
    State("upload-business-name", "value"),
    State("upload-industry", "value"),
    State("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def handle_upload(
    n_clicks: int,
    business_name: str | None,
    industry: str | None,
    contents: str | None,
    filename: str | None,
) -> tuple[Any, Any, Any]:
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update

    if not business_name:
        return _error("Please enter a business name."), dash.no_update, dash.no_update
    if not contents or not filename:
        return _error("Please select a CSV file."), dash.no_update, dash.no_update

    try:
        file_bytes = _decode_upload(contents)
    except Exception:  # noqa: BLE001
        return _error("Could not read the uploaded file."), dash.no_update, dash.no_update

    try:
        result = start_onboarding(
            file_bytes=file_bytes,
            filename=filename,
            business_name=business_name,
            industry=industry or None,
        )
    except APIError as exc:
        return _error(f"Onboarding failed: {exc}"), dash.no_update, dash.no_update

    store_data = {
        "dataset_id": result["dataset_id"],
        "business_name": result["business_name"],
        "industry": result.get("industry"),
        "row_count": result["row_count"],
        "stored_path": result["stored_path"],
        "suggestions": result["suggestions"],
    }

    return (
        _success(
            f"Profiled {result['row_count']} rows across "
            f"{len(result['suggestions'])} columns. Redirecting to review..."
        ),
        store_data,
        "/confirm",
    )


def _decode_upload(contents: str) -> bytes:
    _, content_string = contents.split(",", 1)
    return base64.b64decode(content_string)


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


def _success(message: str) -> html.Div:
    return html.Div(
        message,
        style={
            "color": theme.COLORS["collected"],
            "backgroundColor": "#f0fdf4",
            "border": f"1px solid {theme.COLORS['collected']}",
            "borderRadius": "8px",
            "padding": "0.6rem 0.9rem",
            "fontSize": "0.9rem",
        },
    )