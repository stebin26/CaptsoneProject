from __future__ import annotations

from typing import Any

import dash
from dash import ALL, Input, Output, State, callback, ctx, dcc, html

from app.api_client import APIError, confirm_onboarding, list_domains
from app import theme

dash.register_page(__name__, path="/confirm", name="Confirm mappings")

_ROLE_OPTIONS = [
    {"label": "Metric (a measurable value)", "value": "metric"},
    {"label": "Entity (identifies a thing)", "value": "entity"},
    {"label": "Skip (don't collect)", "value": "skip"},
]


def _domain_options() -> list[dict[str, str]]:
    try:
        domains = list_domains()
        return [{"label": d["domain"].title(), "value": d["domain"]} for d in domains]
    except APIError:
        return []


def layout() -> html.Div:
    return html.Div(
        style=theme.PAGE_STYLE,
        children=[
            html.H1("Review suggested mappings", style=theme.HEADING_STYLE),
            html.P(
                "The platform suggested how each column maps to a universal domain. "
                "Adjust anything, then confirm to load the data into the hub.",
                style=theme.SUBHEADING_STYLE,
            ),
            html.Div(id="confirm-header"),
            html.Div(id="confirm-columns"),
            html.Div(
                style={"marginTop": "1.5rem", "display": "flex", "gap": "1rem"},
                children=[
                    html.Button(
                        "Confirm and load into hub",
                        id="confirm-submit",
                        n_clicks=0,
                        style=theme.PRIMARY_BUTTON_STYLE,
                    ),
                ],
            ),
            html.Div(id="confirm-status", style={"marginTop": "1rem"}),
        ],
    )


@callback(
    Output("confirm-header", "children"),
    Output("confirm-columns", "children"),
    Input("onboarding-store", "data"),
)
def render_columns(store: dict[str, Any] | None) -> tuple[Any, Any]:
    if not store or "suggestions" not in store:
        return (
            html.Div(
                "No dataset in progress. Please upload a file first.",
                style={"color": theme.COLORS["text_muted"]},
            ),
            "",
        )

    header = html.Div(
        style=theme.CARD_STYLE,
        children=[
            html.Span(
                f"{store['business_name']}",
                style={"fontWeight": 600, "fontSize": "1.1rem"},
            ),
            html.Span(
                f"  ·  {store['row_count']} rows  ·  "
                f"{len(store['suggestions'])} columns",
                style={"color": theme.COLORS["text_muted"]},
            ),
        ],
    )

    domain_opts = _domain_options()
    rows = [_column_row(s, domain_opts) for s in store["suggestions"]]
    return header, html.Div(rows)


def _column_row(suggestion: dict[str, Any], domain_opts: list[dict[str, str]]) -> html.Div:
    col_name = suggestion["column_name"]
    samples = suggestion.get("sample_values") or []
    sample_text = ", ".join(str(v) for v in samples[:4]) if samples else "—"
    confidence = suggestion.get("confidence", 0.0)
    source = suggestion.get("source", "none")

    return html.Div(
        style={
            **theme.CARD_STYLE,
            "display": "grid",
            "gridTemplateColumns": "1.4fr 1fr 1fr 1fr",
            "gap": "1rem",
            "alignItems": "center",
        },
        children=[
            html.Div(
                [
                    html.Div(col_name, style={"fontWeight": 600}),
                    html.Div(
                        f"{suggestion.get('data_type', '?')} · samples: {sample_text}",
                        style={"fontSize": "0.8rem", "color": theme.COLORS["text_muted"]},
                    ),
                    html.Div(
                        f"confidence {confidence:.0%} · {source}",
                        style={"fontSize": "0.75rem", "color": theme.COLORS["text_muted"]},
                    ),
                ]
            ),
            dcc.Dropdown(
                id={"type": "confirm-role", "column": col_name},
                options=_ROLE_OPTIONS,
                value=suggestion.get("role", "skip"),
                clearable=False,
            ),
            dcc.Dropdown(
                id={"type": "confirm-domain", "column": col_name},
                options=domain_opts,
                value=suggestion.get("suggested_domain"),
                placeholder="domain",
                clearable=True,
            ),
            dcc.Input(
                id={"type": "confirm-metric", "column": col_name},
                type="text",
                value=suggestion.get("suggested_metric") or "",
                placeholder="metric name",
                style={
                    "width": "100%",
                    "padding": "0.45rem 0.6rem",
                    "border": f"1px solid {theme.COLORS['border']}",
                    "borderRadius": "6px",
                },
            ),
        ],
    )


@callback(
    Output("confirm-status", "children"),
    Output("onboarding-store", "data", allow_duplicate=True),
    Output("url", "pathname", allow_duplicate=True),
    Input("confirm-submit", "n_clicks"),
    State("onboarding-store", "data"),
    State({"type": "confirm-role", "column": ALL}, "value"),
    State({"type": "confirm-role", "column": ALL}, "id"),
    State({"type": "confirm-domain", "column": ALL}, "value"),
    State({"type": "confirm-metric", "column": ALL}, "value"),
    prevent_initial_call=True,
)
def handle_confirm(
    n_clicks: int,
    store: dict[str, Any] | None,
    roles: list[str],
    role_ids: list[dict[str, Any]],
    domains: list[str | None],
    metrics: list[str | None],
) -> tuple[Any, Any, Any]:
    if not n_clicks or not store:
        return dash.no_update, dash.no_update, dash.no_update

    columns: list[dict[str, Any]] = []
    for role, rid, domain, metric in zip(roles, role_ids, domains, metrics):
        col_name = rid["column"]
        entry: dict[str, Any] = {"column_name": col_name, "role": role}
        if role != "skip":
            entry["domain"] = domain
            entry["metric_name"] = (metric or "").strip() or None
        columns.append(entry)

    invalid = [
        c["column_name"]
        for c in columns
        if c["role"] == "metric" and (not c.get("domain") or not c.get("metric_name"))
    ]
    if invalid:
        return (
            _error(
                "These metric columns need both a domain and a metric name: "
                + ", ".join(invalid)
            ),
            dash.no_update,
            dash.no_update,
        )

    try:
        result = confirm_onboarding(
            dataset_id=store["dataset_id"],
            stored_path=store["stored_path"],
            columns=columns,
        )
    except APIError as exc:
        return _error(f"Confirmation failed: {exc}"), dash.no_update, dash.no_update

    validation = result.get("validation", {})
    if not validation.get("ok", True) and result.get("hub_rows_written", 0) == 0:
        issues = validation.get("issues", [])
        msgs = "; ".join(i.get("message", "") for i in issues if i.get("severity") == "error")
        return _error(f"Validation blocked the load: {msgs}"), dash.no_update, dash.no_update

    updated_store = {**store, "config_version": result.get("config_version")}
    return (
        _success(
            f"Loaded {result['hub_rows_written']} rows into the hub. "
            f"{result['features_collected']} features collected, "
            f"{result['features_skipped']} skipped. Opening dashboard..."
        ),
        updated_store,
        "/review",
    )


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