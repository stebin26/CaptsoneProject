"""Callbacks for /confirm -- the human-in-the-loop mapping checkpoint.

This is the only manual step in the pipeline, and the heart of the portability
architecture: we do not model industries, we model the eight universal
functions every operation shares, and onboard through a one-time confirmation
rather than a code change.
"""

from __future__ import annotations

from typing import Any

import dash
from dash import ALL, Input, Output, State, callback, dcc, html

from app import feedback, ids
from app.api_client import APIError, confirm_onboarding, list_domains
from app.components import ui
from app.constants import domain_label

ROLE_OPTIONS: list[dict[str, str]] = [
    {"label": "Metric \u2014 a measurable value", "value": "metric"},
    {"label": "Entity \u2014 identifies a thing", "value": "entity"},
    {"label": "Skip \u2014 do not collect", "value": "skip"},
]


@callback(
    Output(ids.CONFIRM_HEADER, "children"),
    Output(ids.CONFIRM_COLUMNS, "children"),
    Input(ids.ONBOARDING_STORE, "data"),
    State(ids.ACCESS_TOKEN, "data"),
)
def render_columns(store: dict[str, Any] | None, token: str | None) -> tuple[Any, Any]:
    if not store or "suggestions" not in store:
        return feedback.empty(
            "No dataset in progress. Upload a file to start, or open an "
            "existing one from Datasets."
        ), ""

    header = ui.card(
        html.Span(store["business_name"], style={"fontWeight": 600}),
        html.Span(
            f"  \u00b7  {store['row_count']:,} rows"
            f"  \u00b7  {len(store['suggestions'])} columns",
            className="kpi-note",
        ),
    )

    options = _domain_options(token)
    return header, html.Div(
        [_column_row(s, options) for s in store["suggestions"]]
    )


@callback(
    Output(ids.CONFIRM_STATUS, "children"),
    Output(ids.ONBOARDING_STORE, "data", allow_duplicate=True),
    Output(ids.URL, "pathname", allow_duplicate=True),
    Input(ids.CONFIRM_SUBMIT, "n_clicks"),
    State(ids.ONBOARDING_STORE, "data"),
    State(ids.confirm_role(ALL), "value"),
    State(ids.confirm_role(ALL), "id"),
    State(ids.confirm_domain(ALL), "value"),
    State(ids.confirm_metric(ALL), "value"),
    State(ids.ACCESS_TOKEN, "data"),
    prevent_initial_call=True,
)
def handle_confirm(
    n_clicks: int,
    store: dict[str, Any] | None,
    roles: list[str],
    role_ids: list[dict[str, Any]],
    domains: list[str | None],
    metrics: list[str | None],
    token: str | None,
) -> tuple[Any, Any, Any]:
    hold = (dash.no_update, dash.no_update, dash.no_update)

    if not n_clicks or not store:
        return hold

    columns: list[dict[str, Any]] = []
    for role, role_id, domain, metric in zip(roles, role_ids, domains, metrics):
        entry: dict[str, Any] = {"column_name": role_id["column"], "role": role}
        if role != "skip":
            entry["domain"] = domain
            entry["metric_name"] = (metric or "").strip() or None
        columns.append(entry)

    # A metric column without a domain or a name cannot be loaded. Name the
    # offending columns -- never fail silently, never guess for the user.
    invalid = [
        c["column_name"]
        for c in columns
        if c["role"] == "metric" and (not c.get("domain") or not c.get("metric_name"))
    ]
    if invalid:
        return feedback.error(
            "These columns need both a domain and a metric name before they "
            "can be loaded: " + ", ".join(invalid)
        ), *hold[1:]

    try:
        result = confirm_onboarding(
            dataset_id=store["dataset_id"],
            stored_path=store["stored_path"],
            columns=columns,
            token=token,
        )
    except APIError as exc:
        return feedback.error(f"Could not load into the hub: {exc}"), *hold[1:]

    # Zero rows written after a failed validation is a real failure, not a
    # quiet success. Surface what blocked it rather than moving on.
    validation = result.get("validation", {})
    if not validation.get("ok", True) and result.get("hub_rows_written", 0) == 0:
        blocking = "; ".join(
            issue.get("message", "")
            for issue in validation.get("issues", [])
            if issue.get("severity") == "error"
        )
        return feedback.error(
            f"Nothing was loaded. Validation blocked it: {blocking}"
        ), *hold[1:]

    updated = {**store, "config_version": result.get("config_version")}
    return feedback.success(
        f"Loaded {result['hub_rows_written']:,} rows into the hub. "
        f"{result['features_collected']} columns collected, "
        f"{result['features_skipped']} skipped. Opening review\u2026"
    ), updated, "/review"


# ============================================================
# Render helpers
# ============================================================

def _domain_options(token: str | None = None) -> list[dict[str, str]]:
    """Domains come from the API, not a hardcoded list -- the hub is the truth."""
    try:
        return [
            {"label": domain_label(d["domain"]), "value": d["domain"]}
            for d in list_domains(token=token)
        ]
    except APIError:
        return []


def _column_row(
    suggestion: dict[str, Any],
    domain_options: list[dict[str, str]],
) -> html.Div:
    column = suggestion["column_name"]
    samples = suggestion.get("sample_values") or []
    sample_text = ", ".join(str(v) for v in samples[:4]) if samples else "\u2014"
    confidence = suggestion.get("confidence", 0.0)

    return ui.card(
        html.Div(
            [
                html.Div(
                    [
                        html.Div(column, style={"fontWeight": 600}),
                        html.Div(
                            f"{suggestion.get('data_type', 'unknown')} \u00b7 "
                            f"{sample_text}",
                            className="kpi-note",
                        ),
                        html.Div(
                            f"suggested with {confidence:.0%} confidence "
                            f"({suggestion.get('source', 'no source')})",
                            className="kpi-note",
                        ),
                    ]
                ),
                dcc.Dropdown(
                    id=ids.confirm_role(column),
                    options=ROLE_OPTIONS,
                    value=suggestion.get("role", "skip"),
                    clearable=False,
                ),
                dcc.Dropdown(
                    id=ids.confirm_domain(column),
                    options=domain_options,
                    value=suggestion.get("suggested_domain"),
                    placeholder="Domain",
                    clearable=True,
                ),
                dcc.Input(
                    id=ids.confirm_metric(column),
                    type="text",
                    value=suggestion.get("suggested_metric") or "",
                    placeholder="Metric name",
                    className="input",
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "1.6fr 1fr 1fr 1fr",
                "gap": "1rem",
                "alignItems": "center",
            },
        )
    )