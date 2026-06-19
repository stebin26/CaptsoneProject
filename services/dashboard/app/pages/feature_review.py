from __future__ import annotations

from typing import Any

import dash
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, callback, ctx, dcc, html

from app.api_client import (
    APIError,
    add_feature,
    dataset_summary,
    domain_data,
    feature_review,
)
from app import theme

dash.register_page(__name__, path="/review", name="Feature review")


def layout() -> html.Div:
    return html.Div(
        style=theme.PAGE_STYLE,
        children=[
            dcc.Interval(id="review-init", interval=300, max_intervals=1),
            html.H1("Feature review", style=theme.HEADING_STYLE),
            html.P(
                "This is what the platform extracted from your data. "
                "Review what was collected, see what was skipped, and pull in "
                "anything that was missed.",
                style=theme.SUBHEADING_STYLE,
            ),
            html.Div(id="review-header"),
            html.Div(id="review-coverage-chart"),
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "gap": "1.5rem",
                    "marginTop": "1rem",
                },
                children=[
                    html.Div(
                        [
                            html.H3(
                                "Collected features",
                                style={"fontSize": "1.1rem", "color": theme.COLORS["collected"]},
                            ),
                            html.Div(id="review-collected"),
                        ]
                    ),
                    html.Div(
                        [
                            html.H3(
                                "Skipped columns",
                                style={"fontSize": "1.1rem", "color": theme.COLORS["skipped"]},
                            ),
                            html.Div(id="review-missed"),
                        ]
                    ),
                ],
            ),
            html.Hr(style={"margin": "2rem 0", "border": "none",
                           "borderTop": f"1px solid {theme.COLORS['border']}"}),
            html.H3("Explore hub data", style={"fontSize": "1.2rem"}),
            html.Div(id="review-data-charts"),
            html.Div(id="review-add-status", style={"marginTop": "1rem"}),
        ],
    )


# ============================================================
# Main load — triggered once on page open and after each add
# ============================================================

@callback(
    Output("review-header", "children"),
    Output("review-coverage-chart", "children"),
    Output("review-collected", "children"),
    Output("review-missed", "children"),
    Output("review-data-charts", "children"),
    Input("review-init", "n_intervals"),
    Input("review-refresh", "data"),
    State("onboarding-store", "data"),
)
def load_review(
    _init: int | None,
    _refresh: Any,
    store: dict[str, Any] | None,
) -> tuple[Any, Any, Any, Any, Any]:
    if not store or "dataset_id" not in store:
        empty = html.Div(
            "No dataset loaded. Please upload and confirm a file first.",
            style={"color": theme.COLORS["text_muted"]},
        )
        return empty, "", "", "", ""

    dataset_id = store["dataset_id"]

    try:
        review = feature_review(dataset_id)
    except APIError as exc:
        err = _error(f"Could not load review: {exc}")
        return err, "", "", "", ""

    header = _render_header(review)
    coverage = _render_coverage_chart(review.get("coverage", []))
    collected = _render_collected(review.get("collected", []))
    missed = _render_missed(review.get("missed", []), review.get("coverage", []))
    charts = _render_data_charts(dataset_id, review.get("coverage", []))

    return header, coverage, collected, missed, charts


def _render_header(review: dict[str, Any]) -> html.Div:
    return html.Div(
        style=theme.CARD_STYLE,
        children=[
            html.Span(review["business_name"], style={"fontWeight": 600, "fontSize": "1.1rem"}),
            html.Span(
                f"  ·  {review.get('industry') or 'unspecified'}  ·  "
                f"{review.get('row_count') or 0} rows",
                style={"color": theme.COLORS["text_muted"]},
            ),
        ],
    )


def _render_coverage_chart(coverage: list[dict[str, Any]]) -> Any:
    if not coverage:
        return ""

    domains = [c["domain"] for c in coverage]
    collected = [c["features_collected"] for c in coverage]
    skipped = [c["features_skipped"] for c in coverage]

    fig = go.Figure()
    fig.add_bar(name="Collected", x=domains, y=collected, marker_color=theme.COLORS["collected"])
    fig.add_bar(name="Skipped", x=domains, y=skipped, marker_color=theme.COLORS["skipped"])
    fig.update_layout(
        barmode="stack",
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        plot_bgcolor="white",
        title="Feature coverage by domain",
    )
    return html.Div(dcc.Graph(figure=fig), style=theme.CARD_STYLE)


def _render_collected(collected: list[dict[str, Any]]) -> Any:
    if not collected:
        return html.Div("Nothing collected yet.", style={"color": theme.COLORS["text_muted"]})

    by_domain: dict[str, list[dict[str, Any]]] = {}
    for f in collected:
        by_domain.setdefault(f["domain"], []).append(f)

    blocks = []
    for domain, feats in sorted(by_domain.items()):
        items = [
            html.Li(
                [
                    html.Span(f["feature_name"], style={"fontWeight": 500}),
                    html.Span(
                        f"  ← {f['source_column']}",
                        style={"color": theme.COLORS["text_muted"], "fontSize": "0.8rem"},
                    ),
                ],
                style={"marginBottom": "0.25rem"},
            )
            for f in feats
        ]
        blocks.append(
            html.Div(
                style=theme.CARD_STYLE,
                children=[
                    html.Div(
                        domain.title(),
                        style={"fontWeight": 600, "color": theme.domain_color(domain),
                               "marginBottom": "0.4rem"},
                    ),
                    html.Ul(items, style={"margin": 0, "paddingLeft": "1.1rem"}),
                ],
            )
        )
    return html.Div(blocks)


def _render_missed(missed: list[dict[str, Any]], coverage: list[dict[str, Any]]) -> Any:
    if not missed:
        return html.Div(
            "No columns were skipped — everything was collected.",
            style={"color": theme.COLORS["text_muted"]},
        )

    rows = []
    for col in missed:
        col_name = col["column_name"]
        samples = col.get("sample_values") or []
        sample_text = ", ".join(str(v) for v in samples[:3]) if samples else "—"

        rows.append(
            html.Div(
                style=theme.CARD_STYLE,
                children=[
                    html.Div(col_name, style={"fontWeight": 600}),
                    html.Div(
                        f"{col.get('data_type', '?')} · samples: {sample_text}",
                        style={"fontSize": "0.8rem", "color": theme.COLORS["text_muted"],
                               "marginBottom": "0.5rem"},
                    ),
                    html.Div(
                        style={"display": "flex", "gap": "0.5rem", "alignItems": "center"},
                        children=[
                            dcc.Dropdown(
                                id={"type": "add-domain", "column": col_name},
                                options=[
                                    {"label": d.title(), "value": d}
                                    for d in _ALL_DOMAINS
                                ],
                                value=col.get("suggested_domain"),
                                placeholder="domain",
                                style={"minWidth": "150px"},
                                clearable=True,
                            ),
                            dcc.Input(
                                id={"type": "add-metric", "column": col_name},
                                type="text",
                                value=col_name,
                                placeholder="metric name",
                                style={
                                    "padding": "0.4rem 0.6rem",
                                    "border": f"1px solid {theme.COLORS['border']}",
                                    "borderRadius": "6px",
                                    "flex": "1",
                                },
                            ),
                            html.Button(
                                "Add",
                                id={"type": "add-button", "column": col_name},
                                n_clicks=0,
                                style=theme.ADD_BUTTON_STYLE,
                            ),
                        ],
                    ),
                ],
            )
        )
    return html.Div(rows)


def _render_data_charts(dataset_id: int, coverage: list[dict[str, Any]]) -> Any:
    domains_with_data = [c["domain"] for c in coverage if c["features_collected"] > 0]
    if not domains_with_data:
        return html.Div(
            "No hub data to chart yet.", style={"color": theme.COLORS["text_muted"]}
        )

    try:
        summary = dataset_summary(dataset_id)
    except APIError:
        return html.Div(
            "Hub data charts unavailable (analytics layer not ready).",
            style={"color": theme.COLORS["text_muted"]},
        )

    metrics = summary.get("metrics", [])
    if not metrics:
        return html.Div("No metrics summarized yet.", style={"color": theme.COLORS["text_muted"]})

    domains = sorted({m["domain"] for m in metrics})
    names = [f"{m['domain']}.{m['metric_name']}" for m in metrics]
    totals = [m.get("metric_sum") or 0 for m in metrics]
    colors = [theme.domain_color(m["domain"]) for m in metrics]

    fig = go.Figure()
    fig.add_bar(x=names, y=totals, marker_color=colors)
    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=40, b=120),
        plot_bgcolor="white",
        title="Metric totals across the hub",
        xaxis=dict(tickangle=-45),
    )
    return html.Div(dcc.Graph(figure=fig), style=theme.CARD_STYLE)


# ============================================================
# Add a missed feature
# ============================================================

@callback(
    Output("review-add-status", "children"),
    Output("review-refresh", "data"),
    Input({"type": "add-button", "column": ALL}, "n_clicks"),
    State({"type": "add-domain", "column": ALL}, "value"),
    State({"type": "add-domain", "column": ALL}, "id"),
    State({"type": "add-metric", "column": ALL}, "value"),
    State("onboarding-store", "data"),
    State("review-refresh", "data"),
    prevent_initial_call=True,
)
def handle_add(
    n_clicks_list: list[int],
    domains: list[str | None],
    domain_ids: list[dict[str, Any]],
    metrics: list[str | None],
    store: dict[str, Any] | None,
    refresh_token: int | None,
) -> tuple[Any, Any]:
    if not store or not any(n_clicks_list):
        return dash.no_update, dash.no_update

    triggered = ctx.triggered_id
    if not triggered or "column" not in triggered:
        return dash.no_update, dash.no_update

    target_col = triggered["column"]
    idx = next((i for i, d in enumerate(domain_ids) if d["column"] == target_col), None)
    if idx is None:
        return dash.no_update, dash.no_update

    domain = domains[idx]
    metric = (metrics[idx] or "").strip()

    if not domain:
        return _error(f"Pick a domain for '{target_col}' before adding."), dash.no_update
    if not metric:
        return _error(f"Enter a metric name for '{target_col}'."), dash.no_update

    try:
        result = add_feature(
            dataset_id=store["dataset_id"],
            column_name=target_col,
            domain=domain,
            metric_name=metric,
        )
    except APIError as exc:
        return _error(f"Add failed: {exc}"), dash.no_update

    new_token = (refresh_token or 0) + 1
    return (
        _success(
            f"Added '{target_col}' to {domain}. "
            f"{result['features_added']} features generated."
        ),
        new_token,
    )


_ALL_DOMAINS = [
    "assets", "operations", "quality", "maintenance",
    "inventory", "workforce", "finance", "customers",
]


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