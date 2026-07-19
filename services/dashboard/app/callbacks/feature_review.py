"""Callbacks for /review.

Most platforms load your data and go quiet. This page reports back: what was
collected, what was skipped, and how complete the picture actually is -- with
the option to pull in anything missed, without re-uploading.
"""

from __future__ import annotations

from typing import Any

import dash
from dash import ALL, Input, Output, State, callback, ctx, dcc, html

from app import feedback, ids
from app.api_client import APIError, add_feature, dataset_summary, feature_review
from app.charts import review_charts
from app.components import ui
from app.constants import DOMAIN_ORDER, domain_label


@callback(
    Output(ids.REVIEW_HEADER, "children"),
    Output(ids.REVIEW_COVERAGE_CHART, "children"),
    Output(ids.REVIEW_COLLECTED, "children"),
    Output(ids.REVIEW_MISSED, "children"),
    Output(ids.REVIEW_DATA_CHARTS, "children"),
    Input(ids.REVIEW_INIT, "n_intervals"),
    Input(ids.REVIEW_REFRESH, "data"),
    State(ids.ONBOARDING_STORE, "data"),
    State(ids.ACCESS_TOKEN, "data"),
)
def load_review(
    _init: int | None,
    _refresh: Any,
    store: dict[str, Any] | None,
    token: str | None,
) -> tuple[Any, Any, Any, Any, Any]:
    """Load and render the collected, skipped, and coverage breakdown.

    Args:
        _init: Interval tick that triggers the initial load.
        _refresh: Refresh token bumped after a feature is added.
        store: The shared review store holding the dataset id.
        token: Caller's access token.

    Returns:
        The rendered review, or a message when unavailable.
    """
    if not store or "dataset_id" not in store:
        return (
            feedback.empty(
                "No dataset selected. Upload a file, or open one from Datasets."
            ),
            "",
            "",
            "",
            "",
        )

    dataset_id = store["dataset_id"]

    try:
        review = feature_review(dataset_id, token=token)
    except APIError as exc:
        return feedback.error(f"Could not load review: {exc}"), "", "", "", ""

    coverage = review.get("coverage", [])

    return (
        _header(review),
        _coverage(coverage),
        _collected(review.get("collected", [])),
        _missed(review.get("missed", [])),
        _data_charts(dataset_id, coverage, token),
    )


@callback(
    Output(ids.REVIEW_ADD_STATUS, "children"),
    Output(ids.REVIEW_REFRESH, "data"),
    Input(ids.add_button(ALL), "n_clicks"),
    State(ids.add_domain(ALL), "value"),
    State(ids.add_domain(ALL), "id"),
    State(ids.add_metric(ALL), "value"),
    State(ids.ONBOARDING_STORE, "data"),
    State(ids.REVIEW_REFRESH, "data"),
    State(ids.ACCESS_TOKEN, "data"),
    prevent_initial_call=True,
)
def handle_add(
    n_clicks_list: list[int],
    domains: list[str | None],
    domain_ids: list[dict[str, Any]],
    metrics: list[str | None],
    store: dict[str, Any] | None,
    refresh_token: int | None,
    token: str | None,
) -> tuple[Any, Any]:
    """Add the chosen skipped column as a new feature.

    Args:
        n_clicks_list: Click counts for each add button.
        domains: Selected domain for each skipped column.
        domain_ids: Component ids identifying those selectors.
        metrics: Entered metric name for each skipped column.
        store: The shared review store holding the dataset id.
        refresh_token: Refresh counter bumped to reload the review.
        token: Caller's access token.

    Returns:
        The result message and the bumped refresh counter.
    """
    hold = (dash.no_update, dash.no_update)

    if not store or not any(n_clicks_list):
        return hold

    triggered = ctx.triggered_id
    if not triggered or "column" not in triggered:
        return hold

    column = triggered["column"]
    index = next((i for i, d in enumerate(domain_ids) if d["column"] == column), None)
    if index is None:
        return hold

    domain = domains[index]
    metric = (metrics[index] or "").strip()

    if not domain:
        return feedback.error(
            f"Pick a domain for '{column}' before adding it."
        ), dash.no_update
    if not metric:
        return feedback.error(
            f"Give '{column}' a metric name before adding it."
        ), dash.no_update

    try:
        result = add_feature(
            dataset_id=store["dataset_id"],
            column_name=column,
            domain=domain,
            metric_name=metric,
            token=token,
        )
    except APIError as exc:
        return feedback.error(f"Could not add '{column}': {exc}"), dash.no_update

    # Bumping the token re-fires load_review, so the column moves from skipped
    # to collected without a page reload.
    return feedback.success(
        f"Added '{column}' to {domain_label(domain)}. "
        f"{result['features_added']} features generated."
    ), (refresh_token or 0) + 1


# ============================================================
# Render helpers
# ============================================================


def _header(review: dict[str, Any]) -> html.Div:
    return ui.card(
        html.Span(review["business_name"], style={"fontWeight": 600}),
        html.Span(
            f"  \u00b7  {review.get('industry') or 'industry unspecified'}"
            f"  \u00b7  {review.get('row_count') or 0:,} rows",
            className="kpi-note",
        ),
    )


def _coverage(coverage: list[dict[str, Any]]) -> Any:
    if not coverage:
        return ""
    return ui.chart(review_charts.coverage(coverage))


def _collected(collected: list[dict[str, Any]]) -> Any:
    if not collected:
        return feedback.empty("Nothing collected yet.")

    by_domain: dict[str, list[dict[str, Any]]] = {}
    for feature in collected:
        by_domain.setdefault(feature["domain"], []).append(feature)

    cards = []
    for domain in DOMAIN_ORDER:
        features = by_domain.get(domain)
        if not features:
            continue
        cards.append(
            ui.card(
                html.Div(
                    ui.domain_chip(domain, suffix=str(len(features))),
                    style={"marginBottom": "0.75rem"},
                ),
                html.Ul(
                    [
                        html.Li(
                            [
                                html.Span(f["feature_name"]),
                                html.Span(
                                    f"  \u2190 {f['source_column']}",
                                    className="kpi-note",
                                ),
                            ]
                        )
                        for f in features
                    ],
                    style={"margin": 0, "paddingLeft": "1.1rem"},
                ),
            )
        )

    # Anything the hub kept but that is not one of the eight universal domains.
    # It is shown rather than hidden -- a surprise bucket is a data problem
    # worth seeing, not one worth quietly dropping.
    extra = sorted(set(by_domain) - set(DOMAIN_ORDER))
    for domain in extra:
        cards.append(
            ui.card(
                html.Div(
                    ui.domain_chip(domain, suffix=str(len(by_domain[domain]))),
                    style={"marginBottom": "0.75rem"},
                ),
                html.P(
                    "Not one of the eight universal domains.",
                    className="kpi-note",
                ),
            )
        )

    return html.Div(cards)


def _missed(missed: list[dict[str, Any]]) -> Any:
    if not missed:
        return feedback.empty("Nothing was skipped. Every column was collected.")

    options = [{"label": domain_label(d), "value": d} for d in DOMAIN_ORDER]

    cards = []
    for col in missed:
        column = col["column_name"]
        samples = col.get("sample_values") or []
        sample_text = ", ".join(str(v) for v in samples[:3]) if samples else "\u2014"

        cards.append(
            ui.card(
                html.Div(column, style={"fontWeight": 600}),
                html.Div(
                    f"{col.get('data_type', 'unknown')} \u00b7 {sample_text}",
                    className="kpi-note",
                    style={"marginBottom": "0.75rem"},
                ),
                html.Div(
                    [
                        dcc.Dropdown(
                            id=ids.add_domain(column),
                            options=options,
                            value=col.get("suggested_domain"),
                            placeholder="Domain",
                            clearable=True,
                            style={"minWidth": "150px", "flex": "1"},
                        ),
                        dcc.Input(
                            id=ids.add_metric(column),
                            type="text",
                            value=column,
                            placeholder="Metric name",
                            className="input",
                            style={"flex": "1"},
                        ),
                        html.Button(
                            "Add",
                            id=ids.add_button(column),
                            n_clicks=0,
                            className="btn btn-secondary btn-sm",
                        ),
                    ],
                    style={
                        "display": "flex",
                        "gap": "0.5rem",
                        "alignItems": "center",
                    },
                ),
            )
        )

    return html.Div(cards)


def _data_charts(
    dataset_id: int, coverage: list[dict[str, Any]], token: str | None = None
) -> Any:
    if not any(c["features_collected"] > 0 for c in coverage):
        return feedback.empty("No hub data to chart yet.")

    try:
        summary = dataset_summary(dataset_id, token=token)
    except APIError:
        return feedback.empty(
            "Hub charts unavailable \u2014 the analytics layer has not run for "
            "this dataset yet."
        )

    metrics = summary.get("metrics", [])
    if not metrics:
        return feedback.empty("No metrics summarised yet.")

    return ui.chart(review_charts.metric_totals(metrics))
