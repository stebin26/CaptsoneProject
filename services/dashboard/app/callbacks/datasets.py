"""Callbacks for /datasets.

The registry view of the hub. A manufacturing dataset and a telecom dataset
are not two systems -- they are two rows, in the same eight tables, queried by
the same code. This page is that argument made visible.
"""

from __future__ import annotations

from typing import Any

import dash
from dash import Input, Output, State, callback, ctx, html

from app import feedback, ids
from app.api_client import APIError, list_datasets
from app.components import ui
from app.logging_setup import get_logger

logger = get_logger(__name__)


@callback(
    Output(ids.DATASETS_LIST, "children"),
    Input(ids.DATASETS_INIT, "n_intervals"),
    State(ids.ACCESS_TOKEN, "data"),
)
def load_datasets(_init: int | None, token: str | None) -> Any:
    """Load and render the dataset browse list.

    Args:
        _init: Interval tick that triggers the initial load.
        token: Caller's access token.

    Returns:
        The rendered dataset list, or a message when unavailable.
    """
    try:
        datasets = list_datasets(token=token)
    except APIError as exc:
        logger.warning("Callback datasets.load_datasets failed", exc_info=True)
        return feedback.error(f"Could not load datasets: {exc}")

    if not datasets:
        return ui.card(
            html.Div("No datasets yet", className="empty-title"),
            html.P(
                "Upload a CSV and the platform will profile it, map every "
                "column to a universal domain, and load it into the hub.",
                className="empty-body",
            ),
        )

    return html.Div([_dataset_card(d) for d in datasets])


@callback(
    Output(ids.ONBOARDING_STORE, "data", allow_duplicate=True),
    Output(ids.URL, "pathname", allow_duplicate=True),
    Input(ids.open_dataset(dash.ALL), "n_clicks"),
    State(ids.ONBOARDING_STORE, "data"),
    prevent_initial_call=True,
)
def open_dataset(
    n_clicks_list: list[int],
    store: dict[str, Any] | None,
) -> tuple[Any, Any]:
    """Open the clicked dataset's feature review.

    Args:
        n_clicks_list: Click counts for each dataset's open button.
        store: The shared review store to update.

    Returns:
        The updated store and the redirect target.
    """
    if not any(n_clicks_list):
        return dash.no_update, dash.no_update

    triggered = ctx.triggered_id
    if not triggered or "id" not in triggered:
        return dash.no_update, dash.no_update

    new_store = dict(store or {})
    new_store["dataset_id"] = triggered["id"]
    return new_store, "/review"


# ============================================================
# Render helpers
# ============================================================


def _dataset_card(d: dict[str, Any]) -> html.Div:
    collected = d["features_collected"]
    skipped = d["features_skipped"]
    total = collected + skipped

    return ui.card(
        html.Div(
            [
                html.Div(
                    [
                        html.Div(d["business_name"], style={"fontWeight": 600}),
                        html.Div(
                            f"#{d['dataset_id']}"
                            f"  \u00b7  {d.get('industry') or 'industry unspecified'}"
                            f"  \u00b7  {d.get('row_count') or 0:,} rows"
                            f"  \u00b7  {d['uploaded_at'][:10]}",
                            className="kpi-note",
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span(str(collected), className="trend-down"),
                                html.Span(
                                    f" of {total} columns collected",
                                    className="kpi-note",
                                ),
                            ]
                        ),
                        html.Div(
                            f"{skipped} skipped" if skipped else "nothing skipped",
                            className="kpi-note",
                        ),
                    ],
                    style={"textAlign": "right"},
                ),
                html.Button(
                    "Open",
                    id=ids.open_dataset(d["dataset_id"]),
                    n_clicks=0,
                    className="btn btn-secondary btn-sm",
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr auto auto",
                "gap": "1.5rem",
                "alignItems": "center",
            },
        )
    )
