from __future__ import annotations

from typing import Any

import dash
from dash import Input, Output, State, callback, ctx, dcc, html

from app.api_client import APIError, list_datasets
from app import theme

dash.register_page(__name__, path="/datasets", name="Datasets")


def layout() -> html.Div:
    return html.Div(
        style=theme.PAGE_STYLE,
        children=[
            dcc.Interval(id="datasets-init", interval=300, max_intervals=1),
            html.H1("Onboarded datasets", style=theme.HEADING_STYLE),
            html.P(
                "Every dataset loaded into the hub. Click one to open its review.",
                style=theme.SUBHEADING_STYLE,
            ),
            html.Div(id="datasets-list"),
        ],
    )


@callback(
    Output("datasets-list", "children"),
    Input("datasets-init", "n_intervals"),
)
def load_datasets(_init: int | None) -> Any:
    try:
        datasets = list_datasets()
    except APIError as exc:
        return html.Div(
            f"Could not load datasets: {exc}",
            style={"color": theme.COLORS["danger"]},
        )

    if not datasets:
        return html.Div(
            "No datasets onboarded yet. Upload a file to get started.",
            style={"color": theme.COLORS["text_muted"]},
        )

    return html.Div([_dataset_card(d) for d in datasets])


def _dataset_card(d: dict[str, Any]) -> html.Div:
    return html.Div(
        style={
            **theme.CARD_STYLE,
            "display": "grid",
            "gridTemplateColumns": "2fr 1fr 1fr auto",
            "gap": "1rem",
            "alignItems": "center",
        },
        children=[
            html.Div(
                [
                    html.Div(
                        f"{d['business_name']}",
                        style={"fontWeight": 600, "fontSize": "1.05rem"},
                    ),
                    html.Div(
                        f"#{d['dataset_id']} · {d.get('industry') or 'unspecified'} · "
                        f"{d.get('row_count') or 0} rows",
                        style={"fontSize": "0.8rem", "color": theme.COLORS["text_muted"]},
                    ),
                    html.Div(
                        d["uploaded_at"][:19].replace("T", " "),
                        style={"fontSize": "0.75rem", "color": theme.COLORS["text_muted"]},
                    ),
                ]
            ),
            html.Div(
                [
                    html.Span(str(d["features_collected"]),
                              style={"fontWeight": 600, "color": theme.COLORS["collected"]}),
                    html.Span(" collected", style={"fontSize": "0.8rem",
                                                   "color": theme.COLORS["text_muted"]}),
                ]
            ),
            html.Div(
                [
                    html.Span(str(d["features_skipped"]),
                              style={"fontWeight": 600, "color": theme.COLORS["skipped"]}),
                    html.Span(" skipped", style={"fontSize": "0.8rem",
                                                 "color": theme.COLORS["text_muted"]}),
                ]
            ),
            html.Button(
                "Open review",
                id={"type": "open-dataset", "id": d["dataset_id"]},
                n_clicks=0,
                style=theme.SECONDARY_BUTTON_STYLE,
            ),
        ],
    )


@callback(
    Output("onboarding-store", "data", allow_duplicate=True),
    Output("url", "pathname", allow_duplicate=True),
    Input({"type": "open-dataset", "id": dash.ALL}, "n_clicks"),
    State("onboarding-store", "data"),
    prevent_initial_call=True,
)
def open_dataset(n_clicks_list: list[int], store: dict[str, Any] | None) -> tuple[Any, Any]:
    if not any(n_clicks_list):
        return dash.no_update, dash.no_update

    triggered = ctx.triggered_id
    if not triggered or "id" not in triggered:
        return dash.no_update, dash.no_update

    dataset_id = triggered["id"]
    new_store = dict(store or {})
    new_store["dataset_id"] = dataset_id
    return new_store, "/review"
