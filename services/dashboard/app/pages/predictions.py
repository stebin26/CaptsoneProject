"""/predictions -- layout only.

Callbacks live in app/callbacks/predictions.py.
Figure builders live in app/charts/prediction_charts.py.
"""

from __future__ import annotations

import dash
from dash import dcc, html

from app import ids
from app.components import ui
from app.components.nav import BY_HREF

dash.register_page(__name__, path="/predictions", name="Predictions")


def layout() -> html.Div:
    return ui.page(
        html.H1("What happens next", className="page-title"),
        ui.lede(BY_HREF["/predictions"].subtitle),

        ui.field(
            "Dataset",
            dcc.Dropdown(
                id=ids.PRED_DATASET,
                options=[],
                placeholder="Select a dataset",
                clearable=False,
            ),
        ),
        dcc.Interval(id=ids.PRED_INIT, interval=300, max_intervals=1),

        html.Div(id=ids.PRED_KPIS),
        html.Div(id=ids.PRED_DOMAIN_CHARTS),
        html.Div(id=ids.PRED_TABLE),
    )