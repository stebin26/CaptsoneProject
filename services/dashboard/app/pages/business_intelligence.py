"""/intelligence -- layout only.

Callbacks live in app/callbacks/business_intelligence.py.
"""

from __future__ import annotations

import dash
from dash import dcc, html

from app import ids
from app.components import ui
from app.components.nav import BY_HREF

dash.register_page(__name__, path="/intelligence", name="Intelligence")


def layout() -> html.Div:
    return ui.page(
        html.H1("How it all connects", className="page-title"),
        ui.lede(BY_HREF["/intelligence"].subtitle),

        ui.field(
            "Dataset",
            dcc.Dropdown(
                id=ids.BI_DATASET,
                options=[],
                placeholder="Select a dataset",
                clearable=False,
            ),
        ),
        dcc.Interval(id=ids.BI_INIT, interval=300, max_intervals=1),

        html.Div(id=ids.BI_SUMMARY),
        html.Div(id=ids.BI_INSIGHTS),
    )