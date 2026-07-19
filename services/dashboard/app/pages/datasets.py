"""/datasets -- layout only. Callbacks live in app/callbacks/datasets.py."""

from __future__ import annotations

import dash
from dash import dcc, html

from app import ids
from app.components import ui
from app.components.nav import BY_HREF

dash.register_page(__name__, path="/datasets", name="Datasets")


def layout() -> html.Div:
    """Build the Datasets page.

    Lists every onboarded dataset with its collected and skipped feature counts.

    Returns:
        The page layout.
    """
    return ui.page(
        dcc.Interval(id=ids.DATASETS_INIT, interval=300, max_intervals=1),
        ui.lede(BY_HREF["/datasets"].subtitle),
        html.Div(id=ids.DATASETS_LIST),
    )
