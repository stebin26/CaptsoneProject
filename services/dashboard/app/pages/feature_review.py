"""/review -- layout only. Callbacks live in app/callbacks/feature_review.py."""

from __future__ import annotations

import dash
from dash import dcc, html

from app import ids
from app.components import ui
from app.components.nav import BY_HREF

dash.register_page(__name__, path="/review", name="Feature review")


def layout() -> html.Div:
    """Build the Feature Review page.

    Shows what onboarding collected and what it skipped, and lets a skipped column
    be added as a feature.

    Returns:
        The page layout.
    """
    return ui.page(
        dcc.Interval(id=ids.REVIEW_INIT, interval=300, max_intervals=1),
        ui.lede(BY_HREF["/review"].subtitle),
        html.Div(id=ids.REVIEW_HEADER),
        html.Div(id=ids.REVIEW_COVERAGE_CHART),
        ui.grid(
            html.Div(
                [
                    html.H2("Collected", className="section-title"),
                    html.P(
                        "Mapped to a domain and loaded into the hub.",
                        className="page-subtitle",
                    ),
                    html.Div(id=ids.REVIEW_COLLECTED),
                ]
            ),
            html.Div(
                [
                    html.H2("Skipped", className="section-title"),
                    html.P(
                        "Not collected. Add any of these without re-uploading.",
                        className="page-subtitle",
                    ),
                    html.Div(id=ids.REVIEW_MISSED),
                ]
            ),
            cols=2,
        ),
        ui.rule(),
        ui.section(
            "Hub data",
            html.Div(id=ids.REVIEW_DATA_CHARTS),
            note="Metric totals across every domain this dataset populated.",
        ),
        html.Div(id=ids.REVIEW_ADD_STATUS, style={"marginTop": "1rem"}),
    )
