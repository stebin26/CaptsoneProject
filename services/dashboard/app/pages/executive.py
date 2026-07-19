"""/ -- the Executive Dashboard. Layout only.

The one screen that answers "how is the operation doing?" without a single
model call. Every number here is read-and-assembled by one API endpoint
(/executive/{id}/summary); this file only lays out the slots it fills.

Callbacks live in app/callbacks/executive.py.
"""

from __future__ import annotations

import dash
from dash import dcc, html

from app import ids
from app.components import ui

dash.register_page(__name__, path="/", name="Executive")


def layout() -> html.Div:
    """Build the Executive Dashboard page.

    The operation at a glance: risk index, domain health, top risks, open alerts,
    forecasts, and insights.

    Returns:
        The page layout.
    """
    return ui.page(
        html.H1("How the operation is doing", className="page-title"),
        ui.lede(
            "One read across every domain \u2014 risk, alerts, and what is "
            "coming next. Assembled from analytics and the ML layer in a single "
            "call, with no model in the path, so this screen is fast by design."
        ),
        ui.field(
            "Dataset",
            dcc.Dropdown(
                id=ids.EXEC_STORE,
                options=[],
                placeholder="Select a dataset",
                clearable=False,
            ),
        ),
        dcc.Interval(id=ids.EXEC_INIT, interval=300, max_intervals=1),
        html.Div(id=ids.EXEC_ERROR),
        # KPI row -- four headline numbers.
        ui.grid(
            html.Div(id=ids.EXEC_KPI_RISK),
            html.Div(id=ids.EXEC_KPI_ALERTS),
            html.Div(id=ids.EXEC_KPI_INDUSTRY),
            html.Div(id=ids.EXEC_KPI_FRESHNESS),
            cols=4,
        ),
        ui.section(
            "Domain health",
            html.Div(id=ids.EXEC_DOMAIN_HEALTH),
            note=(
                "All eight domains, always. An absent one is shown, not hidden "
                '\u2014 an empty slot is what makes a later "no data for that" '
                "explainable."
            ),
        ),
        ui.grid(
            ui.section(
                "Top risks",
                html.Div(id=ids.EXEC_TOP_RISKS),
                note="Highest relative degradation, with each entity's direction of travel.",
            ),
            ui.section(
                "Active alerts",
                html.Div(id=ids.EXEC_ACTIVE_ALERTS),
                note="Most recent anomalies, worst severity first.",
            ),
            cols=2,
        ),
        ui.section(
            "What is coming next",
            html.Div(id=ids.EXEC_FORECASTS),
            note="Recent history, then the projected next value inside its confidence band.",
        ),
        ui.section(
            "Cross-domain insights",
            html.Div(id=ids.EXEC_INSIGHTS),
            note=(
                "Template-rendered from the same inference engine the "
                "Intelligence page uses \u2014 no model call, so an exec screen "
                "can never take two minutes."
            ),
        ),
    )
