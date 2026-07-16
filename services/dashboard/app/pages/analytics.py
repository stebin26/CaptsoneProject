"""/analytics -- layout only.

Callbacks live in app/callbacks/analytics.py.
Figure builders live in app/charts/domain_charts.py.
"""

from __future__ import annotations

import dash
from dash import dcc, html

from app import ids
from app.components import ui

dash.register_page(__name__, path="/analytics", name="Analytics")


def layout() -> html.Div:
    return ui.page(
        html.H1("What the data says", className="page-title"),
        ui.lede(
            "Domain-by-domain KPIs, trends over time, and the entities that "
            "stand out \u2014 computed by Spark, orchestrated by Airflow. No "
            "model has run yet; this is the ground truth everything else "
            "reasons over."
        ),

        ui.field(
            "Dataset",
            dcc.Dropdown(
                id=ids.ANALYTICS_DATASET,
                options=[],
                placeholder="Select a dataset",
                clearable=False,
            ),
        ),
        dcc.Interval(id=ids.ANALYTICS_INIT, interval=300, max_intervals=1),

        html.Div(id=ids.ANALYTICS_SUMMARY),
        html.Div(id=ids.ANALYTICS_METRICS_CHART),

        ui.section(
            "Trend over time",
            ui.field(
                "Metric",
                dcc.Dropdown(
                    id=ids.ANALYTICS_METRIC,
                    options=[],
                    placeholder="Select a metric",
                    clearable=False,
                ),
            ),
            html.Div(id=ids.ANALYTICS_TREND_CHART),
        ),

        ui.rule(),
        ui.section(
            "Engineered features",
            html.Div(id=ids.ANALYTICS_FEATURES_TABLE),
            note=(
                "One row per entity and metric, with a trend slope \u2014 which "
                "is what separates a degrading machine from a merely "
                "underperforming one. This table is what the ML layer trains on."
            ),
        ),

        ui.rule(),
        ui.section(
            "Every active domain",
            html.Div(id=ids.ANALYTICS_DOMAIN_STATUS),
            html.Div(id=ids.ANALYTICS_DOMAIN_CHARTS),
            note=(
                "Each domain gets the chart its data actually suits. Only "
                "domains present in this dataset are drawn."
            ),
        ),
    )