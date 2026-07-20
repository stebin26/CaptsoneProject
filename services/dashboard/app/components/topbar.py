"""The topbar.

Every control here is wired to something real. The dataset picker sets the
scope every page reads. Refresh bumps a token that re-fires page callbacks.
The bell count is the live open-anomaly count -- the same number the alerts
panel shows. Nothing on this bar is decoration.
"""

from __future__ import annotations

from dash import dcc, html

from app import ids
from app.components.primitives import icon


def render() -> html.Header:
    """Render the top bar with the page title and global controls.

    Returns:
        The rendered top bar.
    """
    return html.Header(
        className="topbar",
        children=[
            html.Button(
                icon("i-menu"),
                id=ids.SHELL_SIDEBAR_TOGGLE + "-mobile",
                n_clicks=0,
                className="topbar-toggle",
                **{"aria-label": "Toggle navigation"},
            ),
            html.Div(
                className="topbar-heading",
                children=[
                    html.Div(id=ids.TOPBAR_TITLE, className="topbar-title"),
                    html.Div(id=ids.TOPBAR_SUBTITLE, className="topbar-subtitle"),
                ],
            ),
            html.Div(
                className="topbar-tools",
                children=[
                    html.Div(
                        dcc.Dropdown(
                            id=ids.TOPBAR_DATASET,
                            options=[],
                            placeholder="Select dataset",
                            clearable=False,
                        ),
                        id=ids.TOPBAR_PICKER,
                        className="topbar-picker",
                    ),
                    dcc.DatePickerRange(
                        id=ids.TOPBAR_DATE_RANGE,
                        display_format="D MMM YYYY",
                        clearable=True,
                        minimum_nights=0,
                    ),
                    html.Div(
                        className="topbar-stamp",
                        children=[
                            html.Span(className="stamp-dot"),
                            html.Span(id=ids.TOPBAR_LAST_REFRESH),
                        ],
                    ),
                    html.Button(
                        icon("i-refresh"),
                        id=ids.TOPBAR_REFRESH,
                        n_clicks=0,
                        className="topbar-icon-btn",
                        title="Recompute this view",
                        **{"aria-label": "Refresh"},
                    ),
                    html.Button(
                        [
                            icon("i-bell"),
                            html.Span(id=ids.TOPBAR_BELL_COUNT, className="bell-count"),
                        ],
                        id=ids.TOPBAR_BELL,
                        n_clicks=0,
                        className="topbar-icon-btn",
                        title="Open alerts",
                        **{"aria-label": "Alerts"},
                    ),
                    html.Div(
                        id=ids.TOPBAR_PROFILE,
                        className="topbar-profile",
                        children=[
                            html.Span("AD", className="avatar"),
                            html.Span("Admin", id=ids.TOPBAR_USER_NAME),
                            html.Button(
                                "Sign out",
                                id=ids.TOPBAR_LOGOUT,
                                n_clicks=0,
                                className="btn btn-ghost btn-sm",
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(id=ids.TOPBAR_BELL_PANEL, className="bell-panel", hidden=True),
        ],
    )
