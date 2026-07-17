"""The shell: rail + topbar + content. Wraps every page."""

from __future__ import annotations

from typing import Any

from dash import dcc, html

from app import ids
from app.components import sidebar, topbar


def render(page_container: Any) -> html.Div:
    return html.Div(
        id=ids.SHELL_ROOT,
        className="shell",
        children=[
            sidebar.render(pathname="/"),
            html.Div(
                className="shell-main",
                children=[
                    topbar.render(),
                    html.Main(
                        [
                            html.Div(page_container, id=ids.SHELL_PAGE_WRAP),
                            html.Div(
                                id=ids.SHELL_GUARD_403,
                                hidden=True,
                                style={
                                    "display": "flex",
                                    "alignItems": "center",
                                    "justifyContent": "center",
                                    "minHeight": "60vh",
                                    "padding": "48px 24px",
                                },
                            ),
                        ],
                        id=ids.SHELL_MAIN,
                        className="content",
                    ),
                ],
            ),
            html.Div(id=ids.SHELL_SCRIM, className="rail-scrim", n_clicks=0),
        ],
    )


def stores() -> list:
    """Global state owned by the shell. Mounted once, read by every page."""
    return [
        dcc.Location(id=ids.URL, refresh=False),
        dcc.Store(id=ids.ONBOARDING_STORE, storage_type="session"),
        dcc.Store(id=ids.REVIEW_REFRESH, storage_type="memory"),
        dcc.Store(id=ids.ACTIVE_DATASET, storage_type="session"),
        dcc.Store(id=ids.ACTIVE_DATE_RANGE, storage_type="session"),
        dcc.Store(id=ids.REFRESH_TOKEN, storage_type="memory", data=0),
        # Auth state (Item 6): session-scoped so a reload keeps you logged in
        # but closing the tab logs you out.
        dcc.Store(id=ids.ACCESS_TOKEN, storage_type="session"),
        dcc.Store(id=ids.AUTH_REFRESH_TOKEN, storage_type="session"),
        dcc.Store(id=ids.AUTH_USER, storage_type="session"),
        # Collapse survives a reload: it is a preference, not view state.
        dcc.Store(id=ids.SHELL_COLLAPSED, storage_type="local", data=False),
        dcc.Interval(id=ids.SHELL_INIT, interval=300, max_intervals=1),
    ]