"""Callbacks for the app shell.

The collapse toggle is clientside: it is pure view state and a server
round-trip for it would be a visible stutter on every click.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from dash import Input, Output, State, callback, clientside_callback, html, no_update

from app import ids
from app.api_client import APIError, list_datasets, ml_anomalies
from app.components import sidebar
from app.components.nav import BY_HREF

# ---- Collapse / mobile drawer. Clientside: no server round-trip. ----

clientside_callback(
    """
    function (deskClicks, mobileClicks, scrimClicks, collapsed) {
        const trigger = dash_clientside.callback_context.triggered_id;
        const root = document.getElementById('shell-root');
        if (!root) { return [window.dash_clientside.no_update]; }

        // Below 760px the rail leaves the flow and slides over the content,
        // so the same button means "open the drawer", not "collapse".
        const isMobile = window.matchMedia('(max-width: 760px)').matches;

        if (trigger === 'shell-scrim') {
            root.classList.remove('is-open');
            return [window.dash_clientside.no_update];
        }

        if (isMobile) {
            root.classList.toggle('is-open');
            return [window.dash_clientside.no_update];
        }

       const next = !root.classList.contains('is-collapsed');
        root.classList.toggle('is-collapsed', next);
        return [next];
    }
    """,
    Output(ids.SHELL_COLLAPSED, "data"),
    Input(ids.SHELL_SIDEBAR_TOGGLE, "n_clicks"),
    Input(ids.SHELL_SIDEBAR_TOGGLE + "-mobile", "n_clicks"),
    Input(ids.SHELL_SCRIM, "n_clicks"),
    State(ids.SHELL_COLLAPSED, "data"),
    prevent_initial_call=True,
)

# Restore the collapsed preference on load, before first paint.
# Restore the collapsed preference once on load, via the init interval.
clientside_callback(
    """
    function (_init, collapsed) {
        const root = document.getElementById('shell-root');
        if (root) { root.classList.toggle('is-collapsed', !!collapsed); }
        return window.dash_clientside.no_update;
    }
    """,
    Output(ids.SHELL_INIT, "n_intervals"),
    Input(ids.SHELL_INIT, "n_intervals"),
    State(ids.SHELL_COLLAPSED, "data"),
    prevent_initial_call=False,
)


# ---- Active nav item + page title ----

@callback(
    Output(ids.SHELL_NAV, "children"),
    Output(ids.TOPBAR_TITLE, "children"),
    Output(ids.TOPBAR_SUBTITLE, "children"),
    Input(ids.URL, "pathname"),
)
def sync_route(pathname: str | None) -> tuple[Any, str, str]:
    path = pathname or "/"
    item = BY_HREF.get(path)
    title = item.label if item else "Not found"
    kicker = item.kicker if item else ""
    return sidebar.nav_children(path), title, kicker

# ---- Dataset scope ----

@callback(
    Output(ids.TOPBAR_DATASET, "options"),
    Output(ids.TOPBAR_DATASET, "value"),
    Input(ids.SHELL_INIT, "n_intervals"),
    State(ids.ACTIVE_DATASET, "data"),
)
def populate_datasets(
    _init: int | None,
    active: int | None,
) -> tuple[list[dict[str, Any]], Any]:
    try:
        datasets = list_datasets()
    except APIError:
        return [], None

    options = [
        {"label": d["business_name"], "value": d["dataset_id"]} for d in datasets
    ]
    if not options:
        return [], None

    known = {o["value"] for o in options}
    value = active if active in known else options[0]["value"]
    return options, value


@callback(
    Output(ids.ACTIVE_DATASET, "data"),
    Input(ids.TOPBAR_DATASET, "value"),
    prevent_initial_call=True,
)
def set_active_dataset(dataset_id: int | None) -> Any:
    return dataset_id


@callback(
    Output(ids.ACTIVE_DATE_RANGE, "data"),
    Input(ids.TOPBAR_DATE_RANGE, "start_date"),
    Input(ids.TOPBAR_DATE_RANGE, "end_date"),
)
def set_date_range(start: str | None, end: str | None) -> dict[str, Any]:
    return {"start": start, "end": end}


# ---- Refresh ----

@callback(
    Output(ids.REFRESH_TOKEN, "data"),
    Output(ids.TOPBAR_LAST_REFRESH, "children"),
    Input(ids.TOPBAR_REFRESH, "n_clicks"),
    Input(ids.ACTIVE_DATASET, "data"),
    State(ids.REFRESH_TOKEN, "data"),
)
def refresh(
    _clicks: int | None,
    _dataset_id: int | None,
    token: int | None,
) -> tuple[int, str]:
    """Bump a token that page callbacks listen to, and stamp the time.

    The stamp says when this view was last computed. It is not 'real-time':
    the platform is a batch snapshot, and the label says so.
    """
    stamp = datetime.now(timezone.utc).astimezone().strftime("%H:%M")
    return (token or 0) + 1, f"Updated {stamp}"


# ---- Alerts ----

@callback(
    Output(ids.TOPBAR_BELL_COUNT, "children"),
    Output(ids.TOPBAR_BELL_PANEL, "children"),
    Input(ids.ACTIVE_DATASET, "data"),
    Input(ids.REFRESH_TOKEN, "data"),
)
def load_alerts(dataset_id: int | None, _token: int | None) -> tuple[Any, Any]:
    """Notifications are alerts. There is no second notification system.

    The count is the number of open anomalies for the active dataset -- the
    same rows the panel lists. If the two ever disagreed, one of them would
    be lying.
    """
    if dataset_id is None:
        return "", _panel([])

    try:
        anomalies = ml_anomalies(dataset_id, limit=200)
    except APIError:
        return "", _panel([])

    if not anomalies:
        return "", _panel([])

    rank = {"high": 0, "medium": 1, "low": 2}
    ordered = sorted(anomalies, key=lambda a: rank.get(a.get("severity"), 3))
    count = len(anomalies)

    return (str(count) if count < 100 else "99+"), _panel(ordered[:8], count)


@callback(
    Output(ids.TOPBAR_BELL_PANEL, "hidden"),
    Input(ids.TOPBAR_BELL, "n_clicks"),
    State(ids.TOPBAR_BELL_PANEL, "hidden"),
    prevent_initial_call=True,
)
def toggle_alerts(_clicks: int, hidden: bool) -> bool:
    return not hidden


def _panel(anomalies: list[dict[str, Any]], total: int = 0) -> Any:
    head = html.Div(
        f"Alerts \u00b7 {total}" if total else "Alerts",
        className="bell-panel-head",
    )

    if not anomalies:
        return [
            head,
            html.Div(
                "No open alerts for this dataset.",
                className="msg-empty",
            ),
        ]

    items = []
    for a in anomalies:
        severity = a.get("severity") or "low"
        items.append(
            html.Div(
                className="bell-item",
                children=[
                    html.Span(severity, className=f"badge badge-{severity}"),
                    html.Div(
                        [
                            html.Div(
                                f"{a.get('metric_name', 'metric')} anomaly",
                                className="bell-item-title",
                            ),
                            html.Div(
                                f"{a.get('entity_ref') or a.get('domain', '')} "
                                f"\u00b7 {str(a.get('detected_at') or '')[:10]}",
                                className="bell-item-meta",
                            ),
                        ]
                    ),
                ],
            )
        )

    return [head, *items]