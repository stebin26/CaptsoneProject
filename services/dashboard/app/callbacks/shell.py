"""Callbacks for the app shell.

The collapse toggle is clientside: it is pure view state and a server
round-trip for it would be a visible stutter on every click.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from dash import (
    Input,
    Output,
    State,
    callback,
    clientside_callback,
    dcc,
    html,
    no_update,
)

from app import ids
from app.api_client import APIError, list_datasets, ml_anomalies
from app.components import sidebar
from app.components.nav import BY_HREF
from app.logging_setup import get_logger

logger = get_logger(__name__)

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
    Input(ids.AUTH_USER, "data"),
)
def sync_route(pathname: str | None, user: Any) -> tuple[Any, str, str]:
    """Keep the rail, page title, and sign-in state in step with the route.

    Args:
        pathname: The current route.
        user: The stored user profile.

    Returns:
        The rendered rail and top bar for this route.
    """
    path = pathname or "/"
    item = BY_HREF.get(path)
    title = item.label if item else "Not found"
    kicker = item.kicker if item else ""
    permissions = (user or {}).get("permissions", [])
    return sidebar.nav_children(path, permissions), title, kicker


# ---- Dataset scope ----


@callback(
    Output(ids.TOPBAR_DATASET, "options"),
    Output(ids.TOPBAR_DATASET, "value"),
    Input(ids.SHELL_INIT, "n_intervals"),
    State(ids.ACTIVE_DATASET, "data"),
    State(ids.ACCESS_TOKEN, "data"),
)
def populate_datasets(
    _init: int | None,
    active: int | None,
    token: str | None,
) -> tuple[list[dict[str, Any]], Any]:
    """Fill the shell's dataset selector, preserving the active choice.

    Args:
        _init: Interval tick that triggers the initial load.
        active: The currently active dataset, kept selected when still present.
        token: Caller's access token.

    Returns:
        The selector options and the dataset to select.
    """
    try:
        datasets = list_datasets(token=token)
    except APIError:
        logger.warning("Callback shell.populate_datasets failed", exc_info=True)
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
    """Store the dataset chosen in the shell selector.

    Args:
        dataset_id: The chosen dataset.

    Returns:
        The dataset id to store.
    """
    return dataset_id


@callback(
    Output(ids.ACTIVE_DATE_RANGE, "data"),
    Input(ids.TOPBAR_DATE_RANGE, "start_date"),
    Input(ids.TOPBAR_DATE_RANGE, "end_date"),
)
def set_date_range(start: str | None, end: str | None) -> dict[str, Any]:
    """Store the date range chosen in the shell.

    Args:
        start: Start of the range.
        end: End of the range.

    Returns:
        The range to store.
    """
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
    stamp = datetime.now(UTC).astimezone().strftime("%H:%M")
    return (token or 0) + 1, f"Updated {stamp}"


# ---- Alerts ----


@callback(
    Output(ids.TOPBAR_BELL_COUNT, "children"),
    Output(ids.TOPBAR_BELL_PANEL, "children"),
    Input(ids.ACTIVE_DATASET, "data"),
    Input(ids.REFRESH_TOKEN, "data"),
    State(ids.ACCESS_TOKEN, "data"),
)
def load_alerts(
    dataset_id: int | None, _token: int | None, access: str | None
) -> tuple[Any, Any]:
    """Notifications are alerts. There is no second notification system.

    The count is the number of open anomalies for the active dataset -- the
    same rows the panel lists. If the two ever disagreed, one of them would
    be lying.
    """
    if dataset_id is None:
        return "", _panel([])

    try:
        anomalies = ml_anomalies(dataset_id, limit=200, token=access)
    except APIError:
        logger.warning("Callback shell.load_alerts failed", exc_info=True)
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
    """Toggle the alerts panel open or closed.

    Args:
        _clicks: Toggle button clicks.
        hidden: Whether the panel is currently hidden.

    Returns:
        The new hidden state.
    """
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


# ---- Page-level permission guard (Item 6, Step 3) ----
#
# The sidebar hides menus a user cannot access, but typing a URL directly
# still loads the page. This is the last UI gate: check the path's required
# permission against the user's grants, show a 403 instead of the page.
# Real enforcement is the backend 401/403 -- this only stops unauthorized
# page chrome from rendering.


@callback(
    Output(ids.SHELL_PAGE_WRAP, "hidden"),
    Output(ids.SHELL_GUARD_403, "hidden"),
    Output(ids.SHELL_GUARD_403, "children"),
    Input(ids.URL, "pathname"),
    Input(ids.AUTH_USER, "data"),
)
def guard_page(pathname: str | None, user: Any) -> tuple[bool, bool, Any]:
    """Block a route the caller lacks permission for.

    The rail already hides forbidden pages, but a user can still type the URL, so
    the same permission is enforced here as well and a refusal is shown in place of
    the page.

    Args:
        pathname: The requested route.
        user: The stored user profile.

    Returns:
        Whether to show the page or the refusal notice.
    """
    path = pathname or "/"
    item = BY_HREF.get(path)

    # Unmapped path (login, 404) or no permission requirement -> allow.
    if item is None or item.permission is None:
        return False, True, no_update

    # Not logged in: the clientside auth guard redirects to /login, so a 403
    # here would only flash. Allow and let the redirect win.
    perms = (user or {}).get("permissions") or []
    if not perms:
        return False, True, no_update

    if item.permission in perms:
        return False, True, no_update

    return True, False, _denied(item)


def _denied(item: Any) -> Any:
    return html.Div(
        style={
            "maxWidth": "440px",
            "textAlign": "center",
            "background": "#fff",
            "border": "1px solid #e5e7eb",
            "borderRadius": "12px",
            "padding": "40px 32px",
            "boxShadow": "0 1px 3px rgba(0,0,0,.06)",
        },
        children=[
            html.Div(
                "403",
                style={
                    "fontSize": "48px",
                    "fontWeight": "700",
                    "color": "#dc2626",
                    "lineHeight": "1",
                    "marginBottom": "12px",
                },
            ),
            html.H2(
                "You don't have access to this page",
                style={
                    "fontSize": "20px",
                    "fontWeight": "600",
                    "color": "#111827",
                    "margin": "0 0 8px",
                },
            ),
            html.P(
                [
                    f"\u201c{item.label}\u201d requires the ",
                    html.Code(
                        item.permission,
                        style={
                            "background": "#f3f4f6",
                            "padding": "1px 6px",
                            "borderRadius": "4px",
                            "fontSize": "13px",
                        },
                    ),
                    " permission, which your account does not have. "
                    "Ask an administrator if you need it.",
                ],
                style={
                    "fontSize": "14px",
                    "color": "#6b7280",
                    "lineHeight": "1.6",
                    "margin": "0 0 20px",
                },
            ),
            dcc.Link(
                "\u2190 Back to Executive Dashboard",
                href="/",
                style={
                    "display": "inline-block",
                    "padding": "9px 18px",
                    "background": "#111827",
                    "color": "#fff",
                    "borderRadius": "8px",
                    "textDecoration": "none",
                    "fontSize": "14px",
                    "fontWeight": "500",
                },
            ),
        ],
    )
