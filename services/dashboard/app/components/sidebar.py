"""The navy rail."""

from __future__ import annotations

from collections.abc import Iterable

from dash import dcc, html

from app import ids
from app.components.nav import NAV, NavItem
from app.components.primitives import icon


def render(pathname: str, permissions: Iterable[str] | None = None) -> html.Aside:
    return html.Aside(
        id=ids.SHELL_SIDEBAR,
        className="rail",
        children=[
            _brand(),
            html.Nav(
                id=ids.SHELL_NAV,
                className="rail-nav",
                children=_nav(pathname, permissions),
            ),
            _foot(),
        ],
    )


def nav_children(pathname: str, permissions: Iterable[str] | None = None) -> list:
    """Exposed so the active-state callback can rebuild only the nav."""
    return _nav(pathname, permissions)


def _visible(item: NavItem, perms: set[str]) -> bool:
    # No permission required -> always visible (covers pending placeholders).
    if item.permission is None:
        return True
    return item.permission in perms


def _brand() -> html.Div:
    return html.Div(
        className="rail-brand",
        children=[
            icon("i-hub", "icon-lg"),
            html.Span("Industrial Data Hub", className="rail-brand-text"),
        ],
    )


def _nav(pathname: str, permissions: Iterable[str] | None) -> list:
    perms = set(permissions or [])
    groups = []
    for group in NAV:
        visible_items = [it for it in group.items if _visible(it, perms)]
        if not visible_items:
            continue
        children = []
        if group.label:
            children.append(html.Div(group.label, className="rail-group-label"))
        children += [_link(item, pathname) for item in visible_items]
        groups.append(html.Div(children, className="rail-group"))
    return groups


def _link(item: NavItem, pathname: str) -> dcc.Link:
    classes = ["rail-link"]
    if item.href == pathname:
        classes.append("is-active")
    if item.pending:
        classes.append("is-pending")

    return dcc.Link(
        href=item.href,
        className=" ".join(classes),
        children=[
            icon(item.icon),
            html.Span(item.label, className="rail-label"),
        ],
    )


def _foot() -> html.Div:
    return html.Div(
        className="rail-foot",
        children=[
            html.Button(
                id=ids.SHELL_SIDEBAR_TOGGLE,
                n_clicks=0,
                className="rail-collapse",
                children=[
                    icon("i-chevron-left", "icon-sm"),
                    html.Span("Collapse", className="rail-label"),
                ],
            )
        ],
    )