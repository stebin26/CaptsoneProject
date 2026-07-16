"""Status messages: error, success, empty.

Errors say what happened and what to do about it. They do not apologise and
they are never vague. Emptiness is a state, not a failure -- an empty screen
is an invitation to act.
"""

from __future__ import annotations

from dash import html


def error(message: str) -> html.Div:
    """A failure the user must see. Never swallow an exception into empty output."""
    return html.Div(message, className="msg msg-error")


def success(message: str) -> html.Div:
    return html.Div(message, className="msg msg-success")


def empty(message: str) -> html.Div:
    return html.Div(message, className="msg-empty")