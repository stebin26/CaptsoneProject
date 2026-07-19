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
    """Render a success message.

    Args:
        message: The text to show.

    Returns:
        The rendered message.
    """
    return html.Div(message, className="msg msg-success")


def empty(message: str) -> html.Div:
    """Render an empty-state message.

    Used where there is nothing to show yet, which is a normal state rather than a
    failure, so it is styled differently from an error.

    Args:
        message: The text to show.

    Returns:
        The rendered message.
    """
    return html.Div(message, className="msg-empty")
