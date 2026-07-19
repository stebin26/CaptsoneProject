"""Callbacks for /login and the global auth guard."""

from __future__ import annotations

import dash
from dash import Input, Output, State, callback, clientside_callback, html

from app import feedback, ids
from app.api_client import APIError, auth_me
from app.api_client import login as api_login
from app.logging_setup import get_logger

logger = get_logger(__name__)

_PUBLIC_PATHS = frozenset({"/login"})


@callback(
    Output(ids.ACCESS_TOKEN, "data"),
    Output(ids.AUTH_REFRESH_TOKEN, "data"),
    Output(ids.AUTH_USER, "data"),
    Output(ids.LOGIN_STATUS, "children"),
    Input(ids.LOGIN_SUBMIT, "n_clicks"),
    Input(ids.LOGIN_PASSWORD, "n_submit"),
    State(ids.LOGIN_EMAIL, "value"),
    State(ids.LOGIN_PASSWORD, "value"),
    prevent_initial_call=True,
)
def handle_login(n_clicks, n_submit, email, password):
    """Authenticate the user and store the issued tokens.

    Args:
        n_clicks: Sign-in button clicks.
        n_submit: Enter presses in the password field.
        email: Entered email address.
        password: Entered password.

    Returns:
        The stored tokens and profile on success, or an error message.
    """
    no = dash.no_update
    if not (n_clicks or n_submit):
        return no, no, no, no
    if not email or not password:
        return no, no, no, feedback.error("Enter your email and password.")

    try:
        tokens = api_login(email, password)
    except APIError as exc:
        logger.warning("Callback auth.handle_login failed", exc_info=True)
        return no, no, no, feedback.error(str(exc))

    access = tokens["access_token"]
    refresh = tokens["refresh_token"]

    try:
        user = auth_me(access)
    except APIError as exc:
        logger.warning("Callback auth.handle_login failed", exc_info=True)
        return no, no, no, feedback.error(f"Loaded tokens, profile failed: {exc}")

    return access, refresh, user, ""


@callback(
    Output(ids.LOGIN_STATUS, "children", allow_duplicate=True),
    Input(ids.LOGIN_GOOGLE, "n_clicks"),
    prevent_initial_call=True,
)
def google_login(n_clicks):
    """Report that Google sign-in is not yet available.

    Args:
        n_clicks: Button clicks.

    Returns:
        A message explaining the option is not enabled.
    """
    if not n_clicks:
        return dash.no_update
    return html.Span("Google sign-in coming soon.", className="msg-success")


clientside_callback(
    """
    function(token, pathname) {
        if (token && pathname === '/login') {
            window.location.href = '/';
            return window.dash_clientside.no_update;
        }
        if (!token && pathname !== '/login') {
            window.location.href = '/login';
            return window.dash_clientside.no_update;
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output(ids.URL, "pathname", allow_duplicate=True),
    Input(ids.ACCESS_TOKEN, "data"),
    Input(ids.URL, "pathname"),
    prevent_initial_call=True,
)

# ---- Logout: clear tokens, guard sends to /login ----


@callback(
    Output(ids.ACCESS_TOKEN, "data", allow_duplicate=True),
    Output(ids.AUTH_REFRESH_TOKEN, "data", allow_duplicate=True),
    Output(ids.AUTH_USER, "data", allow_duplicate=True),
    Input(ids.TOPBAR_LOGOUT, "n_clicks"),
    prevent_initial_call=True,
)
def handle_logout(n_clicks):
    """Revoke the session and clear the stored tokens.

    Args:
        n_clicks: Sign-out button clicks.

    Returns:
        The cleared stores and the redirect target.
    """
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update
    # Clearing the token makes the guard redirect to /login on the next tick.
    return None, None, None


# ---- Show the signed-in user's name in the topbar ----


@callback(
    Output(ids.TOPBAR_USER_NAME, "children"),
    Input(ids.AUTH_USER, "data"),
)
def show_user(user):
    """Render the signed-in user's name and role in the top bar.

    Args:
        user: The stored user profile.

    Returns:
        The rendered identity, or nothing when signed out.
    """
    if not user:
        return ""
    return user.get("full_name") or user.get("email") or "User"
