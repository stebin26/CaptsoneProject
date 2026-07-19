"""/login -- layout only.

Rendered full-bleed over the shell chrome via the .login-screen class, so the
sidebar/topbar behind it are hidden. Auth callbacks live in callbacks/auth.py.
"""

from __future__ import annotations

import dash
from dash import dcc, html

from app import ids
from app.components import ui

dash.register_page(__name__, path="/login", name="Login")


def layout() -> html.Div:
    """Build the sign-in page.

    Rendered without the shell, since there is no session to frame yet.

    Returns:
        The page layout.
    """
    return html.Div(
        html.Div(
            [
                html.H1("Industrial Data Hub", className="login-brand"),
                html.P(
                    "Sign in to continue.",
                    className="login-subtitle",
                ),
                ui.card(
                    ui.field(
                        "Email",
                        ui.text_input(ids.LOGIN_EMAIL, "you@company.com"),
                    ),
                    ui.field(
                        "Password",
                        dcc.Input(
                            id=ids.LOGIN_PASSWORD,
                            type="password",
                            placeholder="\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022",
                            className="input",
                            n_submit=0,
                        ),
                    ),
                    ui.button("Sign in", ids.LOGIN_SUBMIT),
                    html.Div(id=ids.LOGIN_STATUS, className="login-status"),
                    html.Div("or", className="login-divider"),
                    # Phase 2 placeholder — wired when Google OAuth lands.
                    ui.button(
                        "Continue with Google",
                        ids.LOGIN_GOOGLE,
                        variant="secondary",
                    ),
                ),
            ],
            className="login-card-wrap",
        ),
        className="login-screen",
    )
