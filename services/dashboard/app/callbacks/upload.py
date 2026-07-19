"""Callbacks for / (upload)."""

from __future__ import annotations

import base64
from typing import Any

import dash
from dash import Input, Output, State, callback, html

from app import feedback, ids
from app.api_client import APIError, start_onboarding


@callback(
    Output(ids.UPLOAD_FILENAME, "children"),
    Input(ids.UPLOAD_DATA, "filename"),
    prevent_initial_call=True,
)
def show_filename(filename: str | None) -> Any:
    """Show the name of the file the user selected.

    Args:
        filename: The selected filename.

    Returns:
        The rendered filename, or nothing when none is selected.
    """
    if not filename:
        return ""
    return html.Span(f"Selected: {filename}", className="msg-success")


@callback(
    Output(ids.UPLOAD_STATUS, "children"),
    Output(ids.ONBOARDING_STORE, "data"),
    Output(ids.URL, "pathname", allow_duplicate=True),
    Input(ids.UPLOAD_SUBMIT, "n_clicks"),
    State(ids.UPLOAD_BUSINESS_NAME, "value"),
    State(ids.UPLOAD_INDUSTRY, "value"),
    State(ids.UPLOAD_DATA, "contents"),
    State(ids.UPLOAD_DATA, "filename"),
    State(ids.ACCESS_TOKEN, "data"),
    prevent_initial_call=True,
)
def handle_upload(
    n_clicks: int,
    business_name: str | None,
    industry: str | None,
    contents: str | None,
    filename: str | None,
    token: str | None,
) -> tuple[Any, Any, Any]:
    """Upload the CSV, start onboarding, and move to the mapping review.

    Args:
        n_clicks: Upload button clicks.
        business_name: Business the dataset belongs to.
        industry: Optional industry label.
        contents: Base64 contents of the selected file.
        filename: Name of the selected file.
        token: Caller's access token.

    Returns:
        The result message, the onboarding store, and the redirect target.
    """
    hold = (dash.no_update, dash.no_update, dash.no_update)

    if not n_clicks:
        return hold
    if not business_name:
        return feedback.error("Please enter a business name."), *hold[1:]
    if not contents or not filename:
        return feedback.error("Please select a CSV file."), *hold[1:]

    try:
        file_bytes = _decode_upload(contents)
    except Exception:  # noqa: BLE001
        return feedback.error("Could not read the uploaded file."), *hold[1:]

    try:
        result = start_onboarding(
            file_bytes=file_bytes,
            filename=filename,
            business_name=business_name,
            industry=industry or None,
            token=token,
        )
    except APIError as exc:
        return feedback.error(f"Onboarding failed: {exc}"), *hold[1:]

    store_data = {
        "dataset_id": result["dataset_id"],
        "business_name": result["business_name"],
        "industry": result.get("industry"),
        "row_count": result["row_count"],
        "stored_path": result["stored_path"],
        "suggestions": result["suggestions"],
    }

    message = feedback.success(
        f"Profiled {result['row_count']} rows across "
        f"{len(result['suggestions'])} columns. Redirecting to review..."
    )
    return message, store_data, "/confirm"


def _decode_upload(contents: str) -> bytes:
    _, _, payload = contents.partition(",")
    return base64.b64decode(payload)
