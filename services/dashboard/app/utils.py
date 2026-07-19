"""Pure helpers. No Dash, no Plotly, no styling. Safe to unit-test."""

from __future__ import annotations

from typing import Any

EM_DASH = "\u2014"


def fmt(value: Any, places: int = 2) -> str:
    """Format a number for display. None becomes an em dash, never 'None'."""
    if value is None:
        return EM_DASH
    try:
        return f"{float(value):.{places}f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_int(value: Any) -> str:
    """Format a value as a thousands-separated integer.

    Args:
        value: The value to format.

    Returns:
        The formatted string, or a dash when the value is missing.
    """
    if value is None:
        return EM_DASH
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return str(value)


def fmt_pct(value: Any, places: int = 1) -> str:
    """Format a value as a signed percentage.

    Args:
        value: The value to format.
        places: Decimal places to show.

    Returns:
        The formatted string, or a dash when the value is missing.
    """
    if value is None:
        return EM_DASH
    try:
        return f"{float(value):+.{places}f}%"
    except (TypeError, ValueError):
        return str(value)


def avg_by_entity(rows: list[dict[str, Any]]) -> tuple[list[str], list[float]]:
    """Split entity_features rows into (entity labels, average values)."""
    entities = [r["entity_ref"] for r in rows]
    averages = [float(r.get("avg_value") or 0) for r in rows]
    return entities, averages


def group_by_domain(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group rows by their domain.

    Args:
        rows: The rows to group.

    Returns:
        The rows keyed by domain.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["domain"], []).append(row)
    return grouped


def rgba(hex_color: str, alpha: float) -> str:
    """Convert a hex colour into a CSS colour string with transparency.

    Used for chart confidence bands, where the band must sit behind the line
    without hiding it.

    Args:
        hex_color: Colour in #RRGGBB form.
        alpha: Opacity between 0 and 1.

    Returns:
        The colour as a CSS string, or a neutral grey if the input is malformed.
    """
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return f"rgba(120,120,120,{alpha})"
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def group(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    """Group rows by a lowercased string key. Missing keys become the string 'None'."""
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(str(row.get(key)).lower(), []).append(row)
    return out
