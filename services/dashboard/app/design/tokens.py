"""The subset of design tokens Python actually needs.

Mirrors app/assets/00-tokens.css. Plotly renders to canvas and cannot read
CSS variables, so chart colours must exist on the Python side. Nothing else
belongs here -- if a value is only used in CSS, it stays in CSS.

Keep in sync with 00-tokens.css. This duplication is deliberate and bounded.
"""

from __future__ import annotations

# ---- Ink and surfaces ----
INK = "#0F172A"
INK_MUTED = "#64748B"
INK_FAINT = "#94A3B8"
SURFACE = "#FFFFFF"
CANVAS = "#F6F8FB"
LINE = "#E4EAF2"

# ---- Accent ----
ACCENT = "#2563EB"
ACCENT_TINT = "#EFF6FF"

# ---- Semantics. Meaning only. ----
DANGER = "#EF4444"
WARN = "#F59E0B"
OK = "#10B981"

# ---- The eight universal domains ----
DOMAIN_INK: dict[str, str] = {
    "assets": "#4F46E5",
    "operations": "#0EA5E9",
    "quality": "#DC2626",
    "maintenance": "#D97706",
    "inventory": "#16A34A",
    "workforce": "#9333EA",
    "finance": "#0D9488",
    "customers": "#DB2777",
}

DOMAIN_TINT: dict[str, str] = {
    "assets": "#EEF2FF",
    "operations": "#E0F2FE",
    "quality": "#FEE2E2",
    "maintenance": "#FEF3C7",
    "inventory": "#DCFCE7",
    "workforce": "#F3E8FF",
    "finance": "#CCFBF1",
    "customers": "#FCE7F3",
}

DOMAIN_NONE = "#94A3B8"
DOMAIN_NONE_TINT = "#F1F5F9"

# ---- Risk and severity levels ----
LEVEL_INK: dict[str, str] = {
    "critical": DANGER,
    "high": DANGER,
    "medium": WARN,
    "low": OK,
}

# ---- Type ----
FONT = 'Inter, -apple-system, "Segoe UI", Roboto, system-ui, sans-serif'


def domain_ink(domain: str) -> str:
    """Saturated colour for a domain. Used for chart series and glyphs."""
    return DOMAIN_INK.get(domain.lower(), DOMAIN_NONE)


def domain_tint(domain: str) -> str:
    """Soft background for a domain tile or chip."""
    return DOMAIN_TINT.get(domain.lower(), DOMAIN_NONE_TINT)


def level_ink(level: str | None) -> str:
    """Colour for a risk or severity level. Unknown levels are neutral, not red."""
    return LEVEL_INK.get((level or "").lower(), INK_MUTED)
