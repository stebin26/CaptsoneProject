"""The navigation model: what the rail shows, in what order, and where it goes.

One list, four groups. Adding a page means adding one entry here -- the rail,
the active-state highlight and the topbar title all follow from it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NavItem:
    """One navigation entry: where it goes and who may see it.

    The permission string matches the backend's, so the rail and the API enforce
    the same rule from one definition rather than two that can drift.
    """
    label: str
    href: str
    icon: str
    kicker: str
    subtitle: str
    pending: bool = False
    # Permission required to see this item. None = any authenticated user.
    # Matches backend permission strings -- rail and API share one source.
    permission: str | None = None


@dataclass(frozen=True)
class NavGroup:
    """A titled group of navigation entries."""
    label: str | None
    items: tuple[NavItem, ...]


NAV: tuple[NavGroup, ...] = (
    NavGroup(
        label=None,
        items=(
            NavItem(
                "Executive Dashboard",
                "/",
                "i-dashboard",
                "Operational summary",
                "How the operation is doing right now, across every domain "
                "present in your data.",
                permission="analytics:read",
            ),
        ),
    ),
    NavGroup(
        label="Data hub",
        items=(
            NavItem(
                "Upload",
                "/upload",
                "i-upload",
                "Onboard a new business",
                "Drop in any business CSV. The platform reads it, profiles every "
                "column, and works out what kind of data it is \u2014 no schema, "
                "no setup, no code.",
                permission="dataset:upload",
            ),
            NavItem(
                "Mapping",
                "/confirm",
                "i-mapping",
                "Confirm the column mapping",
                "Every column has been mapped to a universal business domain. "
                "Review the suggestions, correct anything wrong, and confirm "
                "\u2014 this is the only manual step in the entire pipeline.",
                permission="mapping:confirm",
            ),
            NavItem(
                "Review",
                "/review",
                "i-review",
                "Coverage and gaps",
                "What we captured, what we missed, and how complete your data "
                "actually is \u2014 with the option to fill the gaps.",
                permission="dataset:read",
            ),
            NavItem(
                "Datasets",
                "/datasets",
                "i-datasets",
                "Everything onboarded",
                "Every business, every upload, side by side \u2014 all sitting in "
                "the same eight tables, queried by the same code.",
                permission="dataset:read",
            ),
        ),
    ),
    NavGroup(
        label="Business analytics",
        items=(
            NavItem(
                "Analytics",
                "/analytics",
                "i-analytics",
                "KPIs, trends and outliers",
                "Domain-by-domain KPIs, trends over time, and the entities that "
                "stand out \u2014 computed by Spark, orchestrated by Airflow.",
                permission="analytics:read",
            ),
            NavItem(
                "Predictions",
                "/predictions",
                "i-predictions",
                "Forecasts, anomalies and risk",
                "Where the numbers are heading, what is behaving abnormally, and "
                "which assets are most at risk.",
                permission="ml:read",
            ),
            NavItem(
                "Intelligence",
                "/intelligence",
                "i-intelligence",
                "Cross-domain relationships",
                "How the different parts of the business move together \u2014 a "
                "knowledge graph across all eight domains, lit up by the ML "
                "signals in this dataset.",
                permission="intelligence:read",
            ),
            NavItem(
                "Evaluation",
                "/evaluation",
                "i-evaluation",
                "How well the models actually work",
                "Measured performance of the RAG assistant and the ML models "
                "\u2014 retrieval accuracy, answer faithfulness, forecast error, "
                "and anomaly detection, scored against held-out ground truth.",
                permission="evaluation:read",
            ),
        ),
    ),
    NavGroup(
        label="Knowledge",
        items=(
            NavItem(
                "Documents",
                "/documents",
                "i-documents",
                "Ask your manuals",
                "Upload your manuals and SOPs, then ask questions in plain "
                "language. Answers are grounded in your own documents \u2014 or "
                "it tells you it found nothing.",
                permission="documents:read",
            ),
            NavItem(
                "AI Copilot",
                "/copilot",
                "i-copilot",
                "Investigate in plain language",
                "Ask anything about your operations. The agent decides which "
                "tools to use, gathers the evidence itself, and shows you exactly "
                "what it looked at.",
                permission="copilot:use",
            ),
        ),
    ),
    NavGroup(
        label="Management",
        items=(
            NavItem(
                "Settings",
                "/settings",
                "i-settings",
                "Platform configuration",
                "Platform configuration.",
                pending=True,
            ),
            NavItem(
                "Users",
                "/users",
                "i-users",
                "Access and roles",
                "People with access to this platform, and what they can do.",
                pending=True,
            ),
            NavItem(
                "Audit Logs",
                "/audit-logs",
                "i-audit",
                "Who did what, and when",
                "Who did what, and when.",
                pending=True,
            ),
        ),
    ),
)

# Flat lookup: path -> item. Used by the topbar to title the current page.
BY_HREF: dict[str, NavItem] = {item.href: item for group in NAV for item in group.items}
