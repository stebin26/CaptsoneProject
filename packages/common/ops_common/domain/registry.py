"""The domain registry -- what each universal domain means and computes.

For each of the eight domains this declares the alias keywords used to
recognise a raw column and the ready-made features that domain owns. It is pure
declaration: the mapping suggester reads the aliases to route columns, and the
analytics layer reads the features to know what to compute. Onboarding a new
industry changes the mapping, never this file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ops_common.domain.models import Domain


# The set of aggregation operations a feature can apply (sum, avg, trend, etc.).
# Each feature below picks one of these to describe HOW it's computed.
class Aggregation(str, Enum): # noqa: UP042
    """How a feature is computed from its underlying metric values."""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    DISTINCT_COUNT = "distinct_count"
    TREND = "trend"
    VARIANCE = "variance"
    RATE = "rate"


# Blueprint of a single feature: its name, how it's aggregated, a description,
# and whether it needs a time column (trends require time).
@dataclass(frozen=True)
class FeatureDef:
    """Blueprint of a single feature within a domain.

    Names the feature, how it aggregates, what it means, and whether it needs a
    time column (trends do).
    """
    name: str
    aggregation: Aggregation
    description: str
    requires_time: bool = False


# Blueprint of one domain: which Domain it is, a description, the alias keywords
# used to recognise it, and the list of features it owns.
@dataclass(frozen=True)
class DomainSpec:
    """Blueprint of one universal domain.

    Carries the domain itself, a description, the alias keywords used to recognise
    it in raw column names, and the features it owns.
    """
    domain: Domain
    description: str
    aliases: tuple[str, ...]
    features: tuple[FeatureDef, ...] = field(default_factory=tuple)

    # Convenience: just the feature names of this domain.
    def feature_names(self) -> list[str]:
        """Return this domain's feature names.

        Returns:
            The names of every feature the domain owns.
        """
        return [f.name for f in self.features]


# ============================================================
# Registry — universal functions every operation shares.
# Aliases are the hints the LLM suggester and keyword fallback
# use to route a raw column into the right domain.
# ============================================================

# THE central config of the whole portability idea: for each of the 8 domains,
# its recognition aliases + its ready-made features. Spark reads this to know
# what to compute; the suggester reads aliases to route columns. Pure declaration.
DOMAIN_REGISTRY: dict[Domain, DomainSpec] = {
    # ASSETS — the physical/logical things being operated. Aliases catch words
    # like machine/tower/vehicle; features count assets and measure utilization.
    Domain.ASSETS: DomainSpec(
        domain=Domain.ASSETS,
        description="Physical or logical resources: machines, towers, vehicles, aircraft, classrooms.",
        aliases=(
            "asset",
            "machine",
            "equipment",
            "device",
            "tower",
            "vehicle",
            "aircraft",
            "engine",
            "unit",
            "resource",
            "plant",
            "node",
            "server",
        ),
        features=(
            FeatureDef("asset_count", Aggregation.DISTINCT_COUNT, "Number of distinct assets."),
            FeatureDef("avg_utilization", Aggregation.AVG, "Average utilization across assets."),
            FeatureDef("max_load", Aggregation.MAX, "Peak load observed on an asset."),
            FeatureDef(
                "availability_rate", Aggregation.RATE, "Share of time assets are available."
            ),
        ),
    ),
    # OPERATIONS — the core work/output. Features sum and trend the throughput.
    Domain.OPERATIONS: DomainSpec(
        domain=Domain.OPERATIONS,
        description="The core work performed: production, throughput, calls handled, trips, sessions.",
        aliases=(
            "production",
            "output",
            "throughput",
            "units_produced",
            "volume",
            "operation",
            "run",
            "trips",
            "sessions",
            "calls",
            "tasks",
            "jobs",
            "flight_hours",
            "sorties",
            "transactions",
        ),
        features=(
            FeatureDef("total_output", Aggregation.SUM, "Total operational output."),
            FeatureDef("avg_output", Aggregation.AVG, "Average output per record."),
            FeatureDef(
                "output_trend", Aggregation.TREND, "Output direction over time.", requires_time=True
            ),
            FeatureDef("peak_output", Aggregation.MAX, "Highest output observed."),
        ),
    ),
    # QUALITY — defects/failures/complaints. Features count and trend defects.
    Domain.QUALITY: DomainSpec(
        domain=Domain.QUALITY,
        description="Defects, failures, SLA breaches, complaints, inspection outcomes.",
        aliases=(
            "defect",
            "failure",
            "reject",
            "scrap",
            "complaint",
            "sla",
            "breach",
            "error",
            "fault",
            "inspection",
            "quality",
            "ncr",
            "incident",
        ),
        features=(
            FeatureDef("defect_count", Aggregation.SUM, "Total defects or failures."),
            FeatureDef("defect_rate", Aggregation.RATE, "Defects relative to output."),
            FeatureDef(
                "defect_trend", Aggregation.TREND, "Defect direction over time.", requires_time=True
            ),
            FeatureDef("worst_offender", Aggregation.MAX, "Highest single defect value."),
        ),
    ),
    # MAINTENANCE — repairs/downtime. Features sum downtime and trend it.
    Domain.MAINTENANCE: DomainSpec(
        domain=Domain.MAINTENANCE,
        description="Repairs, downtime, servicing, maintenance cycles, mean time to repair.",
        aliases=(
            "maintenance",
            "repair",
            "downtime",
            "service",
            "servicing",
            "breakdown",
            "outage",
            "mttr",
            "mtbf",
            "overhaul",
            "cycle",
        ),
        features=(
            FeatureDef("total_downtime", Aggregation.SUM, "Total downtime accumulated."),
            FeatureDef("avg_repair_time", Aggregation.AVG, "Average repair duration."),
            FeatureDef(
                "downtime_trend",
                Aggregation.TREND,
                "Downtime direction over time.",
                requires_time=True,
            ),
            FeatureDef("maintenance_events", Aggregation.COUNT, "Number of maintenance events."),
        ),
    ),
    # INVENTORY — stock/materials. Features sum stock and measure variance.
    Domain.INVENTORY: DomainSpec(
        domain=Domain.INVENTORY,
        description="Stock, materials, spare parts, supply levels, consumables.",
        aliases=(
            "inventory",
            "stock",
            "material",
            "spare",
            "parts",
            "supply",
            "consumable",
            "warehouse",
            "quantity_on_hand",
            "reorder",
        ),
        features=(
            FeatureDef("total_stock", Aggregation.SUM, "Total stock on hand."),
            FeatureDef("avg_stock_level", Aggregation.AVG, "Average stock level."),
            FeatureDef("stock_variance", Aggregation.VARIANCE, "Variability in stock levels."),
            FeatureDef("low_stock_min", Aggregation.MIN, "Lowest stock level observed."),
        ),
    ),
    # WORKFORCE — staff/shifts/hours. Features count heads and sum labor hours.
    Domain.WORKFORCE: DomainSpec(
        domain=Domain.WORKFORCE,
        description="Staff, shifts, crew, headcount, attendance, labor hours.",
        aliases=(
            "workforce",
            "staff",
            "employee",
            "crew",
            "headcount",
            "shift",
            "attendance",
            "labor",
            "labour",
            "operator",
            "personnel",
            "worker",
        ),
        features=(
            FeatureDef("headcount", Aggregation.DISTINCT_COUNT, "Distinct staff count."),
            FeatureDef("total_hours", Aggregation.SUM, "Total labor hours."),
            FeatureDef("avg_hours", Aggregation.AVG, "Average hours per record."),
            FeatureDef("attendance_rate", Aggregation.RATE, "Share of expected attendance met."),
        ),
    ),
    # FINANCE — money: revenue/cost. Features sum, average, trend, vary value.
    Domain.FINANCE: DomainSpec(
        domain=Domain.FINANCE,
        description="Spend, revenue, cost, billing, efficiency, margins.",
        aliases=(
            "finance",
            "revenue",
            "cost",
            "spend",
            "expense",
            "bill",
            "billing",
            "amount",
            "price",
            "margin",
            "budget",
            "fee",
            "payment",
            "invoice",
        ),
        features=(
            FeatureDef("total_revenue", Aggregation.SUM, "Total revenue or amount."),
            FeatureDef("total_cost", Aggregation.SUM, "Total cost or spend."),
            FeatureDef("avg_value", Aggregation.AVG, "Average monetary value per record."),
            FeatureDef(
                "value_trend", Aggregation.TREND, "Monetary trend over time.", requires_time=True
            ),
            FeatureDef("value_variance", Aggregation.VARIANCE, "Variability in monetary values."),
        ),
    ),
    # CUSTOMERS — subscribers/students/clients. Features count and measure churn.
    Domain.CUSTOMERS: DomainSpec(
        domain=Domain.CUSTOMERS,
        description="Subscribers, students, clients, patients, accounts.",
        aliases=(
            "customer",
            "subscriber",
            "student",
            "client",
            "patient",
            "account",
            "member",
            "user",
            "consumer",
            "tenant",
            "buyer",
        ),
        features=(
            FeatureDef("customer_count", Aggregation.DISTINCT_COUNT, "Distinct customers."),
            FeatureDef("repeat_rate", Aggregation.RATE, "Share of repeat customers."),
            FeatureDef("churn_proxy", Aggregation.RATE, "Inactive over total customers."),
            FeatureDef("new_customers", Aggregation.COUNT, "Newly observed customers."),
        ),
    ),
}


# ============================================================
# Lookup helpers
# ============================================================


# Get the full spec for a domain — accepts either a Domain enum or its string.
def get_spec(domain: Domain | str) -> DomainSpec:
    """Return the full specification for a domain.

    Args:
        domain: The domain, as an enum member or its string value.

    Returns:
        The domain's specification.
    """
    if isinstance(domain, str):
        domain = Domain(domain)
    return DOMAIN_REGISTRY[domain]


# Get just the feature definitions for a domain (used by Spark when computing).
def features_for_domain(domain: Domain | str) -> tuple[FeatureDef, ...]:
    """Return the feature definitions a domain owns.

    Args:
        domain: The domain, as an enum member or its string value.

    Returns:
        The domain's feature definitions.
    """
    return get_spec(domain).features


# Flatten every alias across all domains into one {alias: domain} dict.
# This is the lookup table the keyword fallback searches.
def all_aliases() -> dict[str, Domain]:
    """Flatten every domain's aliases into one lookup table.

    Returns:
        A mapping of lowercase alias to its domain.
    """
    mapping: dict[str, Domain] = {}
    for spec in DOMAIN_REGISTRY.values():
        for alias in spec.aliases:
            mapping[alias.lower()] = spec.domain
    return mapping


# Keyword fallback (used when the LLM is off/fails): if any alias appears
# inside the column name, route it to that domain. First match wins.
def match_domain_by_keyword(column_name: str) -> Domain | None:
    """Route a column to a domain by alias keyword.

    The fallback used when the LLM suggester is disabled or fails: the first alias
    found inside the column name wins.

    Args:
        column_name: The raw source column name.

    Returns:
        The matched domain, or None if no alias matched.
    """
    name = column_name.lower()
    for alias, domain in all_aliases().items():
        if alias in name:
            return domain
    return None


# Builds a text summary of all domains + features to feed into the LLM prompt,
# so the suggester knows the available domains when classifying columns.
def registry_as_prompt_context() -> str:
    """Render the registry as prompt context for the LLM suggester.

    Returns:
        A newline-separated summary of every domain and its features.
    """
    lines: list[str] = []
    for spec in DOMAIN_REGISTRY.values():
        feats = ", ".join(spec.feature_names())
        lines.append(f"- {spec.domain.value}: {spec.description} Features: {feats}")
    return "\n".join(lines)
