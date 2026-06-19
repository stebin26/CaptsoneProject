from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ops_common.domain.models import Domain


class Aggregation(str, Enum):
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    DISTINCT_COUNT = "distinct_count"
    TREND = "trend"
    VARIANCE = "variance"
    RATE = "rate"


@dataclass(frozen=True)
class FeatureDef:
    name: str
    aggregation: Aggregation
    description: str
    requires_time: bool = False


@dataclass(frozen=True)
class DomainSpec:
    domain: Domain
    description: str
    aliases: tuple[str, ...]
    features: tuple[FeatureDef, ...] = field(default_factory=tuple)

    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]


# ============================================================
# Registry — universal functions every operation shares.
# Aliases are the hints the LLM suggester and keyword fallback
# use to route a raw column into the right domain.
# ============================================================

DOMAIN_REGISTRY: dict[Domain, DomainSpec] = {
    Domain.ASSETS: DomainSpec(
        domain=Domain.ASSETS,
        description="Physical or logical resources: machines, towers, vehicles, aircraft, classrooms.",
        aliases=(
            "asset", "machine", "equipment", "device", "tower", "vehicle",
            "aircraft", "engine", "unit", "resource", "plant", "node", "server",
        ),
        features=(
            FeatureDef("asset_count", Aggregation.DISTINCT_COUNT, "Number of distinct assets."),
            FeatureDef("avg_utilization", Aggregation.AVG, "Average utilization across assets."),
            FeatureDef("max_load", Aggregation.MAX, "Peak load observed on an asset."),
            FeatureDef("availability_rate", Aggregation.RATE, "Share of time assets are available."),
        ),
    ),
    Domain.OPERATIONS: DomainSpec(
        domain=Domain.OPERATIONS,
        description="The core work performed: production, throughput, calls handled, trips, sessions.",
        aliases=(
            "production", "output", "throughput", "units_produced", "volume",
            "operation", "run", "trips", "sessions", "calls", "tasks", "jobs",
            "flight_hours", "sorties", "transactions",
        ),
        features=(
            FeatureDef("total_output", Aggregation.SUM, "Total operational output."),
            FeatureDef("avg_output", Aggregation.AVG, "Average output per record."),
            FeatureDef("output_trend", Aggregation.TREND, "Output direction over time.", requires_time=True),
            FeatureDef("peak_output", Aggregation.MAX, "Highest output observed."),
        ),
    ),
    Domain.QUALITY: DomainSpec(
        domain=Domain.QUALITY,
        description="Defects, failures, SLA breaches, complaints, inspection outcomes.",
        aliases=(
            "defect", "failure", "reject", "scrap", "complaint", "sla", "breach",
            "error", "fault", "inspection", "quality", "ncr", "incident",
        ),
        features=(
            FeatureDef("defect_count", Aggregation.SUM, "Total defects or failures."),
            FeatureDef("defect_rate", Aggregation.RATE, "Defects relative to output."),
            FeatureDef("defect_trend", Aggregation.TREND, "Defect direction over time.", requires_time=True),
            FeatureDef("worst_offender", Aggregation.MAX, "Highest single defect value."),
        ),
    ),
    Domain.MAINTENANCE: DomainSpec(
        domain=Domain.MAINTENANCE,
        description="Repairs, downtime, servicing, maintenance cycles, mean time to repair.",
        aliases=(
            "maintenance", "repair", "downtime", "service", "servicing",
            "breakdown", "outage", "mttr", "mtbf", "overhaul", "cycle",
        ),
        features=(
            FeatureDef("total_downtime", Aggregation.SUM, "Total downtime accumulated."),
            FeatureDef("avg_repair_time", Aggregation.AVG, "Average repair duration."),
            FeatureDef("downtime_trend", Aggregation.TREND, "Downtime direction over time.", requires_time=True),
            FeatureDef("maintenance_events", Aggregation.COUNT, "Number of maintenance events."),
        ),
    ),
    Domain.INVENTORY: DomainSpec(
        domain=Domain.INVENTORY,
        description="Stock, materials, spare parts, supply levels, consumables.",
        aliases=(
            "inventory", "stock", "material", "spare", "parts", "supply",
            "consumable", "warehouse", "quantity_on_hand", "reorder",
        ),
        features=(
            FeatureDef("total_stock", Aggregation.SUM, "Total stock on hand."),
            FeatureDef("avg_stock_level", Aggregation.AVG, "Average stock level."),
            FeatureDef("stock_variance", Aggregation.VARIANCE, "Variability in stock levels."),
            FeatureDef("low_stock_min", Aggregation.MIN, "Lowest stock level observed."),
        ),
    ),
    Domain.WORKFORCE: DomainSpec(
        domain=Domain.WORKFORCE,
        description="Staff, shifts, crew, headcount, attendance, labor hours.",
        aliases=(
            "workforce", "staff", "employee", "crew", "headcount", "shift",
            "attendance", "labor", "labour", "operator", "personnel", "worker",
        ),
        features=(
            FeatureDef("headcount", Aggregation.DISTINCT_COUNT, "Distinct staff count."),
            FeatureDef("total_hours", Aggregation.SUM, "Total labor hours."),
            FeatureDef("avg_hours", Aggregation.AVG, "Average hours per record."),
            FeatureDef("attendance_rate", Aggregation.RATE, "Share of expected attendance met."),
        ),
    ),
    Domain.FINANCE: DomainSpec(
        domain=Domain.FINANCE,
        description="Spend, revenue, cost, billing, efficiency, margins.",
        aliases=(
            "finance", "revenue", "cost", "spend", "expense", "bill", "billing",
            "amount", "price", "margin", "budget", "fee", "payment", "invoice",
        ),
        features=(
            FeatureDef("total_revenue", Aggregation.SUM, "Total revenue or amount."),
            FeatureDef("total_cost", Aggregation.SUM, "Total cost or spend."),
            FeatureDef("avg_value", Aggregation.AVG, "Average monetary value per record."),
            FeatureDef("value_trend", Aggregation.TREND, "Monetary trend over time.", requires_time=True),
            FeatureDef("value_variance", Aggregation.VARIANCE, "Variability in monetary values."),
        ),
    ),
    Domain.CUSTOMERS: DomainSpec(
        domain=Domain.CUSTOMERS,
        description="Subscribers, students, clients, patients, accounts.",
        aliases=(
            "customer", "subscriber", "student", "client", "patient", "account",
            "member", "user", "consumer", "tenant", "buyer",
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

def get_spec(domain: Domain | str) -> DomainSpec:
    if isinstance(domain, str):
        domain = Domain(domain)
    return DOMAIN_REGISTRY[domain]


def features_for_domain(domain: Domain | str) -> tuple[FeatureDef, ...]:
    return get_spec(domain).features


def all_aliases() -> dict[str, Domain]:
    mapping: dict[str, Domain] = {}
    for spec in DOMAIN_REGISTRY.values():
        for alias in spec.aliases:
            mapping[alias.lower()] = spec.domain
    return mapping


def match_domain_by_keyword(column_name: str) -> Domain | None:
    name = column_name.lower()
    for alias, domain in all_aliases().items():
        if alias in name:
            return domain
    return None


def registry_as_prompt_context() -> str:
    lines: list[str] = []
    for spec in DOMAIN_REGISTRY.values():
        feats = ", ".join(spec.feature_names())
        lines.append(f"- {spec.domain.value}: {spec.description} Features: {feats}")
    return "\n".join(lines)