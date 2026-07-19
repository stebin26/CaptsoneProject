"""ORM models for the universal domain hub.

Defines the eight industry-agnostic business domains, the metadata tables that
record an onboarding (dataset, column profile, mapping config, feature record),
and the eight hub tables that store the data itself. Every hub table shares one
identical column shape -- that sameness is what makes the platform
industry-agnostic: a new industry needs a new mapping, never a new table.
"""
from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# ============================================================
# Universal domains — the stable backbone (industry-agnostic)
# ============================================================

# The 8 fixed business domains every industry maps onto. This enum is the
# single source of truth — change nothing here and the whole hub stays stable.


class Domain(str, enum.Enum): # noqa: UP042
    """The eight universal business domains every industry maps onto.

    This enum is the single source of truth for domain names; keeping it fixed is
    what keeps the hub stable across industries.
    """
    ASSETS = "assets"
    OPERATIONS = "operations"
    QUALITY = "quality"
    MAINTENANCE = "maintenance"
    INVENTORY = "inventory"
    WORKFORCE = "workforce"
    FINANCE = "finance"
    CUSTOMERS = "customers"

    # Helper to get all domain names as a plain list (used in loops/validation).
    @classmethod
    def values(cls) -> list[str]:
        """Return every domain name as a plain list.

        Returns:
            The domain values, for use in loops and validation.
        """
        return [d.value for d in cls]


# Lifecycle of a column's mapping: suggested → confirmed/skipped, or added later.


class MappingStatus(str, enum.Enum): # noqa: UP042
    """Lifecycle of a column's mapping decision during onboarding."""
    SUGGESTED = "suggested"
    CONFIRMED = "confirmed"
    SKIPPED = "skipped"
    ADDED_LATER = "added_later"


# Lifecycle of a feature: was it collected, skipped, or added after onboarding.
class FeatureStatus(str, enum.Enum): # noqa: UP042
    """Lifecycle of a feature: collected, skipped, or added after onboarding."""
    COLLECTED = "collected"
    SKIPPED = "skipped"
    ADDED_LATER = "added_later"


# The parent class all ORM tables inherit from (SQLAlchemy 2.0 style).
class Base(DeclarativeBase):
    """Declarative base class every ORM table inherits from."""
    pass


# ============================================================
# META schema — onboarding, profiling, mapping, features
# ============================================================

# One row per uploaded dataset — the master record of an onboarding.
# Holds business name, source file, row count, upload time.


class Dataset(Base):
    """One uploaded dataset -- the master record of an onboarding.

    Owns its column profiles and feature records; deleting a dataset cascades to
    both.
    """
    __tablename__ = "dataset"
    __table_args__ = {"schema": "meta"}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    business_name: Mapped[str] = mapped_column(Text, nullable=False)
    industry: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_filename: Mapped[str] = mapped_column(Text, nullable=False)
    row_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # A dataset owns many column profiles and many feature records.
    # cascade="all, delete-orphan" → delete the dataset, its children go too.

    columns: Mapped[list[ColumnProfile]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )
    features: Mapped[list[FeatureRecord]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )


# One row per column in the uploaded CSV — the profiling result.
# Stores type, samples, null/distinct counts, and the suggested mapping.


class ColumnProfile(Base):
    """Profiling result and mapping decision for one source column.

    Records the column's type, samples, null and distinct counts, what the
    suggester proposed, and whether the user confirmed or skipped it.
    """
    __tablename__ = "column_profile"
    # Index on dataset_id so "get all columns for this dataset" is fast.
    __table_args__ = (
        Index("idx_column_profile_dataset", "dataset_id"),
        {"schema": "meta"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # FK back to the parent dataset; CASCADE delete with it.
    dataset_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("meta.dataset.id", ondelete="CASCADE"), nullable=False
    )
    column_name: Mapped[str] = mapped_column(Text, nullable=False)
    data_type: Mapped[str] = mapped_column(Text, nullable=False)
    # Sample values stored as JSONB so we can keep a small list/dict inline.
    sample_values: Mapped[dict | list | None] = mapped_column(JSONB, nullable=True)
    distinct_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    null_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # What the suggester proposed for this column (domain + metric).
    suggested_domain: Mapped[str | None] = mapped_column(Text, nullable=True)
    suggested_metric: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Tracks whether the user confirmed/skipped this column's mapping.
    mapping_status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=MappingStatus.SUGGESTED.value
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    dataset: Mapped[Dataset] = relationship(back_populates="columns")


# Saved per-business mapping config (the reusable "declaration, not code").
# JSONB config + version so the same business re-onboards without redoing work.


class MappingConfig(Base):
    """A saved, versioned per-business mapping configuration.

    This is the 'declaration, not code' artifact that lets the same business
    re-onboard without redoing the mapping work.
    """
    __tablename__ = "mapping_config"
    __table_args__ = {"schema": "meta"}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    business_name: Mapped[str] = mapped_column(Text, nullable=False)
    config: Mapped[dict] = mapped_column(JSONB, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# One row per feature — records what was collected/skipped per domain.
# This is what the dashboard's "collected / skipped " view reads from.


class FeatureRecord(Base):
    """One feature and whether it was collected, skipped, or added later.

    This is what the dashboard's collected-versus-skipped review reads from.
    """
    __tablename__ = "feature_record"
    __table_args__ = (
        Index("idx_feature_record_dataset", "dataset_id"),
        {"schema": "meta"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("meta.dataset.id", ondelete="CASCADE"), nullable=False
    )
    domain: Mapped[str] = mapped_column(Text, nullable=False)
    feature_name: Mapped[str] = mapped_column(Text, nullable=False)
    source_column: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=FeatureStatus.COLLECTED.value
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    dataset: Mapped[Dataset] = relationship(back_populates="features")


# ============================================================
# HUB schema — eight universal domain tables (shared shape)
# ============================================================

# The shared column shape for ALL 8 hub tables. Defined once here so every
# domain table is identical — this sameness is what makes the hub industry-agnostic.
# entity_ref = which thing (machine/tower/student), metric_name+value = the measurement,
# attributes = extra JSON, recorded_at = when (drives time-series charts).


class _HubMetricMixin:
    """The shared column shape for all eight hub tables.

    Defined once so every domain table is identical: ``entity_ref`` is which thing
    was measured, ``metric_name`` and ``metric_value`` are the measurement,
    ``attributes`` carries extra JSON, and ``recorded_at`` drives the time series.
    """
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("meta.dataset.id", ondelete="CASCADE"), nullable=False
    )
    entity_ref: Mapped[str] = mapped_column(Text, nullable=False)
    metric_name: Mapped[str] = mapped_column(Text, nullable=False)
    metric_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    attributes: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    recorded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


# The 8 domain tables. Each just inherits the mixin's shape + its own
# name, schema, and dataset_id index. No per-domain custom columns — by design.


class Assets(_HubMetricMixin, Base):
    """Hub table for the Assets domain -- the things being operated."""
    __tablename__ = "assets"
    __table_args__ = (Index("idx_assets_dataset", "dataset_id"), {"schema": "hub"})


class Operations(_HubMetricMixin, Base):
    """Hub table for the Operations domain -- the core work performed."""
    __tablename__ = "operations"
    __table_args__ = (Index("idx_operations_dataset", "dataset_id"), {"schema": "hub"})


class Quality(_HubMetricMixin, Base):
    """Hub table for the Quality domain -- defects, failures, and breaches."""
    __tablename__ = "quality"
    __table_args__ = (Index("idx_quality_dataset", "dataset_id"), {"schema": "hub"})


class Maintenance(_HubMetricMixin, Base):
    """Hub table for the Maintenance domain -- repairs and downtime."""
    __tablename__ = "maintenance"
    __table_args__ = (Index("idx_maintenance_dataset", "dataset_id"), {"schema": "hub"})


class Inventory(_HubMetricMixin, Base):
    """Hub table for the Inventory domain -- stock and materials."""
    __tablename__ = "inventory"
    __table_args__ = (Index("idx_inventory_dataset", "dataset_id"), {"schema": "hub"})


class Workforce(_HubMetricMixin, Base):
    """Hub table for the Workforce domain -- people, shifts, and hours."""
    __tablename__ = "workforce"
    __table_args__ = (Index("idx_workforce_dataset", "dataset_id"), {"schema": "hub"})


class Finance(_HubMetricMixin, Base):
    """Hub table for the Finance domain -- revenue, cost, and spend."""
    __tablename__ = "finance"
    __table_args__ = (Index("idx_finance_dataset", "dataset_id"), {"schema": "hub"})


class Customers(_HubMetricMixin, Base):
    """Hub table for the Customers domain -- whoever the operation serves."""
    __tablename__ = "customers"
    __table_args__ = (Index("idx_customers_dataset", "dataset_id"), {"schema": "hub"})


# ============================================================
# Domain → ORM model lookup (used by loaders to route writes)
# ============================================================

# Maps a domain string → its table class. The loader uses this to decide
# which of the 8 tables a row should be written to. This is the routing table.

DOMAIN_MODELS: dict[str, type[_HubMetricMixin]] = {
    Domain.ASSETS.value: Assets,
    Domain.OPERATIONS.value: Operations,
    Domain.QUALITY.value: Quality,
    Domain.MAINTENANCE.value: Maintenance,
    Domain.INVENTORY.value: Inventory,
    Domain.WORKFORCE.value: Workforce,
    Domain.FINANCE.value: Finance,
    Domain.CUSTOMERS.value: Customers,
}

# Safe lookup: given a domain string, return its table class — or raise a
# clear error if the domain is unknown (guards against bad mapping input).


def model_for_domain(domain: str) -> type[_HubMetricMixin]:
    """Return the hub table class for a domain name.

    Used by the loader to route each row to one of the eight tables.

    Args:
        domain: The domain name to resolve.

    Returns:
        The ORM class backing that domain.

    Raises:
        ValueError: If the domain name is not one of the eight.
    """
    try:
        return DOMAIN_MODELS[domain]
    except KeyError as exc:
        raise ValueError(f"Unknown domain: {domain!r}") from exc
