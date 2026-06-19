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

class Domain(str, enum.Enum):
    ASSETS = "assets"
    OPERATIONS = "operations"
    QUALITY = "quality"
    MAINTENANCE = "maintenance"
    INVENTORY = "inventory"
    WORKFORCE = "workforce"
    FINANCE = "finance"
    CUSTOMERS = "customers"

    @classmethod
    def values(cls) -> list[str]:
        return [d.value for d in cls]


class MappingStatus(str, enum.Enum):
    SUGGESTED = "suggested"
    CONFIRMED = "confirmed"
    SKIPPED = "skipped"
    ADDED_LATER = "added_later"


class FeatureStatus(str, enum.Enum):
    COLLECTED = "collected"
    SKIPPED = "skipped"
    ADDED_LATER = "added_later"


class Base(DeclarativeBase):
    pass


# ============================================================
# META schema — onboarding, profiling, mapping, features
# ============================================================

class Dataset(Base):
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

    columns: Mapped[list["ColumnProfile"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )
    features: Mapped[list["FeatureRecord"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )


class ColumnProfile(Base):
    __tablename__ = "column_profile"
    __table_args__ = (
        Index("idx_column_profile_dataset", "dataset_id"),
        {"schema": "meta"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("meta.dataset.id", ondelete="CASCADE"), nullable=False
    )
    column_name: Mapped[str] = mapped_column(Text, nullable=False)
    data_type: Mapped[str] = mapped_column(Text, nullable=False)
    sample_values: Mapped[dict | list | None] = mapped_column(JSONB, nullable=True)
    distinct_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    null_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    suggested_domain: Mapped[str | None] = mapped_column(Text, nullable=True)
    suggested_metric: Mapped[str | None] = mapped_column(Text, nullable=True)
    mapping_status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=MappingStatus.SUGGESTED.value
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    dataset: Mapped["Dataset"] = relationship(back_populates="columns")


class MappingConfig(Base):
    __tablename__ = "mapping_config"
    __table_args__ = {"schema": "meta"}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    business_name: Mapped[str] = mapped_column(Text, nullable=False)
    config: Mapped[dict] = mapped_column(JSONB, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class FeatureRecord(Base):
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

    dataset: Mapped["Dataset"] = relationship(back_populates="features")


# ============================================================
# HUB schema — eight universal domain tables (shared shape)
# ============================================================

class _HubMetricMixin:
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("meta.dataset.id", ondelete="CASCADE"), nullable=False
    )
    entity_ref: Mapped[str] = mapped_column(Text, nullable=False)
    metric_name: Mapped[str] = mapped_column(Text, nullable=False)
    metric_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    attributes: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    recorded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class Assets(_HubMetricMixin, Base):
    __tablename__ = "assets"
    __table_args__ = (Index("idx_assets_dataset", "dataset_id"), {"schema": "hub"})


class Operations(_HubMetricMixin, Base):
    __tablename__ = "operations"
    __table_args__ = (Index("idx_operations_dataset", "dataset_id"), {"schema": "hub"})


class Quality(_HubMetricMixin, Base):
    __tablename__ = "quality"
    __table_args__ = (Index("idx_quality_dataset", "dataset_id"), {"schema": "hub"})


class Maintenance(_HubMetricMixin, Base):
    __tablename__ = "maintenance"
    __table_args__ = (Index("idx_maintenance_dataset", "dataset_id"), {"schema": "hub"})


class Inventory(_HubMetricMixin, Base):
    __tablename__ = "inventory"
    __table_args__ = (Index("idx_inventory_dataset", "dataset_id"), {"schema": "hub"})


class Workforce(_HubMetricMixin, Base):
    __tablename__ = "workforce"
    __table_args__ = (Index("idx_workforce_dataset", "dataset_id"), {"schema": "hub"})


class Finance(_HubMetricMixin, Base):
    __tablename__ = "finance"
    __table_args__ = (Index("idx_finance_dataset", "dataset_id"), {"schema": "hub"})


class Customers(_HubMetricMixin, Base):
    __tablename__ = "customers"
    __table_args__ = (Index("idx_customers_dataset", "dataset_id"), {"schema": "hub"})


# ============================================================
# Domain → ORM model lookup (used by loaders to route writes)
# ============================================================

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


def model_for_domain(domain: str) -> type[_HubMetricMixin]:
    try:
        return DOMAIN_MODELS[domain]
    except KeyError as exc:
        raise ValueError(f"Unknown domain: {domain!r}") from exc