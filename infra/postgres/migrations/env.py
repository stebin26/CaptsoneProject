"""Alembic environment.

Option A: schema is hand-written SQL applied at container start. Alembic does
NOT own the models -- target_metadata is None. It version-controls forward
changes on top of the existing schema. The DB URL comes from ops_common, the
same engine the app uses, so there is one source of truth.
"""

from __future__ import annotations

from logging.config import fileConfig

from ops_common.db import get_engine

from alembic import context

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Option A: no autogenerate target. We write migrations by hand.
target_metadata = None


def run_migrations_offline() -> None:
    """Emit SQL to stdout without a live DB (alembic upgrade --sql)."""
    engine = get_engine()
    context.configure(
        url=str(engine.url),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table_schema="public",
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against the live DB using the app's own engine."""
    engine = get_engine()
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table_schema="public",
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
