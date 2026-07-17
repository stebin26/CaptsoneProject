"""baseline existing schema

Revision ID: 895e978f0c5a
Revises: 
Create Date: 2026-07-17 11:06:47.932081
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '895e978f0c5a'
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Baseline anchor. The schema (auth, meta, domain, agent, analytics views)
    # is created by the hand-written SQL applied at container start, so there
    # is nothing to build here. This revision marks that starting point;
    # every future change is a new revision on top of it. See `alembic stamp`.
    pass


def downgrade() -> None:
    # No-op: we never tear the baseline schema down via Alembic.
    pass