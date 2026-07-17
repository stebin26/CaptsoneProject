"""add last_login_ip to users

Revision ID: dde071ce0e13
Revises: 895e978f0c5a
Create Date: 2026-07-17 11:09:27.425490
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'dde071ce0e13'
down_revision: str | None = '895e978f0c5a'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Record the IP a session was last authenticated from. Nullable so it
    # applies cleanly to rows that predate the column.
    op.add_column(
        "users",
        sa.Column("last_login_ip", sa.String(length=45), nullable=True),
        schema="auth",
    )


def downgrade() -> None:
    op.drop_column("users", "last_login_ip", schema="auth")