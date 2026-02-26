"""add domain column to signals and crawl_sources

Revision ID: b9e1f2a3c4d5
Revises: a1b2c3d4e5f6
Create Date: 2026-02-26 00:00:00.000000

Adds a free-form `domain` column to both tables so new world layers
(e.g. "market", "news", "social", "crypto", "politics") can be added
without any further migrations or code changes.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b9e1f2a3c4d5"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # signals — nullable, indexed for filtering by domain
    op.add_column(
        "signals",
        sa.Column("domain", sa.String(100), nullable=True),
    )
    op.create_index("ix_signals_domain", "signals", ["domain"])

    # crawl_sources — nullable, no index needed (low cardinality, small table)
    op.add_column(
        "crawl_sources",
        sa.Column("domain", sa.String(100), nullable=True),
    )


def downgrade() -> None:
    op.drop_index("ix_signals_domain", table_name="signals")
    op.drop_column("signals", "domain")
    op.drop_column("crawl_sources", "domain")
