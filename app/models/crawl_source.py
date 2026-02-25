"""CrawlSource — user-configured web pages/RSS feeds for the web crawler."""
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, String, Text, text
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class CrawlSource(Base):
    """A web page or RSS feed the crawler should poll."""

    __tablename__ = "crawl_sources"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    label: Mapped[str] = mapped_column(String(255))
    url: Mapped[str] = mapped_column(Text, unique=True)
    # Which IntuOne layer this source feeds
    layer: Mapped[str] = mapped_column(String(50), default="news")
    topic_tags: Mapped[list | None] = mapped_column(ARRAY(String), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )
