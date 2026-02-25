from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

# ── Engine (supports PostgreSQL and SQLite) ───────────────────────────────────
_url = settings.DATABASE_URL

# SQLite: swap the scheme so SQLAlchemy uses aiosqlite driver
if _url.startswith("sqlite://"):
    _url = _url.replace("sqlite://", "sqlite+aiosqlite://", 1)
elif _url.startswith("sqlite+aiosqlite://"):
    pass  # already correct

_kwargs: dict = {"echo": settings.DEBUG}
# connect_args only needed for SQLite (prevents thread-sharing warnings)
if "sqlite" in _url:
    _kwargs["connect_args"] = {"check_same_thread": False}

engine = create_async_engine(_url, **_kwargs)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session
