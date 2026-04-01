from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from lexdrift.config import settings


def _build_engine() -> AsyncEngine:
    """Create the async engine with appropriate pool settings.

    SQLite does not support connection pool parameters (pool_size, max_overflow),
    so those are only applied for non-SQLite backends (PostgreSQL, etc.).
    """
    is_sqlite = settings.database_url.startswith("sqlite")

    kwargs: dict = {
        "echo": False,
        "future": True,
        "pool_pre_ping": True,
    }

    if is_sqlite:
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs["pool_size"] = 5
        kwargs["max_overflow"] = 10
        kwargs["pool_recycle"] = 3600

    return create_async_engine(settings.database_url, **kwargs)


engine: AsyncEngine = _build_engine()
