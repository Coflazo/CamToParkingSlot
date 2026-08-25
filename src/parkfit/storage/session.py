"""Engine and session management for both SQLite and PostgreSQL.

The application code never branches on database vendor. Where behaviour genuinely
differs -- spatial predicates, upsert syntax -- the difference lives in
:mod:`parkfit.storage.dialects` behind a common interface.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from parkfit.config import get_settings
from parkfit.storage.models import Base

_async_engine: AsyncEngine | None = None
_sync_engine: Engine | None = None


def _tune_sqlite(dbapi_connection, _connection_record) -> None:
    """Apply the pragmas that make SQLite behave sensibly for this workload.

    WAL lets the ingest workers write while the API reads, which matters because a full
    national refresh takes minutes and must not block search. The busy timeout turns
    lock contention into a short wait instead of an immediate ``database is locked``.
    """

    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA busy_timeout=10000")
    cursor.execute("PRAGMA cache_size=-64000")  # 64 MB page cache
    cursor.close()


def get_sync_engine() -> Engine:
    global _sync_engine
    if _sync_engine is None:
        settings = get_settings()
        url = settings.sync_database_url
        kwargs: dict = {"future": True, "echo": False}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
        else:
            kwargs.update(pool_size=10, max_overflow=20, pool_pre_ping=True)
        _sync_engine = create_engine(url, **kwargs)
        if url.startswith("sqlite"):
            event.listen(_sync_engine, "connect", _tune_sqlite)
    return _sync_engine


def get_async_engine() -> AsyncEngine:
    global _async_engine
    if _async_engine is None:
        settings = get_settings()
        url = settings.resolved_database_url
        kwargs: dict = {"future": True, "echo": False}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
        else:
            kwargs.update(pool_size=10, max_overflow=20, pool_pre_ping=True)
        _async_engine = create_async_engine(url, **kwargs)
        if url.startswith("sqlite"):
            event.listen(_async_engine.sync_engine, "connect", _tune_sqlite)
    return _async_engine


SyncSessionFactory = sessionmaker(bind=None, class_=Session, expire_on_commit=False)
AsyncSessionFactory = async_sessionmaker(bind=None, class_=AsyncSession, expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    """A synchronous transactional scope, for ingest workers and the CLI."""
    session = Session(bind=get_sync_engine(), expire_on_commit=False)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@asynccontextmanager
async def async_session_scope() -> AsyncIterator[AsyncSession]:
    session = AsyncSession(bind=get_async_engine(), expire_on_commit=False)
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency."""
    async with async_session_scope() as session:
        yield session


def create_all() -> None:
    """Create the schema directly, without Alembic.

    Used by tests and by first-run bootstrap. Migrations remain the path for an
    existing database with data in it.
    """
    Base.metadata.create_all(get_sync_engine())


def drop_all() -> None:
    Base.metadata.drop_all(get_sync_engine())


def reset_engines() -> None:
    """Dispose cached engines. Tests call this when they repoint the database URL."""
    global _async_engine, _sync_engine
    if _sync_engine is not None:
        _sync_engine.dispose()
        _sync_engine = None
    _async_engine = None
