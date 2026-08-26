"""Engine and session management for both SQLite and PostgreSQL.

The application code never branches on database vendor. Where behaviour genuinely
differs, spatial predicates, upsert syntax, the difference lives in
:mod:`parkfit.storage.dialects` behind a common interface.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
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
    # Default autocheckpoint is 1000 pages, about 4 MB. A national ingest leaves a WAL
    # far larger than that, and the *next* ordinary write inherits the checkpoint.
    # That is how a 10-row recommendation insert came to take 1.4 seconds, and why
    # search latency alternated between 140 ms and 4 s for no visible reason.
    # Checkpointing is moved to maintenance instead; see `checkpoint()`.
    cursor.execute("PRAGMA wal_autocheckpoint=8000")
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


def checkpoint(*, analyze: bool = False) -> dict[str, int | float]:
    """Fold the write-ahead log back into the database and reset it.

    Run after an ingest and at startup. Leaving a large WAL in place makes the next
    unlucky request pay for it: SQLite checkpoints on a commit once the log exceeds the
    autocheckpoint threshold, so a request that happens to cross it stalls for seconds
    while every other request is fast. Doing it deliberately, off the request path,
    turns an unpredictable multi-second stall into one predictable maintenance pause.
    """
    import time

    from sqlalchemy import text

    settings = get_settings()
    if settings.is_postgres:
        return {"skipped": 1}

    stats: dict[str, int | float] = {}
    engine = get_sync_engine()
    started = time.perf_counter()
    with engine.connect() as connection:
        connection.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))
        stats["checkpoint_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
        if analyze:
            started = time.perf_counter()
            connection.execute(text("ANALYZE"))
            connection.commit()
            stats["analyze_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
    return stats
