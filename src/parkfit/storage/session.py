"""Engine and session management for both SQLite and PostgreSQL.

The application code never branches on database vendor. Where behaviour genuinely
differs, spatial predicates, upsert syntax, the difference lives in
:mod:`parkfit.storage.dialects` behind a common interface.
"""

from __future__ import annotations

import logging
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

log = logging.getLogger(__name__)

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
    """Bring the database up to what the models declare.

    ``metadata.create_all`` creates missing *tables* and silently ignores missing
    *columns* on tables that already exist, which is the failure mode that matters here:
    adding a field to a model leaves every existing database broken with "no such column"
    at the first query, and the error names the column rather than the cause.

    This used to say migrations were the path for a database with data in it. They were
    not: ``database/migrations/versions`` is empty and there is no Alembic configuration
    anywhere in the tree, so that sentence described a plan rather than a mechanism.
    :func:`sync_schema` is the mechanism, and its limits are stated on it.
    """
    Base.metadata.create_all(get_sync_engine())
    added = sync_schema()
    if added:
        log.info("schema sync added %d column(s): %s", len(added), ", ".join(added))


def sync_schema() -> list[str]:
    """Add columns the models declare and the database lacks. Returns what it added.

    **This is not a migration system, and it is not trying to be one.** It only ever adds
    a column, never renames, drops, retypes or backfills one, because those need to know
    intent and this only knows the difference between two schemas. A column that cannot
    be added safely, meaning one that is NOT NULL with no server default, is reported and
    skipped rather than guessed at: filling it would mean inventing values for every
    existing row.

    What it buys is that adding a field to a model does not brick every developer's
    database. What it does not buy is safety on a production database with real data in
    it, which still wants a considered migration.
    """
    from sqlalchemy import inspect, text

    engine = get_sync_engine()
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    added: list[str] = []

    with engine.begin() as connection:
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue  # create_all just made it, so it is already current
            present = {column["name"] for column in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in present:
                    continue

                default_sql = _default_sql(column)
                if not column.nullable and default_sql is None:
                    log.warning(
                        "cannot add %s.%s automatically: it is NOT NULL with no default, "
                        "so existing rows have no value to take",
                        table.name,
                        column.name,
                    )
                    continue

                type_sql = column.type.compile(engine.dialect)
                clause = f"ALTER TABLE {table.name} ADD COLUMN {column.name} {type_sql}"
                if default_sql is not None:
                    clause += f" DEFAULT {default_sql}"
                    # SQLite requires a default on a NOT NULL added column, and having
                    # one makes the constraint safe to carry over on every dialect.
                    if not column.nullable:
                        clause += " NOT NULL"
                connection.execute(text(clause))
                added.append(f"{table.name}.{column.name}")

    return added


def _default_sql(column) -> str | None:
    """A literal for the column's default, or None when there is nothing to use.

    Only scalar defaults are rendered. A callable default (``default=utcnow``) is a
    Python-side value that SQL cannot evaluate, so a nullable column simply gets NULL for
    existing rows, which is the honest answer: those rows genuinely have no value.
    """
    from sqlalchemy import String

    default = column.default
    if default is None or getattr(default, "is_callable", False):
        return "NULL" if column.nullable else None
    value = getattr(default, "arg", None)
    if callable(value):
        return "NULL" if column.nullable else None
    if value is None:
        return "NULL" if column.nullable else None
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(column.type, String) or isinstance(value, str):
        escaped = str(value).replace("'", "''")
        return f"'{escaped}'"
    return None


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
