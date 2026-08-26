"""In-memory ledger of live recommendations, for anti-herding.

If the app can see one free kerb space and tells every driver who asks about it, it
manufactures its own congestion: five cars converge on one space, four waste the trip,
and all five learn not to trust the app. So before offering an exact space the ranking
counts how many other live recommendations already point at it and decays its survival
probability accordingly.

That count is kept in memory rather than queried from the database, for two reasons.

**Latency.** Recording recommendations synchronously made every search block on a write
it did not need, and nothing in the response depends on those rows being durable
before it returns. Bookkeeping does not belong on the request path.

**Accuracy.** The ledger sees a recommendation issued microseconds ago; a query against
committed rows does not. For a signal whose entire purpose is to notice near-simultaneous
requests, that difference is the whole point.

Rows are still persisted, in batches, off the request path, so the history survives a
restart and can be analysed later.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LedgerEntry:
    target_kind: str
    target_id: int
    search_id: str
    user_id: int | None
    rank: int
    generalised_cost: float
    probability_at_eta: float
    confidence_label: str
    fit_verdict: str
    created_at: float
    expires_at: float


class RecommendationLedger:
    """Thread-safe ring of live recommendations with time-based expiry."""

    #: Hard cap so a runaway process cannot grow the ledger without bound.
    MAX_ENTRIES = 20000

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: deque[LedgerEntry] = deque(maxlen=self.MAX_ENTRIES)
        self._pending: list[LedgerEntry] = []

    def record(self, entries: list[LedgerEntry]) -> None:
        if not entries:
            return
        with self._lock:
            self._entries.extend(entries)
            self._pending.extend(entries)

    def counts(self, keys: list[tuple[str, int]]) -> dict[tuple[str, int], int]:
        """How many live recommendations currently point at each target."""
        if not keys:
            return {}
        wanted = set(keys)
        now = time.time()
        counts: dict[tuple[str, int], int] = {}
        with self._lock:
            self._expire_locked(now)
            for entry in self._entries:
                key = (entry.target_kind, entry.target_id)
                if key in wanted:
                    counts[key] = counts.get(key, 0) + 1
        return counts

    def drain_pending(self) -> list[LedgerEntry]:
        """Take everything not yet persisted. Called by the background flusher."""
        with self._lock:
            pending, self._pending = self._pending, []
        return pending

    def _expire_locked(self, now: float) -> None:
        # Entries are appended in time order, so expiry is a prefix removal.
        while self._entries and self._entries[0].expires_at <= now:
            self._entries.popleft()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._entries)


_LEDGER: RecommendationLedger | None = None
_LEDGER_LOCK = threading.Lock()


def get_ledger() -> RecommendationLedger:
    global _LEDGER
    if _LEDGER is None:
        with _LEDGER_LOCK:
            if _LEDGER is None:
                _LEDGER = RecommendationLedger()
    return _LEDGER


def flush_ledger() -> int:
    """Persist buffered recommendations. Safe to call from a background task."""
    from datetime import UTC, datetime

    from parkfit.storage.models import Recommendation
    from parkfit.storage.session import session_scope

    pending = get_ledger().drain_pending()
    if not pending:
        return 0

    with session_scope() as session:
        for entry in pending:
            session.add(
                Recommendation(
                    search_id=entry.search_id,
                    user_id=entry.user_id,
                    target_kind=entry.target_kind,
                    target_id=entry.target_id,
                    rank=entry.rank,
                    generalised_cost=entry.generalised_cost,
                    probability_at_eta=entry.probability_at_eta,
                    confidence_label=entry.confidence_label,
                    fit_verdict=entry.fit_verdict,
                    created_at=datetime.fromtimestamp(entry.created_at, UTC).replace(tzinfo=None),
                    expires_at=datetime.fromtimestamp(entry.expires_at, UTC).replace(tzinfo=None),
                )
            )
    log.debug("persisted %d recommendations", len(pending))
    return len(pending)
