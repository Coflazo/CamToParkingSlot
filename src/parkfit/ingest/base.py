"""Shared machinery for upstream data adapters.

Every adapter gets the same three guarantees:

* **Identified, retried HTTP.** Overpass rejects anonymous clients outright and Socrata
  throttles them, so a User-Agent is mandatory rather than polite. Retries use
  exponential backoff and only on transient status codes -- retrying a 400 just
  hammers a server with the same broken request.
* **On-disk response caching.** A national RDW refresh is ~130k rows across eight
  datasets. Re-fetching that on every run during development is slow and rude; the
  cache makes iteration cheap and makes the test suite reproducible offline.
* **Recorded provenance.** Nothing enters the database without its source name, fetch
  time and licence. When a recommendation turns out to be wrong, the first question is
  always "where did that come from", and this is the answer.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from parkfit.config import Settings, get_settings

log = logging.getLogger(__name__)

# Only these are worth retrying. A 4xx other than 429 means the request itself is wrong.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


@dataclass
class SourceMeta:
    """Licence and attribution facts for one upstream dataset."""

    name: str
    url: str
    licence: str
    licence_url: str | None = None
    attribution: str | None = None
    commercial_use: bool | None = None
    share_alike: bool = False
    refresh: str | None = None
    contact: str | None = None
    notes: str | None = None


@dataclass
class IngestResult:
    """What one adapter run actually did."""

    source: str
    fetched: int = 0
    created: int = 0
    updated: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None

    @property
    def duration_s(self) -> float:
        end = self.finished_at or datetime.now(UTC)
        return (end - self.started_at).total_seconds()

    def summary(self) -> str:
        parts = [
            f"{self.source}: fetched={self.fetched}",
            f"created={self.created}",
            f"updated={self.updated}",
        ]
        if self.skipped:
            parts.append(f"skipped={self.skipped}")
        if self.errors:
            parts.append(f"errors={len(self.errors)}")
        parts.append(f"in {self.duration_s:.1f}s")
        return " ".join(parts)


class HttpCache:
    """Content-addressed cache of upstream responses."""

    def __init__(self, directory: Path, ttl: timedelta = timedelta(hours=12)):
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        return self.dir / f"{digest}.json"

    def get(self, key: str) -> Any | None:
        p = self._path(key)
        if not p.exists():
            return None
        age = time.time() - p.stat().st_mtime
        if age > self.ttl.total_seconds():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A truncated cache entry is not worth diagnosing; drop it and re-fetch.
            p.unlink(missing_ok=True)
            return None

    def set(self, key: str, value: Any) -> None:
        try:
            self._path(key).write_text(json.dumps(value), encoding="utf-8")
        except (OSError, TypeError) as exc:
            log.debug("cache write failed for %s: %s", key[:60], exc)

    def clear(self) -> int:
        n = 0
        for p in self.dir.glob("*.json"):
            p.unlink(missing_ok=True)
            n += 1
        return n


class BaseAdapter(ABC):
    """Base class for every upstream source adapter."""

    #: Licence and attribution facts, registered into ``source_licences`` on first run.
    meta: SourceMeta

    def __init__(self, settings: Settings | None = None, use_cache: bool = True):
        self.settings = settings or get_settings()
        self.use_cache = use_cache
        self.cache = HttpCache(self.settings.cache_dir(self.meta.name.lower().replace(" ", "_")))
        self._client: httpx.Client | None = None

    # -- HTTP ---------------------------------------------------------------
    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.settings.http_timeout_s,
                headers={"User-Agent": self.settings.user_agent},
                follow_redirects=True,
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def fetch_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        *,
        cache_key: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        # Normalise an empty mapping to None. httpx replaces a URL query string with
        # whatever `params` holds, so passing {} strips any cursor already in the URL.
        params = params or None
        key = cache_key or f"{url}?{json.dumps(params or {}, sort_keys=True)}"
        if self.use_cache:
            hit = self.cache.get(key)
            if hit is not None:
                return hit

        payload = self._request_with_retry(url, params, headers)
        if self.use_cache:
            self.cache.set(key, payload)
        return payload

    def fetch_text(
        self, url: str, params: dict[str, Any] | None = None, *, headers: dict[str, str] | None = None
    ) -> str:
        last: Exception | None = None
        for attempt in range(self.settings.http_max_retries):
            try:
                r = self.client.get(url, params=params, headers=headers)
                if r.status_code in RETRYABLE_STATUS:
                    raise httpx.HTTPStatusError(
                        f"retryable {r.status_code}", request=r.request, response=r
                    )
                r.raise_for_status()
                return r.text
            except (httpx.HTTPError, httpx.TimeoutException) as exc:
                last = exc
                if not self._should_retry(exc):
                    raise
                self._backoff(attempt)
        raise RuntimeError(f"{self.meta.name}: GET {url} failed") from last

    def _request_with_retry(
        self, url: str, params: dict[str, Any] | None, headers: dict[str, str] | None
    ) -> Any:
        last: Exception | None = None
        for attempt in range(self.settings.http_max_retries):
            try:
                r = self.client.get(url, params=params, headers=headers)
                if r.status_code in RETRYABLE_STATUS:
                    raise httpx.HTTPStatusError(
                        f"retryable {r.status_code}", request=r.request, response=r
                    )
                r.raise_for_status()
                return r.json()
            except (httpx.HTTPError, httpx.TimeoutException, json.JSONDecodeError) as exc:
                last = exc
                if not self._should_retry(exc):
                    raise
                self._backoff(attempt)
        raise RuntimeError(f"{self.meta.name}: GET {url} failed after retries") from last

    @staticmethod
    def _should_retry(exc: Exception) -> bool:
        if isinstance(exc, httpx.TimeoutException | httpx.TransportError):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in RETRYABLE_STATUS
        return False

    def _backoff(self, attempt: int) -> None:
        delay = min(30.0, 1.5 * (2**attempt))
        log.debug("%s: retrying in %.1fs", self.meta.name, delay)
        time.sleep(delay)

    # -- contract -----------------------------------------------------------
    @abstractmethod
    def run(self, **kwargs: Any) -> IngestResult:
        """Fetch from upstream and write to the database."""


class SocrataAdapter(BaseAdapter):
    """Adapter for a Socrata-hosted dataset, as used by RDW.

    Socrata caps a single response at 50 000 rows, so anything larger has to be paged.
    The TIJDVAK dataset alone has 93 407 rows, which makes paging mandatory rather than
    defensive.
    """

    PAGE_SIZE = 50000

    def socrata_rows(
        self,
        dataset_id: str,
        *,
        select: str | None = None,
        where: str | None = None,
        order: str | None = None,
        limit: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        url = f"{self.settings.rdw_base_url}/{dataset_id}.json"
        headers: dict[str, str] = {}
        if self.settings.rdw_app_token:
            headers["X-App-Token"] = self.settings.rdw_app_token

        offset = 0
        yielded = 0
        while True:
            page_size = self.PAGE_SIZE
            if limit is not None:
                page_size = min(page_size, limit - yielded)
                if page_size <= 0:
                    return

            params: dict[str, Any] = {"$limit": page_size, "$offset": offset}
            if select:
                params["$select"] = select
            if where:
                params["$where"] = where
            # A stable sort is required for correct paging: without it Socrata may
            # return rows in a different order per page and silently drop or repeat
            # records across the boundary.
            params["$order"] = order or ":id"

            rows = self.fetch_json(url, params, headers=headers)
            if not rows:
                return
            for row in rows:
                yield row
                yielded += 1
            if len(rows) < page_size:
                return
            offset += len(rows)

    def socrata_count(self, dataset_id: str, where: str | None = None) -> int:
        url = f"{self.settings.rdw_base_url}/{dataset_id}.json"
        params: dict[str, Any] = {"$select": "count(*)"}
        if where:
            params["$where"] = where
        rows = self.fetch_json(url, params)
        if not rows:
            return 0
        first = rows[0]
        for key in ("count", "count_1", "count_id"):
            if key in first:
                return int(first[key])
        return int(next(iter(first.values())))


# ---------------------------------------------------------------------------
# Small parsing helpers shared by the Dutch sources
# ---------------------------------------------------------------------------
def parse_rdw_datetime(value: str | None) -> datetime | None:
    """RDW mixes ``YYYYMMDD``, ``YYYYMMDDHHMMSS`` and ISO dates in the same feeds."""
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y%m%d%H%M%S", "%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None


def parse_int(value: Any) -> int | None:
    f = parse_float(value)
    return int(f) if f is not None else None


def parse_bool(value: Any) -> bool | None:
    """RDW encodes booleans as J/N, Y/N, 1/0 and true/false depending on the dataset."""
    if value is None or value == "":
        return None
    text = str(value).strip().lower()
    if text in {"j", "y", "ja", "yes", "true", "1"}:
        return True
    if text in {"n", "nee", "no", "false", "0"}:
        return False
    return None


#: RDW day names, mapped to Python weekday numbers (Monday = 0).
DUTCH_WEEKDAYS = {
    "maandag": 0,
    "dinsdag": 1,
    "woensdag": 2,
    "donderdag": 3,
    "vrijdag": 4,
    "zaterdag": 5,
    "zondag": 6,
}


def parse_dutch_weekday(value: str | None) -> int | None:
    if not value:
        return None
    return DUTCH_WEEKDAYS.get(str(value).strip().lower())


def parse_hhmm(value: Any) -> int | None:
    """Convert RDW's ``HHMM``-as-integer times to minutes past midnight.

    ``2400`` is a legitimate value meaning end-of-day, and ``500`` means 05:00 rather
    than 500 minutes, so this cannot be a plain integer division.
    """
    n = parse_int(value)
    if n is None:
        return None
    hours, minutes = divmod(n, 100)
    return hours * 60 + minutes
