"""In-memory spatial index over parking supply.

Radius search is deliberately *not* a database query. A ``lat BETWEEN ... AND lon
BETWEEN ...`` predicate can only use the leading column of a composite index for a range
scan, so SQLite narrows on latitude and then filters roughly 30,000 rows by longitude.
With a warm page cache that costs 200 ms; with a cold one -- which is every fresh
connection, and the API opens one per request -- it costs **four seconds**.

The C++ grid answers the same question over 200,000 bays in microseconds and hands back
ids, which turns the whole thing into a few hundred primary-key lookups. The index is
built once per process and refreshed on demand, because parking supply changes on the
timescale of an ingest run, not a request.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from parkfit.native import native
from parkfit.storage.models import ParkingBay, ParkingFacility

log = logging.getLogger(__name__)

#: Rebuild after this long. Supply changes when an ingest runs, not between requests.
DEFAULT_TTL_S = 900.0


@dataclass(frozen=True)
class IndexedTarget:
    target_id: int
    lat: float
    lon: float
    distance_m: float


class _PythonGrid:
    """Fallback grid for a checkout that has not been compiled.

    Same cell-bucketing idea as the C++ version, just slower. It exists so the product
    still runs without a build step, not to be fast.
    """

    def __init__(self, cell_size_m: float = 250.0):
        self.lat_step = math.degrees(cell_size_m / 6371008.8)
        self.lon_step = self.lat_step / max(0.05, math.cos(math.radians(52.1)))
        self.cells: dict[tuple[int, int], list[tuple[float, float, int]]] = {}

    def insert_many(self, items) -> None:
        for lat, lon, payload in items:
            key = (int(lon // self.lon_step), int(lat // self.lat_step))
            self.cells.setdefault(key, []).append((lat, lon, payload))

    def build(self) -> None:
        return None

    def __len__(self) -> int:
        return sum(len(v) for v in self.cells.values())

    def query_radius(self, lat: float, lon: float, radius_m: float, _max: int = 0):
        from parkfit.geo.rd import haversine_m

        span_lat = int(math.ceil(math.degrees(radius_m / 6371008.8) / self.lat_step))
        span_lon = int(
            math.ceil(
                (math.degrees(radius_m / 6371008.8) / max(0.05, math.cos(math.radians(lat))))
                / self.lon_step
            )
        )
        cx, cy = int(lon // self.lon_step), int(lat // self.lat_step)
        hits = []
        for dx in range(-span_lon, span_lon + 1):
            for dy in range(-span_lat, span_lat + 1):
                for plat, plon, payload in self.cells.get((cx + dx, cy + dy), ()):
                    d = haversine_m(lat, lon, plat, plon)
                    if d <= radius_m:
                        hits.append(_Hit(payload, d))
        hits.sort(key=lambda h: h.distance_m)
        return hits


@dataclass(frozen=True)
class _Hit:
    payload: int
    distance_m: float


class CandidateIndex:
    """Spatial index over facilities and bays, shared across requests."""

    def __init__(self, ttl_s: float = DEFAULT_TTL_S):
        self.ttl_s = ttl_s
        self._lock = threading.Lock()
        self._built_at = 0.0
        self._facility_grid = None
        self._bay_grid = None
        self._facility_ids: list[int] = []
        self._bay_ids: list[int] = []

    # -- building -----------------------------------------------------------
    def ensure_built(self, session: Session, *, force: bool = False) -> None:
        with self._lock:
            fresh = (time.monotonic() - self._built_at) < self.ttl_s
            if not force and self._bay_grid is not None and fresh:
                return
            self._build(session)

    def _build(self, session: Session) -> None:
        started = time.perf_counter()

        facility_rows = session.execute(
            select(ParkingFacility.id, ParkingFacility.lat, ParkingFacility.lon).where(
                ParkingFacility.active.is_(True),
                ParkingFacility.lat.is_not(None),
                ParkingFacility.lon.is_not(None),
            )
        ).all()
        bay_rows = session.execute(select(ParkingBay.id, ParkingBay.lat, ParkingBay.lon)).all()

        self._facility_ids = [row[0] for row in facility_rows]
        self._bay_ids = [row[0] for row in bay_rows]

        make = (lambda size: native.SpatialGrid(size)) if native is not None else _PythonGrid
        self._facility_grid = make(400.0)
        # Bays are dense: a smaller cell keeps the per-cell list short, which is what
        # makes the sweep cheap.
        self._bay_grid = make(150.0)

        self._facility_grid.insert_many(
            [(row[1], row[2], i) for i, row in enumerate(facility_rows)]
        )
        self._bay_grid.insert_many([(row[1], row[2], i) for i, row in enumerate(bay_rows)])
        self._facility_grid.build()
        self._bay_grid.build()

        self._built_at = time.monotonic()
        log.info(
            "candidate index built: %d facilities, %d bays in %.0f ms (%s)",
            len(self._facility_ids),
            len(self._bay_ids),
            (time.perf_counter() - started) * 1000.0,
            "native" if native is not None else "python fallback",
        )

    def invalidate(self) -> None:
        """Force a rebuild on next use. Call after an ingest run."""
        with self._lock:
            self._built_at = 0.0

    # -- querying -----------------------------------------------------------
    def facilities_within(
        self, lat: float, lon: float, radius_m: float, limit: int
    ) -> list[IndexedTarget]:
        return self._query(self._facility_grid, self._facility_ids, lat, lon, radius_m, limit)

    def bays_within(
        self, lat: float, lon: float, radius_m: float, limit: int
    ) -> list[IndexedTarget]:
        return self._query(self._bay_grid, self._bay_ids, lat, lon, radius_m, limit)

    @staticmethod
    def _query(grid, ids: list[int], lat: float, lon: float, radius_m: float, limit: int):
        if grid is None or not ids:
            return []
        hits = grid.query_radius(lat, lon, radius_m, limit)
        return [
            IndexedTarget(target_id=ids[hit.payload], lat=lat, lon=lon, distance_m=hit.distance_m)
            for hit in hits
        ]

    @property
    def size(self) -> tuple[int, int]:
        return len(self._facility_ids), len(self._bay_ids)


_INDEX: CandidateIndex | None = None
_INDEX_LOCK = threading.Lock()


def get_candidate_index() -> CandidateIndex:
    """The process-wide index. Built lazily on first use."""
    global _INDEX
    if _INDEX is None:
        with _INDEX_LOCK:
            if _INDEX is None:
                _INDEX = CandidateIndex()
    return _INDEX
