"""Global geocoding, for the destinations PDOK has never heard of.

PDOK is authoritative for the Netherlands and knows nothing outside it. Asking it for
"Beşiktaş" returns nothing, which is correct behaviour for a Dutch address register and
useless for a product that now has 248 live sites in Istanbul. Nominatim is the OSM
geocoder: global, free, and the natural counterpart to a product whose road graph and
legal anchors already come from OSM.

It is deliberately the **last** tier. PDOK indexes the BAG register with rooftop
coordinates and stays the better answer for any Dutch address; Nominatim is what answers
everywhere PDOK does not reach.

**Its usage policy is a constraint, not a suggestion.** The public instance asks for at
most one request per second, a User-Agent identifying the application, and no bulk
geocoding. All three are honoured here: requests are serialised behind a lock with a real
sleep between them, the User-Agent names the project, and this is only ever reached for
one interactive destination at a time. A product that ignored that would deserve to be
blocked, and would be.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from parkfit.ingest.base import BaseAdapter, IngestResult, SourceMeta
from parkfit.ingest.pdok import GeocodeHit

log = logging.getLogger(__name__)

BASE_URL = "https://nominatim.openstreetmap.org"

#: The public instance asks for at most one request per second. A little over is used so
#: clock jitter cannot put two calls inside the same second.
MIN_INTERVAL_SECONDS = 1.1

#: Nominatim's place_rank, mapped to how precisely a result locates a destination.
#:
#: Rank runs from 0 (a continent) to 30 (a building). The threshold that matters is
#: ``GeocodeHit.is_precise`` at 0.70: a street is precise enough to route a car to, and a
#: city centroid is not, because routing to the middle of Istanbul is not an answer.
_RANK_PRECISION = (
    (30, 0.95),  # building, house number
    (26, 0.85),  # street
    (20, 0.75),  # neighbourhood, suburb
    (16, 0.55),  # city, town
    (12, 0.40),  # county, state district
    (0, 0.25),   # anything larger
)

#: OSM classes that name a destination somebody actually drives to. A result in one of
#: these is worth more than its rank alone suggests, because "Galata Kulesi" is a precise
#: destination even though it is not an address.
_DESTINATION_CLASSES = frozenset(
    {"amenity", "tourism", "leisure", "shop", "historic", "railway", "aeroway", "building"}
)


def precision_for(place_rank: int, osm_class: str) -> float:
    """How precisely a Nominatim result locates a destination."""
    precision = _RANK_PRECISION[-1][1]
    for threshold, value in _RANK_PRECISION:
        if place_rank >= threshold:
            precision = value
            break
    if osm_class in _DESTINATION_CLASSES:
        # A named venue is a real point on the ground even when its rank is low, so it is
        # lifted over the routable threshold rather than treated as an area.
        precision = max(precision, 0.80)
    return precision


class NominatimClient(BaseAdapter):
    """Rate-limited client over the public Nominatim search API."""

    meta = SourceMeta(
        name="Nominatim",
        url=f"{BASE_URL}/search",
        licence="ODbL-1.0",
        licence_url="https://opendatacommons.org/licenses/odbl/1-0/",
        attribution="Geocoding: (c) OpenStreetMap contributors, via Nominatim",
        commercial_use=True,
        share_alike=True,
        refresh="continuous",
        contact="https://operations.osmfoundation.org/policies/nominatim/",
        notes=(
            "Public instance. Usage policy caps requests at one per second and requires "
            "an identifying User-Agent. Both are enforced in this client."
        ),
    )

    #: Shared across instances on purpose. The rate limit belongs to the remote service,
    #: not to any one client object, so two clients in one process must still queue.
    _lock = threading.Lock()
    _last_request_at = 0.0

    def _throttle(self) -> None:
        with NominatimClient._lock:
            elapsed = time.monotonic() - NominatimClient._last_request_at
            if elapsed < MIN_INTERVAL_SECONDS:
                time.sleep(MIN_INTERVAL_SECONDS - elapsed)
            NominatimClient._last_request_at = time.monotonic()

    def search(
        self,
        query: str,
        *,
        country_codes: str | None = None,
        rows: int = 5,
        viewbox: tuple[float, float, float, float] | None = None,
    ) -> list[GeocodeHit]:
        """Resolve free text to locations, best first.

        ``country_codes`` is a comma-separated ISO 3166-1 alpha-2 list. Passing it turns a
        worldwide search into a national one, which matters for short or ambiguous names:
        without it "Merkez" matches somewhere in half the countries on earth.

        ``viewbox`` biases results towards an area without excluding others, which is how
        a city hint is expressed to this API.
        """
        if not query.strip():
            return []

        params: dict[str, Any] = {
            "q": query,
            "format": "jsonv2",
            "limit": max(1, min(rows, 20)),
            "addressdetails": 0,
        }
        if country_codes:
            params["countrycodes"] = country_codes
        if viewbox:
            south, west, north, east = viewbox
            params["viewbox"] = f"{west},{north},{east},{south}"

        self._throttle()
        try:
            payload = self.fetch_json(f"{BASE_URL}/search", params)
        except Exception as exc:
            log.warning("Nominatim search failed for %r: %s", query, exc)
            return []

        rows_out: list[GeocodeHit] = []
        for row in payload if isinstance(payload, list) else []:
            hit = _to_hit(row)
            if hit is not None:
                rows_out.append(hit)
        return rows_out

    def run(self, **kwargs: Any) -> IngestResult:
        """Nominatim is queried live and never bulk-ingested, which its policy forbids."""
        raise NotImplementedError(
            "Nominatim is a live lookup service, not an ingest source. Its usage policy "
            "explicitly prohibits bulk geocoding."
        )


def _to_hit(row: dict[str, Any]) -> GeocodeHit | None:
    try:
        lat = float(row["lat"])
        lon = float(row["lon"])
    except (KeyError, TypeError, ValueError):
        return None

    osm_class = str(row.get("class") or "")
    place_rank = int(row.get("place_rank") or 0)
    precision = precision_for(place_rank, osm_class)

    # `importance` is Nominatim's own relevance, roughly 0 to 1 and usually well below
    # 0.5. It orders results sensibly but is not a confidence, so it is blended with
    # precision rather than used raw: a famous but vague place should not outrank an
    # exact address.
    importance = float(row.get("importance") or 0.0)
    score = min(1.0, 0.65 * precision + 0.35 * min(1.0, importance * 2.0))

    return GeocodeHit(
        label=str(row.get("display_name") or "").strip(),
        lat=lat,
        lon=lon,
        kind=str(row.get("addresstype") or row.get("type") or osm_class or "place"),
        score=score,
        source="Nominatim",
        precision=precision,
    )
