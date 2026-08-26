"""PDOK Locatieserver client.

PDOK is the official Dutch government geocoder, built on the BAG address register and
the NWB road network. It is excellent at what it indexes and completely blind to what
it does not.

That distinction is not academic. Searching PDOK for **"Rembrandthuis"** returns *zero*
results; searching for "Jodenbreestraat 4, Amsterdam", the museum's actual address,
returns an exact match. PDOK indexes addresses, streets, postcodes and place names, not
points of interest. A parking app whose users type destinations like "Rembrandt House
Museum", "Vondelpark" or "Ziggo Dome" therefore cannot be built on PDOK alone, which is
why :mod:`parkfit.services.geocoding` layers an OpenStreetMap point-of-interest index in
front of it.

What PDOK *is* uniquely good at is resolving a real Dutch address to an exact rooftop
coordinate, which is what the facility-geocoding pass needs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from parkfit.ingest.base import BaseAdapter, IngestResult, SourceMeta

log = logging.getLogger(__name__)

#: PDOK returns geometry as WKT, e.g. ``POINT(4.90124965 52.36936916)``.
_POINT_RE = re.compile(r"POINT\(\s*([-\d.]+)\s+([-\d.]+)\s*\)", re.IGNORECASE)

#: Result types, ranked by how precisely they locate a destination. An exact address
#: beats a street, which beats a postcode district, which beats a whole town.
TYPE_PRECISION = {
    "adres": 1.00,
    "postcode": 0.80,
    "weg": 0.70,
    "gemeente": 0.35,
    "woonplaats": 0.40,
    "provincie": 0.10,
}


@dataclass(frozen=True)
class GeocodeHit:
    """One resolved location."""

    label: str
    lat: float
    lon: float
    kind: str
    score: float
    source: str
    precision: float

    @property
    def is_precise(self) -> bool:
        """Precise enough to route a car to. A town centroid is not."""
        return self.precision >= 0.70


def parse_wkt_point(wkt: str | None) -> tuple[float, float] | None:
    """Parse ``POINT(lon lat)`` into ``(lat, lon)``."""
    if not wkt:
        return None
    m = _POINT_RE.search(wkt)
    if not m:
        return None
    lon, lat = float(m.group(1)), float(m.group(2))
    return lat, lon


class PdokClient(BaseAdapter):
    """Thin client over the PDOK Locatieserver search API."""

    meta = SourceMeta(
        name="PDOK-Locatieserver",
        url="https://api.pdok.nl/bzk/locatieserver/search/v3_1",
        licence="CC0-1.0 (Publieke Dienstverlening Op de Kaart)",
        licence_url="https://www.pdok.nl/",
        attribution="Geocoding: PDOK Locatieserver (Kadaster / BZK)",
        commercial_use=True,
        share_alike=False,
        refresh="continuous",
        contact="https://www.pdok.nl/",
        notes="Address register only. Contains no points of interest.",
    )

    FIELDS = "id,weergavenaam,centroide_ll,type,score,woonplaatsnaam,gemeentenaam"

    def search(self, query: str, *, rows: int = 8) -> list[GeocodeHit]:
        """Free-text search across the address register."""
        if not query or not query.strip():
            return []
        payload = self.fetch_json(
            f"{self.settings.pdok_base_url}/free",
            {"q": query.strip(), "rows": rows, "fl": self.FIELDS},
        )
        return self._hits(payload)

    def suggest(self, query: str, *, rows: int = 8) -> list[GeocodeHit]:
        """Type-ahead suggestions. Returns identifiers without coordinates.

        The suggest endpoint deliberately omits geometry, so a suggestion must be
        resolved through :meth:`lookup` before it can be used as a destination.
        """
        if not query or not query.strip():
            return []
        payload = self.fetch_json(
            f"{self.settings.pdok_base_url}/suggest",
            {"q": query.strip(), "rows": rows},
        )
        return self._hits(payload, allow_missing_geometry=True)

    def lookup(self, pdok_id: str) -> GeocodeHit | None:
        """Resolve a suggestion identifier to a full record with coordinates."""
        payload = self.fetch_json(
            f"{self.settings.pdok_base_url}/lookup", {"id": pdok_id, "fl": self.FIELDS}
        )
        hits = self._hits(payload)
        return hits[0] if hits else None

    def geocode_address(
        self, street: str | None, house_number: str | None, city: str | None
    ) -> GeocodeHit | None:
        """Resolve a structured Dutch address to a rooftop coordinate."""
        parts = [p for p in (street, house_number, city) if p]
        if not parts:
            return None
        hits = self.search(" ".join(str(p) for p in parts), rows=3)
        for hit in hits:
            if hit.kind == "adres":
                return hit
        return hits[0] if hits else None

    def _hits(self, payload: Any, *, allow_missing_geometry: bool = False) -> list[GeocodeHit]:
        docs = ((payload or {}).get("response") or {}).get("docs") or []
        out: list[GeocodeHit] = []
        for doc in docs:
            point = parse_wkt_point(doc.get("centroide_ll"))
            if point is None:
                if not allow_missing_geometry:
                    continue
                continue
            kind = str(doc.get("type") or "onbekend")
            out.append(
                GeocodeHit(
                    label=str(doc.get("weergavenaam") or ""),
                    lat=point[0],
                    lon=point[1],
                    kind=kind,
                    score=float(doc.get("score") or 0.0),
                    source=self.meta.name,
                    precision=TYPE_PRECISION.get(kind, 0.5),
                )
            )
        return out

    def run(self, **kwargs: Any) -> IngestResult:
        """PDOK is queried on demand; there is nothing to bulk-ingest."""
        result = IngestResult(source=self.meta.name)
        result.skipped = 1
        return result
