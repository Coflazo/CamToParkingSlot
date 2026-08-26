"""Hybrid destination geocoding.

The single most important correction to the original design of this product: **the
official Dutch geocoder cannot find the destinations users type.**

Verified against the live PDOK Locatieserver:

* ``"Rembrandthuis"`` -> **0 results**
* ``"Jodenbreestraat 4, Amsterdam"`` -> exact match, rooftop coordinate

PDOK indexes the BAG address register and the NWB road network. It does not index
museums, parks, stadiums or restaurants. A driver setting off for the Rembrandt House
does not know it is at Jodenbreestraat 4, so a parking app built on PDOK alone returns
nothing for its own headline use case.

The resolution order is therefore:

1. **Local OpenStreetMap point-of-interest index**, exact name, then alias, then
   token-subset match. This is what answers "Rembrandt House Museum".
2. **PDOK free search**, authoritative for anything that is a real address.
3. **PDOK suggest**, last resort for partial or misspelled input.

Results are ranked by how precisely they locate a destination, not by which service
answered. An exact address always outranks a fuzzy point-of-interest guess, because
routing a car to the wrong Vondelstraat is worse than asking the user to be specific.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from parkfit.ingest.pdok import GeocodeHit, PdokClient
from parkfit.storage.models import PointOfInterest

log = logging.getLogger(__name__)

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")

#: Words that carry no discriminating power in Dutch destination names. Dropping them
#: lets "Rembrandt House Museum" match "Museum Het Rembrandthuis".
STOPWORDS = frozenset(
    {
        "de",
        "het",
        "een",
        "der",
        "den",
        "van",
        "in",
        "op",
        "aan",
        "te",
        "the",
        "a",
        "of",
        "at",
    }
)

#: Words a user might type for a category whose OSM tag spells it differently.
CATEGORY_SYNONYMS: dict[str, set[str]] = {
    "attraction": {"museum", "centre", "center"},
    "stadium": {"arena", "stadion", "dome"},
    "station": {"centraal", "central", "gare"},
    "theme_park": {"park"},
    "arts_centre": {"arts", "centre", "center"},
    "events_venue": {"arena", "dome", "hall", "venue"},
    "sports_centre": {"sport", "sports", "centre", "center"},
    "conference_centre": {"congress", "convention", "centre", "center"},
    "mall": {"shopping", "centre", "center"},
    "university": {"campus", "universiteit"},
    "memorial": {"monument"},
    "monument": {"memorial"},
}

#: Category prominence, used only to break ties between identically-named places.
#: A museum called "Centraal" is a likelier destination than a bench called "Centraal".
CATEGORY_IMPORTANCE = {
    "museum": 0.95,
    "attraction": 0.90,
    "stadium": 0.90,
    "theatre": 0.85,
    "station": 0.85,
    "zoo": 0.85,
    "theme_park": 0.85,
    "aquarium": 0.80,
    "gallery": 0.80,
    "hospital": 0.80,
    "university": 0.75,
    "mall": 0.75,
    "park": 0.70,
    "arts_centre": 0.70,
    "cinema": 0.70,
    "conference_centre": 0.70,
    "sports_centre": 0.65,
    "townhall": 0.65,
    "castle": 0.65,
    "monument": 0.60,
    "memorial": 0.55,
    "garden": 0.55,
    "place": 0.40,
}


def normalise(text: str) -> str:
    """Fold accents, strip punctuation, collapse whitespace, lower-case."""
    if not text:
        return ""
    folded = unicodedata.normalize("NFKD", text)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    folded = _PUNCT.sub(" ", folded.lower())
    return _WS.sub(" ", folded).strip()


def content_tokens(text: str) -> set[str]:
    """Meaningful tokens: normalised, stopwords removed, single letters dropped."""
    return {t for t in normalise(text).split() if t not in STOPWORDS and len(t) > 1}


def name_similarity(query: str, candidate: str, category: str | None = None) -> float:
    """Score how well a query names a candidate place, in ``[0, 1]``.

    Deliberately token-based rather than edit-distance based. Dutch place names glue
    words together and reorder freely, "Rembrandt House Museum" against "Museum Het
    Rembrandthuis" is nearly maximal edit distance but an obvious match once you look
    at which meaningful words appear on both sides, and allow for compounds.
    """
    q_norm, c_norm = normalise(query), normalise(candidate)
    if not q_norm or not c_norm:
        return 0.0
    if q_norm == c_norm:
        return 1.0

    q_tokens, c_tokens = content_tokens(query), content_tokens(candidate)
    if not q_tokens or not c_tokens:
        return 0.0

    # Direct token overlap.
    matched = q_tokens & c_tokens

    # Credit words that describe what the place *is* rather than what it is called.
    # Users type "Museum Het Rembrandthuis" and "NEMO Science Museum", the word
    # "museum" is real information, not noise, and penalising it as an unmatched token
    # pushed both queries below a wrong PDOK street match.
    if category:
        category_words = content_tokens(category.replace("_", " ")) | CATEGORY_SYNONYMS.get(
            category, set()
        )
        matched = matched | (q_tokens & category_words)

    # Compound matching: Dutch fuses words, so "rembrandthuis" contains "rembrandt".
    # Only tokens of four or more characters qualify, or "de" would match everything.
    for q in q_tokens - matched:
        if len(q) < 4:
            continue
        if any(q in c or c in q for c in c_tokens if len(c) >= 4):
            matched.add(q)

    if not matched:
        return 0.0

    coverage = len(matched) / len(q_tokens)  # how much of the query was explained
    density = len(matched) / len(c_tokens)  # how much of the name was used
    # Coverage matters more: a query fully explained by a longer name is a good hit,
    # whereas a name fully consumed by half the query usually is not.
    score = 0.72 * coverage + 0.28 * density

    if c_norm.startswith(q_norm) or q_norm in c_norm:
        score = min(1.0, score + 0.12)
    return min(0.99, score)


@dataclass(frozen=True)
class Destination:
    """A resolved destination, with enough context to explain the choice."""

    label: str
    lat: float
    lon: float
    kind: str
    source: str
    confidence: float
    city: str | None = None

    @property
    def is_precise(self) -> bool:
        return self.confidence >= 0.60


class HybridGeocoder:
    """Resolves a free-text destination using OSM points of interest, then PDOK."""

    #: A local point-of-interest scoring at least this is returned without asking PDOK.
    STRONG_POI_SCORE = 0.72
    #: Below this a point-of-interest match is not offered at all.
    MIN_POI_SCORE = 0.42

    def __init__(self, session: Session, pdok: PdokClient | None = None):
        self.session = session
        self._pdok = pdok
        self._owns_pdok = pdok is None

    @property
    def pdok(self) -> PdokClient:
        if self._pdok is None:
            self._pdok = PdokClient()
        return self._pdok

    def close(self) -> None:
        if self._owns_pdok and self._pdok is not None:
            self._pdok.close()
            self._pdok = None

    # public API ---------------------------------------------------------
    def geocode(self, query: str, *, city: str | None = None, limit: int = 5) -> list[Destination]:
        """Resolve a destination, best match first."""
        query = (query or "").strip()
        if not query:
            return []

        results = self._search_pois(query, city=city, limit=limit)
        if results and results[0].confidence >= self.STRONG_POI_SCORE:
            # A confident local hit. Asking PDOK as well would only add addresses that
            # happen to share a word with the museum the user meant.
            return results[:limit]

        try:
            for hit in self.pdok.search(query if city is None else f"{query} {city}", rows=limit):
                results.append(self._from_pdok(hit, query))
        except Exception as exc:
            log.warning("PDOK search failed for %r: %s", query, exc)

        if not results:
            try:
                for hit in self.pdok.suggest(query, rows=limit):
                    results.append(self._from_pdok(hit, query))
            except Exception as exc:
                log.warning("PDOK suggest failed for %r: %s", query, exc)

        results.sort(key=lambda d: d.confidence, reverse=True)
        return self._dedupe(results)[:limit]

    def geocode_one(self, query: str, *, city: str | None = None) -> Destination | None:
        hits = self.geocode(query, city=city, limit=1)
        return hits[0] if hits else None

    # point of interest search ------------------------------------------
    def _search_pois(self, query: str, *, city: str | None, limit: int) -> list[Destination]:
        tokens = content_tokens(query)
        if not tokens:
            return []

        # Pull a candidate set with a cheap SQL prefilter, then score in Python. Doing
        # the scoring in SQL would mean reimplementing compound matching in every
        # dialect, and the candidate set here is small enough that it does not pay.
        clauses = [PointOfInterest.normalised_name.contains(t) for t in tokens if len(t) >= 3]
        if not clauses:
            clauses = [PointOfInterest.normalised_name.contains(normalise(query))]

        stmt = select(PointOfInterest).where(or_(*clauses))
        if city:
            stmt = stmt.where(func.lower(PointOfInterest.city) == city.lower())
        candidates = self.session.execute(stmt.limit(400)).scalars().all()

        scored: list[Destination] = []
        for poi in candidates:
            score = name_similarity(query, poi.name, poi.category)
            for alias in self._aliases(poi):
                score = max(score, name_similarity(query, alias, poi.category))
            if score < self.MIN_POI_SCORE:
                continue
            # Prominence only ever breaks ties; it must not promote a weak name match.
            confidence = min(0.99, score * (0.88 + 0.12 * poi.importance))
            scored.append(
                Destination(
                    label=poi.name,
                    lat=poi.lat,
                    lon=poi.lon,
                    kind=poi.category,
                    source="OpenStreetMap",
                    confidence=confidence,
                    city=poi.city,
                )
            )
        scored.sort(key=lambda d: d.confidence, reverse=True)
        return scored[: limit * 2]

    @staticmethod
    def _aliases(poi: PointOfInterest) -> list[str]:
        if not poi.aliases_json:
            return []
        try:
            data = json.loads(poi.aliases_json)
        except json.JSONDecodeError:
            return []
        return [str(a) for a in data if a]

    @staticmethod
    def _from_pdok(hit: GeocodeHit, query: str) -> Destination:
        """Score a PDOK result on precision *and* relevance.

        Precision alone is not enough. It answers "how exactly do I know where this
        is", never "is this the place they meant". Scoring on precision alone let
        "Van Gogh Allee" in Rhoon, a street 30 km away, score 0.70 and outrank the
        actual Van Gogh Museum.
        """
        relevance = name_similarity(query, hit.label)
        # An exact address keeps a floor: someone who types a full address wants that
        # address, even when the returned label adds a postcode they did not type.
        if hit.kind == "adres":
            relevance = max(relevance, 0.80)
        return Destination(
            label=hit.label,
            lat=hit.lat,
            lon=hit.lon,
            kind=hit.kind,
            source=hit.source,
            confidence=min(0.99, hit.precision * relevance),
        )

    @staticmethod
    def _dedupe(results: list[Destination]) -> list[Destination]:
        """Drop near-duplicates. Two hits within ~30 m are the same place."""
        out: list[Destination] = []
        for candidate in results:
            duplicate = any(
                abs(candidate.lat - kept.lat) < 3e-4 and abs(candidate.lon - kept.lon) < 5e-4
                for kept in out
            )
            if not duplicate:
                out.append(candidate)
        return out
