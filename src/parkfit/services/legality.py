"""Whether a driver may legally park where the car happens to fit.

The fit engine answers a physical question and the ranking answers an economic one.
Neither can see the question this module exists for: a space can be empty, measured, wide
enough and cheap, and still be one the law forbids. Offering it costs a fine, and unlike
a missed space the driver acts on it before finding out.

The statutes live in ``cpp/core/include/parkfit/legal/`` as rule tables, one per country,
each rule carrying the article it came from. The anchors they measure against come from
``parkfit.ingest.anchors``. This module is the join: it holds the built index, picks the
book for the country, and hands the whole candidate list across the boundary in one call.

**Nothing here ever upgrades Unknown to Legal.** An uncompiled checkout, an empty anchor
cache, a country whose statute has not been transcribed: all of them answer Unknown, and
the search treats Unknown as "shown with lowered confidence" rather than "cleared". That
asymmetry is the entire point. A system that guesses "probably fine" when it cannot see
is worse than one that says nothing, because the guess is indistinguishable from a check.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import lru_cache

from parkfit.config import Settings, get_settings
from parkfit.ingest import anchors as anchor_ingest
from parkfit.native import native

log = logging.getLogger(__name__)

#: Country the rulebook defaults to when a candidate does not carry one. The pilot is
#: Dutch, and guessing a country is exactly the kind of silent substitution the C++ side
#: refuses to do, so this is a stated default rather than an inference.
DEFAULT_COUNTRY = "NL"


@dataclass(frozen=True)
class LegalVerdict:
    """One legality answer, in plain Python, ready to serialise.

    Mirrors ``parkfit_native.LegalFinding`` rather than exposing it directly, so the API
    layer never handles a pybind object and a checkout with no extension can still build
    the same shape.
    """

    verdict: str
    #: True for LEGAL and CONDITIONAL, false for PROHIBITED and UNKNOWN. Unknown is not
    #: permission, so it does not count as allowed.
    allowed: bool
    anchor: str = ""
    citation: str = ""
    reason: str = ""
    distance_cm: float = -1.0
    required_cm: float = -1.0
    #: Anchor kinds this country's book has rules about that nothing supplied for this
    #: area, so those rules could not fire.
    #:
    #: A "legal" verdict with a non-empty list here means "no rule I could check was
    #: broken", which is a weaker claim than "no rule was broken", and the difference is
    #: exactly the kind a product like this must not paper over. A "prohibited" verdict
    #: is unaffected: a rule that did fire is sound whatever else was missing.
    unchecked_anchors: tuple[str, ...] = ()

    @property
    def fully_checked(self) -> bool:
        """Whether every rule in the book had the data it needed."""
        return not self.unchecked_anchors

    @property
    def is_unknown(self) -> bool:
        return self.verdict == "unknown"

    @property
    def slack_cm(self) -> float:
        """How far outside the prohibited distance this point sits. Negative means inside."""
        if self.distance_cm < 0.0 or self.required_cm < 0.0:
            return 0.0
        return self.distance_cm - self.required_cm


UNKNOWN = LegalVerdict(
    verdict="unknown",
    allowed=False,
    reason="no restriction data loaded for this area, so legality is unknown",
)

OUTSIDE_COVERAGE = LegalVerdict(
    verdict="unknown",
    allowed=False,
    reason="this location is outside the area the loaded restriction data covers",
)

_NOT_BUILT = LegalVerdict(
    verdict="unknown",
    allowed=False,
    reason="the native legality engine is not built, so legality was not checked",
)


class LegalityService:
    """Holds the anchor index and answers legality for a whole candidate list at once."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._index = None
        self._anchor_sets: list[anchor_ingest.AnchorSet] = []
        self._loaded = False

    # ---------------------------------------------------------------- state
    @property
    def available(self) -> bool:
        """True when a verdict here can be better than Unknown."""
        self._ensure_loaded()
        return native is not None and self._index is not None and len(self._index) > 0

    def anchor_counts(self) -> dict[str, int]:
        """Anchors by kind, summed across every cached region."""
        self._ensure_loaded()
        totals: dict[str, int] = {}
        for anchor_set in self._anchor_sets:
            for kind, count in anchor_set.counts().items():
                totals[kind] = totals.get(kind, 0) + count
        return dict(sorted(totals.items(), key=lambda kv: -kv[1]))

    @property
    def regions(self) -> list[tuple[str, tuple[float, float, float, float] | None]]:
        """Which areas are loaded, as (country, bbox) pairs."""
        self._ensure_loaded()
        return [(a.country, a.bbox) for a in self._anchor_sets]

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True

        if native is None:
            log.info("parkfit_native is not built; legality answers Unknown")
            return

        anchor_sets = [a for a in anchor_ingest.load_all(self.settings) if a.anchors]
        if not anchor_sets:
            log.info(
                "no legal anchors cached under %s; legality answers Unknown. "
                "Populate them with: pf ingest anchors",
                anchor_ingest.anchor_dir(self.settings),
            )
            return

        # One index across every region. The grid does not care that the points come
        # from different cities, and a single index means a search near a border sees
        # anchors from both sides without any special handling.
        index = native.AnchorIndex()
        rows = []
        unknown_kinds: set[str] = set()
        for anchor_set in anchor_sets:
            for kind, lat, lon in anchor_set.anchors:
                enum_value = getattr(native.AnchorKind, kind, None)
                if enum_value is None:
                    # A cache written by a newer version than this binary. Dropping the
                    # row is right; inventing a kind would apply the wrong rule.
                    unknown_kinds.add(kind)
                    continue
                rows.append((enum_value, lat, lon))
        if unknown_kinds:
            log.warning("anchor cache holds kinds this build does not know: %s", unknown_kinds)

        index.add_many(rows)
        index.build()
        self._index = index
        self._anchor_sets = anchor_sets
        log.info(
            "legal anchors loaded: %d across %d region(s) %s",
            len(index),
            len(anchor_sets),
            [a.country for a in anchor_sets],
        )

    # ------------------------------------------------------------ querying
    def covers(self, lat: float, lon: float, *, country: str | None = None) -> bool:
        """Whether the loaded anchors can actually say anything about this point.

        This exists because of a bug that was invisible from the output. The index held
        4,477 Amsterdam anchors; an Istanbul point swept it, found nothing within a
        hundred metres because everything in it was two thousand kilometres away, broke
        no rules, and came back **legal**. A non-empty index is not the same thing as
        coverage, and "no anchors near this point" and "no anchors near this point
        because none were ever collected here" are opposite answers.

        The usable area is the ingested bounding box **eroded by the book's own reach**.
        A point three metres inside the edge of the box could have a hydrant four metres
        away on the other side of it, never ingested and therefore never seen, so the
        band around the edge is honestly outside coverage even though anchors exist there.
        """
        self._ensure_loaded()
        if not self._anchor_sets:
            return False

        book = self.rulebook(country)
        reach_m = (book.max_distance_cm / 100.0) if book is not None else 0.0

        for anchor_set in self._anchor_sets:
            if anchor_set.bbox is None:
                # An older cache with no recorded extent. Its coverage cannot be
                # established, so it is accepted only on the strength of the index
                # itself, which is how this behaved before regions existed.
                if self._index is not None and len(self._index) > 0:
                    return True
                continue

            south, west, north, east = anchor_set.bbox
            # Degrees per metre. Longitude is scaled by latitude, and the cosine is
            # floored so a pole never produces an infinite margin.
            lat_margin = reach_m / 111_320.0
            lon_margin = lat_margin / max(0.05, math.cos(math.radians((south + north) / 2.0)))
            if (
                south + lat_margin <= lat <= north - lat_margin
                and west + lon_margin <= lon <= east - lon_margin
            ):
                return True
        return False

    @property
    def coverage_bbox(self) -> tuple[float, float, float, float] | None:
        """The first loaded region's extent. Prefer :attr:`regions` when more than one."""
        self._ensure_loaded()
        return self._anchor_sets[0].bbox if self._anchor_sets else None

    def unchecked_anchors(self, lat: float, lon: float, *, country: str | None = None):
        """Anchor kinds this book needs that nothing supplied for this point's region.

        Compared against what the ingest **asked for**, not what it found. A district
        with no fire hydrants and a district nobody looked for hydrants in produce the
        same empty list, and only the query record separates them.

        Kinds an area never queried are a real gap even when the area is otherwise
        covered, and Turkey is the clear case: KTK 2918 has a ten-metre rule for bridges
        and underpasses, nothing sources either yet, so a Turkish space near a bridge
        comes back clean because that rule never ran rather than because it passed.
        """
        self._ensure_loaded()
        book = self.rulebook(country)
        if book is None or not book.complete:
            return ()

        required = {name.upper() for name in book.anchor_names}
        for anchor_set in self._anchor_sets:
            if anchor_set.bbox is None:
                continue
            south, west, north, east = anchor_set.bbox
            if south <= lat <= north and west <= lon <= east:
                return tuple(sorted(required - set(anchor_set.queried_kinds)))
        return tuple(sorted(required))

    def rulebook(self, country: str | None = None):
        """The book for a country. Unknown codes get an incomplete one, never a substitute."""
        if native is None:
            return None
        return native.rulebook_for((country or DEFAULT_COUNTRY).upper())

    def evaluate(
        self,
        points: list[tuple[float, float]],
        *,
        country: str | None = None,
        contexts: list[object] | None = None,
    ) -> list[LegalVerdict]:
        """One verdict per point, in order.

        The whole list crosses the boundary once. A search scores a few hundred
        candidates and each needs its own anchor sweep, so doing this one call at a time
        would spend more in pybind than in the sweep.
        """
        if not points:
            return []
        self._ensure_loaded()

        if native is None:
            return [_NOT_BUILT] * len(points)
        if self._index is None or len(self._index) == 0:
            return [UNKNOWN] * len(points)

        code = (country or DEFAULT_COUNTRY).upper()
        # A point the anchors do not cover is not swept at all. Sweeping it would find
        # nothing and report legal, which is the same answer a genuinely clear space
        # gets and therefore indistinguishable from one that was actually checked.
        inside = [i for i, (lat, lon) in enumerate(points) if self.covers(lat, lon, country=code)]
        out: list[LegalVerdict] = [OUTSIDE_COVERAGE] * len(points)
        if not inside:
            return out

        book = native.rulebook_for(code)
        findings = native.legal_evaluate_many(
            book,
            native.Manoeuvre.PARKING,
            self._index,
            [points[i] for i in inside],
            [contexts[i] for i in inside] if contexts else [],
        )
        # The gap depends only on which region a point falls in, so it is computed
        # once per distinct region rather than once per candidate.
        gaps: dict[tuple[float, float], tuple[str, ...]] = {}
        for position, finding in zip(inside, findings, strict=True):
            lat, lon = points[position]
            key = (round(lat, 2), round(lon, 2))
            if key not in gaps:
                gaps[key] = self.unchecked_anchors(lat, lon, country=code)
            out[position] = _to_verdict(finding, gaps[key])
        return out

    def evaluate_one(
        self, lat: float, lon: float, *, country: str | None = None, context: object | None = None
    ) -> LegalVerdict:
        self._ensure_loaded()
        if native is None:
            return _NOT_BUILT
        if self._index is None or len(self._index) == 0:
            return UNKNOWN
        code = (country or DEFAULT_COUNTRY).upper()
        if not self.covers(lat, lon, country=code):
            return OUTSIDE_COVERAGE
        book = native.rulebook_for(code)
        finding = native.legal_evaluate_at(
            book,
            native.Manoeuvre.PARKING,
            self._index,
            lat,
            lon,
            context if context is not None else native.LegalContext(),
        )
        return _to_verdict(finding, self.unchecked_anchors(lat, lon, country=code))

    def context(
        self,
        *,
        built_up: bool = True,
        road_has_marked_bays: bool = False,
        inside_marked_bay: bool = False,
        permit_zone_without_permit: bool = False,
        disc_zone: bool = False,
    ):
        """Build a native context, or None on an uncompiled checkout."""
        if native is None:
            return None
        context = native.LegalContext()
        context.built_up = built_up
        context.road_has_marked_bays = road_has_marked_bays
        context.inside_marked_bay = inside_marked_bay
        context.permit_zone_without_permit = permit_zone_without_permit
        context.disc_zone = disc_zone
        return context

    def attribution(self, country: str | None = None) -> dict[str, object]:
        """What a result should say about the law it was judged against."""
        book = self.rulebook(country)
        if book is None:
            return {"instrument": "", "complete": False, "citations": []}
        return {
            "country": book.country,
            "instrument": book.instrument,
            "complete": book.complete,
            "rule_count": book.rule_count,
            "citations": list(book.citations),
        }


def _to_verdict(finding, unchecked: tuple[str, ...] = ()) -> LegalVerdict:
    # A verdict that broke no rule has no anchor. The C++ struct carries Junction there
    # as a filler because the field is not optional, and passing that on would tell a
    # reader a legal space was judged against a junction it was never near.
    distance_based = finding.distance_cm >= 0.0
    return LegalVerdict(
        verdict=finding.verdict_name,
        allowed=finding.allowed,
        anchor=finding.anchor_name if distance_based else "",
        citation=finding.citation,
        reason=finding.reason,
        distance_cm=finding.distance_cm,
        required_cm=finding.required_cm,
        # Only reported on a clean verdict. A refusal already stands on a rule that
        # fired, and listing what was not checked beside it would just be noise.
        unchecked_anchors=unchecked if finding.verdict_name == "legal" else (),
    )


@lru_cache(maxsize=1)
def get_legality_service() -> LegalityService:
    """Process-wide singleton. The anchor index is built once and reused per request."""
    return LegalityService()


def reset_legality_service() -> None:
    """Drop the singleton, so a test or a fresh ingest picks up new anchors."""
    get_legality_service.cache_clear()
