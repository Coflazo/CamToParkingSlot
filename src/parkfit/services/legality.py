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
        self._anchor_set: anchor_ingest.AnchorSet | None = None
        self._loaded = False

    # ---------------------------------------------------------------- state
    @property
    def available(self) -> bool:
        """True when a verdict here can be better than Unknown."""
        self._ensure_loaded()
        return native is not None and self._index is not None and len(self._index) > 0

    def anchor_counts(self) -> dict[str, int]:
        self._ensure_loaded()
        return self._anchor_set.counts() if self._anchor_set else {}

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True

        if native is None:
            log.info("parkfit_native is not built; legality answers Unknown")
            return

        anchor_set = anchor_ingest.load(self.settings)
        if anchor_set is None or not anchor_set.anchors:
            log.info(
                "no legal anchors cached at %s; legality answers Unknown. "
                "Populate them with: pf ingest anchors",
                anchor_ingest.cache_path(self.settings),
            )
            return

        index = native.AnchorIndex()
        rows = []
        unknown_kinds: set[str] = set()
        for kind, lat, lon in anchor_set.anchors:
            enum_value = getattr(native.AnchorKind, kind, None)
            if enum_value is None:
                # A cache written by a newer version than this binary. Dropping the row is
                # right; inventing a kind for it would apply the wrong rule.
                unknown_kinds.add(kind)
                continue
            rows.append((enum_value, lat, lon))
        if unknown_kinds:
            log.warning("anchor cache holds kinds this build does not know: %s", unknown_kinds)

        index.add_many(rows)
        index.build()
        self._index = index
        self._anchor_set = anchor_set
        log.info("legal anchors loaded: %d (%s)", len(index), anchor_set.counts())

    # ------------------------------------------------------------ querying
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

        book = native.rulebook_for((country or DEFAULT_COUNTRY).upper())
        findings = native.legal_evaluate_many(
            book,
            native.Manoeuvre.PARKING,
            self._index,
            points,
            contexts or [],
        )
        return [_to_verdict(f) for f in findings]

    def evaluate_one(
        self, lat: float, lon: float, *, country: str | None = None, context: object | None = None
    ) -> LegalVerdict:
        self._ensure_loaded()
        if native is None:
            return _NOT_BUILT
        if self._index is None or len(self._index) == 0:
            return UNKNOWN
        book = native.rulebook_for((country or DEFAULT_COUNTRY).upper())
        finding = native.legal_evaluate_at(
            book,
            native.Manoeuvre.PARKING,
            self._index,
            lat,
            lon,
            context if context is not None else native.LegalContext(),
        )
        return _to_verdict(finding)

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


def _to_verdict(finding) -> LegalVerdict:
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
    )


@lru_cache(maxsize=1)
def get_legality_service() -> LegalityService:
    """Process-wide singleton. The anchor index is built once and reused per request."""
    return LegalityService()


def reset_legality_service() -> None:
    """Drop the singleton, so a test or a fresh ingest picks up new anchors."""
    get_legality_service.cache_clear()
