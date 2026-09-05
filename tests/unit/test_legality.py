"""The legality layer: statutory setbacks, and what happens when it cannot see.

The tests that matter most here are the negative ones. A legality engine that answers
"legal" when it has no data is worse than no legality engine at all, because the answer
is indistinguishable from one that was actually checked. So several of these assert that
the system stays quiet rather than that it speaks.

The C++ rule tables have their own suite in ``cpp/tests/test_legal.cpp``. This covers the
Python side: the service, its degradation paths, and the shape that reaches the API.
"""

from __future__ import annotations

import pytest

from parkfit.config import Settings
from parkfit.ingest import anchors as anchor_ingest
from parkfit.native import native
from parkfit.services.legality import LegalityService, LegalVerdict

pytestmark = pytest.mark.skipif(native is None, reason="parkfit_native is not built")

#: A quiet corner of Amsterdam with nothing near it, used as the "clear" control.
CLEAR_LAT, CLEAR_LON = 52.3600, 4.8700
#: One hydrant, and a point standing right on it.
HYDRANT_LAT, HYDRANT_LON = 52.3700, 4.9000


@pytest.fixture
def service(tmp_path) -> LegalityService:
    """A service over a tiny hand-built anchor set, so the assertions are exact."""
    settings = Settings(data_dir=tmp_path)
    anchor_ingest.save(
        anchor_ingest.AnchorSet(
            anchors=[
                ("FIRE_HYDRANT", HYDRANT_LAT, HYDRANT_LON),
                ("PEDESTRIAN_CROSSING", 52.3800, 4.9000),
                ("BUS_STOP_SIGN", 52.3850, 4.9000),
                ("JUNCTION", 52.3900, 4.9000),
                ("JUNCTION_WITH_CYCLE_PATH", 52.3900, 4.9000),
            ],
            country="NL",
        ),
        settings,
    )
    return LegalityService(settings)


# ------------------------------------------------------------- degradation

def test_no_anchor_cache_answers_unknown_not_legal(tmp_path):
    """The single most important property in this module."""
    empty = LegalityService(Settings(data_dir=tmp_path))
    assert not empty.available

    verdicts = empty.evaluate([(CLEAR_LAT, CLEAR_LON)])
    assert len(verdicts) == 1
    assert verdicts[0].verdict == "unknown"
    assert verdicts[0].allowed is False
    assert verdicts[0].is_unknown


def test_an_empty_point_list_is_not_an_error(service):
    assert service.evaluate([]) == []


def test_unknown_is_never_counted_as_allowed():
    unknown = LegalVerdict(verdict="unknown", allowed=False)
    assert not unknown.allowed
    assert unknown.is_unknown


# ------------------------------------------------------------- the verdicts

def test_standing_on_a_hydrant_is_prohibited_in_turkiye_and_legal_in_the_netherlands(service):
    """Not a bug: the two statutes genuinely differ.

    KTK 2918 article 61(d) protects fire hydrants explicitly. RVV 1990 articles 23 and 24
    do not mention them at all, and StVO paragraph 12 protects marked fire-brigade access
    ways rather than hydrants. Making all three agree would mean inventing law.
    """
    turkish = service.evaluate_one(HYDRANT_LAT, HYDRANT_LON, country="TR")
    assert turkish.verdict == "prohibited"
    assert turkish.anchor == "fire_hydrant"
    assert "61(d)" in turkish.citation
    assert turkish.required_cm == pytest.approx(500.0)

    dutch = service.evaluate_one(HYDRANT_LAT, HYDRANT_LON, country="NL")
    assert dutch.verdict == "legal"

    german = service.evaluate_one(HYDRANT_LAT, HYDRANT_LON, country="DE")
    assert german.verdict == "legal"


def test_a_dutch_crossing_refusal_cites_its_article(service):
    verdict = service.evaluate_one(52.3800, 4.9000, country="NL")
    assert verdict.verdict == "prohibited"
    assert verdict.anchor == "pedestrian_crossing"
    assert "RVV 1990 art. 23(1)(c)" == verdict.citation
    assert verdict.required_cm == pytest.approx(500.0)


def test_a_clear_point_is_legal_and_claims_no_anchor(service):
    """A verdict that broke no rule must not name a feature it was never near."""
    verdict = service.evaluate_one(CLEAR_LAT, CLEAR_LON, country="NL")
    assert verdict.verdict == "legal"
    assert verdict.allowed
    assert verdict.anchor == ""
    assert verdict.distance_cm < 0
    assert verdict.slack_cm == 0.0


def test_germany_refuses_a_cycle_path_junction_the_netherlands_allows(service):
    """StVO 12(3): 5 m becomes 8 m where a separate cycle path runs on the right."""
    # About 6.7 m south of the junction: inside the German 8 m rule, outside the Dutch 5 m.
    lat, lon = 52.3899400, 4.9000
    assert service.evaluate_one(lat, lon, country="NL").verdict == "legal"

    german = service.evaluate_one(lat, lon, country="DE")
    assert german.verdict == "prohibited"
    assert german.anchor == "junction_with_cycle_path"
    assert german.required_cm == pytest.approx(800.0)


def test_slack_reports_how_far_outside_the_rule_a_point_sits(service):
    verdict = service.evaluate_one(HYDRANT_LAT, HYDRANT_LON, country="TR")
    # Standing on it: 5 m inside a 5 m rule.
    assert verdict.slack_cm == pytest.approx(-500.0, abs=50.0)


# ------------------------------------------------------------- the rulebooks

def test_france_answers_unknown_because_its_statute_is_not_transcribed(service):
    verdict = service.evaluate_one(HYDRANT_LAT, HYDRANT_LON, country="FR")
    assert verdict.verdict == "unknown"
    assert not verdict.allowed


def test_an_unknown_country_never_borrows_another_countrys_law(service):
    """Falling back to the Dutch rules here would apply Dutch law in Belgium, silently."""
    verdict = service.evaluate_one(HYDRANT_LAT, HYDRANT_LON, country="BE")
    assert verdict.verdict == "unknown"

    book = service.rulebook("BE")
    assert not book.complete
    assert book.rule_count == 0


@pytest.mark.parametrize(
    ("country", "complete", "reach_m"),
    [("NL", True, 12.0), ("DE", True, 15.0), ("TR", True, 100.0), ("FR", False, 0.0)],
)
def test_each_book_reports_its_own_reach(service, country, complete, reach_m):
    """The sweep radius comes from the book, so a long rule is never missed."""
    book = service.rulebook(country)
    assert book.complete is complete
    assert book.max_distance_cm == pytest.approx(reach_m * 100.0)


def test_every_rule_in_a_complete_book_carries_a_citation(service):
    for country in ("NL", "DE", "TR"):
        attribution = service.attribution(country)
        assert attribution["complete"]
        assert attribution["rule_count"] > 0
        assert attribution["instrument"]
        for citation in attribution["citations"]:
            assert len(citation) > 4


def test_batched_and_single_evaluation_agree(service):
    points = [
        (HYDRANT_LAT, HYDRANT_LON),
        (CLEAR_LAT, CLEAR_LON),
        (52.3800, 4.9000),
    ]
    batched = service.evaluate(points, country="TR")
    singles = [service.evaluate_one(lat, lon, country="TR") for lat, lon in points]
    assert batched == singles


def test_an_anchor_kind_this_build_does_not_know_is_dropped_not_guessed(tmp_path, caplog):
    """A cache from a newer build must not have its unknown kinds mapped onto a wrong rule."""
    settings = Settings(data_dir=tmp_path)
    anchor_ingest.save(
        anchor_ingest.AnchorSet(
            anchors=[
                ("FIRE_HYDRANT", HYDRANT_LAT, HYDRANT_LON),
                ("SOME_FUTURE_KIND", HYDRANT_LAT, HYDRANT_LON),
            ],
            country="NL",
        ),
        settings,
    )
    service = LegalityService(settings)
    assert service.available
    # The known anchor still works; the unknown one simply is not there.
    assert service.evaluate_one(HYDRANT_LAT, HYDRANT_LON, country="TR").verdict == "prohibited"
