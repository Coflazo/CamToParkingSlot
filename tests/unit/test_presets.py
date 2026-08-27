"""The test fleet.

These are real registered vehicles, so the tests assert facts about the Dutch register
rather than about numbers this project invented. The one that matters is the last: a
7 metre Sprinter must not be offered a 5.7 metre kerb bay, and the only way to know the
fit engine still believes that is to ask it.
"""

from __future__ import annotations

import pytest

from parkfit.domain import presets


def test_every_segment_a_driver_might_have_is_represented():
    segments = set(presets.by_segment())
    for expected in ("city car", "hatchback", "sedan", "estate", "SUV", "MPV", "van"):
        assert any(expected in s for s in segments), f"no {expected} in the fleet"


def test_the_fleet_spans_the_range_the_fit_engine_has_to_get_right():
    """From a car that fits anything to a van that fits almost nothing."""
    lengths = [p.length_cm for p in presets.PRESETS]
    assert min(lengths) < 380, "nothing small enough to stress the short end"
    assert max(lengths) > 650, "nothing long enough to stress the long end"


def test_dimensions_are_plausible_for_a_registered_road_vehicle():
    for p in presets.PRESETS:
        assert 300 <= p.length_cm <= 800, f"{p.key} length {p.length_cm}"
        assert 150 <= p.body_width_cm <= 260, f"{p.key} width {p.body_width_cm}"
        assert 120 <= p.height_cm <= 400, f"{p.key} height {p.height_cm}"
        assert 700 <= p.weight_kg <= 4000, f"{p.key} weight {p.weight_kg}"


def test_mirrors_are_added_to_bodywork_not_baked_into_it():
    """RDW's breedte excludes mirrors, and the fit engine relies on that.

    Bodywork is checked against painted lines, mirrors against apertures. Conflating them
    rejects an ordinary car from an ordinary bay by the width of two mirrors.
    """
    polo = presets.get("polo")
    assert polo.body_width_cm == 175
    assert polo.width_with_mirrors_cm == 211


def test_rdw_has_no_suv_category_and_the_preset_says_so():
    """A real quirk of the register, worth pinning so nobody "fixes" it later.

    The register files a BMW X5 and a Range Rover Evoque as `stationwagen`, the same
    class as a Skoda Octavia estate. The everyday word lives in `segment` instead.
    """
    x5 = presets.get("x5")
    assert x5.rdw_body_type == "stationwagen"
    assert "SUV" in x5.segment

    octavia = presets.get("octavia")
    assert octavia.rdw_body_type == "stationwagen"
    assert octavia.segment == "estate"


def test_electric_vehicles_are_flagged_so_charging_bays_resolve_correctly():
    """An EV bay is reserved for a charging EV; a diesel in one is a fine."""
    assert presets.get("zoe").is_ev
    assert presets.get("modely").is_ev
    assert not presets.get("s60").is_ev


def test_a_preset_converts_to_a_profile_with_its_dimensions_confirmed():
    profile = presets.get("s60").to_profile()
    assert profile.length_cm == 460
    assert profile.height_cm == 143
    # Length, height and weight come from the register, so they are confirmed. Mirror
    # span is inferred, so width is not.
    assert profile.length_confirmed
    assert profile.height_confirmed
    assert not profile.width_confirmed


def test_an_unknown_key_returns_nothing_rather_than_a_default_car():
    assert presets.get("delorean") is None


@pytest.mark.parametrize("key", [p.key for p in presets.PRESETS])
def test_every_preset_produces_a_usable_profile(key):
    profile = presets.get(key).to_profile()
    assert profile.effective_height_cm > 0
    assert profile.width_with_mirrors_cm > profile.body_width_cm


def test_a_van_cannot_be_offered_a_kerb_bay_a_hatchback_fits():
    """The discrimination the whole fit engine exists for.

    A 5.7 by 2.2 m Waterlooplein bay takes a Polo comfortably. A 6.97 m Sprinter does not
    fit in it and must not be offered it, whatever else is true about the space.
    """
    from parkfit.native import native

    bay_length_cm, bay_width_cm = 570.0, 220.0
    margins = native.Margins()

    polo = presets.get("polo").to_profile().to_native()
    sprinter = presets.get("sprinter").to_profile().to_native()

    polo_verdict = native.check_bay(
        polo, bay_length_cm, bay_width_cm, native.BayOrientation.PARALLEL, margins
    )
    sprinter_verdict = native.check_bay(
        sprinter, bay_length_cm, bay_width_cm, native.BayOrientation.PARALLEL, margins
    )

    assert polo_verdict.verdict != native.Verdict.DOES_NOT_FIT
    assert sprinter_verdict.verdict == native.Verdict.DOES_NOT_FIT
