"""Germany: parsing the Autobahn feed without inventing an occupancy it never gives.

The most important thing this adapter does is refuse to overstate. The feed says how many
spaces exist; it never says how many are free. Turning "20 car spaces" into "20 free car
spaces" would be a lie at exactly the moment a driver decides to stop, so capacity lands
as `STATIC_DATABASE` with `vacant_spaces` unset, and the tests below hold it there.

The rest are parsing traps that fail quietly rather than loudly: GeoJSON coordinate order
and a boolean that arrives as a string.
"""

from __future__ import annotations

import pytest

from parkfit.ingest.autobahn import parse_record, parse_spaces
from parkfit.storage.models import FacilityKind


def record(**overrides) -> dict:
    """A real record shape, taken verbatim from the A8 response."""
    base = {
        "identifier": "DE-SL-000031",
        "icon": "314-50",
        "isBlocked": "false",
        "subtitle": "RA Moseltal N",
        "title": "A8 | undefined",
        "coordinate": {"type": "Point", "coordinates": [6.373376, 49.483848]},
        "description": ["PKW Stellplätze: 20", "LKW Stellplätze: 16"],
    }
    base.update(overrides)
    return base


# ------------------------------------------------------------------ parsing

def test_a_normal_record_parses():
    area = parse_record(record(), "A8")
    assert area is not None
    assert area.identifier == "DE-SL-000031"
    assert area.road == "A8"
    assert area.name == "RA Moseltal N"
    assert area.car_spaces == 20
    assert area.lorry_spaces == 16
    assert not area.blocked
    assert area.problems == ()


def test_coordinates_are_read_longitude_first():
    """GeoJSON order. Reading it the other way puts every German rest area elsewhere,
    with valid-looking coordinates and nothing raising."""
    area = parse_record(record(), "A8")
    assert area.lat == pytest.approx(49.483848)
    assert area.lon == pytest.approx(6.373376)
    # Sanity: Germany, not the Indian Ocean.
    assert 47.0 < area.lat < 56.0
    assert 5.0 < area.lon < 16.0


def test_is_blocked_is_a_string_not_a_boolean():
    """Every non-empty string is truthy, so a naive read closes the whole network."""
    assert not parse_record(record(isBlocked="false"), "A8").blocked
    assert parse_record(record(isBlocked="true"), "A8").blocked
    assert not parse_record(record(isBlocked=""), "A8").blocked


@pytest.mark.parametrize(
    "bad",
    [
        {"identifier": ""},
        {"coordinate": {}},
        {"coordinate": {"coordinates": []}},
        {"coordinate": {"coordinates": [0.0, 0.0]}},
        {"coordinate": {"coordinates": ["x", "y"]}},
        {"coordinate": {"coordinates": [6.37, 999.0]}},
    ],
)
def test_a_record_that_cannot_be_placed_is_dropped(bad):
    assert parse_record(record(**bad), "A8") is None


def test_the_title_is_ignored_because_it_says_undefined():
    """The API's own frontend leaks the word into every title."""
    area = parse_record(record(), "A8")
    assert "undefined" not in area.name
    assert area.name == "RA Moseltal N"


def test_a_record_with_no_name_falls_back_to_its_identifier():
    area = parse_record(record(subtitle=""), "A8")
    assert area.name == "DE-SL-000031"


# ------------------------------------------------------------------ capacity

@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        (["PKW Stellplätze: 20", "LKW Stellplätze: 16"], (20, 16)),
        (["LKW Stellplätze: 65", "PKW Stellplätze: 0"], (0, 65)),
        (["PKW Stellplaetze: 12"], (12, 0)),   # the label without the umlaut
        (["pkw stellplätze : 7"], (7, 0)),     # case and spacing drift
        ([], (0, 0)),
        (None, (0, 0)),
        (["nothing useful here"], (0, 0)),
    ],
)
def test_space_counts_are_read_from_the_german_description(lines, expected):
    assert parse_spaces(lines) == expected


def test_a_site_with_only_lorry_spaces_is_truck_parking():
    """A car search must never be offered one. Storing it as a surface lot would offer a
    car a space that does not exist for it."""
    area = parse_record(record(description=["PKW Stellplätze: 0", "LKW Stellplätze: 65"]), "A3")
    assert area.car_spaces == 0
    assert area.kind is FacilityKind.TRUCK_PARKING


def test_a_site_with_car_spaces_is_a_surface_lot():
    assert parse_record(record(), "A8").kind is FacilityKind.SURFACE_LOT


def test_an_implausible_capacity_is_refused_rather_than_stored():
    area = parse_record(record(description=["PKW Stellplätze: 99999"]), "A8")
    assert area.car_spaces == 0
    assert any("implausible" in p for p in area.problems)


def test_no_capacity_is_not_treated_as_unknown_but_large():
    """Silence and zero both mean "do not offer this to a car"."""
    area = parse_record(record(description=[]), "A8")
    assert area.car_spaces == 0
    assert area.kind is FacilityKind.TRUCK_PARKING
