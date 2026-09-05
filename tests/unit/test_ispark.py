"""Istanbul: parsing the ISPARK feed, and refusing to believe it when it is wrong.

The parsing here is mostly small, and small parsing is where quiet errors live. Three of
these tests exist because the obvious implementation is wrong in a way that produces
plausible output rather than an exception: Turkish case folding, comma decimals, and WKT
coordinate order. A latitude and longitude swapped puts Istanbul in Somalia, and nothing
downstream would raise; it would just route people to the Indian Ocean.

The rest cover the feed telling us something impossible. A site cannot have more free
spaces than spaces, and passing that through would put an impossible number in front of a
driver rather than in a log where somebody can see it.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from parkfit.ingest.ispark import (
    ParkReading,
    parse_reading,
    parse_tariff,
    parse_updated_at,
    parse_wkt_polygon,
)
from parkfit.storage.models import FacilityKind, OccupancyState


def row(**overrides) -> dict:
    """A realistic ``/Park`` row, matching the live feed's field names and types."""
    base = {
        "parkID": 3068,
        # Verbatim from the live feed. The Turkish characters are the point of the
        # folding tests below, so they are not sanitised to please the linter.
        "parkName": "15 Temmuz Şehitler Meydanı Zeminaltı Otoparkı",  # noqa: RUF001
        "lat": "41.0246",
        "lng": "29.0915",
        "capacity": 1029,
        "emptyCapacity": 509,
        "workHours": "24 Saat",
        "parkType": "KAPALI OTOPARK",
        "freeTime": 15,
        "district": "ÜMRANİYE",
        "isOpen": 0,
    }
    base.update(overrides)
    return base


# ------------------------------------------------------------------ parsing

def test_a_normal_row_parses_with_the_numbers_intact():
    reading = parse_reading(row())
    assert reading is not None
    assert reading.park_id == 3068
    assert reading.capacity == 1029
    assert reading.empty == 509
    assert reading.occupied == 520
    assert reading.ratio == pytest.approx(520 / 1029)
    assert reading.state is OccupancyState.VACANT
    assert reading.problems == ()


def test_coordinates_arrive_as_strings():
    reading = parse_reading(row())
    assert reading.lat == pytest.approx(41.0246)
    assert reading.lon == pytest.approx(29.0915)


@pytest.mark.parametrize("bad", [{"lat": ""}, {"lat": "0", "lng": "0"}, {"lng": "999"}, {"parkID": None}])
def test_a_site_with_no_usable_position_is_dropped(bad):
    """Zero, zero is in the Atlantic. Placing a car park there is worse than losing it."""
    assert parse_reading(row(**bad)) is None


def test_turkish_case_folding_classifies_open_lots_correctly():
    """``str.upper()`` mishandles the dotted and dotless i, so folding is explicit.

    Without it, every ``AÇIK OTOPARK`` falls through to UNKNOWN, which is 111 of the 248
    live sites and would silently lose them from the surface-lot category.
    """
    assert parse_reading(row(parkType="AÇIK OTOPARK")).kind is FacilityKind.SURFACE_LOT
    assert parse_reading(row(parkType="KAPALI OTOPARK")).kind is FacilityKind.GARAGE
    assert parse_reading(row(parkType="YOL ÜSTÜ")).kind is FacilityKind.ON_STREET_ZONE
    assert parse_reading(row(parkType="something else")).kind is FacilityKind.UNKNOWN


def test_on_street_sites_stay_on_street():
    """The 51 YOL ÜSTÜ sites are the ones this product is actually about.

    Flattening them into surface lots would lose the distinction that decides whether the
    statutory setbacks in KTK 2918 apply: a kerbside bay is subject to them, a lot behind
    a barrier is not.
    """
    reading = parse_reading(row(parkType="YOL ÜSTÜ"))
    assert reading.kind is FacilityKind.ON_STREET_ZONE


# ----------------------------------------------------- implausible readings

def test_more_free_spaces_than_spaces_is_clamped_and_reported():
    reading = parse_reading(row(capacity=100, emptyCapacity=140))
    assert reading.empty == 100
    assert reading.occupied == 0
    assert any("exceeds capacity" in p for p in reading.problems)


def test_a_negative_free_count_is_clamped_and_reported():
    reading = parse_reading(row(capacity=100, emptyCapacity=-3))
    assert reading.empty == 0
    assert any("negative" in p for p in reading.problems)


def test_an_absurd_capacity_is_refused_rather_than_stored():
    reading = parse_reading(row(capacity=999999, emptyCapacity=10))
    assert reading.capacity == 0
    assert reading.state is OccupancyState.UNKNOWN
    assert any("plausible" in p for p in reading.problems)


def test_a_full_site_reports_occupied_not_unknown():
    assert parse_reading(row(capacity=100, emptyCapacity=0)).state is OccupancyState.OCCUPIED


def test_a_site_with_no_capacity_is_unknown_not_full():
    """Zero capacity means the feed said nothing, not that every space is taken."""
    reading = ParkReading(
        park_id=1, name="x", lat=41.0, lon=29.0, capacity=0, empty=0,
        park_type="", district="", work_hours="", free_minutes=0,
    )
    assert reading.state is OccupancyState.UNKNOWN
    assert reading.ratio is None


# ------------------------------------------------------------------ tariffs

def test_the_first_hour_is_read_from_a_comma_decimal_ladder():
    """``float("110,00")`` raises and ``float("1.234,00")`` would read as 1.234."""
    hourly, ladder = parse_tariff(
        "0-1 Saat : 110,00;1-2 Saat : 140,00;2-4 Saat : 170,00;Tam Gün : 370,00"
    )
    assert hourly == pytest.approx(110.0)
    assert ladder.startswith("0-1 Saat")


def test_a_thousands_separator_does_not_become_a_decimal_point():
    hourly, _ = parse_tariff("0-1 Saat : 1.250,00")
    assert hourly == pytest.approx(1250.0)


def test_the_full_ladder_is_kept_because_it_is_not_linear():
    """110 lira for one hour and 370 for the day: multiplying out would more than double it."""
    text = "0-1 Saat : 110,00;8-12 Saat : 260,00;Tam Gün : 370,00"
    hourly, ladder = parse_tariff(text)
    assert hourly == pytest.approx(110.0)
    assert ladder == text
    assert hourly * 8 > 370.0  # the reason a single rate cannot stand alone


@pytest.mark.parametrize("text", ["", "free of charge", "Saat"])
def test_an_unparseable_tariff_yields_no_number_rather_than_a_wrong_one(text):
    hourly, _ = parse_tariff(text)
    assert hourly is None


# ------------------------------------------------------------------- geometry

def test_wkt_is_read_longitude_first_and_returned_latitude_first():
    """The swap that would put Istanbul in Somalia if it were missed."""
    ring = parse_wkt_polygon(
        "POLYGON ((29.091051 41.025152, 29.092017 41.025142, "
        "29.092051 41.023770, 29.091051 41.025152))"
    )
    assert len(ring) == 4
    for lat, lon in ring:
        # Istanbul is around 41 N, 29 E. If these were swapped, lat would be 29.
        assert 40.0 < lat < 42.0
        assert 28.0 < lon < 30.0


@pytest.mark.parametrize("bad", ["", "not wkt", "POINT (29 41)", "POLYGON ((", "POLYGON (())"])
def test_unusable_wkt_yields_no_ring_rather_than_a_broken_one(bad):
    assert parse_wkt_polygon(bad) == []


def test_a_malformed_coordinate_pair_is_skipped_not_fatal():
    ring = parse_wkt_polygon("POLYGON ((29.0 41.0, oops, 29.1 41.1))")
    assert ring == [(41.0, 29.0), (41.1, 29.1)]


# ----------------------------------------------------------------- timestamps

def test_the_update_stamp_is_converted_from_istanbul_time_to_utc():
    """Turkey has been on permanent UTC+3 since 2016, with no daylight saving."""
    parsed = parse_updated_at("05.09.2026 02:45:17")
    assert parsed == datetime(2026, 9, 4, 23, 45, 17, tzinfo=UTC)


@pytest.mark.parametrize("bad", ["", "2026-09-05T02:45:17", "not a date"])
def test_an_unreadable_stamp_is_none_rather_than_now(bad):
    """Defaulting to now would make a stale reading look fresh, which is the one lie
    this product must never tell about a timestamp."""
    assert parse_updated_at(bad) is None
