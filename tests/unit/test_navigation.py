"""Handing a space to a navigation app.

The failure this guards against is quiet and expensive: a driver taps "take me there",
the link works, the app opens, a route appears, and it goes to the wrong place. Nothing
errors. So the tests are about precision surviving the trip and about the product being
honest that a car park centroid is not a door.
"""

from __future__ import annotations

import pytest

from parkfit.native import native
from parkfit.services.navigation import build_handoff

# A real Amsterdam bay, to the seventh decimal.
BAY_LAT = 52.3677861
BAY_LON = 4.9026825


def test_the_exact_coordinate_survives_into_every_link():
    """Seven decimals is about a centimetre. Losing them is losing the whole product.

    Amsterdam surveys these bays. If the handoff rounds to five decimals the destination
    moves by roughly a metre, which is most of the gap between one bay and the next.
    """
    handoff = build_handoff(lat=BAY_LAT, lon=BAY_LON, label="Waterlooplein bay")

    assert handoff.available
    for link in handoff.links:
        assert "52.3677861" in link.url, f"{link.provider} lost latitude precision"
        assert "4.9026825" in link.url, f"{link.provider} lost longitude precision"


def test_two_adjacent_bays_do_not_collapse_to_the_same_point():
    """Neighbouring bays are metres apart; the link must be able to tell them apart."""
    a = build_handoff(lat=52.3677861, lon=4.9026825, label="bay A")
    b = build_handoff(lat=52.3677016, lon=4.9025885, label="bay B")

    urls_a = {link.url for link in a.links}
    urls_b = {link.url for link in b.links}
    assert not (urls_a & urls_b)


def test_every_provider_the_product_claims_is_actually_offered():
    handoff = build_handoff(lat=BAY_LAT, lon=BAY_LON, label="bay")
    providers = {link.provider for link in handoff.links}
    assert providers == {
        "google_maps",
        "apple_maps",
        "waze",
        "yandex",
        "openstreetmap",
        "geo",
    }


def test_an_origin_is_only_forwarded_when_one_is_given():
    """Absent is not the same as empty.

    Most navigation apps have a better position fix than a web page can forward. Omitting
    the origin lets them use it; sending a stale one routes from where the driver was.
    """
    without = build_handoff(lat=BAY_LAT, lon=BAY_LON, label="bay")
    google_without = next(link for link in without.links if link.provider == "google_maps")
    assert "origin=" not in google_without.url

    with_origin = build_handoff(
        lat=BAY_LAT, lon=BAY_LON, label="bay", origin_lat=52.3789, origin_lon=4.9002
    )
    google_with = next(link for link in with_origin.links if link.provider == "google_maps")
    assert "origin=52.3789,4.9002" in google_with.url


def test_a_label_that_would_break_a_url_is_escaped():
    """"Q-Park Bijenkorf & Dam" unescaped ends the query string at the ampersand."""
    handoff = build_handoff(lat=BAY_LAT, lon=BAY_LON, label="Q-Park Bijenkorf & Dam")
    apple = next(link for link in handoff.links if link.provider == "apple_maps")
    assert "%26" in apple.url
    assert "& Dam" not in apple.url


def test_a_bay_is_described_as_the_surveyed_point_it_is():
    handoff = build_handoff(
        lat=BAY_LAT, lon=BAY_LON, label="bay", point_description="exact surveyed bay location"
    )
    assert handoff.is_entrance is False
    assert "surveyed" in handoff.point_description


def test_an_entrance_says_so_and_a_centroid_admits_it_is_not_one():
    """The honesty requirement.

    A driver routed to a garage centroid arrives somewhere inside a building outline and
    still has to find the ramp. Telling them that is the difference between a product that
    is wrong and one that is merely imprecise.
    """
    entrance = build_handoff(
        lat=52.3702,
        lon=4.8952,
        label="Q-Park",
        is_entrance=True,
        point_description="routing to the Marnixstraat entrance",
    )
    assert entrance.is_entrance
    assert "entrance" in entrance.point_description

    centroid = build_handoff(
        lat=52.3702,
        lon=4.8952,
        label="Q-Park",
        is_entrance=False,
        point_description="no entrance recorded; routing to the car park itself",
    )
    assert not centroid.is_entrance
    assert "no entrance recorded" in centroid.point_description


def test_an_impossible_coordinate_produces_no_links_at_all():
    """Better a missing button than a button that navigates to the Gulf of Guinea."""
    handoff = build_handoff(lat=91.0, lon=4.9, label="nowhere")
    assert not handoff.available
    assert handoff.links == []


def test_waze_drops_an_origin_it_cannot_express():
    """Waze always routes from the device. Pretending otherwise would be worse."""
    handoff = build_handoff(
        lat=BAY_LAT, lon=BAY_LON, label="bay", origin_lat=52.3789, origin_lon=4.9002
    )
    waze = next(link for link in handoff.links if link.provider == "waze")
    assert "52.3789" not in waze.url
    assert "navigate=yes" in waze.url


def test_coordinates_never_use_a_comma_decimal_separator():
    """A Dutch locale would print 52,3677861, and that is two coordinates, not one."""
    assert native.format_coordinate(52.3677861) == "52.3677861"
    assert native.format_coordinate(-4.5) == "-4.5"


def test_coordinates_never_use_exponent_notation():
    """1e-07 is not something any provider parses."""
    printed = native.format_coordinate(0.0000001)
    assert "e" not in printed.lower()


def test_the_handoff_carries_a_coordinate_not_an_address():
    """The rule the whole module exists for.

    A street string is re-geocoded by the receiving app against its own database, and the
    same text resolves differently in different apps. None of those results is the bay this
    product measured.
    """
    handoff = build_handoff(lat=BAY_LAT, lon=BAY_LON, label="Jodenbreestraat 4, Amsterdam")
    google = next(link for link in handoff.links if link.provider == "google_maps")
    assert "destination=52.3677861,4.9026825" in google.url
    assert "Jodenbreestraat" not in google.url


@pytest.mark.parametrize(
    ("lat", "lon"),
    [(52.0, 4.0), (-33.8688, 151.2093), (0.0, 0.0), (52.3677861, -4.9026825)],
)
def test_links_are_produced_for_any_real_coordinate(lat, lon):
    handoff = build_handoff(lat=lat, lon=lon, label="somewhere")
    assert len(handoff.links) == 6
