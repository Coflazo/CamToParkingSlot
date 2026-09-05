"""France: the national base, and the height column that decides whether a van fits.

Height is the reason this dataset matters. It is the dimension a barrier physically stops
a vehicle at, and 84 % of French sites publish one against 29 % of Dutch RDW rows. It is
also the column with the most ways to be quietly wrong, so most of these tests are about
refusing to believe it rather than about reading it.
"""

from __future__ import annotations

import pytest

from parkfit.ingest.france import parse_csv, parse_height_cm, parse_row
from parkfit.storage.models import FacilityKind

HEADER = (
    "id;nom;insee;adresse;url;type_usagers;gratuit;nb_places;nb_pr;nb_pmr;"
    "nb_voitures_electriques;nb_velo;nb_2r_el;nb_autopartage;nb_2_rm;nb_covoit;"
    "hauteur_max;num_siret;Xlong;Ylat;tarif_pmr;tarif_1h;tarif_2h;tarif_3h;tarif_4h;"
    "tarif_24h;abo_resident;abo_non_resident;type_ouvrage;info"
)
#: Taken verbatim from the published file.
ROW = (
    "04070-P-001;Parking Gassendi;04070;Place Général de Gaulle;;tous;0;222;;6;;;;0;;;"
    "200;21040070100012;6.2366534;44.0932836;;2.00;;;;18.00;;;ouvrage;En travaux en 2020"
)


def row(**overrides) -> dict[str, str]:
    fields = dict(zip(HEADER.split(";"), ROW.split(";"), strict=False))
    fields.update(overrides)
    return fields


# ------------------------------------------------------------------ parsing

def test_a_real_row_parses():
    site = parse_row(row())
    assert site is not None
    assert site.external_id == "04070-P-001"
    assert site.name == "Parking Gassendi"
    assert site.spaces == 222
    assert site.disabled_spaces == 6
    assert site.kind is FacilityKind.GARAGE
    assert site.problems == ()


def test_the_columns_say_which_coordinate_is_which():
    """Xlong and Ylat are named, unlike the German and Turkish feeds."""
    site = parse_row(row())
    assert site.lat == pytest.approx(44.0932836)
    assert site.lon == pytest.approx(6.2366534)
    # Metropolitan France, not the Gulf of Guinea.
    assert 41.0 < site.lat < 52.0
    assert -6.0 < site.lon < 10.0


def test_the_file_is_semicolon_delimited_with_a_bom():
    """Both are the French CSV convention and both break a default reader.

    A comma reader sees one enormous column, and a BOM left in place glues itself to the
    first column name so `id` silently never matches.
    """
    sites = parse_csv("﻿" + HEADER + "\n" + ROW + "\n")
    assert len(sites) == 1
    assert sites[0].external_id == "04070-P-001"


@pytest.mark.parametrize(
    "bad",
    [
        {"id": ""},
        {"Ylat": "", "Xlong": ""},
        {"Ylat": "0", "Xlong": "0"},
        {"Ylat": "not a number"},
        {"Ylat": "999"},
    ],
)
def test_a_row_that_cannot_be_placed_is_dropped(bad):
    assert parse_row(row(**bad)) is None


def test_a_comma_decimal_is_accepted():
    """French files use both separators, sometimes in the same file."""
    assert parse_row(row(Xlong="6,2366534")).lon == pytest.approx(6.2366534)


# ------------------------------------------------------------------- height

def test_the_height_is_centimetres_not_metres():
    """The median across the file is 190, the classic French underground limit."""
    height, complaint = parse_height_cm("190")
    assert height == pytest.approx(190.0)
    assert complaint is None


def test_an_unpublished_height_is_none_not_unlimited():
    """The same trap RDW sets. Zero means nobody wrote a limit down."""
    for value in ("0", "", None, "  "):
        height, complaint = parse_height_cm(value)
        assert height is None
        # Silence is not a data-quality complaint, it is just silence.
        assert complaint is None


def test_an_absurdly_tall_limit_is_refused():
    """The real file contains a 1905, which is not a 19 metre car park."""
    height, complaint = parse_height_cm("1905")
    assert height is None
    assert "not a barrier" in complaint


def test_an_absurdly_short_limit_is_refused_because_it_would_reject_every_car():
    """Six real Paris rows publish 20, 40 and 100 cm.

    Stored as-is, those car parks would fail every vehicle on earth, which reads as a
    confident answer rather than as missing data. Unverified is the honest verdict.
    """
    for value in ("20", "40", "100"):
        height, complaint = parse_height_cm(value)
        assert height is None
        assert "bollard" in complaint


def test_an_implausible_height_is_reported_as_a_problem_not_swallowed():
    site = parse_row(row(hauteur_max="1905"))
    assert site.max_height_cm is None
    assert any("not a barrier" in p for p in site.problems)


# ------------------------------------------------------------------ the rest

def test_free_is_only_free_when_the_column_says_one():
    """Telling a driver a paid car park is free is the more expensive mistake."""
    assert parse_row(row(gratuit="1")).free
    for value in ("0", "", "oui", "true"):
        assert not parse_row(row(gratuit=value)).free


@pytest.mark.parametrize(
    ("type_ouvrage", "kind"),
    [
        ("ouvrage", FacilityKind.GARAGE),
        ("enclos_en_surface", FacilityKind.SURFACE_LOT),
        ("", FacilityKind.UNKNOWN),
        ("something new", FacilityKind.UNKNOWN),
    ],
)
def test_the_structure_type_maps_or_stays_unknown(type_ouvrage, kind):
    """178 of 826 rows say nothing, and unknown is the right answer for those."""
    assert parse_row(row(type_ouvrage=type_ouvrage)).kind is kind


def test_an_implausible_capacity_is_refused():
    site = parse_row(row(nb_places="999999"))
    assert site.spaces == 0
    assert any("plausible" in p for p in site.problems)


def test_tariffs_are_read_as_euros():
    site = parse_row(row())
    assert site.tariff_1h_eur == pytest.approx(2.00)
    assert site.tariff_24h_eur == pytest.approx(18.00)


def test_a_missing_count_is_zero_not_an_error():
    site = parse_row(row(nb_pmr="", nb_voitures_electriques=""))
    assert site.disabled_spaces == 0
    assert site.ev_spaces == 0
