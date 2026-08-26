"""Merging the same car park described by several sources.

RDW, OpenStreetMap and NDW all describe overlapping reality. Garage The Bank in
Amsterdam appears twice: once from RDW as "Garage The Bank (Amsterdam)" with a 210 cm
barrier and a 110-space capacity, and once from OSM as "The Bank" with neither. Showing
both is bad in three separate ways, it wastes result slots, it makes the list look
padded, and worst of all the OSM copy has no published height, so the same garage shows
up as both FITS and UNVERIFIED in one result set.

Merging keeps the *union* of what is known rather than picking a winner outright: RDW
usually has the dimensions and capacity, OSM usually has the better name and the more
precise entrance. Taking the best field from each beats taking the best record.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from parkfit.geo.rd import haversine_m

#: Two records this close together, describing the same kind of thing, are the same
#: car park. Deliberately generous: a garage centroid from OSM and an entrance
#: coordinate from RDW can easily sit 60 m apart while being the same building.
MERGE_RADIUS_M = 85.0

#: Trust order when two sources disagree on a scalar. RDW is the national register and
#: is authoritative for physical limits; OSM is community data and better for names.
SOURCE_RANK = {
    "RDW-NPR": 3,
    "NDW": 3,
    "Amsterdam-Parkeervakken": 2,
    "OpenStreetMap": 1,
}

_NOISE = re.compile(r"\b(parking|parkeergarage|parkeren|garage|p\+r|pr|car park)\b", re.I)
_PUNCT = re.compile(r"[^\w\s]")


class MergeableCandidate(Protocol):
    key: tuple[str, int]
    kind: str
    name: str
    lat: float
    lon: float
    capacity: int | None
    max_height_cm: float | None
    source_name: str


def _core_name(name: str) -> str:
    """Strip the generic words so "Garage The Bank" and "The Bank" compare equal."""
    cleaned = _PUNCT.sub(" ", (name or "").lower())
    cleaned = re.sub(r"\(.*?\)", " ", cleaned)
    cleaned = _NOISE.sub(" ", cleaned)
    return " ".join(cleaned.split())


def _same_place(a: MergeableCandidate, b: MergeableCandidate) -> bool:
    if a.kind != b.kind and {a.kind, b.kind} != {"garage", "surface_lot"}:
        return False
    distance = haversine_m(a.lat, a.lon, b.lat, b.lon)
    if distance > MERGE_RADIUS_M:
        return False

    name_a, name_b = _core_name(a.name), _core_name(b.name)
    if not name_a or not name_b:
        # An unnamed OSM lot 30 m from a named garage is almost certainly the same
        # thing seen twice. Beyond that, without a name there is nothing to compare.
        return distance <= 40.0
    if name_a == name_b:
        return True
    if name_a in name_b or name_b in name_a:
        return True

    tokens_a, tokens_b = set(name_a.split()), set(name_b.split())
    shared = tokens_a & tokens_b
    if shared and len(shared) >= min(len(tokens_a), len(tokens_b)):
        return True
    # Same spot, unrelated names: two genuinely different car parks in one block is
    # possible, so proximity alone is not enough once both are named.
    return distance <= 25.0 and bool(shared)


def _informativeness(c: MergeableCandidate) -> tuple[int, int, int, int]:
    """Rank records by how much they actually tell us."""
    return (
        1 if c.max_height_cm else 0,
        1 if c.capacity else 0,
        SOURCE_RANK.get(c.source_name, 0),
        len(_core_name(c.name)),
    )


def merge_duplicates(candidates: Sequence[MergeableCandidate]) -> list[MergeableCandidate]:
    """Collapse records describing the same car park, keeping the union of their facts.

    Runs in the search path rather than at ingest so that each source stays intact in
    the database. Provenance is the point of keeping them separate: when a claim turns
    out to be wrong we need to know which source made it.
    """
    if len(candidates) < 2:
        return list(candidates)

    # Bays are never merged. Individual marked bays genuinely sit metres apart and are
    # distinct spaces; collapsing them would delete real parking supply.
    bays = [c for c in candidates if c.key[0] == "bay"]
    facilities = [c for c in candidates if c.key[0] != "bay"]

    kept: list[MergeableCandidate] = []
    for candidate in sorted(facilities, key=_informativeness, reverse=True):
        duplicate_of = next((k for k in kept if _same_place(k, candidate)), None)
        if duplicate_of is None:
            kept.append(candidate)
            continue
        # Fill any gap in the record we are keeping from the one we are dropping.
        if duplicate_of.max_height_cm is None and candidate.max_height_cm is not None:
            duplicate_of.max_height_cm = candidate.max_height_cm
        if not duplicate_of.capacity and candidate.capacity:
            duplicate_of.capacity = candidate.capacity
        if len(_core_name(candidate.name)) > len(_core_name(duplicate_of.name)):
            duplicate_of.name = candidate.name

    return kept + bays
