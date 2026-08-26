"""DATEX II XML helpers.

DATEX II is heavily namespaced and its namespace URIs change between minor versions, so
everything here matches on local tag name instead of a fixed URI.

The important rule this module exists to enforce: **navigate by direct child, never by
subtree search**. A DATEX II ``parkingRecordStatus`` carries a site-level
``parkingOccupancy`` *and* a nested ``groupOfParkingSpacesStatus`` for each sub-area,
each with its own vacant-space count. ``ElementTree.iter()`` walks the whole subtree and
happily returns a sub-area's figure as though it were the site's, a real NDW record
has four ``parkingNumberOfVacantSpaces`` elements reading 8, 4, 4 and 0. Only the first
describes the car park.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterator
from datetime import datetime


def local_name(tag: str) -> str:
    """``{http://datex2.eu/schema/3/parking}parkingRecord`` -> ``parkingRecord``."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def direct_children(element: ET.Element, name: str) -> list[ET.Element]:
    """Immediate children with this local name. Never descends."""
    return [child for child in element if local_name(child.tag) == name]


def direct_child(element: ET.Element, name: str) -> ET.Element | None:
    for child in element:
        if local_name(child.tag) == name:
            return child
    return None


def direct_text(element: ET.Element | None, name: str) -> str | None:
    """Text of an immediate child, or ``None``."""
    if element is None:
        return None
    child = direct_child(element, name)
    if child is None or child.text is None:
        return None
    text = child.text.strip()
    return text or None


def path_text(element: ET.Element | None, *names: str) -> str | None:
    """Follow a chain of direct children and return the final element's text."""
    node = element
    for name in names[:-1]:
        if node is None:
            return None
        node = direct_child(node, name)
    return direct_text(node, names[-1])


def iter_descendants(element: ET.Element, name: str) -> Iterator[ET.Element]:
    """Subtree search. Only for genuinely repeated records, never for scalar fields."""
    for child in element.iter():
        if child is not element and local_name(child.tag) == name:
            yield child


def find_records(root: ET.Element, name: str) -> list[ET.Element]:
    return [e for e in root.iter() if local_name(e.tag) == name]


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_int(value: str | None) -> int | None:
    f = parse_float(value)
    return int(f) if f is not None else None


def element_id(element: ET.Element) -> str | None:
    """DATEX II puts ``id`` either bare or namespaced depending on the element."""
    if element.get("id"):
        return element.get("id")
    for key, value in element.attrib.items():
        if local_name(key) == "id":
            return value
    return None
