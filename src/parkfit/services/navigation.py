"""Handing a chosen space over to the driver's navigation app.

Two decisions live here, and both are about *which point* to send.

**A car park is entered at its entrance, not at its middle.** The centroid of an Amsterdam
garage is regularly a canal, a tram line, or the wrong end of a one-way street. Where an
entrance is known it is used, and the response says so, so an interface can show "routing
to the entrance on Marnixstraat" rather than implying a precision the data does not have.
Where no entrance is recorded the centroid is used and the response admits that too.

**A coordinate is handed over as a coordinate.** Never as a street address. The receiving
app would re-geocode the text against its own database, and the same string resolves to
different points in different apps; none of them is the bay this product measured. Amsterdam
publishes bays as surveyed polygons, and throwing that away at the last step to save a few
characters of URL would be the single most expensive shortcut in the product.

The URL building itself is in ``parkfit::nav`` on the C++ side, so there is one
implementation of the coordinate formatting and the provider templates rather than one
here, one in the web client, and a slow divergence between them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from parkfit.native import native
from parkfit.storage.models import FacilityEntrance

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class NavigationLink:
    provider: str
    display_name: str
    url: str


@dataclass(frozen=True)
class NavigationHandoff:
    """Everything an interface needs for a "take me there" button."""

    lat: float
    lon: float
    label: str
    #: True when the point is a recorded entrance rather than a centroid or bay point.
    is_entrance: bool
    #: What the point actually is, in words, for display next to the button.
    point_description: str
    links: list[NavigationLink] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return bool(self.links)


def _entrances(session: Session, facility_ids: list[int]) -> dict[int, FacilityEntrance]:
    """Preferred entry point per facility, in one query.

    An entrance flagged ``is_entry`` wins over one that is exit-only; a garage with a
    separate exit ramp will happily route a driver to the ramp otherwise, and the ramp is
    usually a one-way street pointing the wrong direction.
    """
    if not facility_ids:
        return {}

    rows = (
        session.execute(
            select(FacilityEntrance).where(FacilityEntrance.facility_id.in_(facility_ids))
        )
        .scalars()
        .all()
    )

    best: dict[int, FacilityEntrance] = {}
    for row in rows:
        if row.lat is None or row.lon is None:
            continue
        current = best.get(row.facility_id)
        if current is None or (row.is_entry and not current.is_entry):
            best[row.facility_id] = row
    return best


def build_handoff(
    *,
    lat: float,
    lon: float,
    label: str,
    is_entrance: bool = False,
    point_description: str = "",
    origin_lat: float | None = None,
    origin_lon: float | None = None,
) -> NavigationHandoff:
    """Build the handoff for one already-resolved point.

    On a checkout that has never been compiled there is no C++ URL builder, so the
    handoff comes back with no links and ``available`` False. That is the honest answer:
    the point itself is still correct and still shown, and the button is simply absent
    rather than the whole search failing on an AttributeError.
    """
    if native is None:
        log.warning(
            "parkfit_native is not built, so navigation links are unavailable; "
            "build it with: .\\tasks.ps1 build"
        )
        return NavigationHandoff(
            lat=float(lat),
            lon=float(lon),
            label=label,
            is_entrance=is_entrance,
            point_description=point_description,
        )

    target = native.NavTarget()
    target.lat = float(lat)
    target.lon = float(lon)
    target.label = label
    target.is_entrance = is_entrance

    origin = native.NavOrigin()
    if origin_lat is not None and origin_lon is not None:
        origin.lat = float(origin_lat)
        origin.lon = float(origin_lon)
        # Deliberately opt-in. Most navigation apps have a better position fix than a web
        # page can forward, and a stale origin routes from where the driver was, not where
        # they are.
        origin.present = True

    links = [
        NavigationLink(provider=link.provider, display_name=link.display_name, url=link.url)
        for link in native.build_nav_links(target, origin)
    ]

    return NavigationHandoff(
        lat=float(lat),
        lon=float(lon),
        label=label,
        is_entrance=is_entrance,
        point_description=point_description
        or ("entrance" if is_entrance else "exact parking location"),
        links=links,
    )


def attach_to_candidates(
    session: Session,
    candidates: list,
    *,
    origin_lat: float | None = None,
    origin_lon: float | None = None,
) -> dict[tuple[str, int], NavigationHandoff]:
    """Resolve the destination point for many candidates in one pass.

    Returns a mapping keyed the same way candidates are, so the caller attaches without a
    second lookup. Facilities are resolved to entrances in a single batched query rather
    than one per result.
    """
    facility_ids = [c.key[1] for c in candidates if c.key[0] == "facility"]
    entrances = _entrances(session, facility_ids)

    out: dict[tuple[str, int], NavigationHandoff] = {}
    for candidate in candidates:
        kind, target_id = candidate.key

        if kind == "facility":
            entrance = entrances.get(target_id)
            if entrance is not None:
                where = entrance.label or "entrance"
                out[candidate.key] = build_handoff(
                    lat=entrance.lat,
                    lon=entrance.lon,
                    label=candidate.name,
                    is_entrance=True,
                    point_description=f"routing to the {where}",
                    origin_lat=origin_lat,
                    origin_lon=origin_lon,
                )
                continue

            out[candidate.key] = build_handoff(
                lat=candidate.lat,
                lon=candidate.lon,
                label=candidate.name,
                is_entrance=False,
                # Said plainly, because a driver arriving at a centroid needs to know they
                # are looking for a way in rather than standing at it.
                point_description="no entrance recorded; routing to the car park itself",
                origin_lat=origin_lat,
                origin_lon=origin_lon,
            )
            continue

        # A bay is a surveyed polygon. Its point is the destination, exactly.
        out[candidate.key] = build_handoff(
            lat=candidate.lat,
            lon=candidate.lon,
            label=candidate.name,
            is_entrance=False,
            point_description="exact surveyed bay location",
            origin_lat=origin_lat,
            origin_lon=origin_lon,
        )

    return out
