"""Showing the driver a live view near the space we are recommending.

The honest version of this feature is narrower than the obvious one, and the narrowing is
the whole design.

**A camera near a bay is not a camera of that bay.** The four public feeds this project can
open are wide pan-tilt-zoom views of squares and forecourts. One of them may happen to
include the bay being recommended, and it may equally be pointed the other way, because the
operator moves it. So nothing here claims the footage shows the space. It says how far the
camera is from it and lets the driver look.

**Distance is the gate.** A camera two kilometres away tells a driver nothing about a
street they are about to drive to, and showing it would be decoration pretending to be
evidence. Beyond :data:`MAX_USEFUL_DISTANCE_M` no view is offered at all.

**This is never evidence.** The occupancy pipeline publishes camera observations through
``AvailabilityObservation`` with ``CAMERA_OBSERVATION`` evidence, and that path is
calibrated, gated on permission, and measured. This module is a convenience: a window onto
a street. It does not feed ranking, it does not change a confidence label, and a result is
ranked exactly the same whether or not a camera happens to be nearby.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from parkfit.geo.rd import haversine_m
from parkfit.storage.models import CameraSource

log = logging.getLogger(__name__)

#: Past this, a public camera says nothing useful about the street in question.
MAX_USEFUL_DISTANCE_M = 400.0

#: Within this, the camera is close enough that it plausibly overlooks the space. Still
#: only "plausibly": these are steerable cameras and the operator points them where they
#: like.
CLOSE_ENOUGH_M = 120.0


@dataclass(frozen=True)
class CameraView:
    """A live view a driver can open, with an honest description of what it is."""

    camera_id: str
    name: str
    operator: str
    lat: float
    lon: float
    distance_m: float
    #: An embeddable player URL, or empty when the feed cannot be embedded.
    embed_url: str
    #: The page a person can open to watch it themselves.
    page_url: str
    #: What this view actually is, in words, for display next to the player.
    relationship: str

    @property
    def overlooks_plausibly(self) -> bool:
        return self.distance_m <= CLOSE_ENOUGH_M


def _youtube_id(stream_url: str) -> str:
    """Pull the video id out of a watch URL. Empty when it is not a YouTube feed."""
    marker = "watch?v="
    at = stream_url.find(marker)
    if at < 0:
        return ""
    tail = stream_url[at + len(marker) :]
    for separator in ("&", "#", "?"):
        cut = tail.find(separator)
        if cut >= 0:
            tail = tail[:cut]
    return tail


def _openable_cameras(session: Session) -> list[CameraSource]:
    """Cameras that actually have a stream, which is a tiny subset of the registry.

    The registry holds twelve thousand mapped camera *locations* from OpenStreetMap. Almost
    none of them are feeds anyone may open, so this filters to the ones with a URL rather
    than pretending the map is a network.
    """
    rows = (
        session.execute(
            select(CameraSource).where(
                CameraSource.stream_url.is_not(None),
                CameraSource.stream_url != "",
                CameraSource.lat.is_not(None),
                CameraSource.lon.is_not(None),
            )
        )
        .scalars()
        .all()
    )
    return [row for row in rows if _youtube_id(row.stream_url or "")]


def nearest_views(
    session: Session,
    targets: list[tuple[tuple[str, int], float, float]],
    *,
    max_distance_m: float = MAX_USEFUL_DISTANCE_M,
) -> dict[tuple[str, int], CameraView]:
    """Nearest openable camera per target, when one is close enough to be worth showing.

    Takes ``(key, lat, lon)`` triples and returns a mapping keyed the same way. The camera
    list is loaded once for the whole page rather than per result, because it is a handful
    of rows and a query per candidate would cost more than the search does.
    """
    cameras = _openable_cameras(session)
    if not cameras or not targets:
        return {}

    out: dict[tuple[str, int], CameraView] = {}
    for key, lat, lon in targets:
        best: CameraSource | None = None
        best_distance = float("inf")
        for camera in cameras:
            distance = haversine_m(lat, lon, camera.lat, camera.lon)
            if distance < best_distance:
                best, best_distance = camera, distance

        if best is None or best_distance > max_distance_m:
            continue

        video_id = _youtube_id(best.stream_url or "")
        close = best_distance <= CLOSE_ENOUGH_M
        out[key] = CameraView(
            camera_id=best.camera_id,
            name=(best.notes or best.camera_id).split(".")[0][:80],
            operator=best.operator or best.owner or "unknown",
            lat=best.lat,
            lon=best.lon,
            distance_m=round(best_distance, 1),
            # Embedded muted and without autoplay: a page that starts making noise, or
            # pulls a 4K stream nobody asked for, is a page people close.
            embed_url=f"https://www.youtube-nocookie.com/embed/{video_id}?autoplay=0&mute=1",
            page_url=best.stream_url or "",
            relationship=(
                f"public camera {best_distance:.0f} m away; it may or may not be pointed at "
                "this space"
                if close
                else f"nearest public camera, {best_distance:.0f} m away, showing the area "
                "rather than this space"
            ),
        )

    return out
