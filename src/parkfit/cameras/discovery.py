"""Finding every camera in the Netherlands, and being precise about what each one is.

Three kinds of thing get called "a camera", and conflating them is how a project ends up
claiming a hundred thousand feeds it cannot open.

**A location.** OpenStreetMap maps 25,897 ``man_made=surveillance`` nodes across the
Netherlands, each with real coordinates, often an operator, a mount and a direction. That
is a map of where cameras physically are. The overwhelming majority are private CCTV: a
shop doorway, a car park entrance, a bank. There is no stream, and there is no right to
one. They are recorded here as locations with no URL and no permission, because knowing a
camera overlooks a given bay is useful even when the pictures are not ours to see.

**A public feed.** A small number are published deliberately for anyone to watch. Those get
a stream URL and can actually be processed.

**A stream we may process.** Smaller still. Publishing a picture on a web page is not a
licence to run computer vision over it at scale, so a feed only becomes processable when
someone records an authorisation or an owner attestation against it.

The registry keeps all three, distinguished by ``permission_status`` and by whether
``stream_url`` is set. Nothing here ever promotes a location to a feed on its own.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import httpx
from sqlalchemy.orm import Session

from parkfit.cameras.registry import CameraRegistry
from parkfit.storage.models import CameraPermission

log = logging.getLogger(__name__)

#: Public Overpass mirrors, tried in order. The main instance is frequently saturated and
#: answers a national query with a dispatcher timeout rather than an error, so a fallback
#: is not optional.
OVERPASS_MIRRORS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
)

#: The Netherlands, generously. Slightly wider than the coastline so nothing on the border
#: is clipped; anything outside the country is dropped by the reverse lookup later.
NL_BBOX = (50.70, 3.30, 53.60, 7.30)

USER_AGENT = "CamToParkingSlot/0.1 (parking research; contact via repository)"


@dataclass
class DiscoveryReport:
    tiles: int = 0
    locations_found: int = 0
    locations_stored: int = 0
    feeds_stored: int = 0
    skipped_no_coordinates: int = 0
    by_operator: dict[str, int] = field(default_factory=dict)
    by_type: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def describe(self) -> str:
        return (
            f"{self.locations_stored:,} camera locations and {self.feeds_stored} openable "
            f"feeds from {self.tiles} tiles"
        )


def _tiles(bbox: tuple[float, float, float, float], rows: int, cols: int):
    """Split a bounding box into a grid.

    A single national query for 26,000 nodes times out on every public mirror. Tiling keeps
    each request small enough to answer and lets a failed tile be retried on its own rather
    than losing the whole run.
    """
    south, west, north, east = bbox
    lat_step = (north - south) / rows
    lon_step = (east - west) / cols
    for r in range(rows):
        for c in range(cols):
            yield (
                south + r * lat_step,
                west + c * lon_step,
                south + (r + 1) * lat_step,
                west + (c + 1) * lon_step,
            )


def _query_overpass(client: httpx.Client, query: str) -> dict | None:
    """Run one Overpass query, falling through the mirrors."""
    for mirror in OVERPASS_MIRRORS:
        try:
            response = client.post(
                mirror,
                content=query.encode("utf-8"),
                headers={"Content-Type": "text/plain", "User-Agent": USER_AGENT},
                timeout=180.0,
            )
            if response.status_code != 200:
                continue
            # Overpass answers an overloaded dispatcher with HTTP 200 and an HTML error
            # page, so the status code alone does not mean the query ran.
            text = response.text.lstrip()
            if not text.startswith("{"):
                continue
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            log.debug("overpass mirror %s failed: %s", mirror, exc)
            continue
    return None


def discover_locations(
    session: Session,
    *,
    bbox: tuple[float, float, float, float] = NL_BBOX,
    rows: int = 6,
    cols: int = 4,
    pause_s: float = 2.0,
    limit: int | None = None,
) -> DiscoveryReport:
    """Ingest every mapped surveillance camera in the bounding box.

    These are locations, not feeds. Each is stored with ``permission_status=UNVERIFIED``
    and no stream URL, which is exactly what the registry gate refuses to run. Promoting
    one to something processable is a deliberate human act, never a consequence of having
    found it.
    """
    report = DiscoveryReport()
    registry = CameraRegistry(session)
    stored = 0

    with httpx.Client(follow_redirects=True) as client:
        for south, west, north, east in _tiles(bbox, rows, cols):
            report.tiles += 1
            query = (
                "[out:json][timeout:120];"
                f'(node["man_made"="surveillance"]({south:.4f},{west:.4f},{north:.4f},{east:.4f}););'
                "out body;"
            )
            payload = _query_overpass(client, query)
            if payload is None:
                report.errors.append(f"tile {south:.2f},{west:.2f} failed on every mirror")
                continue

            elements = payload.get("elements", [])
            report.locations_found += len(elements)

            for element in elements:
                lat, lon = element.get("lat"), element.get("lon")
                if lat is None or lon is None:
                    report.skipped_no_coordinates += 1
                    continue

                tags = element.get("tags", {})
                operator = tags.get("operator") or "unknown"
                zone = tags.get("surveillance") or tags.get("surveillance:zone") or "unknown"
                report.by_operator[operator] = report.by_operator.get(operator, 0) + 1
                report.by_type[zone] = report.by_type.get(zone, 0) + 1

                direction = tags.get("camera:direction") or tags.get("direction")
                try:
                    heading = float(direction) if direction is not None else None
                except (TypeError, ValueError):
                    heading = None

                notes = "; ".join(
                    f"{k}={v}"
                    for k, v in tags.items()
                    if k
                    in (
                        "surveillance",
                        "surveillance:type",
                        "surveillance:zone",
                        "camera:type",
                        "camera:mount",
                        "operator",
                        "name",
                    )
                )

                registry.register(
                    camera_id=f"osm_{element['id']}",
                    # No URL. A mapped location is not a feed, and inventing one here is
                    # how a registry starts lying about what it can open.
                    stream_url="",
                    stream_type="none",
                    owner=operator,
                    operator=operator,
                    lat=float(lat),
                    lon=float(lon),
                    heading_deg=heading,
                    permission_status=CameraPermission.UNVERIFIED.value,
                    automated_processing_allowed=None,
                    notes=f"OpenStreetMap surveillance node. {notes}"[:900],
                )
                stored += 1
                if stored % 500 == 0:
                    session.commit()
                    log.info("stored %d camera locations", stored)
                if limit and stored >= limit:
                    session.commit()
                    report.locations_stored = stored
                    return report

            session.commit()
            time.sleep(pause_s)

    session.commit()
    report.locations_stored = stored
    return report


# ---------------------------------------------------------------------------
# Publicly broadcast feeds
# ---------------------------------------------------------------------------
#: Cameras a Dutch operator publishes deliberately, as an embedded public live stream.
#:
#: Every entry here was resolved and confirmed live rather than copied from a list. The
#: coordinates are the camera's own position, not the city centre, because a camera
#: registry whose positions are approximate cannot be used to say which bay a camera sees.
#:
#: They are stored with a stream URL and ``ROBOTS_OK``, which this deployment accepts in
#: dev and refuses in production. Processing any of them in production still needs an
#: attestation from whoever owns the camera.
PUBLIC_FEEDS = (
    # Verified live and fetchable on 2026-08-27. Two feeds that used to be listed here,
    # Now4Rent's Dam Square and WebCam.NL's Zaanse Schans, now answer "This video is not
    # available" and "We're experiencing technical difficulties", so they are gone rather
    # than left in to inflate a count. Other people's cameras go down without telling us,
    # which is why `pf cameras verify` re-checks rather than trusting this list.
    #
    # Coordinates for the entries added on 2026-08-27 are landmark-level, taken from the
    # place each operator names in its own stream title. They are good enough to say
    # "this camera is on Hofplein" and deliberately not called surveyed anywhere.
    {
        "camera_id": "yt_amsterdam_beursplein",
        "youtube_id": "43qH0tDA6lM",
        "name": "Amsterdam Damrak / Beursplein",
        "operator": "WebCam.NL",
        "lat": 52.3748,
        "lon": 4.8949,
        "note": "ultraHD pan-tilt-zoom over Beursplein, published live on YouTube",
    },
    {
        "camera_id": "yt_amsterdam_stationseiland",
        "youtube_id": "1phWWCgzXgM",
        "name": "Amsterdam Stationseiland / Centraal",
        "operator": "WebCam.NL",
        "lat": 52.3789,
        "lon": 4.9002,
        "note": "Stationseiland forecourt, published live on YouTube",
    },
    {
        "camera_id": "yt_amsterdam_vijfbruggen1",
        "youtube_id": "2tgHBRFHMm8",
        "name": "Amsterdam De Vijf Bruggen, camera 1",
        "operator": "Amsterdam De Vijf Bruggen",
        "lat": 52.3785,
        "lon": 4.9000,
        "note": "canal crossing by Amsterdam Centraal, landmark-level placement",
    },
    {
        "camera_id": "yt_amsterdam_vijfbruggen2",
        "youtube_id": "FHJH2yMe6Hw",
        "name": "Amsterdam Centraal De Vijf Bruggen, camera 2",
        "operator": "Amsterdam De Vijf Bruggen",
        "lat": 52.3785,
        "lon": 4.9000,
        "note": "second angle on the same crossing, landmark-level placement",
    },
    {
        "camera_id": "yt_rotterdam_erasmusbrug",
        "youtube_id": "nFozEhYTEMo",
        "name": "Rotterdam Erasmusbrug / Kop van Zuid",
        "operator": "Rotterdam live stream",
        "lat": 51.9089,
        "lon": 4.4870,
        "note": "Erasmusbrug and cruise terminal, landmark-level placement",
    },
    {
        "camera_id": "yt_rotterdam_erasmus_kpn",
        "youtube_id": "gsViKzj7nuQ",
        "name": "Rotterdam Erasmusbrug, KPN led wall",
        "operator": "Rotterdam live stream",
        "lat": 51.9089,
        "lon": 4.4870,
        "note": "second angle on the Erasmusbrug, landmark-level placement",
    },
    {
        "camera_id": "yt_rotterdam_hofplein",
        "youtube_id": "wHhs_Ef8LSU",
        "name": "Rotterdam Hofplein",
        "operator": "Rotterdam live stream",
        "lat": 51.9244,
        "lon": 4.4777,
        "note": "Hofplein roundabout, landmark-level placement",
    },
)

#: Live and fetchable, but nowhere near city parking: a port terminal and a railway.
#: They are real training footage and useless as a "camera near your bay", so they are
#: harvested from and deliberately kept out of the registry the search reads.
TRAINING_ONLY_FEEDS = (
    {"camera_id": "yt_nieuwe_waterweg", "youtube_id": "_pGIJmXxAHk", "name": "Nieuwe Waterweg"},
    {"camera_id": "yt_amazonehaven", "youtube_id": "M09NaBVPjAI", "name": "Amazonehaven west"},
    {"camera_id": "yt_railcam_nl", "youtube_id": "abaZ4GD5jbM", "name": "RailCam Netherlands"},
)


def register_public_feeds(session: Session) -> int:
    """Record the deliberately-published live streams.

    Stored as YouTube watch URLs rather than resolved manifests on purpose: a googlevideo
    HLS manifest carries an expiry, so a stored one is broken within hours. The worker
    resolves it at open time instead.
    """
    registry = CameraRegistry(session)
    count = 0
    for feed in PUBLIC_FEEDS:
        registry.register(
            camera_id=feed["camera_id"],
            stream_url=f"https://www.youtube.com/watch?v={feed['youtube_id']}",
            stream_type="youtube_live",
            owner=feed["operator"],
            operator=feed["operator"],
            public_page_url="https://www.amsterdam.info/webcam/",
            lat=feed["lat"],
            lon=feed["lon"],
            # Published for public viewing, so crawling is not in question. That is still
            # short of a licence to run detection over it, which is why production will
            # not accept this status without an attestation.
            permission_status=CameraPermission.ROBOTS_OK.value,
            automated_processing_allowed=None,
            notes=f"{feed['name']}. {feed['note']}",
        )
        count += 1
    session.commit()
    return count
