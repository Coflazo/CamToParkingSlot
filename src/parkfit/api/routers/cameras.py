"""The public cameras, so a driver can look at the street before driving to it.

Every camera the vision pipeline is allowed to read is a camera the user is allowed to
watch. That is not a courtesy feature: a system that says "this bay is free" on the
strength of a camera should be willing to show you the camera, and if it will not, the
claim is worth less.

Only feeds whose operator publishes them ever reach this endpoint. The twelve thousand
mapped camera locations that arrive from OpenStreetMap carry no stream and permission
``UNVERIFIED``, and the registry gate refuses to run them, so they are not offered here
either. What is listed is exactly what anybody could already open in a browser tab.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from pydantic import BaseModel

from parkfit.cameras.discovery import PUBLIC_FEEDS
from parkfit.services import camera_analysis

log = logging.getLogger(__name__)
router = APIRouter(prefix="/cameras", tags=["cameras"])


class PublicCamera(BaseModel):
    """One watchable feed, with an honest description of what its position means."""

    camera_id: str
    name: str
    operator: str
    lat: float
    lon: float
    #: Player URL for embedding. nocookie, muted, and never autoplaying: a page that
    #: starts making noise or pulls a 4K stream nobody asked for is a page people close.
    embed_url: str
    #: The operator's own page, so the feed can be opened outside this product.
    watch_url: str
    note: str
    #: Free spaces the last analysis of this feed found, or -1 if it has not run yet.
    #: Never triggers one, so listing the cameras stays instant.
    free_spaces_seen: int = -1


class CameraList(BaseModel):
    cameras: list[PublicCamera]
    count: int
    #: Said plainly because the map cannot say it: these are wide pan-tilt-zoom views of
    #: streets and squares, and the operator moves them. A camera near a bay is not a
    #: camera of that bay.
    disclaimer: str


_DISCLAIMER = (
    "These are wide pan-tilt-zoom views published by their operators. The operator "
    "points them, so a camera near a parking space is not necessarily showing it."
)


def _to_model(feed: dict) -> PublicCamera:
    video_id = feed["youtube_id"]
    return PublicCamera(
        camera_id=feed["camera_id"],
        name=feed["name"],
        operator=feed["operator"],
        lat=feed["lat"],
        lon=feed["lon"],
        embed_url=f"https://www.youtube-nocookie.com/embed/{video_id}?autoplay=0&mute=1",
        watch_url=f"https://www.youtube.com/watch?v={video_id}",
        note=feed["note"],
    )


class DetectedBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    score: float


class FreeSpaceBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    #: Estimated, from a scale derived from the detected cars. Never a measurement.
    length_m: float
    depth_m: float
    fits: list[str]


class CameraAnalysisResponse(BaseModel):
    """What the vision pipeline currently sees through one camera."""

    camera_id: str
    ok: bool
    reason: str = ""
    #: Seconds since this frame was analysed. Shown to the reader, because a picture of a
    #: street is worth very different amounts at two seconds and at two minutes.
    age_seconds: float = -1.0
    frame_width: int = 0
    frame_height: int = 0
    frame_data_uri: str = ""
    vehicles: list[DetectedBox] = []
    free_spaces: list[FreeSpaceBox] = []
    pixels_per_metre: float = 0.0
    scale_confident: bool = False
    note: str = ""


@router.get("", response_model=CameraList)
async def list_cameras() -> CameraList:
    """Every camera a user may watch, for plotting on the map."""
    cameras = [_to_model(feed) for feed in PUBLIC_FEEDS]
    for camera in cameras:
        camera.free_spaces_seen = camera_analysis.cached_spot_count(camera.camera_id)
    return CameraList(cameras=cameras, count=len(cameras), disclaimer=_DISCLAIMER)


@router.get("/{camera_id}/analysis", response_model=CameraAnalysisResponse)
async def camera_analysis_endpoint(camera_id: str) -> CameraAnalysisResponse:
    """Where a car would fit in what this camera can see right now.

    Asking marks the camera as watched, so the background watcher keeps refreshing it for
    the next minute and the next request is served from a reading a second or two old
    rather than waiting for a fresh grab.

    The work runs in a thread. It is a subprocess, a socket read and a forward pass, all
    blocking, and doing that on the event loop would stall every other request.
    """
    import anyio

    camera_analysis.mark_interest(camera_id)
    analysis = await anyio.to_thread.run_sync(camera_analysis.analyse, camera_id)

    return CameraAnalysisResponse(
        camera_id=analysis.camera_id,
        ok=analysis.ok,
        reason=analysis.reason,
        age_seconds=camera_analysis.age_seconds(camera_id),
        frame_width=analysis.frame_width,
        frame_height=analysis.frame_height,
        frame_data_uri=analysis.frame_data_uri,
        vehicles=[
            DetectedBox(x1=v.x1, y1=v.y1, x2=v.x2, y2=v.y2, label=v.label, score=round(v.score, 3))
            for v in analysis.vehicles
        ],
        free_spaces=[
            FreeSpaceBox(
                x1=f.x1,
                y1=f.y1,
                x2=f.x2,
                y2=f.y2,
                length_m=f.length_m,
                depth_m=f.depth_m,
                fits=f.fits,
            )
            for f in analysis.free_spaces
        ],
        pixels_per_metre=analysis.pixels_per_metre,
        scale_confident=analysis.scale_confident,
        note=analysis.note,
    )
