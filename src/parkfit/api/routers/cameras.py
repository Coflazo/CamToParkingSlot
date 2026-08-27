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


@router.get("", response_model=CameraList)
async def list_cameras() -> CameraList:
    """Every camera a user may watch, for plotting on the map."""
    cameras = [_to_model(feed) for feed in PUBLIC_FEEDS]
    return CameraList(cameras=cameras, count=len(cameras), disclaimer=_DISCLAIMER)
