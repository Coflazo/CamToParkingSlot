"""Pulling real frames off the live public cameras.

This exists because the detector was trained on rendered boxes and fell over the first
time it saw a real street. Rendered scenes give exact ground truth, which is what makes
gap-length error measurable, but nothing in them teaches a model what a wet tram rail or
a tree shadow looks like. The only cure is real frames, and these four cameras are the
ones whose operators publish them.

**Transport is deliberately split.** yt-dlp resolves the manifest, urllib fetches the
playlist and the segments, and ffmpeg only ever opens a local file. ffmpeg cannot
complete a TLS handshake against googlevideo from here (it fails with "Error in the pull
function" before any HTTP happens) while urllib and curl both manage it fine, so handing
ffmpeg a URL fails and handing it bytes on disk works. Splitting it this way also means
one flaky segment is a retry rather than a dead ffmpeg process.

Frames land on disk here, unlike the production vision worker, which holds them in RAM
and discards them. That difference is deliberate and narrow: a training set has to be
inspectable, and a public square at 720p contains no more than a passer-by already sees.
No face or plate recognition is run on them at any point.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

#: 1280x720 video-only. The audio-muxed selectors YouTube offers for a live broadcast do
#: not exist for these feeds, which is why a plain "best" selector fails here.
DEFAULT_ITAG = "232"

#: A live playlist only ever lists the last handful of segments, so asking for frames
#: faster than the stream produces them returns the same picture twice.
SEGMENT_SECONDS = 5.0

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) CamToParkingSlot/0.1 (research)"


@dataclass(frozen=True)
class LiveCamera:
    """A feed whose operator publishes it, with the location its frames depict."""

    camera_id: str
    youtube_id: str
    name: str
    lat: float
    lon: float

    @property
    def watch_url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.youtube_id}"


def _run(args: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)


def resolve_manifest(camera: LiveCamera, *, itag: str = DEFAULT_ITAG, timeout: float = 90.0) -> str:
    """Ask yt-dlp for the HLS manifest URL. Empty string when the feed is not live."""
    exe = shutil.which("yt-dlp")
    args = (
        [exe, "-g", "-f", itag, camera.watch_url]
        if exe
        else ["python", "-m", "yt_dlp", "-g", "-f", itag, camera.watch_url]
    )
    try:
        done = _run(args, timeout)
    except subprocess.TimeoutExpired:
        log.warning("%s: yt-dlp timed out resolving the manifest", camera.camera_id)
        return ""
    url = done.stdout.strip().splitlines()[-1] if done.stdout.strip() else ""
    if not url.startswith("http"):
        log.warning("%s: no manifest (%s)", camera.camera_id, (done.stderr or "").strip()[:120])
        return ""
    return url


def _get(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def segment_urls(manifest_url: str, *, timeout: float = 45.0) -> list[str]:
    """The segment URLs a live HLS playlist is currently advertising."""
    try:
        body = _get(manifest_url, timeout).decode("utf-8", "replace")
    except Exception as exc:
        log.warning("playlist fetch failed: %s", exc)
        return []
    return [line.strip() for line in body.splitlines() if line.strip().startswith("http")]


def _decode_frame(segment: Path, destination: Path) -> bool:
    """Pull a single frame out of a local transport-stream chunk."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        log.error("ffmpeg is not on PATH, cannot decode frames")
        return False
    done = _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(segment),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-y",
            str(destination),
        ],
        timeout=120.0,
    )
    if done.returncode != 0 or not destination.exists():
        log.warning("decode failed for %s: %s", segment.name, (done.stderr or "").strip()[:120])
        return False
    return True


def harvest(
    camera: LiveCamera,
    out_dir: Path,
    *,
    frames: int = 12,
    spacing_seconds: float = SEGMENT_SECONDS,
) -> list[Path]:
    """Grab ``frames`` real frames from one live camera, spread out in time.

    Spread matters more than count. Twelve frames from twelve consecutive seconds are
    twelve pictures of the same parked cars, which teaches a detector almost nothing; the
    same twelve spread across an hour catch different traffic, different occlusion and a
    different sun angle.

    Two feeds behave quite differently. Some advertise only the last handful of segments,
    so the only way to get spread is to wait for the stream to produce it. Others carry a
    DVR window of several hundred segments, which is already an hour of footage sitting
    there, and sampling evenly across it gets the same spread immediately. The DVR case is
    both faster and better data, so it is preferred whenever the playlist offers it.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = resolve_manifest(camera)
    if not manifest:
        return []

    written: list[Path] = []
    scratch = out_dir / f".{camera.camera_id}.ts"
    stamp = int(time.time())

    def take(url: str) -> bool:
        try:
            scratch.write_bytes(_get(url, 60.0))
        except Exception as exc:
            log.warning("%s: segment fetch failed: %s", camera.camera_id, exc)
            return False
        target = out_dir / f"{camera.camera_id}_{stamp}_{len(written):03d}.jpg"
        if _decode_frame(scratch, target):
            written.append(target)
            log.info("%s: %d/%d", camera.camera_id, len(written), frames)
            return True
        return False

    available = segment_urls(manifest)
    if len(available) >= frames * 2:
        # DVR window: sample evenly across everything on offer rather than crawling the
        # live edge one segment at a time.
        step = len(available) / float(frames)
        picked = [available[min(len(available) - 1, int(i * step))] for i in range(frames)]
        log.info(
            "%s: DVR playlist with %d segments, sampling %d evenly",
            camera.camera_id,
            len(available),
            frames,
        )
        for url in picked:
            take(url)
        scratch.unlink(missing_ok=True)
        return written

    # Live edge only: the wait is the mechanism, not a fallback.
    seen: set[str] = set()
    while len(written) < frames:
        urls = [u for u in segment_urls(manifest) if u not in seen]
        if not urls:
            time.sleep(spacing_seconds)
            urls = [u for u in segment_urls(manifest) if u not in seen]
            if not urls:
                log.warning("%s: playlist stopped advertising new segments", camera.camera_id)
                break
        url = urls[-1]
        seen.add(url)
        take(url)
        if len(written) < frames:
            time.sleep(spacing_seconds)

    scratch.unlink(missing_ok=True)
    return written


def live_cameras() -> list[LiveCamera]:
    """The published feeds, taken from the registry's own list so the two cannot drift."""
    from parkfit.cameras.discovery import PUBLIC_FEEDS

    return [
        LiveCamera(
            camera_id=feed["camera_id"],
            youtube_id=feed["youtube_id"],
            name=feed["name"],
            lat=feed["lat"],
            lon=feed["lon"],
        )
        for feed in PUBLIC_FEEDS
    ]


def harvest_all(
    out_dir: Path, *, frames_per_camera: int = 12, spacing_seconds: float = SEGMENT_SECONDS
) -> dict[str, list[Path]]:
    """Harvest from every published feed. A camera that is offline is skipped, not fatal."""
    result: dict[str, list[Path]] = {}
    for camera in live_cameras():
        log.info("harvesting %s (%s)", camera.name, camera.camera_id)
        got = harvest(camera, out_dir, frames=frames_per_camera, spacing_seconds=spacing_seconds)
        result[camera.camera_id] = got
        if not got:
            log.warning("%s produced no frames", camera.camera_id)
    return result
