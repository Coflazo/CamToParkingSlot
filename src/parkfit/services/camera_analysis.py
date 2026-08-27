"""Looking at a live camera and marking where a car would fit.

This runs when somebody opens a camera in the interface, which is now and then rather than
once per frame, so it lives in Python. The per-frame path is the C++ occupancy classifier.

**What is measured, what is assumed, and what is not available.** The distinction is the
whole design, because a green box drawn on a street is a claim and it should be one the
project can stand behind.

Measured. The vehicle boxes come from a COCO-pretrained Faster R-CNN run on a frame pulled
from that camera seconds earlier. Those are real detections on a real street, and the gaps
between them are real distances in the image.

Assumed. Turning a gap in pixels into a gap in metres needs a scale, and a public webcam
carries no survey: nobody has recorded which pixel is which point on the ground. So the
scale comes from the cars themselves. A passenger car is about 1.80 m across the bodywork,
the median of the fourteen real RDW vehicles in this project, so the median detected car
width in pixels divides into that. Every metric number here inherits that assumption and is
labelled an estimate.

Not available at all. Height cannot be recovered from one uncalibrated view of a ground
plane, so this never reports one. Where a height limit matters it comes from the parking
register and the signage.

The honest summary is that this answers "roughly how much kerb is free here, and where"
and does not answer "will your car fit". The fit engine answers the second, from surveyed
bay polygons, in centimetres.
"""

from __future__ import annotations

import base64
import itertools
import logging
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

#: Median bodywork width of the RDW test fleet, in metres. The scale anchor.
TYPICAL_CAR_WIDTH_M = 1.80

#: Shorter than this and the smallest car in the fleet, a Fiat 500 at 3.63 m, has no room
#: to get into it.
MIN_USEFUL_GAP_M = 4.2

#: Longer than this and it is almost certainly not a parking space.
#:
#: Three car lengths of continuous empty kerb on a city street usually means the kerb is
#: not for parking: a bus stop, a crossing, a loading bay, or simply the carriageway. The
#: camera cannot tell the difference, because whether parking is permitted lives in the
#: sign code and the time regime rather than in the picture. On a Groningen frame this
#: cap is what stopped 37.6 m of open road being offered as a space.
MAX_CREDIBLE_GAP_M = 15.0

#: A parked car occupies at least this much depth in the picture, in estimated metres.
#:
#: The depth comes from how tall the flanking cars are in the frame, so a very oblique or
#: very distant row produces boxes a few pixels high and a "space" half a metre deep. That
#: is not a parking space, it is the algorithm reading a row it cannot really see, and a
#: floor here is cheaper than trying to recover the camera angle.
MIN_CREDIBLE_DEPTH_M = 1.2

#: How long an analysis is served before it counts as stale.
#:
#: A parking space can be taken in the time it takes to read a sentence, so this is
#: deliberately short. It is not shorter because these feeds emit one HLS segment every
#: five seconds, and refreshing faster than the camera produces pictures would spend the
#: operator's bandwidth re-analysing a frame already seen. Measured here a full refresh is
#: about 0.85 s: a 0.25 s frame grab and a 0.6 s forward pass.
CACHE_SECONDS = 5.0

#: A camera nobody has looked at recently does not need refreshing at all. Opening one in
#: the interface marks it live for this long, and the watcher spends its time only there.
INTEREST_SECONDS = 60.0

#: Below this the teacher is guessing, and a phantom car invents a gap on each side of
#: itself.
SCORE_THRESHOLD = 0.60

#: Only a gap with one of these parked on each side is treated as a parking space.
#:
#: A bicycle leaning against a wall and a car twenty metres away have a large empty space
#: between them, and it is a pavement. Two parked cars with a hole between them is what a
#: parking space actually looks like from a camera, and requiring both flanks to be a
#: motor vehicle is the cheapest way to say so.
FLANKING_CLASSES = frozenset({"car", "van", "truck"})

#: Three cars is where a median starts to mean something. Fewer and the scale is still
#: returned, because a rough number beats none, but it is not called confident.
MIN_CARS_FOR_CONFIDENT_SCALE = 3


@dataclass(frozen=True)
class DetectedVehicle:
    """One vehicle the teacher found, in frame pixels."""

    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    score: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def centre_y(self) -> float:
        return (self.y1 + self.y2) / 2.0


@dataclass(frozen=True)
class FreeSpace:
    """A stretch of kerb with nothing on it, in frame pixels and estimated metres."""

    x1: float
    y1: float
    x2: float
    y2: float
    length_m: float
    depth_m: float
    #: Which fleet vehicles clear it on length. Named rather than counted, because
    #: "a Fiat 500 but not a Sprinter" is the useful shape of the answer.
    fits: list[str] = field(default_factory=list)


@dataclass
class CameraAnalysis:
    camera_id: str
    ok: bool = False
    reason: str = ""
    captured_at: float = 0.0
    frame_width: int = 0
    frame_height: int = 0
    #: The frame as a data URI, so the interface can draw on it without a second request
    #: and without this service needing anywhere to host files.
    frame_data_uri: str = ""
    vehicles: list[DetectedVehicle] = field(default_factory=list)
    free_spaces: list[FreeSpace] = field(default_factory=list)
    pixels_per_metre: float = 0.0
    scale_confident: bool = False
    note: str = ""


_cache: dict[str, tuple[float, CameraAnalysis]] = {}


def _fleet_lengths() -> list[tuple[str, float]]:
    """Fleet vehicles by length in metres, shortest first."""
    from parkfit.domain import presets

    return sorted(
        ((v.label, v.length_cm / 100.0) for v in presets.BY_KEY.values()),
        key=lambda pair: pair[1],
    )


def estimate_scale(vehicles: list[DetectedVehicle]) -> tuple[float, bool]:
    """Pixels per metre, from the median detected car width.

    Median rather than mean. One lorry, or one box that swallowed two cars at once, drags
    a mean far enough to make every distance wrong, and a median simply ignores it.
    """
    widths = sorted(v.width for v in vehicles if v.label == "car" and v.width > 4.0)
    if not widths:
        return 0.0, False
    middle = widths[len(widths) // 2]
    if middle <= 0.0:
        return 0.0, False
    return middle / TYPICAL_CAR_WIDTH_M, len(widths) >= MIN_CARS_FOR_CONFIDENT_SCALE


def kerb_band(vehicles: list[DetectedVehicle]) -> tuple[float, float] | None:
    """The horizontal band the parked cars occupy.

    Cars at one kerb sit at roughly one depth in the image, so their vertical centres
    cluster. Taking the median and a tolerance from the cars' own heights keeps a car
    crossing the far side of the square out of the row being measured, which would
    otherwise produce a gap spanning the whole picture.
    """
    if len(vehicles) < 2:
        return None
    centres = sorted(v.centre_y for v in vehicles)
    median = centres[len(centres) // 2]
    heights = sorted(v.y2 - v.y1 for v in vehicles)
    tolerance = max(12.0, heights[len(heights) // 2])
    return median - tolerance, median + tolerance


def find_free_spaces(
    vehicles: list[DetectedVehicle], pixels_per_metre: float, frame_width: int
) -> list[FreeSpace]:
    """Gaps between consecutive vehicles along the kerb, left to right.

    Built for parallel kerb parking, which is what a street camera usually looks at. A
    perpendicular car park breaks the one-depth assumption the band relies on, so every
    candidate is checked against the full detection list and thrown away if any vehicle
    is standing inside it. That guard is what stops a row of parked cars being reported
    as one long empty space.
    """
    if pixels_per_metre <= 0.0:
        return []

    band = kerb_band(vehicles)
    if band is None:
        return []
    low, high = band

    row = sorted(
        (v for v in vehicles if low <= v.centre_y <= high and v.label in FLANKING_CLASSES),
        key=lambda v: v.x1,
    )
    if len(row) < 2:
        return []

    fleet = _fleet_lengths()
    spaces: list[FreeSpace] = []

    for left, right in itertools.pairwise(row):
        gap_px = right.x1 - left.x2
        if gap_px <= 0:
            continue
        length_m = gap_px / pixels_per_metre
        if not (MIN_USEFUL_GAP_M <= length_m <= MAX_CREDIBLE_GAP_M):
            continue

        # The box spans the gap and takes its vertical extent from the cars either side,
        # which is the depth a car parked there would occupy.
        y1 = min(left.y1, right.y1)
        y2 = max(left.y2, right.y2)

        # Nothing may be standing in it. The band above assumes cars parked along a kerb
        # at one depth, and in a perpendicular car park that assumption breaks: cars sit
        # at many depths, the band keeps a scattered subset, and the "gap" between two of
        # them runs straight over the cars in between. On a real Kijkduin frame that
        # produced a confident 22.4 m space across two dozen parked cars. Checking the
        # whole detection list rather than the band is the point, because the vehicles
        # standing in the way are exactly the ones the band discarded.
        if any(
            other is not left
            and other is not right
            and other.x2 > space_x1
            and other.x1 < space_x2
            and other.y2 > y1
            and other.y1 < y2
            for other in vehicles
            for space_x1, space_x2 in ((left.x2, right.x1),)
        ):
            continue

        depth_m = (y2 - y1) / pixels_per_metre
        if depth_m < MIN_CREDIBLE_DEPTH_M:
            continue

        spaces.append(
            FreeSpace(
                x1=left.x2,
                y1=y1,
                x2=right.x1,
                y2=y2,
                length_m=round(length_m, 1),
                depth_m=round(depth_m, 1),
                # 60 cm on top of the car's own length, which is roughly what it takes to
                # get into a parallel space without a dozen shuffles.
                fits=[name for name, length in fleet if length + 0.6 <= length_m],
            )
        )

    # A gap running off the right of the frame has its far end outside the picture, so its
    # length is a lower bound rather than a length, and drawing it would overstate.
    return [s for s in spaces if s.x2 < frame_width - 2]


def analyse(camera_id: str, *, device: str = "cpu", force: bool = False) -> CameraAnalysis:
    """Grab a frame from one live camera and work out where a car would fit."""
    now = time.time()
    if not force:
        cached = _cache.get(camera_id)
        if cached is not None and now - cached[0] < CACHE_SECONDS:
            return cached[1]

    camera = _camera(camera_id)
    if camera is None:
        return CameraAnalysis(camera_id=camera_id, reason="no such published camera")

    from parkfit.cameras.harvest import grab_latest_frame

    out_dir = Path(tempfile.gettempdir()) / "camtoparkingslot-analysis"
    frame_path = out_dir / f"{camera_id}.jpg"
    if not grab_latest_frame(camera, frame_path):
        result = CameraAnalysis(camera_id=camera_id, reason="the camera is not live right now")
        _cache[camera_id] = (now, result)
        return result

    try:
        analysis = _analyse_frame(camera_id, frame_path, device=device)
    finally:
        frame_path.unlink(missing_ok=True)

    _cache[camera_id] = (time.time(), analysis)
    return analysis


def _analyse_frame(camera_id: str, frame_path: Path, *, device: str) -> CameraAnalysis:
    from PIL import Image

    from parkfit.ml.datasets import real as real_ds
    from parkfit.ml.datasets import scenes

    with Image.open(frame_path) as handle:
        width, height = handle.convert("RGB").size

    labelled = real_ds.label_frames([frame_path], device=device)
    if not labelled:
        return CameraAnalysis(camera_id=camera_id, reason="the frame could not be read")

    # label_frames returns boxes against the model input, so they come back to frame
    # pixels here. Scaling twice is silly, and the alternative is a second code path
    # through the teacher, which is worse.
    sx = width / float(scenes.INPUT_WIDTH)
    sy = height / float(scenes.INPUT_HEIGHT)

    vehicles = [
        DetectedVehicle(
            x1=box["x1"] * sx,
            y1=box["y1"] * sy,
            x2=box["x2"] * sx,
            y2=box["y2"] * sy,
            label=scenes.CLASS_NAMES[box["class"]],
            score=float(box.get("score", 0.0)),
        )
        for box in labelled[0].boxes
        if float(box.get("score", 0.0)) >= SCORE_THRESHOLD
    ]

    pixels_per_metre, confident = estimate_scale(vehicles)
    spaces = find_free_spaces(vehicles, pixels_per_metre, width)

    note = (
        "Boxes are real detections on a frame from this camera. Distances are estimates: "
        f"the scale assumes a typical car is {TYPICAL_CAR_WIDTH_M:.2f} m wide, because a "
        "public webcam carries no survey. Height cannot be recovered from one uncalibrated "
        "view and is never reported. An empty stretch of kerb is not the same as a legal "
        "space: whether you may park there comes from the sign code and the time regime, "
        "which the search already checks and the camera cannot see."
    )
    if not confident:
        note += " Fewer than three cars were visible, so the scale is rough."

    return CameraAnalysis(
        camera_id=camera_id,
        ok=True,
        captured_at=time.time(),
        frame_width=width,
        frame_height=height,
        frame_data_uri="data:image/jpeg;base64,"
        + base64.b64encode(frame_path.read_bytes()).decode("ascii"),
        vehicles=vehicles,
        free_spaces=spaces,
        pixels_per_metre=round(pixels_per_metre, 3),
        scale_confident=confident,
        note=note,
    )


def _camera(camera_id: str):
    """The LiveCamera for one published feed, or None."""
    from parkfit.cameras.discovery import PUBLIC_FEEDS
    from parkfit.cameras.harvest import LiveCamera

    feed = next((f for f in PUBLIC_FEEDS if f["camera_id"] == camera_id), None)
    if feed is None:
        return None
    return LiveCamera(
        camera_id=feed["camera_id"],
        youtube_id=feed["youtube_id"],
        name=feed["name"],
        lat=feed["lat"],
        lon=feed["lon"],
    )


def age_seconds(camera_id: str) -> float:
    """How old the served analysis is, or -1 when there is none.

    Shown to the reader rather than kept internal. A picture of a street is worth very
    different amounts at two seconds old and at two minutes, and only the timestamp says
    which one this is.
    """
    entry = _cache.get(camera_id)
    return -1.0 if entry is None else round(time.time() - entry[0], 1)


def cached_spot_count(camera_id: str) -> int:
    """Free spaces the last analysis found, or -1 if it has not run.

    Never triggers an analysis, so it is cheap enough for a map marker.
    """
    entry = _cache.get(camera_id)
    if entry is None:
        return -1
    return len(entry[1].free_spaces) if entry[1].ok else -1


# --------------------------------------------------------------------------- watcher
_interest: dict[str, float] = {}
_watcher: threading.Thread | None = None
_stop = threading.Event()


def mark_interest(camera_id: str) -> None:
    """Say that somebody is looking at this camera, so the watcher keeps it fresh."""
    _interest[camera_id] = time.time()


def _watched() -> list[str]:
    now = time.time()
    return [c for c, seen in _interest.items() if now - seen < INTEREST_SECONDS]


def _loop(device: str, interval: float) -> None:
    while not _stop.is_set():
        watched = _watched()
        if not watched:
            # Waiting on the event rather than the clock, so stop() returns at once
            # instead of after a full idle interval.
            _stop.wait(1.0)
            continue

        for camera_id in watched:
            if _stop.is_set():
                return
            entry = _cache.get(camera_id)
            if entry is not None and time.time() - entry[0] < interval:
                continue
            try:
                analyse(camera_id, device=device, force=True)
            except Exception as exc:
                log.warning("watcher: %s failed: %s", camera_id, exc)
        _stop.wait(0.2)


def start_watcher(*, device: str = "cuda", interval: float = 2.0) -> None:
    """Keep every watched camera's analysis fresh in the background.

    A daemon thread rather than a task on the event loop. The work is a subprocess, a
    socket read and a forward pass, all of which block, and running them on the loop would
    stall every other request the API is serving while a camera is parsed.
    """
    global _watcher
    if _watcher is not None and _watcher.is_alive():
        return
    _stop.clear()
    _watcher = threading.Thread(
        target=_loop, args=(device, interval), name="camera-watcher", daemon=True
    )
    _watcher.start()
    log.info("camera watcher started, refreshing watched feeds every %.1fs", interval)


def stop_watcher() -> None:
    _stop.set()
