"""Synthetic parking-scene generator.

This exists because accuracy claims need ground truth, and ground truth for kerb-gap
measurement is genuinely hard to obtain: you would have to stand in a Dutch street with
a tape measure while a camera watched, for every lighting condition you care about.

So the scenes are *rendered from* the geometry instead. A scene starts as a kerb with
vehicles at exactly-known positions and lengths, and the image is produced by projecting
that geometry through a real camera matrix. The gap lengths are therefore known to the
millimetre by construction, which makes the headline metric, gap-length mean absolute
error, measurable rather than asserted.

What it models, because each one breaks a different part of the pipeline:

* **Perspective.** Distant metres occupy fewer pixels than near ones. This is the reason
  a homography is needed at all, and the reason a bounding box in pixel space cannot
  tell you whether a car fits.
* **Occlusion.** A van hides the car behind it. A system that reports the hidden space
  as free is committing the exact error this product is built to avoid.
* **Night, rain, glare.** These are not cosmetic. They are what the frame-health checks
  exist to catch, and a model evaluated only on bright afternoons has not been evaluated.
* **Motorcycles.** Two in one car bay leave no room for a car, and a detector that only
  knows "car" will report that bay as empty.

This is not a substitute for real imagery. It is a substitute for *unmeasured* accuracy:
it gives a floor that can be regression-tested, and real footage raises the ceiling.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CameraModel:
    """A pinhole camera looking down at the ground plane.

    Defaults describe a plausible kerb-overlooking installation: mounted on a building
    at 6 m, pitched 25 degrees down, 1280x720.
    """

    height_m: float = 6.0
    tilt_deg: float = 25.0
    focal_px: float = 900.0
    width_px: int = 1280
    height_px: int = 720
    origin_x: float = 121000.0
    origin_y: float = 487000.0

    @property
    def cx(self) -> float:
        return self.width_px / 2.0

    @property
    def cy(self) -> float:
        return self.height_px / 2.0

    def project(self, world_x: float, world_y: float) -> tuple[float, float] | None:
        """Project a ground point (RD metres) to pixels, or ``None`` if behind the camera.

        Camera axes in world terms are forward = (0, cos, -sin) and down = (0, sin, cos),
        so for a ground point the relative vector is (dx, dy, -h) and depth is
        ``dy·cos + h·sin``, always positive ahead of the camera.
        """
        dx = world_x - self.origin_x
        dy = world_y - self.origin_y
        tilt = math.radians(self.tilt_deg)
        depth = dy * math.cos(tilt) + self.height_m * math.sin(tilt)
        if depth < 0.5:
            return None
        down = dy * math.sin(tilt) - self.height_m * math.cos(tilt)
        return (self.cx + self.focal_px * dx / depth, self.cy + self.focal_px * down / depth)

    def control_points(self, spread_m: float = 8.0) -> list[dict]:
        """Surveyed correspondences a calibration would be built from.

        In production these come from Amsterdam bay corners, which are published in the
        same RD frame, so a camera overlooking marked bays needs no field survey.
        """
        offsets = [
            (-spread_m, 12.0),
            (spread_m, 12.0),
            (-spread_m, 26.0),
            (spread_m, 26.0),
            (0.0, 19.0),
            (-spread_m / 2, 33.0),
        ]
        points = []
        for ox, oy in offsets:
            world = (self.origin_x + ox, self.origin_y + oy)
            pixel = self.project(*world)
            if pixel is None:
                continue
            points.append(
                {
                    "image": [round(pixel[0], 2), round(pixel[1], 2)],
                    "world": [round(world[0], 3), round(world[1], 3)],
                }
            )
        return points


@dataclass
class ParkedVehicle:
    """A vehicle at an exactly-known position along the kerb."""

    along_m: float  # centre, measured from the segment start
    length_m: float
    width_m: float
    height_m: float
    kind: str = "car"

    @property
    def start_m(self) -> float:
        return self.along_m - self.length_m / 2.0

    @property
    def end_m(self) -> float:
        return self.along_m + self.length_m / 2.0


#: Real vehicle footprints, so a scene contains the mix a Dutch street actually holds.
VEHICLE_TYPES: dict[str, tuple[float, float, float]] = {
    # kind: (length, width, height) in metres
    "car": (4.35, 1.80, 1.48),
    "compact": (3.85, 1.70, 1.45),
    "estate": (4.75, 1.85, 1.50),
    "van": (5.30, 1.95, 2.10),
    "truck": (7.20, 2.45, 3.20),
    "motorcycle": (2.10, 0.80, 1.30),
}


@dataclass
class Scene:
    """One rendered scene with its exact ground truth."""

    image: np.ndarray
    vehicles: list[ParkedVehicle]
    gaps: list[tuple[float, float]]  # (start_m, end_m) along the kerb
    kerb_length_m: float
    kerb_y: float
    camera: CameraModel
    condition: str
    occluded_gaps: list[int] = field(default_factory=list)

    @property
    def gap_lengths_m(self) -> list[float]:
        return [end - start for start, end in self.gaps]

    def detections(self) -> list[dict]:
        """Bounding boxes a perfect detector would produce.

        Used to evaluate the *geometry* in isolation. Feeding a real detector's output
        instead measures the two together, and the difference between the two runs is
        exactly the detector's contribution to the error.
        """
        boxes = []
        for vehicle in self.vehicles:
            box = _vehicle_box(self.camera, vehicle, self.kerb_y)
            if box is None:
                continue
            boxes.append(
                {
                    "x1": round(box[0], 1),
                    "y1": round(box[1], 1),
                    "x2": round(box[2], 1),
                    "y2": round(box[3], 1),
                    "score": 0.93,
                    "label": vehicle.kind,
                }
            )
        return boxes


def _vehicle_box(
    camera: CameraModel, vehicle: ParkedVehicle, kerb_y: float
) -> tuple[float, float, float, float] | None:
    """Axis-aligned image box for a vehicle standing on the kerb."""
    left = camera.project(camera.origin_x + vehicle.start_m, kerb_y)
    right = camera.project(camera.origin_x + vehicle.end_m, kerb_y)
    if left is None or right is None:
        return None

    # The top edge is the roof, which sits above the ground contact by the vehicle
    # height scaled by the same perspective factor. Only the bottom edge is used for
    # ground projection, but the top must be plausible for a detector to train on.
    dy = kerb_y - camera.origin_y
    tilt = math.radians(camera.tilt_deg)
    depth = dy * math.cos(tilt) + camera.height_m * math.sin(tilt)
    roof_px = camera.focal_px * vehicle.height_m / max(depth, 0.5)

    y2 = (left[1] + right[1]) / 2.0
    return (min(left[0], right[0]), y2 - roof_px, max(left[0], right[0]), y2)


class SceneGenerator:
    """Builds scenes with known geometry and renders them."""

    CONDITIONS = ("day", "overcast", "dusk", "night", "rain", "glare")

    def __init__(self, camera: CameraModel | None = None, seed: int = 0):
        self.camera = camera or CameraModel()
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def build(
        self,
        *,
        kerb_length_m: float = 40.0,
        occupancy: float = 0.6,
        condition: str | None = None,
        kerb_offset_m: float = 18.0,
    ) -> Scene:
        """Place vehicles along a kerb, leaving gaps of known length."""
        condition = condition or self.rng.choice(self.CONDITIONS)
        kerb_y = self.camera.origin_y + kerb_offset_m

        vehicles: list[ParkedVehicle] = []
        cursor = self.rng.uniform(0.0, 3.0)
        while cursor < kerb_length_m - 2.0:
            if self.rng.random() < occupancy:
                kind = self.rng.choices(
                    ["car", "compact", "estate", "van", "truck", "motorcycle"],
                    weights=[42, 18, 16, 12, 4, 8],
                )[0]
                length, width, height = VEHICLE_TYPES[kind]
                if cursor + length > kerb_length_m:
                    break
                vehicles.append(ParkedVehicle(cursor + length / 2, length, width, height, kind))
                # A small random bumper interval, as real parking has.
                cursor += length + self.rng.uniform(0.35, 0.9)
            else:
                cursor += self.rng.uniform(2.5, 7.0)

        gaps = self._gaps(vehicles, kerb_length_m)
        image = self._render(vehicles, kerb_y, kerb_length_m, condition)
        return Scene(
            image=image,
            vehicles=vehicles,
            gaps=gaps,
            kerb_length_m=kerb_length_m,
            kerb_y=kerb_y,
            camera=self.camera,
            condition=condition,
        )

    @staticmethod
    def _gaps(vehicles: list[ParkedVehicle], kerb_length_m: float) -> list[tuple[float, float]]:
        """Free stretches between vehicles. Exact by construction."""
        if not vehicles:
            return [(0.0, kerb_length_m)]
        ordered = sorted(vehicles, key=lambda v: v.start_m)
        gaps: list[tuple[float, float]] = []
        cursor = 0.0
        for vehicle in ordered:
            if vehicle.start_m > cursor:
                gaps.append((cursor, vehicle.start_m))
            cursor = max(cursor, vehicle.end_m)
        if cursor < kerb_length_m:
            gaps.append((cursor, kerb_length_m))
        return [(a, b) for a, b in gaps if b - a > 0.05]

    # rendering ----------------------------------------------------------
    def _render(
        self,
        vehicles: list[ParkedVehicle],
        kerb_y: float,
        kerb_length_m: float,
        condition: str,
    ) -> np.ndarray:
        cam = self.camera
        image = np.zeros((cam.height_px, cam.width_px, 3), dtype=np.uint8)

        base, road, kerb_tone, noise_sigma = _palette(condition)
        image[:, :] = base

        # Road surface: everything below the horizon the kerb sits on.
        horizon = self._horizon_row()
        image[horizon:, :] = road

        self._draw_kerb(image, kerb_y, kerb_length_m, kerb_tone)

        # Painted bay dividers every 6 m, which is what a calibration is clicked against.
        for metre in range(0, int(kerb_length_m) + 1, 6):
            self._draw_line_world(
                image,
                cam.origin_x + metre,
                kerb_y - 1.0,
                cam.origin_x + metre,
                kerb_y + 1.1,
                (200, 200, 190),
                2,
            )

        # Far vehicles first, so nearer ones paint over them and produce real occlusion.
        for vehicle in sorted(vehicles, key=lambda v: -v.along_m):
            self._draw_vehicle(image, vehicle, kerb_y, condition)

        return self._apply_condition(image, condition, noise_sigma)

    def _horizon_row(self) -> int:
        cam = self.camera
        tilt = math.radians(cam.tilt_deg)
        # The ground plane meets the image where depth goes to infinity: down/depth
        # tends to tan(tilt), so the horizon sits that far below the principal point.
        return max(0, min(cam.height_px - 1, int(cam.cy - cam.focal_px * math.tan(tilt) * -1)))

    def _draw_kerb(
        self, image: np.ndarray, kerb_y: float, kerb_length_m: float, tone: tuple[int, int, int]
    ) -> None:
        cam = self.camera
        for metre in np.arange(0.0, kerb_length_m, 0.25):
            a = cam.project(cam.origin_x + metre, kerb_y + 1.15)
            b = cam.project(cam.origin_x + metre + 0.25, kerb_y + 1.15)
            if a is None or b is None:
                continue
            _line(image, a, b, tone, 3)

    def _draw_line_world(
        self,
        image: np.ndarray,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        colour: tuple[int, int, int],
        width: int,
    ) -> None:
        a = self.camera.project(x1, y1)
        b = self.camera.project(x2, y2)
        if a is None or b is None:
            return
        _line(image, a, b, colour, width)

    def _draw_vehicle(
        self, image: np.ndarray, vehicle: ParkedVehicle, kerb_y: float, condition: str
    ) -> None:
        """Draw a vehicle as a projected box, so its silhouette obeys perspective."""
        cam = self.camera
        near_y = kerb_y - vehicle.width_m / 2
        far_y = kerb_y + vehicle.width_m / 2

        corners_ground = [
            (cam.origin_x + vehicle.start_m, near_y),
            (cam.origin_x + vehicle.end_m, near_y),
            (cam.origin_x + vehicle.end_m, far_y),
            (cam.origin_x + vehicle.start_m, far_y),
        ]
        ground = [cam.project(x, y) for x, y in corners_ground]
        if any(p is None for p in ground):
            return

        tilt = math.radians(cam.tilt_deg)
        body = _vehicle_colour(self.rng, condition)

        # Roof corners: the same ground points raised by the vehicle height, which in
        # this projection is a vertical pixel offset scaled by depth.
        roof = []
        for (_x, y), g in zip(corners_ground, ground, strict=True):
            dy = y - cam.origin_y
            depth = dy * math.cos(tilt) + cam.height_m * math.sin(tilt)
            roof.append((g[0], g[1] - cam.focal_px * vehicle.height_m / max(depth, 0.5)))

        _fill_quad(image, [ground[0], ground[1], roof[1], roof[0]], body)  # near side
        _fill_quad(image, [roof[0], roof[1], roof[2], roof[3]], _shade(body, 1.18))  # roof
        _fill_quad(image, [ground[1], ground[2], roof[2], roof[1]], _shade(body, 0.78))  # end

        # Glazing, so the silhouette is not a flat slab and a detector has something
        # to key on beyond the outline.
        glass = _shade(body, 0.45)
        gx1 = roof[0][0] + (roof[1][0] - roof[0][0]) * 0.22
        gx2 = roof[0][0] + (roof[1][0] - roof[0][0]) * 0.78
        gy1 = roof[0][1] + 2
        gy2 = gy1 + max(4.0, (ground[0][1] - roof[0][1]) * 0.33)
        _fill_quad(image, [(gx1, gy1), (gx2, gy1), (gx2, gy2), (gx1, gy2)], glass)

        if condition in {"night", "dusk"}:
            # Tail lights: a real cue at night, and the reason a night model behaves
            # differently from a daytime one.
            for corner in (ground[0], ground[1]):
                _disc(image, corner[0], corner[1] - 6, 3, (40, 40, 220))

    def _apply_condition(self, image: np.ndarray, condition: str, noise_sigma: float) -> np.ndarray:
        out = image.astype(np.float32)

        if condition == "rain":
            # Rain on the lens suppresses edges, which is what the sharpness check sees.
            out = _box_blur(out, 3)
            streaks = self.np_rng.random(out.shape[:2]) < 0.012
            out[streaks] = np.clip(out[streaks] + 55, 0, 255)
        elif condition == "glare":
            # A blown-out sun patch: bright, and carrying no information.
            yy, xx = np.mgrid[0 : out.shape[0], 0 : out.shape[1]]
            cx = out.shape[1] * 0.72
            cy = out.shape[0] * 0.30
            radial = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (out.shape[1] * 0.16) ** 2))
            out += (radial * 190)[:, :, None]
        elif condition == "night":
            out *= 0.32
        elif condition == "dusk":
            out *= 0.62

        if noise_sigma > 0:
            out += self.np_rng.normal(0.0, noise_sigma, out.shape)
        return np.clip(out, 0, 255).astype(np.uint8)


def _palette(
    condition: str,
) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int], float]:
    """(sky, road, kerb, noise sigma) for a condition."""
    table = {
        "day": ((150, 165, 180), (78, 80, 84), (150, 148, 142), 3.0),
        "overcast": ((128, 132, 138), (70, 72, 75), (135, 134, 130), 4.0),
        "dusk": ((92, 88, 102), (58, 58, 62), (110, 108, 104), 6.0),
        "night": ((24, 26, 34), (38, 38, 42), (78, 77, 74), 9.0),
        "rain": ((104, 108, 114), (58, 60, 64), (118, 117, 114), 7.0),
        "glare": ((178, 180, 186), (92, 93, 96), (162, 160, 154), 4.0),
    }
    return table.get(condition, table["day"])


def _vehicle_colour(rng: random.Random, condition: str) -> tuple[int, int, int]:
    palette = [
        (60, 60, 62),
        (185, 185, 190),
        (140, 30, 30),
        (30, 60, 140),
        (95, 95, 100),
        (25, 70, 45),
        (200, 200, 205),
        (40, 40, 45),
    ]
    colour = rng.choice(palette)
    if condition in {"night", "dusk"}:
        colour = _shade(colour, 0.55)
    return colour


def _shade(colour: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(int(min(255, max(0, c * factor))) for c in colour)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Minimal rasterisation. OpenCV would do this, but the generator is used by the
# evaluation harness, and a harness that cannot run without a heavy optional
# dependency will eventually stop being run.
# ---------------------------------------------------------------------------
def _line(image: np.ndarray, a, b, colour, width: int) -> None:
    x1, y1 = round(a[0]), round(a[1])
    x2, y2 = round(b[0]), round(b[1])
    steps = max(abs(x2 - x1), abs(y2 - y1), 1)
    for i in range(steps + 1):
        x = int(x1 + (x2 - x1) * i / steps)
        y = int(y1 + (y2 - y1) * i / steps)
        _dot(image, x, y, width, colour)


def _dot(image: np.ndarray, x: int, y: int, size: int, colour) -> None:
    h, w = image.shape[:2]
    half = max(1, size // 2)
    x0, x1 = max(0, x - half), min(w, x + half + 1)
    y0, y1 = max(0, y - half), min(h, y + half + 1)
    if x0 < x1 and y0 < y1:
        image[y0:y1, x0:x1] = colour


def _disc(image: np.ndarray, cx: float, cy: float, radius: int, colour) -> None:
    h, w = image.shape[:2]
    for y in range(max(0, int(cy - radius)), min(h, int(cy + radius + 1))):
        for x in range(max(0, int(cx - radius)), min(w, int(cx + radius + 1))):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius**2:
                image[y, x] = colour


def _fill_quad(image: np.ndarray, points, colour) -> None:
    """Scanline fill of a convex quadrilateral."""
    h, w = image.shape[:2]
    ys = [p[1] for p in points]
    y_start = max(0, int(min(ys)))
    y_end = min(h - 1, int(max(ys)))
    if y_end < y_start:
        return

    for y in range(y_start, y_end + 1):
        crossings = []
        for i in range(len(points)):
            ax, ay = points[i]
            bx, by = points[(i + 1) % len(points)]
            if (ay > y) == (by > y):
                continue
            crossings.append(ax + (y - ay) * (bx - ax) / (by - ay))
        if len(crossings) < 2:
            continue
        x_start = max(0, int(min(crossings)))
        x_end = min(w - 1, int(max(crossings)))
        if x_start <= x_end:
            image[y, x_start : x_end + 1] = colour


def _box_blur(image: np.ndarray, radius: int) -> np.ndarray:
    """Separable box blur. Enough to model a wet lens."""
    kernel = 2 * radius + 1
    padded = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    out = np.zeros_like(image, dtype=np.float32)
    for dy in range(kernel):
        for dx in range(kernel):
            out += padded[dy : dy + image.shape[0], dx : dx + image.shape[1]]
    return out / (kernel * kernel)


def write_dataset(
    output_dir: Path, count: int = 40, seed: int = 0, camera: CameraModel | None = None
) -> dict:
    """Render a dataset with its ground truth, for the evaluation harness.

    Writes one PPM per scene, one detection sidecar, and a manifest carrying the exact
    gap lengths. PPM because it needs no image library to read or write, and this must
    keep working when Pillow is not installed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = SceneGenerator(camera=camera, seed=seed)

    scenes = []
    frames = []
    for index in range(count):
        scene = generator.build(condition=SceneGenerator.CONDITIONS[index % 6])
        image_path = output_dir / f"scene_{index:03d}.ppm"
        _write_ppm(image_path, scene.image)
        frames.append({"index": index, "detections": scene.detections()})
        scenes.append(
            {
                "index": index,
                "image": image_path.name,
                "condition": scene.condition,
                "vehicles": len(scene.vehicles),
                "kerb_length_m": scene.kerb_length_m,
                "gaps_m": [[round(a, 4), round(b, 4)] for a, b in scene.gaps],
                "gap_lengths_m": [round(g, 4) for g in scene.gap_lengths_m],
            }
        )

    manifest = {
        "camera": {
            "height_m": generator.camera.height_m,
            "tilt_deg": generator.camera.tilt_deg,
            "focal_px": generator.camera.focal_px,
            "width_px": generator.camera.width_px,
            "height_px": generator.camera.height_px,
            "origin_x": generator.camera.origin_x,
            "origin_y": generator.camera.origin_y,
        },
        "control_points": generator.camera.control_points(),
        "scenes": scenes,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (output_dir / "detections.json").write_text(
        json.dumps({"frames": frames}, indent=1), encoding="utf-8"
    )
    return manifest


def _write_ppm(path: Path, image: np.ndarray) -> None:
    height, width = image.shape[:2]
    with path.open("wb") as handle:
        handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        # The renderer works in BGR to match the worker's frame format.
        handle.write(image[:, :, ::-1].tobytes())
