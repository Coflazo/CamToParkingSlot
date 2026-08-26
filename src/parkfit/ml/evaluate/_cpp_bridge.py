"""Thin bridge to the C++ vision geometry for evaluation.

The gap estimator and temporal state machine live in headers rather than in the pybind11
module, because the search path does not need them — only the worker and this harness do.
Rather than widen the extension module for a test-only path, the evaluation compiles a
small helper on demand and caches it.

If the helper cannot be built, the harness falls back to a faithful Python port of the
same algorithms. The port is not a second implementation to maintain in parallel: it is
checked against the C++ tests by construction, and the fallback exists so an accuracy
report is always available rather than only on a machine with a compiler.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Python port of the C++ geometry, kept deliberately literal
# ---------------------------------------------------------------------------
@dataclass
class _Interval:
    start: float
    end: float

    @property
    def length(self) -> float:
        return max(0.0, self.end - self.start)


def _solve_homography(image_points, world_points):
    """Normalised DLT, mirroring parkfit::vision::solve_homography."""
    import numpy as np

    def normalise(points):
        arr = np.asarray(points, dtype=float)
        centre = arr.mean(axis=0)
        distances = np.linalg.norm(arr - centre, axis=1)
        mean_distance = distances.mean()
        scale = np.sqrt(2.0) / mean_distance if mean_distance > 1e-12 else 1.0
        matrix = np.array([[scale, 0, -scale * centre[0]],
                           [0, scale, -scale * centre[1]],
                           [0, 0, 1.0]])
        return matrix, (arr - centre) * scale

    ti, src = normalise(image_points)
    tw, dst = normalise(world_points)

    rows = []
    for (x, y), (u, v) in zip(src, dst, strict=True):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])

    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=float))
    h = vt[-1].reshape(3, 3)
    h = np.linalg.inv(tw) @ h @ ti
    return h / h[2, 2]


def _apply(h, x: float, y: float) -> tuple[float, float]:
    w = h[2, 0] * x + h[2, 1] * y + h[2, 2]
    if abs(w) < 1e-12:
        return 0.0, 0.0
    return ((h[0, 0] * x + h[0, 1] * y + h[0, 2]) / w,
            (h[1, 0] * x + h[1, 1] * y + h[1, 2]) / w)


def _project_onto_segment(px, py, ax, ay, bx, by) -> tuple[float, float]:
    """Distance along the segment, and perpendicular offset from it."""
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return 0.0, ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    cx, cy = ax + t * dx, ay + t * dy
    return t * length_sq**0.5, ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5


def _merge(intervals: list[_Interval], tolerance: float) -> list[_Interval]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda i: i.start)
    merged = [ordered[0]]
    for current in ordered[1:]:
        last = merged[-1]
        if current.start <= last.end + tolerance:
            last.end = max(last.end, current.end)
        else:
            merged.append(current)
    return merged


def measure_gaps(
    *,
    image_points,
    world_points,
    detections,
    kerb_start: tuple[float, float],
    kerb_end: tuple[float, float],
    camera_world: tuple[float, float],
    max_offset_m: float = 3.2,
    merge_tolerance_m: float = 0.35,
    min_gap_m: float = 3.0,
    max_range_m: float = 45.0,
) -> list[float]:
    """Measure free kerb stretches, mirroring parkfit::vision::CurbGapEstimator.

    Only the *bottom* edge of a detection is projected. That edge is where the vehicle
    meets the ground, and the homography maps the ground plane and nothing else —
    projecting the box centre would place every car metres further away than it is, with
    the error growing with distance.
    """
    h = _solve_homography(image_points, world_points)
    ax, ay = kerb_start
    bx, by = kerb_end
    total = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5

    blocked: list[_Interval] = []
    for detection in detections:
        left = _apply(h, detection["x1"], detection["y2"])
        right = _apply(h, detection["x2"], detection["y2"])
        centre = _apply(h, (detection["x1"] + detection["x2"]) / 2.0, detection["y2"])

        distance = ((centre[0] - camera_world[0]) ** 2 + (centre[1] - camera_world[1]) ** 2) ** 0.5
        if distance > max_range_m:
            continue

        along_c, offset_c = _project_onto_segment(centre[0], centre[1], ax, ay, bx, by)
        if offset_c > max_offset_m:
            continue

        along_l, _ = _project_onto_segment(left[0], left[1], ax, ay, bx, by)
        along_r, _ = _project_onto_segment(right[0], right[1], ax, ay, bx, by)
        start, end = sorted((along_l, along_r))
        if end - start < 1.0:
            start, end = along_c - 2.1, along_c + 2.1

        blocked.append(_Interval(max(0.0, start), min(total, end)))

    merged = _merge(blocked, merge_tolerance_m)

    free: list[float] = []
    cursor = 0.0
    for interval in merged:
        if interval.start > cursor:
            free.append(min(interval.start, total) - cursor)
        cursor = max(cursor, interval.end)
        if cursor >= total:
            break
    if cursor < total:
        free.append(total - cursor)

    return [length for length in free if length >= min_gap_m]


def run_state_machine(
    scores: list[float],
    *,
    vacant_confirmations: int = 3,
    vacant_window: int = 4,
    occupied_min_score: float = 0.45,
    vacant_max_score: float = 0.22,
) -> str:
    """Mirror parkfit::vision::TemporalStateMachine and return the final published state.

    The asymmetry is the point: OCCUPIED on a single confident detection, VACANT only
    when enough of the recent window is clean *and* the latest observation is clean,
    UNKNOWN whenever neither holds.

    The window tolerates one outlier rather than demanding a strict run. A strictly
    consecutive rule makes the answer depend entirely on the last three frames, which
    caps recall at (1 - false alarm rate)^3 however long the camera has been watching.
    """
    state = "UNKNOWN"
    window: list[bool] = []

    def push(clean: bool) -> None:
        window.append(clean)
        if len(window) > vacant_window:
            window.pop(0)

    for score in scores:
        if score >= occupied_min_score:
            push(False)
            state = "OCCUPIED"
        elif score <= vacant_max_score:
            push(True)
            enough_history = len(window) >= vacant_confirmations
            if enough_history and sum(window) >= vacant_confirmations and window[-1]:
                state = "VACANT"
            elif state == "OCCUPIED":
                # The car appears to have gone, but this is not yet proof. Until vacancy
                # is confirmed the honest answer is that we no longer know.
                state = "UNKNOWN"
        else:
            push(False)
            state = "UNKNOWN"

    return state
