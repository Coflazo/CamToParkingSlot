"""Recompute stored bay geometry in place.

Bay dimensions are derived once at ingest and stored, so a change to the measurement
has to be applied to the rows already in the database. Re-fetching a quarter of a
million bays from the city API to recompute a number we can derive from geometry we
already hold would be slow and rude.

Run after any change to :func:`parkfit.geo.shapes.measure_bay` or to the orientation
inference.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter

from sqlalchemy import select

from parkfit.geo.shapes import measure_bay
from parkfit.ingest.amsterdam import ORIENTATION_BY_TYPE, infer_orientation
from parkfit.storage.models import BayOrientation, ParkingBay
from parkfit.storage.session import session_scope

log = logging.getLogger(__name__)

BATCH = 5000


def recompute(reinfer_orientation: bool = True) -> dict[str, int]:
    stats: Counter[str] = Counter()
    changes: list[tuple[float, float]] = []

    with session_scope() as session:
        total = session.execute(select(ParkingBay.id)).scalars().all()
        log.info("recomputing %d bays", len(total))

        for offset in range(0, len(total), BATCH):
            ids = total[offset : offset + BATCH]
            rows = (
                session.execute(select(ParkingBay).where(ParkingBay.id.in_(ids)))
                .scalars()
                .all()
            )
            for bay in rows:
                try:
                    ring = json.loads(bay.geometry_rd_json)
                except (TypeError, json.JSONDecodeError):
                    stats["unparseable"] += 1
                    continue
                if not ring or len(ring) < 3:
                    stats["degenerate"] += 1
                    continue

                measurement = measure_bay(ring)
                changes.append((bay.length_cm - measurement.length_cm,
                                bay.width_cm - measurement.width_cm))

                bay.length_cm = measurement.length_cm
                bay.width_cm = measurement.width_cm
                bay.max_length_cm = measurement.max_length_m * 100.0
                bay.max_width_cm = measurement.max_width_m * 100.0
                bay.fill_ratio = measurement.fill_ratio
                bay.angle_rad = measurement.angle_rad

                if reinfer_orientation and bay.orientation == BayOrientation.UNKNOWN.value:
                    inferred = infer_orientation(measurement.length_m, measurement.width_m)
                    if inferred is not BayOrientation.UNKNOWN:
                        bay.orientation = inferred.value
                        stats["orientation_inferred"] += 1

                stats["updated"] += 1

            session.commit()
            log.info("  %d / %d", min(offset + BATCH, len(total)), len(total))

    if changes:
        widths = sorted(abs(c[1]) for c in changes)
        stats["median_width_change_cm"] = int(widths[len(widths) // 2])
        stats["p90_width_change_cm"] = int(widths[9 * len(widths) // 10])
    return dict(stats)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    stats = recompute()
    for key, value in sorted(stats.items()):
        print(f"  {key:<28} {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
