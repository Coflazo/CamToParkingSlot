"""What the occupancy model is allowed to see.

Everything here comes from the database: coordinates, capacity, tariff, bay geometry,
and the clock. Nothing comes from :mod:`parkfit.prediction.demand`, which holds the
latent parameters that generated the synthetic history. Keeping that boundary sharp is
what makes the evaluation mean anything.

One input does overlap: distance to the city centre. The demand model uses it to set how
residential a street is, and the feature set exposes it too, because a production model
built on real history would obviously be allowed to know where a bay is. So the model's
advantage over a flat prior cannot come from distance alone; a prior could use that.
It has to come from the **interaction** between distance and time of day: inner-city bays
peak in the evening, outer residential bays peak overnight, and the two curves cross. The
model is never told that, and no per-target constant can express it.

Time is encoded as sine and cosine of the daily angle rather than as a raw hour. A tree
can split on a raw hour perfectly well, but it cannot represent that 23:50 and 00:10 are
twenty minutes apart; it would need a split at every wrap point to approximate what one
pair of cyclic features gives exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from parkfit.prediction.demand import CITY_CENTRE_LAT, CITY_CENTRE_LON, haversine_km
from parkfit.storage.models import ParkingBay, ParkingFacility

FEATURE_NAMES = (
    "hour_sin",
    "hour_cos",
    "week_sin",
    "week_cos",
    "weekday",
    "is_weekend",
    "minutes_since_midnight",
    "is_facility",
    "metered",
    "km_to_centre",
    "lat",
    "lon",
    "capacity",
    "tariff_eur_per_hour",
    "bay_length_cm",
    "bay_width_cm",
    "fill_ratio",
)

#: LightGBM handles these natively as categorical splits rather than as ordered numbers,
#: which matters for weekday: Sunday is not "one more than Saturday".
CATEGORICAL_FEATURES = ("weekday", "is_facility", "metered")


@dataclass(frozen=True)
class TargetStatics:
    """The parts of a feature row that do not change with time.

    Loaded once per target and reused across every timestamp, because a training set of
    a quarter of a million rows covers only a few hundred distinct targets.
    """

    key: tuple[str, int]
    lat: float
    lon: float
    is_facility: bool
    metered: bool
    capacity: float
    tariff_eur_per_hour: float
    bay_length_cm: float
    bay_width_cm: float
    fill_ratio: float

    @property
    def km_to_centre(self) -> float:
        return haversine_km(self.lat, self.lon, CITY_CENTRE_LAT, CITY_CENTRE_LON)


def load_statics(
    session: Session, keys: list[tuple[str, int]]
) -> dict[tuple[str, int], TargetStatics]:
    """Fetch the static half of the feature row for many targets in two queries."""
    out: dict[tuple[str, int], TargetStatics] = {}
    bay_ids = [k[1] for k in keys if k[0] == "bay"]
    fac_ids = [k[1] for k in keys if k[0] == "facility"]

    if bay_ids:
        rows = session.execute(
            select(
                ParkingBay.id,
                ParkingBay.lat,
                ParkingBay.lon,
                ParkingBay.fiscal,
                ParkingBay.length_cm,
                ParkingBay.width_cm,
                ParkingBay.fill_ratio,
            ).where(ParkingBay.id.in_(bay_ids))
        ).all()
        for bid, lat, lon, fiscal, length, width, fill in rows:
            out[("bay", bid)] = TargetStatics(
                key=("bay", bid),
                lat=lat or 0.0,
                lon=lon or 0.0,
                is_facility=False,
                metered=bool(fiscal),
                capacity=1.0,
                tariff_eur_per_hour=0.0,
                bay_length_cm=length or 0.0,
                bay_width_cm=width or 0.0,
                fill_ratio=fill if fill is not None else 1.0,
            )

    if fac_ids:
        rows = session.execute(
            select(
                ParkingFacility.id,
                ParkingFacility.lat,
                ParkingFacility.lon,
                ParkingFacility.capacity,
                ParkingFacility.tariff_eur_per_hour,
            ).where(ParkingFacility.id.in_(fac_ids))
        ).all()
        for fid, lat, lon, capacity, tariff in rows:
            out[("facility", fid)] = TargetStatics(
                key=("facility", fid),
                lat=lat or 0.0,
                lon=lon or 0.0,
                is_facility=True,
                metered=True,
                capacity=float(capacity or 0),
                tariff_eur_per_hour=float(tariff or 0.0),
                bay_length_cm=0.0,
                bay_width_cm=0.0,
                fill_ratio=1.0,
            )

    return out


def row(statics: TargetStatics, when: datetime) -> list[float]:
    """One feature row. Order must match :data:`FEATURE_NAMES` exactly."""
    minutes = when.hour * 60 + when.minute
    day_angle = 2.0 * math.pi * minutes / 1440.0
    week_angle = 2.0 * math.pi * (when.weekday() * 1440 + minutes) / (7.0 * 1440.0)
    return [
        math.sin(day_angle),
        math.cos(day_angle),
        math.sin(week_angle),
        math.cos(week_angle),
        float(when.weekday()),
        1.0 if when.weekday() >= 5 else 0.0,
        float(minutes),
        1.0 if statics.is_facility else 0.0,
        1.0 if statics.metered else 0.0,
        statics.km_to_centre,
        statics.lat,
        statics.lon,
        statics.capacity,
        statics.tariff_eur_per_hour,
        statics.bay_length_cm,
        statics.bay_width_cm,
        statics.fill_ratio,
    ]


def matrix(statics: TargetStatics, times: list[datetime]) -> np.ndarray:
    return np.array([row(statics, t) for t in times], dtype=np.float64)


def batch(pairs: list[tuple[TargetStatics, datetime]]) -> np.ndarray:
    """Feature matrix for a heterogeneous batch, as a search produces."""
    if not pairs:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
    return np.array([row(s, t) for s, t in pairs], dtype=np.float64)
