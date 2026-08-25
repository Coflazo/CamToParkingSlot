"""Rijksdriehoek (RD New, EPSG:28992) support.

Amsterdam publishes parking-bay polygons in RD New, so this module is on the ingest
critical path. Two implementations are available:

* :func:`rd_to_wgs84` uses ``pyproj`` and the rigorous EPSG pipeline. It is what ingest
  uses, because a one-off conversion can afford to be exact.
* :func:`rd_to_wgs84_approx` reproduces the Kadaster polynomial used by the C++ core.
  It lands about 0.23 m north and 0.18 m east of the rigorous answer, consistently
  across the country. Keeping it here lets the contract tests prove the two
  implementations agree to within that documented bound.

Metric work stays in RD. RD is a conformal metric projection, so a length measured
there is a true length; measuring the same bay in WGS84 degrees would fold in both the
datum offset and cosine-latitude distortion.
"""

from __future__ import annotations

import math
from functools import lru_cache

RD_ORIGIN_X = 155000.0
RD_ORIGIN_Y = 463000.0
REF_LAT = 52.15517440
REF_LON = 5.38720621

# RD -> WGS84 polynomial coefficients (p, q, coefficient), yielding arc-seconds.
_K = (
    (0, 1, 3235.65389), (2, 0, -32.58297), (0, 2, -0.24750), (2, 1, -0.84978),
    (0, 3, -0.06550), (2, 2, -0.01709), (1, 0, -0.00738), (4, 0, 0.00530),
    (2, 3, -0.00039), (4, 1, 0.00033), (1, 1, -0.00012),
)
_L = (
    (1, 0, 5260.52916), (1, 1, 105.94684), (1, 2, 2.45656), (3, 0, -0.81885),
    (1, 3, 0.05594), (3, 1, -0.05607), (0, 1, 0.01199), (3, 2, -0.00256),
    (1, 4, 0.00128), (0, 2, 0.00022), (2, 0, -0.00022),
)
# WGS84 -> RD.
_R = (
    (0, 1, 190094.945), (1, 1, -11832.228), (2, 1, -114.221), (0, 3, -32.391),
    (1, 0, -0.705), (3, 1, -2.340), (1, 3, -0.608), (0, 2, -0.008), (2, 3, 0.148),
)
_S = (
    (1, 0, 309056.544), (0, 2, 3638.893), (2, 0, 73.077), (1, 2, -157.984),
    (3, 0, 59.788), (0, 1, 0.433), (2, 2, -6.439), (1, 1, -0.032), (0, 4, 0.092),
    (1, 4, -0.054),
)

# Valid envelope of the RD system. Outside it the approximation degrades quickly, and
# a coordinate outside it is far more likely to be a unit error than a real place.
RD_MIN_X, RD_MAX_X = -7000.0, 300000.0
RD_MIN_Y, RD_MAX_Y = 289000.0, 629000.0


def rd_in_range(x: float, y: float) -> bool:
    return RD_MIN_X <= x <= RD_MAX_X and RD_MIN_Y <= y <= RD_MAX_Y


def rd_to_wgs84_approx(x: float, y: float) -> tuple[float, float]:
    """Kadaster polynomial approximation. Returns ``(lat, lon)``."""
    dx = (x - RD_ORIGIN_X) * 1e-5
    dy = (y - RD_ORIGIN_Y) * 1e-5
    lat = REF_LAT + sum(c * dx**p * dy**q for p, q, c in _K) / 3600.0
    lon = REF_LON + sum(c * dx**p * dy**q for p, q, c in _L) / 3600.0
    return lat, lon


def wgs84_to_rd_approx(lat: float, lon: float) -> tuple[float, float]:
    """Kadaster polynomial approximation. Returns ``(x, y)`` in metres."""
    dl = 0.36 * (lat - REF_LAT)
    dp = 0.36 * (lon - REF_LON)
    x = RD_ORIGIN_X + sum(c * dl**p * dp**q for p, q, c in _R)
    y = RD_ORIGIN_Y + sum(c * dl**p * dp**q for p, q, c in _S)
    return x, y


@lru_cache(maxsize=2)
def _transformers():
    """Build the pyproj transformers once; construction is far costlier than use."""
    try:
        from pyproj import Transformer
    except ImportError:  # pragma: no cover - pyproj is a hard dependency
        return None, None
    return (
        Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=False),
        Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=False),
    )


def rd_to_wgs84(x: float, y: float) -> tuple[float, float]:
    """Rigorous RD -> WGS84. Returns ``(lat, lon)``."""
    to_wgs, _ = _transformers()
    if to_wgs is None:
        return rd_to_wgs84_approx(x, y)
    lat, lon = to_wgs.transform(x, y)
    return float(lat), float(lon)


def wgs84_to_rd(lat: float, lon: float) -> tuple[float, float]:
    """Rigorous WGS84 -> RD. Returns ``(x, y)`` in metres."""
    _, to_rd = _transformers()
    if to_rd is None:
        return wgs84_to_rd_approx(lat, lon)
    x, y = to_rd.transform(lat, lon)
    return float(x), float(y)


def ring_rd_to_wgs84(ring: list[list[float]]) -> list[list[float]]:
    """Convert an RD ring to GeoJSON ``[lon, lat]`` order."""
    out = []
    for pt in ring:
        lat, lon = rd_to_wgs84(pt[0], pt[1])
        out.append([lon, lat])
    return out


# ---------------------------------------------------------------------------
# Planar helpers, operating directly in RD metres
# ---------------------------------------------------------------------------
def ring_centroid_rd(ring: list[list[float]]) -> tuple[float, float]:
    """Area-weighted centroid of an RD ring, falling back to the vertex mean."""
    n = len(ring)
    if n == 0:
        return 0.0, 0.0
    if n < 3:
        return (sum(p[0] for p in ring) / n, sum(p[1] for p in ring) / n)

    a2 = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(n):
        x0, y0 = ring[i][0], ring[i][1]
        x1, y1 = ring[(i + 1) % n][0], ring[(i + 1) % n][1]
        f = x0 * y1 - x1 * y0
        a2 += f
        cx += (x0 + x1) * f
        cy += (y0 + y1) * f
    if abs(a2) < 1e-12:
        return (sum(p[0] for p in ring) / n, sum(p[1] for p in ring) / n)
    return cx / (3.0 * a2), cy / (3.0 * a2)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres, matching the C++ core exactly."""
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(min(1.0, h)))
