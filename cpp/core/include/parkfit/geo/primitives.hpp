// SPDX-License-Identifier: MIT
//
// Geodetic primitives. Distances use the spherical earth model: for the ~2 km radii
// this product searches, the ellipsoidal correction is under 0.2 %, well inside the
// noise of the parking data itself, and haversine is branch-free and cheap enough to
// call millions of times during a candidate sweep.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "parkfit/geo/rd.hpp"

namespace parkfit::geo {

inline constexpr double kEarthRadiusM = 6371008.8;  // IUGG mean radius
inline constexpr double kPi = 3.14159265358979323846;

inline constexpr double deg2rad(double d) { return d * kPi / 180.0; }
inline constexpr double rad2deg(double r) { return r * 180.0 / kPi; }

/// Great-circle distance in metres.
inline double haversine_m(const LatLon& a, const LatLon& b) {
    const double phi1 = deg2rad(a.lat);
    const double phi2 = deg2rad(b.lat);
    const double dphi = phi2 - phi1;
    const double dlam = deg2rad(b.lon - a.lon);
    const double s1 = std::sin(dphi * 0.5);
    const double s2 = std::sin(dlam * 0.5);
    const double h = s1 * s1 + std::cos(phi1) * std::cos(phi2) * s2 * s2;
    return 2.0 * kEarthRadiusM * std::asin(std::sqrt(std::min(1.0, h)));
}

/// Initial bearing from `a` to `b`, in degrees clockwise from true north [0, 360).
inline double bearing_deg(const LatLon& a, const LatLon& b) {
    const double phi1 = deg2rad(a.lat);
    const double phi2 = deg2rad(b.lat);
    const double dlam = deg2rad(b.lon - a.lon);
    const double y = std::sin(dlam) * std::cos(phi2);
    const double x = std::cos(phi1) * std::sin(phi2) - std::sin(phi1) * std::cos(phi2) * std::cos(dlam);
    double brg = rad2deg(std::atan2(y, x));
    if (brg < 0.0) brg += 360.0;
    return brg;
}

/// Move `dist_m` metres from `origin` along `bearing` degrees.
inline LatLon offset_m(const LatLon& origin, double bearing_degrees, double dist_m) {
    const double d = dist_m / kEarthRadiusM;
    const double brg = deg2rad(bearing_degrees);
    const double phi1 = deg2rad(origin.lat);
    const double lam1 = deg2rad(origin.lon);
    const double phi2 =
        std::asin(std::sin(phi1) * std::cos(d) + std::cos(phi1) * std::sin(d) * std::cos(brg));
    const double lam2 =
        lam1 + std::atan2(std::sin(brg) * std::sin(d) * std::cos(phi1),
                          std::cos(d) - std::sin(phi1) * std::sin(phi2));
    return LatLon{rad2deg(phi2), rad2deg(lam2)};
}

/// An axis-aligned latitude/longitude bounding box.
struct BBox {
    double min_lat{90.0};
    double min_lon{180.0};
    double max_lat{-90.0};
    double max_lon{-180.0};

    void extend(const LatLon& p) {
        min_lat = std::min(min_lat, p.lat);
        max_lat = std::max(max_lat, p.lat);
        min_lon = std::min(min_lon, p.lon);
        max_lon = std::max(max_lon, p.lon);
    }
    [[nodiscard]] bool contains(const LatLon& p) const {
        return p.lat >= min_lat && p.lat <= max_lat && p.lon >= min_lon && p.lon <= max_lon;
    }
    [[nodiscard]] bool valid() const { return min_lat <= max_lat && min_lon <= max_lon; }
    [[nodiscard]] LatLon centre() const {
        return LatLon{(min_lat + max_lat) * 0.5, (min_lon + max_lon) * 0.5};
    }
};

/// Bounding box that certainly encloses every point within `radius_m` of `centre`.
/// Longitude degrees shrink with latitude, so the longitude half-width is divided by
/// cos(lat); near the poles that blows up, hence the clamp to a full longitude span.
inline BBox bbox_around(const LatLon& centre, double radius_m) {
    const double dlat = rad2deg(radius_m / kEarthRadiusM);
    const double coslat = std::cos(deg2rad(centre.lat));
    BBox b;
    b.min_lat = centre.lat - dlat;
    b.max_lat = centre.lat + dlat;
    if (coslat < 1e-6) {
        b.min_lon = -180.0;
        b.max_lon = 180.0;
    } else {
        const double dlon = dlat / coslat;
        b.min_lon = centre.lon - dlon;
        b.max_lon = centre.lon + dlon;
    }
    return b;
}

/// Shortest distance in metres from point `p` to segment `a`-`b`, computed in a local
/// planar frame. Accurate to well under a centimetre for the segment lengths involved.
inline double point_to_segment_m(const LatLon& p, const LatLon& a, const LatLon& b) {
    const double coslat = std::cos(deg2rad(a.lat));
    const double mx = 111320.0 * coslat;  // metres per degree longitude
    const double my = 110540.0;           // metres per degree latitude
    const double px = (p.lon - a.lon) * mx;
    const double py = (p.lat - a.lat) * my;
    const double bx = (b.lon - a.lon) * mx;
    const double by = (b.lat - a.lat) * my;
    const double len2 = bx * bx + by * by;
    if (len2 < 1e-12) return std::sqrt(px * px + py * py);
    double t = (px * bx + py * by) / len2;
    t = std::clamp(t, 0.0, 1.0);
    const double dx = px - t * bx;
    const double dy = py - t * by;
    return std::sqrt(dx * dx + dy * dy);
}

}  // namespace parkfit::geo
