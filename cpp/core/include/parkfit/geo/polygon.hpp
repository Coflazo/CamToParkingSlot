// SPDX-License-Identifier: MIT
//
// Planar polygon geometry, operating in RD New metres.
//
// Why RD and not lat/lon: the City of Amsterdam publishes each parking bay as a polygon
// in EPSG:28992, which is a conformal metric projection. Lengths measured directly in RD
// are true metres. Converting to WGS84 first and measuring there would inject both the
// systematic ~0.23 m datum offset and cosine-latitude distortion into the very number
// that decides whether a car fits, so metric work stays here, in RD.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace parkfit::geo {

/// A point in a planar metric frame (RD New metres).
struct Point2 {
    double x{};
    double y{};
};

using Ring = std::vector<Point2>;

inline double cross(const Point2& o, const Point2& a, const Point2& b) {
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

inline double dist2(const Point2& a, const Point2& b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return dx * dx + dy * dy;
}

/// Signed area via the shoelace formula. Positive for counter-clockwise rings.
inline double signed_area(const Ring& r) {
    if (r.size() < 3) return 0.0;
    double s = 0.0;
    for (std::size_t i = 0, n = r.size(); i < n; ++i) {
        const Point2& a = r[i];
        const Point2& b = r[(i + 1) % n];
        s += a.x * b.y - b.x * a.y;
    }
    return s * 0.5;
}

inline double area(const Ring& r) { return std::abs(signed_area(r)); }

/// Area-weighted centroid. Falls back to the vertex mean for degenerate rings.
inline Point2 centroid(const Ring& r) {
    const double a = signed_area(r);
    if (std::abs(a) < 1e-12) {
        Point2 c{};
        if (r.empty()) return c;
        for (const auto& p : r) {
            c.x += p.x;
            c.y += p.y;
        }
        c.x /= static_cast<double>(r.size());
        c.y /= static_cast<double>(r.size());
        return c;
    }
    double cx = 0.0, cy = 0.0;
    for (std::size_t i = 0, n = r.size(); i < n; ++i) {
        const Point2& p = r[i];
        const Point2& q = r[(i + 1) % n];
        const double f = p.x * q.y - q.x * p.y;
        cx += (p.x + q.x) * f;
        cy += (p.y + q.y) * f;
    }
    return Point2{cx / (6.0 * a), cy / (6.0 * a)};
}

/// Ray-casting point-in-polygon test. Points exactly on an edge are unspecified,
/// which is fine: bay membership is never decided on a knife edge.
inline bool contains(const Ring& r, const Point2& p) {
    if (r.size() < 3) return false;
    bool inside = false;
    for (std::size_t i = 0, j = r.size() - 1, n = r.size(); i < n; j = i++) {
        const bool straddles = (r[i].y > p.y) != (r[j].y > p.y);
        if (!straddles) continue;
        const double xint = (r[j].x - r[i].x) * (p.y - r[i].y) / (r[j].y - r[i].y) + r[i].x;
        if (p.x < xint) inside = !inside;
    }
    return inside;
}

/// Convex hull via the monotone-chain method of A. M. Andrew.
/// Returns counter-clockwise order with no repeated endpoint.
inline Ring convex_hull(Ring pts) {
    if (pts.size() < 3) return pts;
    std::sort(pts.begin(), pts.end(), [](const Point2& a, const Point2& b) {
        return a.x < b.x || (a.x == b.x && a.y < b.y);
    });
    pts.erase(std::unique(pts.begin(), pts.end(),
                          [](const Point2& a, const Point2& b) {
                              return std::abs(a.x - b.x) < 1e-9 && std::abs(a.y - b.y) < 1e-9;
                          }),
              pts.end());
    if (pts.size() < 3) return pts;

    const std::size_t n = pts.size();
    Ring hull(2 * n);
    std::size_t k = 0;
    for (std::size_t i = 0; i < n; ++i) {
        while (k >= 2 && cross(hull[k - 2], hull[k - 1], pts[i]) <= 0.0) --k;
        hull[k++] = pts[i];
    }
    const std::size_t lower = k + 1;
    for (std::size_t i = n - 1; i-- > 0;) {
        while (k >= lower && cross(hull[k - 2], hull[k - 1], pts[i]) <= 0.0) --k;
        hull[k++] = pts[i];
    }
    hull.resize(k - 1);
    return hull;
}

/// The minimum-area enclosing rectangle of a polygon.
///
/// This is what converts a raw parking-bay polygon into the two numbers that decide
/// vehicle fit. `length_m` is the longer side (the direction a car drives in) and
/// `width_m` the shorter. `angle_rad` is the orientation of the long side, which
/// tells us how the bay is aligned to the kerb.
struct MinAreaRect {
    double length_m{};
    double width_m{};
    double angle_rad{};
    Point2 centre{};
    std::array<Point2, 4> corners{};
};

/// Rotating-calipers minimum-area rectangle.
///
/// By the Freeman-Shapira theorem the minimum-area enclosing rectangle of a convex
/// polygon shares an edge with that polygon, so testing one orientation per hull edge
/// is exhaustive rather than a sampled approximation.
inline MinAreaRect min_area_rect(const Ring& poly) {
    MinAreaRect best;
    const Ring h = convex_hull(poly);
    if (h.size() < 3) {
        if (h.size() == 2) {
            best.length_m = std::sqrt(dist2(h[0], h[1]));
            best.angle_rad = std::atan2(h[1].y - h[0].y, h[1].x - h[0].x);
            best.centre = Point2{(h[0].x + h[1].x) * 0.5, (h[0].y + h[1].y) * 0.5};
        } else if (h.size() == 1) {
            best.centre = h[0];
        }
        best.corners = {best.centre, best.centre, best.centre, best.centre};
        return best;
    }

    double best_area = std::numeric_limits<double>::max();
    const std::size_t n = h.size();
    for (std::size_t i = 0; i < n; ++i) {
        const Point2& a = h[i];
        const Point2& b = h[(i + 1) % n];
        const double ex = b.x - a.x;
        const double ey = b.y - a.y;
        const double elen = std::sqrt(ex * ex + ey * ey);
        if (elen < 1e-9) continue;
        const double ux = ex / elen;
        const double uy = ey / elen;
        const double vx = -uy;
        const double vy = ux;

        double min_u = std::numeric_limits<double>::max(), max_u = -min_u;
        double min_v = std::numeric_limits<double>::max(), max_v = -min_v;
        for (const auto& p : h) {
            const double du = (p.x - a.x) * ux + (p.y - a.y) * uy;
            const double dv = (p.x - a.x) * vx + (p.y - a.y) * vy;
            min_u = std::min(min_u, du);
            max_u = std::max(max_u, du);
            min_v = std::min(min_v, dv);
            max_v = std::max(max_v, dv);
        }
        const double su = max_u - min_u;
        const double sv = max_v - min_v;
        const double ar = su * sv;
        if (ar >= best_area) continue;

        best_area = ar;
        const double cu = (min_u + max_u) * 0.5;
        const double cv = (min_v + max_v) * 0.5;
        best.centre = Point2{a.x + ux * cu + vx * cv, a.y + uy * cu + vy * cv};
        best.length_m = std::max(su, sv);
        best.width_m = std::min(su, sv);
        best.angle_rad = (su >= sv) ? std::atan2(uy, ux) : std::atan2(vy, vx);

        const auto corner = [&](double u, double v) {
            return Point2{a.x + ux * u + vx * v, a.y + uy * u + vy * v};
        };
        best.corners = {corner(min_u, min_v), corner(max_u, min_v), corner(max_u, max_v),
                        corner(min_u, max_v)};
    }
    return best;
}

}  // namespace parkfit::geo
