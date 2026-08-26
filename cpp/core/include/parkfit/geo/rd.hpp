// SPDX-License-Identifier: MIT
//
// Rijksdriehoek (RD New, EPSG:28992) <-> WGS84 conversion.
//
// Every parking bay published by the City of Amsterdam is expressed in RD New,
// so this transform sits on the critical path of the whole product. We use the
// approximation published by the Dutch Kadaster ("Benaderingsformules", Schreutelkamp
// & Strang van Hees), which is accurate to a few decimetres across the Netherlands.
// That is far below the ~10 cm precision that parking-fit decisions actually need, and it
// avoids dragging a full datum-shift grid into the hot path.
//
// Reference point (Onze Lieve Vrouwetoren, Amersfoort):
//   RD    X = 155000.000, Y = 463000.000
//   WGS84 lat = 52.15517440, lon = 5.38720621

#pragma once

#include <array>
#include <cmath>

namespace parkfit::geo {

/// A WGS84 geographic coordinate in decimal degrees.
struct LatLon {
    double lat{};
    double lon{};
};

/// A Rijksdriehoek (EPSG:28992) coordinate in metres.
struct RdPoint {
    double x{};
    double y{};
};

namespace detail {

// Origin of the RD system, in both frames.
inline constexpr double kRdOriginX = 155000.0;
inline constexpr double kRdOriginY = 463000.0;
inline constexpr double kRefLat = 52.15517440;
inline constexpr double kRefLon = 5.38720621;

// Coefficients for RD -> WGS84. Index pairs are the (p, q) powers of the scaled
// RD offsets; the value is the coefficient K (for latitude) or L (for longitude).
struct PolyTerm {
    int p;
    int q;
    double c;
};

inline constexpr std::array<PolyTerm, 11> kLatTerms{{
    {0, 1, 3235.65389},  {2, 0, -32.58297}, {0, 2, -0.24750}, {2, 1, -0.84978},
    {0, 3, -0.06550},    {2, 2, -0.01709},  {1, 0, -0.00738}, {4, 0, 0.00530},
    {2, 3, -0.00039},    {4, 1, 0.00033},   {1, 1, -0.00012},
}};

inline constexpr std::array<PolyTerm, 11> kLonTerms{{
    {1, 0, 5260.52916}, {1, 1, 105.94684}, {1, 2, 2.45656},  {3, 0, -0.81885},
    {1, 3, 0.05594},    {3, 1, -0.05607},  {0, 1, 0.01199},  {3, 2, -0.00256},
    {1, 4, 0.00128},    {0, 2, 0.00022},   {2, 0, -0.00022},
}};

// Coefficients for WGS84 -> RD.
inline constexpr std::array<PolyTerm, 11> kXTerms{{
    {0, 1, 190094.945}, {1, 1, -11832.228}, {2, 1, -114.221}, {0, 3, -32.391},
    {1, 0, -0.705},     {3, 1, -2.340},     {1, 3, -0.608},   {0, 2, -0.008},
    {2, 3, 0.148},      {4, 1, 0.0},        {0, 5, 0.0},
}};

inline constexpr std::array<PolyTerm, 12> kYTerms{{
    {1, 0, 309056.544}, {0, 2, 3638.893}, {2, 0, 73.077},   {1, 2, -157.984},
    {3, 0, 59.788},     {0, 1, 0.433},    {2, 2, -6.439},   {1, 1, -0.032},
    {0, 4, 0.092},      {1, 4, -0.054},   {3, 2, 0.0},      {5, 0, 0.0},
}};

inline double ipow(double base, int exp) {
    double r = 1.0;
    for (int i = 0; i < exp; ++i) r *= base;
    return r;
}

}  // namespace detail

/// Convert RD New (EPSG:28992) metres to WGS84 degrees.
inline LatLon rd_to_wgs84(const RdPoint& rd) {
    const double dx = (rd.x - detail::kRdOriginX) * 1e-5;
    const double dy = (rd.y - detail::kRdOriginY) * 1e-5;

    double dlat = 0.0;
    for (const auto& t : detail::kLatTerms) {
        dlat += t.c * detail::ipow(dx, t.p) * detail::ipow(dy, t.q);
    }
    double dlon = 0.0;
    for (const auto& t : detail::kLonTerms) {
        dlon += t.c * detail::ipow(dx, t.p) * detail::ipow(dy, t.q);
    }
    // Polynomials yield arc-seconds; convert to degrees.
    return LatLon{detail::kRefLat + dlat / 3600.0, detail::kRefLon + dlon / 3600.0};
}

/// Convert WGS84 degrees to RD New (EPSG:28992) metres.
inline RdPoint wgs84_to_rd(const LatLon& ll) {
    const double dlat = 0.36 * (ll.lat - detail::kRefLat);
    const double dlon = 0.36 * (ll.lon - detail::kRefLon);

    double x = detail::kRdOriginX;
    for (const auto& t : detail::kXTerms) {
        x += t.c * detail::ipow(dlat, t.p) * detail::ipow(dlon, t.q);
    }
    double y = detail::kRdOriginY;
    for (const auto& t : detail::kYTerms) {
        y += t.c * detail::ipow(dlat, t.p) * detail::ipow(dlon, t.q);
    }
    return RdPoint{x, y};
}

/// True when a coordinate falls inside the RD system's valid envelope. Outside this
/// box the approximation degrades quickly, so callers should reject rather than guess.
inline bool rd_in_range(const RdPoint& rd) {
    return rd.x >= -7000.0 && rd.x <= 300000.0 && rd.y >= 289000.0 && rd.y <= 629000.0;
}

}  // namespace parkfit::geo
