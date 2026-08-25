// SPDX-License-Identifier: MIT
//
// Uniform-grid spatial index over parking features.
//
// Why a grid rather than an R-tree: the workload is a radius sweep around a destination
// over a static, uniformly-dense point set (roughly 250k Amsterdam bays plus ~15k national
// facilities). A grid gives O(1) cell lookup with no tree descent, rebuilds in one pass
// after each ingest, and has no rebalancing to get wrong. An R-tree would win on wildly
// non-uniform data or frequent mutation; we have neither.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "parkfit/geo/primitives.hpp"

namespace parkfit::index {

using geo::LatLon;

/// One indexed feature. `payload` is an opaque handle into the caller table, so the
/// index never owns or copies domain objects.
struct Entry {
    LatLon position{};
    std::uint32_t payload{};
};

struct Hit {
    std::uint32_t payload{};
    double distance_m{};
};

/// A grid of fixed angular cell size covering the Netherlands.
///
/// Cell size is expressed in degrees of latitude and converted for longitude at the
/// mean latitude of the data, which keeps cells roughly square in metres. The country
/// spans only ~3.5 degrees of latitude, so a single conversion factor is accurate enough
/// that no cell is more than a few percent from square.
class SpatialGrid {
  public:
    /// `cell_size_m` trades memory against the number of candidates scanned per query.
    /// 250 m keeps a typical 1 km query to ~80 cells while holding a handful of entries
    /// each, which is comfortably cache-friendly.
    explicit SpatialGrid(double cell_size_m = 250.0) : cell_size_m_(cell_size_m) {}

    void reserve(std::size_t n) { entries_.reserve(n); }

    void insert(const LatLon& p, std::uint32_t payload) {
        entries_.push_back(Entry{p, payload});
        dirty_ = true;
    }

    void clear() {
        entries_.clear();
        cells_.clear();
        dirty_ = false;
    }

    [[nodiscard]] std::size_t size() const { return entries_.size(); }

    /// Build the cell map. Called automatically by queries when needed.
    void build() {
        cells_.clear();
        if (entries_.empty()) {
            dirty_ = false;
            return;
        }
        double sum_lat = 0.0;
        for (const auto& e : entries_) sum_lat += e.position.lat;
        mean_lat_ = sum_lat / static_cast<double>(entries_.size());

        lat_step_ = geo::rad2deg(cell_size_m_ / geo::kEarthRadiusM);
        const double coslat = std::max(0.05, std::cos(geo::deg2rad(mean_lat_)));
        lon_step_ = lat_step_ / coslat;

        cells_.reserve(entries_.size());
        for (std::uint32_t i = 0; i < entries_.size(); ++i) {
            cells_[key_of(entries_[i].position)].push_back(i);
        }
        dirty_ = false;
    }

    /// All entries within `radius_m`, sorted nearest first.
    [[nodiscard]] std::vector<Hit> query_radius(const LatLon& centre, double radius_m,
                                                std::size_t max_results = 0) {
        std::vector<Hit> hits;
        if (dirty_) build();
        if (entries_.empty() || radius_m <= 0.0) return hits;

        const int span_lat = static_cast<int>(std::ceil(geo::rad2deg(radius_m / geo::kEarthRadiusM) /
                                                        lat_step_));
        const int span_lon = static_cast<int>(
            std::ceil((geo::rad2deg(radius_m / geo::kEarthRadiusM) /
                       std::max(0.05, std::cos(geo::deg2rad(centre.lat)))) /
                      lon_step_));

        const auto [c_ix, c_iy] = index_of(centre);
        for (int dx = -span_lon; dx <= span_lon; ++dx) {
            for (int dy = -span_lat; dy <= span_lat; ++dy) {
                auto it = cells_.find(pack(c_ix + dx, c_iy + dy));
                if (it == cells_.end()) continue;
                for (std::uint32_t idx : it->second) {
                    const double d = geo::haversine_m(centre, entries_[idx].position);
                    if (d <= radius_m) hits.push_back(Hit{entries_[idx].payload, d});
                }
            }
        }
        std::sort(hits.begin(), hits.end(),
                  [](const Hit& a, const Hit& b) { return a.distance_m < b.distance_m; });
        if (max_results > 0 && hits.size() > max_results) hits.resize(max_results);
        return hits;
    }

    /// Nearest `k` entries, expanding the radius until enough are found or the cap is hit.
    ///
    /// Expanding search is what makes a destination in a sparse area still return
    /// something: a fixed radius would come back empty outside the cities, which is
    /// exactly where the driver most needs a suggestion.
    [[nodiscard]] std::vector<Hit> query_knn(const LatLon& centre, std::size_t k,
                                             double start_radius_m = 500.0,
                                             double max_radius_m = 20000.0) {
        std::vector<Hit> hits;
        double r = start_radius_m;
        while (r <= max_radius_m) {
            hits = query_radius(centre, r);
            if (hits.size() >= k) break;
            r *= 2.0;
        }
        if (hits.size() > k) hits.resize(k);
        return hits;
    }

  private:
    static std::uint64_t pack(std::int32_t ix, std::int32_t iy) {
        return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(ix)) << 32) |
               static_cast<std::uint32_t>(iy);
    }

    [[nodiscard]] std::pair<std::int32_t, std::int32_t> index_of(const LatLon& p) const {
        return {static_cast<std::int32_t>(std::floor(p.lon / lon_step_)),
                static_cast<std::int32_t>(std::floor(p.lat / lat_step_))};
    }

    [[nodiscard]] std::uint64_t key_of(const LatLon& p) const {
        const auto [ix, iy] = index_of(p);
        return pack(ix, iy);
    }

    double cell_size_m_;
    double lat_step_{0.00225};
    double lon_step_{0.00366};
    double mean_lat_{52.1};
    bool dirty_{true};
    std::vector<Entry> entries_;
    std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> cells_;
};

}  // namespace parkfit::index
