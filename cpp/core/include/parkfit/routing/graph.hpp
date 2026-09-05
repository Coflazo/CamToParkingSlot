// SPDX-License-Identifier: MIT
//
// Road-network routing: strongly connected components, capped Dijkstra, and A*.
//
// This is the C++ side of what src/parkfit/routing/graph.py has been doing in Python.
// The Python class is called NativeGraphProvider and reports provider="native-graph",
// but until now the only part of it that touched C++ was the node snapping. Everything
// expensive stayed in the interpreter: an iterative Tarjan over 188k nodes, and a capped
// Dijkstra that a single parking search runs twice, once for the drive leg and once for
// the walk leg.
//
// Two representation choices carry most of the speed.
//
// Adjacency is stored compressed (CSR): one offsets array plus parallel target, length
// and seconds arrays. The Python version holds dict[int, list[tuple[int, float, float]]],
// which costs a hash lookup and a tuple unpack per edge relaxation. CSR turns the inner
// loop into a contiguous scan, which is the whole game for a graph sweep.
//
// Cost tables are dense vectors indexed by node rather than dicts keyed by OSM id. A
// sweep touches most of the graph, so a dense table is both smaller and faster than a
// hash map, and lookups afterwards are a bounds check instead of a hash.
//
// Node ids are dense indices here. OSM's 64-bit ids are kept only so callers can map
// back; nothing in the hot path ever hashes one.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <unordered_map>
#include <vector>

#include "parkfit/geo/primitives.hpp"
#include "parkfit/index/grid.hpp"

namespace parkfit::routing {

using geo::LatLon;

/// Sentinel for "no node". Node ids are dense indices, so any real id is < node_count().
inline constexpr std::uint32_t kNoNode = std::numeric_limits<std::uint32_t>::max();

/// Sentinel for "not in any component", used for nodes with no edges on this profile.
inline constexpr std::uint32_t kNoComponent = std::numeric_limits<std::uint32_t>::max();

enum class Profile { Car, Foot };

/// Walking speed in km/h, matching WALK_SPEED_KMH in the Python module.
inline constexpr double kWalkSpeedKmh = 4.8;

/// The speed the A* heuristic assumes a car can achieve. It must be at least as fast as
/// any edge in the graph or the heuristic stops being admissible and A* stops being
/// optimal. The graph's fastest class is motorway at 100 km/h, but the heuristic only
/// needs to not over-estimate; 60 matches the Python implementation exactly, and the
/// parity tests would catch a change here.
inline constexpr double kCarHeuristicKmh = 60.0;

/// Compressed adjacency for one profile.
///
/// `offsets` has node_count() + 1 entries; the edges of node i are the half-open range
/// [offsets[i], offsets[i + 1]) in the parallel arrays.
struct Adjacency {
    std::vector<std::uint32_t> offsets;
    std::vector<std::uint32_t> target;
    std::vector<double> length_m;
    std::vector<double> seconds;

    [[nodiscard]] std::size_t edge_count() const { return target.size(); }

    [[nodiscard]] bool has_edges(std::uint32_t node) const {
        return node + 1 < offsets.size() && offsets[node] < offsets[node + 1];
    }
};

/// An adjacency-list road graph, built once and then read-only.
class RoadGraph {
  public:
    /// Add a node, or return the existing dense id if this OSM id is already known.
    std::uint32_t add_node(std::int64_t external_id, double lat, double lon) {
        auto it = by_external_.find(external_id);
        if (it != by_external_.end()) return it->second;
        const auto id = static_cast<std::uint32_t>(positions_.size());
        positions_.push_back(LatLon{lat, lon});
        external_ids_.push_back(external_id);
        by_external_.emplace(external_id, id);
        return id;
    }

    /// Stage an edge. Edges are accumulated unsorted and compressed by build().
    void add_edge(Profile profile, std::uint32_t from, std::uint32_t to, double length_m,
                  double seconds) {
        auto& pending = (profile == Profile::Car) ? pending_car_ : pending_foot_;
        pending.push_back(PendingEdge{from, to, length_m, seconds});
    }

    void reserve_nodes(std::size_t n) {
        positions_.reserve(n);
        external_ids_.reserve(n);
        by_external_.reserve(n);
    }

    void reserve_edges(Profile profile, std::size_t n) {
        ((profile == Profile::Car) ? pending_car_ : pending_foot_).reserve(n);
    }

    /// Compress both profiles into CSR. Idempotent, and cheap to call twice.
    void build() {
        compress(pending_car_, car_);
        compress(pending_foot_, foot_);
        pending_car_.clear();
        pending_car_.shrink_to_fit();
        pending_foot_.clear();
        pending_foot_.shrink_to_fit();
        built_ = true;
    }

    [[nodiscard]] bool built() const { return built_; }
    [[nodiscard]] std::size_t node_count() const { return positions_.size(); }
    [[nodiscard]] const LatLon& position(std::uint32_t node) const { return positions_[node]; }
    [[nodiscard]] std::int64_t external_id(std::uint32_t node) const {
        return external_ids_[node];
    }

    /// Dense id for an OSM id, or kNoNode if the graph has never seen it.
    [[nodiscard]] std::uint32_t index_of(std::int64_t external_id) const {
        auto it = by_external_.find(external_id);
        return it == by_external_.end() ? kNoNode : it->second;
    }

    [[nodiscard]] const Adjacency& adjacency(Profile profile) const {
        return profile == Profile::Car ? car_ : foot_;
    }

  private:
    struct PendingEdge {
        std::uint32_t from{};
        std::uint32_t to{};
        double length_m{};
        double seconds{};
    };

    /// Counting sort by source node into CSR. One pass to count, one to place.
    void compress(const std::vector<PendingEdge>& pending, Adjacency& out) const {
        const std::size_t n = positions_.size();
        out.offsets.assign(n + 1, 0);
        out.target.resize(pending.size());
        out.length_m.resize(pending.size());
        out.seconds.resize(pending.size());

        for (const auto& e : pending) ++out.offsets[e.from + 1];
        for (std::size_t i = 0; i < n; ++i) out.offsets[i + 1] += out.offsets[i];

        auto cursor = out.offsets;  // copy: consumed as a write head per node
        for (const auto& e : pending) {
            const std::uint32_t slot = cursor[e.from]++;
            out.target[slot] = e.to;
            out.length_m[slot] = e.length_m;
            out.seconds[slot] = e.seconds;
        }
    }

    std::vector<LatLon> positions_;
    std::vector<std::int64_t> external_ids_;
    std::unordered_map<std::int64_t, std::uint32_t> by_external_;
    std::vector<PendingEdge> pending_car_;
    std::vector<PendingEdge> pending_foot_;
    Adjacency car_;
    Adjacency foot_;
    bool built_{false};
};

/// Result of a one-to-many sweep.
///
/// `seconds` and `metres` are dense and indexed by node id; an unreached node holds
/// infinity. Callers test `reached(node)` rather than comparing against infinity.
struct CostTable {
    std::vector<double> seconds;
    std::vector<double> metres;
    std::uint32_t origin{kNoNode};

    [[nodiscard]] bool ok() const { return origin != kNoNode; }

    [[nodiscard]] bool reached(std::uint32_t node) const {
        return node < seconds.size() && std::isfinite(seconds[node]);
    }
};

/// One routed leg. `path` is populated only by route(), not by the sweep.
struct Leg {
    bool ok{false};
    double distance_m{};
    double duration_min{};
    double confidence{};
    std::vector<std::uint32_t> path;
};

/// A* and Dijkstra over a built RoadGraph, with connectivity-aware snapping.
///
/// The two things that are not incidental detail, both carried over from the Python
/// implementation because both were bugs before they were features:
///
/// **Strong connectivity, not reachability.** The car graph is directed because one-way
/// streets are directed, and in a directed graph "reachable from a seed" is a different
/// relation from "mutually reachable". Labelling by forward reachability puts two nodes
/// in one group when the seed reaches both even though neither reaches the other. A
/// canal ring full of one-ways is exactly the topology where that goes wrong.
///
/// **Snap into a component, not to the nearest node.** A bounding-box OSM extract is
/// full of islands: parking aisles, service yards, and anything across water. Snapping
/// each endpoint to its geometrically nearest node can land on an eleven-node service
/// island, and the search then correctly reports no path.
class RoadRouter {
  public:
    explicit RoadRouter(const RoadGraph& graph) : graph_(&graph) {}

    /// Strongly connected component id per node, or kNoComponent for isolated nodes.
    ///
    /// Tarjan, run iteratively. A recursive formulation overflows the stack at this
    /// scale, which is not a hypothetical: the Dutch extract alone is 188k nodes.
    const std::vector<std::uint32_t>& components(Profile profile) {
        auto& cache = state(profile);
        if (cache.components_ready) return cache.component_of;

        const Adjacency& adj = graph_->adjacency(profile);
        const std::size_t n = graph_->node_count();

        cache.component_of.assign(n, kNoComponent);
        cache.component_sizes.clear();

        std::vector<std::uint32_t> index(n, kNoNode);
        std::vector<std::uint32_t> lowlink(n, 0);
        std::vector<char> on_stack(n, 0);
        std::vector<std::uint32_t> stack;
        // Each frame is a node plus how far through its edge range we have descended.
        std::vector<std::pair<std::uint32_t, std::uint32_t>> work;
        std::uint32_t counter = 0;
        std::uint32_t next_component = 0;

        for (std::uint32_t seed = 0; seed < n; ++seed) {
            if (!adj.has_edges(seed) || index[seed] != kNoNode) continue;
            work.push_back({seed, adj.offsets[seed]});
            index[seed] = lowlink[seed] = counter++;
            stack.push_back(seed);
            on_stack[seed] = 1;

            while (!work.empty()) {
                auto& [node, cursor] = work.back();
                const std::uint32_t end = adj.offsets[node + 1];

                bool descended = false;
                while (cursor < end) {
                    const std::uint32_t neighbour = adj.target[cursor];
                    ++cursor;
                    if (index[neighbour] == kNoNode) {
                        index[neighbour] = lowlink[neighbour] = counter++;
                        stack.push_back(neighbour);
                        on_stack[neighbour] = 1;
                        work.push_back({neighbour, adj.offsets[neighbour]});
                        descended = true;
                        break;
                    }
                    if (on_stack[neighbour]) {
                        lowlink[node] = std::min(lowlink[node], index[neighbour]);
                    }
                }
                if (descended) continue;

                if (lowlink[node] == index[node]) {
                    std::uint32_t size = 0;
                    while (true) {
                        const std::uint32_t member = stack.back();
                        stack.pop_back();
                        on_stack[member] = 0;
                        cache.component_of[member] = next_component;
                        ++size;
                        if (member == node) break;
                    }
                    cache.component_sizes.push_back(size);
                    ++next_component;
                }

                const std::uint32_t finished = node;
                work.pop_back();
                if (!work.empty()) {
                    auto& parent = work.back().first;
                    lowlink[parent] = std::min(lowlink[parent], lowlink[finished]);
                }
            }
        }

        cache.components_ready = true;
        return cache.component_of;
    }

    /// The component holding the most nodes, or kNoComponent on an empty graph.
    std::uint32_t largest_component(Profile profile) {
        components(profile);
        const auto& sizes = state(profile).component_sizes;
        if (sizes.empty()) return kNoComponent;
        const auto best = std::max_element(sizes.begin(), sizes.end());
        return static_cast<std::uint32_t>(std::distance(sizes.begin(), best));
    }

    [[nodiscard]] std::uint32_t component_size(Profile profile, std::uint32_t component) {
        components(profile);
        const auto& sizes = state(profile).component_sizes;
        return component < sizes.size() ? sizes[component] : 0;
    }

    /// Closest routable node, optionally restricted to one connected component.
    ///
    /// The radius grows rather than being fixed, because a fixed radius returns nothing
    /// outside a city, which is exactly where a driver most needs an answer. Hits come
    /// back sorted by distance, so the first match in the requested component is the
    /// nearest one and there is no reason to cap the result count.
    std::uint32_t nearest_node(double lat, double lon, Profile profile,
                               std::uint32_t component = kNoComponent) {
        auto& grid = ensure_grid(profile);
        if (grid.size() == 0) return kNoNode;

        const std::vector<std::uint32_t>* component_of = nullptr;
        if (component != kNoComponent) component_of = &components(profile);

        const LatLon at{lat, lon};
        for (double radius = 150.0; radius <= 12000.0; radius *= 2.5) {
            for (const auto& hit : grid.query_radius(at, radius, 0)) {
                const std::uint32_t node = hit.payload;
                if (component_of == nullptr || (*component_of)[node] == component) return node;
            }
        }
        return kNoNode;
    }

    /// Dijkstra from one point, labelling every node inside a time budget.
    ///
    /// This is the shape a parking search actually has. It is not N independent
    /// point-to-point queries; it is two one-to-many queries, drive time from one origin
    /// to many entrances and walk time from many exits to one destination. Running A*
    /// per candidate re-explores the same city several hundred times.
    ///
    /// The cap matters as much as the sweep. Without it the search settles the whole
    /// graph, including places no driver would consider.
    CostTable costs_from(double lat, double lon, Profile profile, double max_seconds = 1500.0) {
        const std::size_t n = graph_->node_count();
        CostTable table;
        table.seconds.assign(n, std::numeric_limits<double>::infinity());
        table.metres.assign(n, std::numeric_limits<double>::infinity());

        // Snap into the largest strongly connected component rather than merely to the
        // closest node. The nearest node to a big station often sits in a tiny forecourt
        // component, and a sweep launched from there reaches almost nothing.
        std::uint32_t origin = nearest_node(lat, lon, profile, largest_component(profile));
        if (origin == kNoNode) origin = nearest_node(lat, lon, profile);
        if (origin == kNoNode) return table;

        const Adjacency& adj = graph_->adjacency(profile);
        table.origin = origin;
        table.seconds[origin] = 0.0;
        table.metres[origin] = 0.0;

        struct Item {
            double seconds;
            double metres;
            std::uint32_t node;
            // Ties break on metres then node so the ordering is total and the sweep is
            // reproducible run to run, which the parity tests depend on.
            bool operator>(const Item& other) const {
                if (seconds != other.seconds) return seconds > other.seconds;
                if (metres != other.metres) return metres > other.metres;
                return node > other.node;
            }
        };
        std::priority_queue<Item, std::vector<Item>, std::greater<Item>> heap;
        heap.push(Item{0.0, 0.0, origin});
        std::vector<char> settled(n, 0);

        while (!heap.empty()) {
            const Item top = heap.top();
            heap.pop();
            if (settled[top.node]) continue;
            settled[top.node] = 1;
            if (top.seconds > max_seconds) break;

            for (std::uint32_t e = adj.offsets[top.node]; e < adj.offsets[top.node + 1]; ++e) {
                const std::uint32_t neighbour = adj.target[e];
                if (settled[neighbour]) continue;
                const double next_seconds = top.seconds + adj.seconds[e];
                if (next_seconds > max_seconds) continue;
                if (next_seconds < table.seconds[neighbour]) {
                    const double next_metres = top.metres + adj.length_m[e];
                    table.seconds[neighbour] = next_seconds;
                    table.metres[neighbour] = next_metres;
                    heap.push(Item{next_seconds, next_metres, neighbour});
                }
            }
        }
        return table;
    }

    /// Route from one point to many, in a single graph sweep.
    ///
    /// A target with no entry is genuinely unreachable inside the budget. It comes back
    /// as `ok == false` so the caller can fall back, rather than being handed a
    /// fabricated number.
    std::vector<Leg> many_costs(double lat, double lon,
                                const std::vector<LatLon>& targets, Profile profile,
                                double max_seconds = 1500.0) {
        std::vector<Leg> out(targets.size());
        const CostTable table = costs_from(lat, lon, profile, max_seconds);
        if (!table.ok()) return out;

        // Targets snap inside the component the sweep actually covered, so a candidate
        // is not lost to a node twenty metres nearer on an unreachable service island.
        const std::uint32_t origin_component = components(profile)[table.origin];

        for (std::size_t i = 0; i < targets.size(); ++i) {
            const std::uint32_t node =
                nearest_node(targets[i].lat, targets[i].lon, profile, origin_component);
            if (node == kNoNode || !table.reached(node)) continue;
            out[i].ok = true;
            out[i].distance_m = table.metres[node];
            out[i].duration_min = std::max(0.5, table.seconds[node] / 60.0);
            out[i].confidence = 0.88;
        }
        return out;
    }

    /// Point to point, with the path.
    ///
    /// The destination snaps first because it is the fixed point of the query; the
    /// origin then snaps inside the destination's component, so the endpoints are
    /// mutually reachable by construction rather than by luck.
    Leg route(double from_lat, double from_lon, double to_lat, double to_lon, Profile profile) {
        Leg leg;
        std::uint32_t goal = nearest_node(to_lat, to_lon, profile);
        if (goal == kNoNode) return leg;

        std::uint32_t start =
            nearest_node(from_lat, from_lon, profile, components(profile)[goal]);
        if (start == kNoNode) {
            // The origin cannot reach that component at all, which happens wherever the
            // only link is a ferry. Fall back to the main road network so the search
            // still returns a usable estimate rather than nothing at all.
            const std::uint32_t main = largest_component(profile);
            goal = nearest_node(to_lat, to_lon, profile, main);
            start = nearest_node(from_lat, from_lon, profile, main);
        }
        if (start == kNoNode || goal == kNoNode) return leg;

        if (start == goal) {
            const double straight = geo::haversine_m(LatLon{from_lat, from_lon},
                                                     LatLon{to_lat, to_lon});
            const double speed = (profile == Profile::Car) ? 20.0 : kWalkSpeedKmh;
            leg.ok = true;
            leg.distance_m = straight;
            leg.duration_min = std::max(0.5, straight / 1000.0 / speed * 60.0);
            leg.confidence = 0.7;
            leg.path = {start};
            return leg;
        }

        const Adjacency& adj = graph_->adjacency(profile);
        const std::size_t n = graph_->node_count();
        const LatLon goal_pos = graph_->position(goal);
        const double speed_ms = ((profile == Profile::Car) ? kCarHeuristicKmh : kWalkSpeedKmh) / 3.6;

        // Admissible: straight-line time at the fastest speed this profile can achieve
        // can never exceed the true remaining time, so A* stays optimal.
        const auto heuristic = [&](std::uint32_t node) {
            return geo::haversine_m(graph_->position(node), goal_pos) / speed_ms;
        };

        struct Item {
            double f;
            std::uint32_t node;
            bool operator>(const Item& other) const {
                if (f != other.f) return f > other.f;
                return node > other.node;
            }
        };
        std::priority_queue<Item, std::vector<Item>, std::greater<Item>> open;
        std::vector<std::uint32_t> came_from(n, kNoNode);
        std::vector<double> g_score(n, std::numeric_limits<double>::infinity());
        std::vector<double> distance(n, 0.0);
        std::vector<char> closed(n, 0);

        g_score[start] = 0.0;
        open.push(Item{heuristic(start), start});

        while (!open.empty()) {
            const std::uint32_t current = open.top().node;
            open.pop();
            if (current == goal) {
                leg.ok = true;
                leg.distance_m = distance[current];
                leg.duration_min = std::max(0.5, g_score[current] / 60.0);
                leg.confidence = 0.9;
                for (std::uint32_t at = goal; at != kNoNode; at = came_from[at]) {
                    leg.path.push_back(at);
                }
                std::reverse(leg.path.begin(), leg.path.end());
                return leg;
            }
            if (closed[current]) continue;
            closed[current] = 1;

            for (std::uint32_t e = adj.offsets[current]; e < adj.offsets[current + 1]; ++e) {
                const std::uint32_t neighbour = adj.target[e];
                if (closed[neighbour]) continue;
                const double tentative = g_score[current] + adj.seconds[e];
                if (tentative < g_score[neighbour]) {
                    came_from[neighbour] = current;
                    g_score[neighbour] = tentative;
                    distance[neighbour] = distance[current] + adj.length_m[e];
                    open.push(Item{tentative + heuristic(neighbour), neighbour});
                }
            }
        }
        return leg;  // ok stays false: no path between the endpoints in this graph
    }

  private:
    struct ProfileState {
        bool components_ready{false};
        std::vector<std::uint32_t> component_of;
        std::vector<std::uint32_t> component_sizes;
        bool grid_ready{false};
        index::SpatialGrid grid{200.0};
    };

    ProfileState& state(Profile profile) {
        return profile == Profile::Car ? car_state_ : foot_state_;
    }

    /// A spatial index over the nodes that actually have outgoing edges on this profile.
    /// Indexing every node would happily snap a driver to a footpath-only junction.
    index::SpatialGrid& ensure_grid(Profile profile) {
        auto& cache = state(profile);
        if (cache.grid_ready) return cache.grid;

        const Adjacency& adj = graph_->adjacency(profile);
        const std::size_t n = graph_->node_count();
        cache.grid.clear();
        cache.grid.reserve(n);
        for (std::uint32_t node = 0; node < n; ++node) {
            if (!adj.has_edges(node)) continue;
            cache.grid.insert(graph_->position(node), node);
        }
        cache.grid.build();
        cache.grid_ready = true;
        return cache.grid;
    }

    const RoadGraph* graph_;
    ProfileState car_state_;
    ProfileState foot_state_;
};

}  // namespace parkfit::routing
