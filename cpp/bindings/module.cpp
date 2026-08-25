// SPDX-License-Identifier: MIT
//
// Python bindings for the ParkFit core.
//
// The split of work is deliberate. Python owns orchestration, I/O and anything that
// talks to a network or a database. C++ owns the arithmetic that runs per candidate,
// per search: coordinate transforms, the spatial sweep over ~250k bays, vehicle fit,
// and the generalised-cost ranking. A single search touches a few hundred candidates
// and each one needs a fit verdict and a score, so this is the hot path.
//
// It also removes two external dependencies. Radius search happens here rather than in
// PostGIS, and the ranking is a pure function rather than a service call, which is why
// the whole product runs on a laptop with no containers.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <string>
#include <vector>

#include "parkfit/fit/vehicle.hpp"
#include "parkfit/fit/vehicle_fit.hpp"
#include "parkfit/geo/polygon.hpp"
#include "parkfit/geo/primitives.hpp"
#include "parkfit/geo/rd.hpp"
#include "parkfit/index/grid.hpp"
#include "parkfit/rank/score.hpp"

namespace py = pybind11;
using namespace parkfit;

PYBIND11_MODULE(parkfit_native, m) {
    m.doc() = "ParkFit NL native core: geodesy, spatial index, vehicle fit and ranking.";
    m.attr("__version__") = "0.1.0";

    // ----------------------------------------------------------------- geo
    py::class_<geo::LatLon>(m, "LatLon")
        .def(py::init<>())
        .def(py::init([](double lat, double lon) { return geo::LatLon{lat, lon}; }),
             py::arg("lat"), py::arg("lon"))
        .def_readwrite("lat", &geo::LatLon::lat)
        .def_readwrite("lon", &geo::LatLon::lon)
        .def("__repr__", [](const geo::LatLon& p) {
            return "LatLon(" + std::to_string(p.lat) + ", " + std::to_string(p.lon) + ")";
        });

    py::class_<geo::RdPoint>(m, "RdPoint")
        .def(py::init<>())
        .def(py::init([](double x, double y) { return geo::RdPoint{x, y}; }), py::arg("x"),
             py::arg("y"))
        .def_readwrite("x", &geo::RdPoint::x)
        .def_readwrite("y", &geo::RdPoint::y);

    m.def(
        "rd_to_wgs84",
        [](double x, double y) {
            const auto ll = geo::rd_to_wgs84(geo::RdPoint{x, y});
            return py::make_tuple(ll.lat, ll.lon);
        },
        py::arg("x"), py::arg("y"),
        "Convert RD New (EPSG:28992) metres to (lat, lon) degrees.");

    m.def(
        "wgs84_to_rd",
        [](double lat, double lon) {
            const auto rd = geo::wgs84_to_rd(geo::LatLon{lat, lon});
            return py::make_tuple(rd.x, rd.y);
        },
        py::arg("lat"), py::arg("lon"), "Convert WGS84 degrees to RD New (x, y) metres.");

    m.def("rd_in_range",
          [](double x, double y) { return geo::rd_in_range(geo::RdPoint{x, y}); },
          py::arg("x"), py::arg("y"));

    m.def(
        "haversine_m",
        [](double lat1, double lon1, double lat2, double lon2) {
            return geo::haversine_m(geo::LatLon{lat1, lon1}, geo::LatLon{lat2, lon2});
        },
        py::arg("lat1"), py::arg("lon1"), py::arg("lat2"), py::arg("lon2"),
        "Great-circle distance in metres.");

    m.def(
        "bearing_deg",
        [](double lat1, double lon1, double lat2, double lon2) {
            return geo::bearing_deg(geo::LatLon{lat1, lon1}, geo::LatLon{lat2, lon2});
        },
        py::arg("lat1"), py::arg("lon1"), py::arg("lat2"), py::arg("lon2"));

    py::class_<geo::BayMeasurement>(m, "BayMeasurement")
        .def_readonly("length_m", &geo::BayMeasurement::length_m)
        .def_readonly("width_m", &geo::BayMeasurement::width_m)
        .def_readonly("max_length_m", &geo::BayMeasurement::max_length_m)
        .def_readonly("max_width_m", &geo::BayMeasurement::max_width_m)
        .def_readonly("angle_rad", &geo::BayMeasurement::angle_rad)
        .def_readonly("fill_ratio", &geo::BayMeasurement::fill_ratio)
        .def_property_readonly("length_cm",
                               [](const geo::BayMeasurement& b) { return b.length_m * 100.0; })
        .def_property_readonly("width_cm",
                               [](const geo::BayMeasurement& b) { return b.width_m * 100.0; })
        .def_property_readonly(
            "centre", [](const geo::BayMeasurement& b) {
                return py::make_tuple(b.centre.x, b.centre.y);
            });

    m.def(
        "measure_bay",
        [](const std::vector<std::pair<double, double>>& ring) {
            geo::Ring r;
            r.reserve(ring.size());
            for (const auto& [x, y] : ring) r.push_back(geo::Point2{x, y});
            return geo::measure_bay(r);
        },
        py::arg("ring"),
        "Conservative usable dimensions of a bay polygon given in RD metres.");

    m.def(
        "ring_area",
        [](const std::vector<std::pair<double, double>>& ring) {
            geo::Ring r;
            r.reserve(ring.size());
            for (const auto& [x, y] : ring) r.push_back(geo::Point2{x, y});
            return geo::area(r);
        },
        py::arg("ring"));

    // --------------------------------------------------------------- index
    py::class_<index::Hit>(m, "Hit")
        .def_readonly("payload", &index::Hit::payload)
        .def_readonly("distance_m", &index::Hit::distance_m)
        .def("__repr__", [](const index::Hit& h) {
            return "Hit(payload=" + std::to_string(h.payload) + ", distance_m=" +
                   std::to_string(h.distance_m) + ")";
        });

    py::class_<index::SpatialGrid>(m, "SpatialGrid")
        .def(py::init<double>(), py::arg("cell_size_m") = 250.0)
        .def("reserve", &index::SpatialGrid::reserve, py::arg("n"))
        .def(
            "insert",
            [](index::SpatialGrid& g, double lat, double lon, std::uint32_t payload) {
                g.insert(geo::LatLon{lat, lon}, payload);
            },
            py::arg("lat"), py::arg("lon"), py::arg("payload"))
        .def(
            "insert_many",
            [](index::SpatialGrid& g,
               const std::vector<std::tuple<double, double, std::uint32_t>>& items) {
                g.reserve(g.size() + items.size());
                for (const auto& [lat, lon, payload] : items) {
                    g.insert(geo::LatLon{lat, lon}, payload);
                }
            },
            py::arg("items"),
            "Bulk insert (lat, lon, payload) triples. One crossing of the boundary "
            "instead of one per row.")
        .def("build", &index::SpatialGrid::build)
        .def("clear", &index::SpatialGrid::clear)
        .def("__len__", &index::SpatialGrid::size)
        .def(
            "query_radius",
            [](index::SpatialGrid& g, double lat, double lon, double radius_m,
               std::size_t max_results) {
                return g.query_radius(geo::LatLon{lat, lon}, radius_m, max_results);
            },
            py::arg("lat"), py::arg("lon"), py::arg("radius_m"), py::arg("max_results") = 0)
        .def(
            "query_knn",
            [](index::SpatialGrid& g, double lat, double lon, std::size_t k,
               double start_radius_m, double max_radius_m) {
                return g.query_knn(geo::LatLon{lat, lon}, k, start_radius_m, max_radius_m);
            },
            py::arg("lat"), py::arg("lon"), py::arg("k"), py::arg("start_radius_m") = 500.0,
            py::arg("max_radius_m") = 20000.0);

    // ----------------------------------------------------------------- fit
    py::enum_<fit::Verdict>(m, "Verdict")
        .value("FITS", fit::Verdict::Fits)
        .value("TIGHT_FIT", fit::Verdict::TightFit)
        .value("DOES_NOT_FIT", fit::Verdict::DoesNotFit)
        .value("UNVERIFIED", fit::Verdict::Unverified)
        .def("__str__", [](fit::Verdict v) { return std::string(fit::to_string(v)); });

    py::enum_<fit::BayOrientation>(m, "BayOrientation")
        .value("PARALLEL", fit::BayOrientation::Parallel)
        .value("PERPENDICULAR", fit::BayOrientation::Perpendicular)
        .value("ANGLED", fit::BayOrientation::Angled)
        .value("UNKNOWN", fit::BayOrientation::Unknown);

    m.def("orientation_from_dutch", &fit::orientation_from_dutch, py::arg("value"),
          "Map the Amsterdam parkeervakken type field (Langs/Haaks/Visgraat).");

    py::class_<fit::DimensionProvenance>(m, "DimensionProvenance")
        .def(py::init<>())
        .def_readwrite("length_confirmed", &fit::DimensionProvenance::length_confirmed)
        .def_readwrite("width_confirmed", &fit::DimensionProvenance::width_confirmed)
        .def_readwrite("height_confirmed", &fit::DimensionProvenance::height_confirmed)
        .def_readwrite("weight_confirmed", &fit::DimensionProvenance::weight_confirmed);

    py::class_<fit::Vehicle>(m, "Vehicle")
        .def(py::init<>())
        .def_readwrite("id", &fit::Vehicle::id)
        .def_readwrite("nickname", &fit::Vehicle::nickname)
        .def_readwrite("length_cm", &fit::Vehicle::length_cm)
        .def_readwrite("body_width_cm", &fit::Vehicle::body_width_cm)
        .def_readwrite("width_with_mirrors_cm", &fit::Vehicle::width_with_mirrors_cm)
        .def_readwrite("height_cm", &fit::Vehicle::height_cm)
        .def_readwrite("height_with_accessories_cm", &fit::Vehicle::height_with_accessories_cm)
        .def_readwrite("weight_kg", &fit::Vehicle::weight_kg)
        .def_readwrite("is_ev", &fit::Vehicle::is_ev)
        .def_readwrite("has_trailer", &fit::Vehicle::has_trailer)
        .def_readwrite("has_roof_box", &fit::Vehicle::has_roof_box)
        .def_readwrite("extra_parallel_clearance_cm", &fit::Vehicle::extra_parallel_clearance_cm)
        .def_readwrite("provenance", &fit::Vehicle::provenance)
        .def("effective_height_cm", &fit::Vehicle::effective_height_cm)
        .def("effective_width_cm", &fit::Vehicle::effective_width_cm,
             "Width across the mirrors: governs physical apertures.")
        .def("effective_body_width_cm", &fit::Vehicle::effective_body_width_cm,
             "Width across the bodywork: governs painted bays, which mirrors overhang.")
        .def("has_usable_dimensions", &fit::Vehicle::has_usable_dimensions);

    py::class_<fit::Margins>(m, "Margins")
        .def(py::init<>())
        .def_readwrite("vertical_cm", &fit::Margins::vertical_cm)
        .def_readwrite("lateral_total_cm", &fit::Margins::lateral_total_cm)
        .def_readwrite("bay_lateral_total_cm", &fit::Margins::bay_lateral_total_cm)
        .def_readwrite("longitudinal_total_cm", &fit::Margins::longitudinal_total_cm)
        .def_readwrite("parallel_front_cm", &fit::Margins::parallel_front_cm)
        .def_readwrite("parallel_rear_cm", &fit::Margins::parallel_rear_cm)
        .def_readwrite("tight_threshold_cm", &fit::Margins::tight_threshold_cm)
        .def("clamped", &fit::Margins::clamped,
             "Apply the safety floors. Clearances can be raised but never lowered past them.");

    py::class_<fit::FacilityLimits>(m, "FacilityLimits")
        .def(py::init<>())
        .def_readwrite("max_height_cm", &fit::FacilityLimits::max_height_cm)
        .def_readwrite("max_width_cm", &fit::FacilityLimits::max_width_cm)
        .def_readwrite("max_length_cm", &fit::FacilityLimits::max_length_cm)
        .def_readwrite("max_weight_kg", &fit::FacilityLimits::max_weight_kg);

    py::class_<fit::FitReason>(m, "FitReason")
        .def_readonly("constraint", &fit::FitReason::constraint)
        .def_readonly("required_cm", &fit::FitReason::required_cm)
        .def_readonly("available_cm", &fit::FitReason::available_cm)
        .def_readonly("slack_cm", &fit::FitReason::slack_cm)
        .def_readonly("binding", &fit::FitReason::binding);

    py::class_<fit::FitResult>(m, "FitResult")
        .def_readonly("verdict", &fit::FitResult::verdict)
        .def_readonly("min_slack_cm", &fit::FitResult::min_slack_cm)
        .def_readonly("reasons", &fit::FitResult::reasons)
        .def_readonly("unverified_dimensions", &fit::FitResult::unverified_dimensions)
        .def_property_readonly("acceptable", &fit::FitResult::acceptable)
        .def_property_readonly(
            "verdict_name",
            [](const fit::FitResult& r) { return std::string(fit::to_string(r.verdict)); })
        .def_property_readonly("binding_constraint", [](const fit::FitResult& r) -> py::object {
            for (const auto& reason : r.reasons) {
                if (reason.binding) return py::cast(reason);
            }
            return py::none();
        });

    m.def("check_facility", &fit::check_facility, py::arg("vehicle"), py::arg("limits"),
          py::arg("margins") = fit::Margins{});
    m.def("check_bay", &fit::check_bay, py::arg("vehicle"), py::arg("bay_length_cm"),
          py::arg("bay_width_cm"), py::arg("orientation"), py::arg("margins") = fit::Margins{});
    m.def("check_gap", &fit::check_gap, py::arg("vehicle"), py::arg("gap_length_cm"),
          py::arg("gap_width_cm"), py::arg("margins") = fit::Margins{});
    m.def("required_gap_length_cm", &fit::required_gap_length_cm, py::arg("vehicle"),
          py::arg("margins") = fit::Margins{});

    // ---------------------------------------------------------------- rank
    py::enum_<rank::EvidenceSource>(m, "EvidenceSource")
        .value("OSM_ONLY", rank::EvidenceSource::OsmOnly)
        .value("STATIC_DATABASE", rank::EvidenceSource::StaticDatabase)
        .value("PREDICTIVE_MODEL", rank::EvidenceSource::PredictiveModel)
        .value("USER_CONFIRMATION", rank::EvidenceSource::UserConfirmation)
        .value("MUNICIPAL_SENSOR", rank::EvidenceSource::MunicipalSensor)
        .value("CAMERA_OBSERVATION", rank::EvidenceSource::CameraObservation)
        .value("OPERATOR_FEED", rank::EvidenceSource::OperatorFeed);

    py::enum_<rank::ConfidenceLabel>(m, "ConfidenceLabel")
        .value("CAMERA_CONFIRMED", rank::ConfidenceLabel::CameraConfirmed)
        .value("REPORTED_BY_OPERATOR", rank::ConfidenceLabel::ReportedByOperator)
        .value("LIKELY_AVAILABLE", rank::ConfidenceLabel::LikelyAvailable)
        .value("STATIC_INFORMATION_ONLY", rank::ConfidenceLabel::StaticInformationOnly)
        .value("DATA_STALE", rank::ConfidenceLabel::DataStale)
        .def("__str__",
             [](rank::ConfidenceLabel c) { return std::string(rank::to_string(c)); });

    py::class_<rank::ScoringConfig>(m, "ScoringConfig")
        .def(py::init<>())
        .def_readwrite("value_of_time_eur_per_min", &rank::ScoringConfig::value_of_time_eur_per_min)
        .def_readwrite("failure_penalty_min", &rank::ScoringConfig::failure_penalty_min)
        .def_readwrite("walk_discomfort_multiplier", &rank::ScoringConfig::walk_discomfort_multiplier)
        .def_readwrite("tight_fit_penalty_eur", &rank::ScoringConfig::tight_fit_penalty_eur)
        .def_readwrite("unverified_fit_penalty_eur", &rank::ScoringConfig::unverified_fit_penalty_eur)
        .def_readwrite("staleness_penalty_eur_per_min",
                       &rank::ScoringConfig::staleness_penalty_eur_per_min)
        .def_readwrite("max_staleness_penalty_eur", &rank::ScoringConfig::max_staleness_penalty_eur)
        .def_readwrite("stale_after_s", &rank::ScoringConfig::stale_after_s)
        .def_readwrite("herding_decay_per_recommendation",
                       &rank::ScoringConfig::herding_decay_per_recommendation)
        .def_readwrite("exact_space_ttl_s", &rank::ScoringConfig::exact_space_ttl_s);

    py::class_<rank::Candidate>(m, "Candidate")
        .def(py::init<>())
        .def_readwrite("id", &rank::Candidate::id)
        .def_readwrite("group_key", &rank::Candidate::group_key)
        .def_readwrite("drive_time_min", &rank::Candidate::drive_time_min)
        .def_readwrite("walk_time_min", &rank::Candidate::walk_time_min)
        .def_readwrite("price_eur", &rank::Candidate::price_eur)
        .def_readwrite("p_available_now", &rank::Candidate::p_available_now)
        .def_readwrite("lambda_per_min", &rank::Candidate::lambda_per_min)
        .def_readwrite("eta_min", &rank::Candidate::eta_min)
        .def_readwrite("observation_age_s", &rank::Candidate::observation_age_s)
        .def_readwrite("evidence", &rank::Candidate::evidence)
        .def_readwrite("fit_verdict", &rank::Candidate::fit_verdict)
        .def_readwrite("recent_recommendation_count", &rank::Candidate::recent_recommendation_count)
        .def_readwrite("is_exact_space", &rank::Candidate::is_exact_space);

    py::class_<rank::ScoredCandidate>(m, "ScoredCandidate")
        .def_readonly("id", &rank::ScoredCandidate::id)
        .def_readonly("generalised_cost", &rank::ScoredCandidate::generalised_cost)
        .def_readonly("expected_time_min", &rank::ScoredCandidate::expected_time_min)
        .def_readonly("p_available_at_eta", &rank::ScoredCandidate::p_available_at_eta)
        .def_readonly("confidence", &rank::ScoredCandidate::confidence)
        .def_readonly("expired", &rank::ScoredCandidate::expired)
        .def_readonly("fit_penalty_eur", &rank::ScoredCandidate::fit_penalty_eur)
        .def_readonly("uncertainty_penalty_eur", &rank::ScoredCandidate::uncertainty_penalty_eur)
        .def_property_readonly("confidence_name", [](const rank::ScoredCandidate& s) {
            return std::string(rank::to_string(s.confidence));
        });

    m.def("survival_probability", &rank::survival_probability, py::arg("p_now"),
          py::arg("lambda_per_min"), py::arg("eta_min"),
          "P(available at ETA) = P(now) * exp(-lambda * t).");
    m.def("apply_anti_herding", &rank::apply_anti_herding, py::arg("p"),
          py::arg("recent_recommendations"), py::arg("decay"));
    m.def("score", &rank::score, py::arg("candidate"), py::arg("config") = rank::ScoringConfig{});
    m.def("rank_and_diversify", &rank::rank_and_diversify, py::arg("candidates"),
          py::arg("config") = rank::ScoringConfig{}, py::arg("max_results") = 10,
          py::arg("max_per_group") = 2,
          "Score, sort and spread results across distinct streets or facilities.");
}
