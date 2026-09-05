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

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "parkfit/fit/vehicle.hpp"
#include "parkfit/fit/clearance.hpp"
#include "parkfit/fit/vehicle_fit.hpp"
#include "parkfit/geo/polygon.hpp"
#include "parkfit/geo/primitives.hpp"
#include "parkfit/geo/rd.hpp"
#include "parkfit/index/grid.hpp"
#include "parkfit/legal/rulebook.hpp"
#include "parkfit/legal/rulebook_de.hpp"
#include "parkfit/legal/rulebook_fr.hpp"
#include "parkfit/legal/rulebook_nl.hpp"
#include "parkfit/legal/rulebook_tr.hpp"
#include "parkfit/nav/deeplink.hpp"
#include "parkfit/rank/score.hpp"
#include "parkfit/routing/graph.hpp"
#include "parkfit/vision/uncalibrated_gap.hpp"

namespace py = pybind11;
using namespace parkfit;

PYBIND11_MODULE(parkfit_native, m) {
    m.doc() = "CamToParkingSlot native core: geodesy, spatial index, vehicle fit and ranking.";
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
    m.def("orientation_from_string", &fit::orientation_from_string, py::arg("value"),
          "Map the normalised orientation (parallel/perpendicular/angled). Prefer this: "
          "it does not require a non-Dutch bay to pretend to be Dutch.");


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
        .def_readwrite("parallel_lateral_total_cm", &fit::Margins::parallel_lateral_total_cm)
        .def_readwrite("bay_parallel_end_cm", &fit::Margins::bay_parallel_end_cm)
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
    py::class_<fit::ClearancePolicy>(m, "ClearancePolicy")
        .def_readonly("country", &fit::ClearancePolicy::country)
        .def_readonly("standard", &fit::ClearancePolicy::standard)
        .def_readonly("verified", &fit::ClearancePolicy::verified)
        .def_readonly("note", &fit::ClearancePolicy::note)
        .def_readonly("margins", &fit::ClearancePolicy::margins)
        .def("__repr__", [](const fit::ClearancePolicy& p) {
            return std::string("ClearancePolicy(") + p.country + ", " + p.standard +
                   ", verified=" + (p.verified ? "True" : "False") + ")";
        });

    m.def("clearance_for", &fit::clearance_for, py::arg("country"),
          "Clearance policy for a country. The margins are physical and apply "
          "everywhere; `verified` says whether a national standard was actually read.");

    m.def("required_gap_length_cm", &fit::required_gap_length_cm, py::arg("vehicle"),
          py::arg("margins") = fit::Margins{});

    // Batched fit checks.
    //
    // A search scores a few hundred candidates and each one needs a verdict. Done one at
    // a time that is a few hundred crossings of the language boundary for arithmetic
    // that takes nanoseconds, so the crossing dominates the work. These take flat lists
    // and cross once, matching the SpatialGrid.insert_many idiom above.

    m.def(
        "check_bays",
        [](const fit::Vehicle& vehicle, const std::vector<double>& length_cm,
           const std::vector<double>& width_cm,
           const std::vector<std::string>& orientation, const fit::Margins& margins) {
            const std::size_t n = length_cm.size();
            if (width_cm.size() != n || orientation.size() != n) {
                throw std::invalid_argument("check_bays: all three lists must be equal length");
            }
            std::vector<fit::FitResult> out;
            out.reserve(n);
            for (std::size_t i = 0; i < n; ++i) {
                out.push_back(fit::check_bay(vehicle, length_cm[i], width_cm[i],
                                             fit::orientation_from_string(orientation[i]),
                                             margins));
            }
            return out;
        },
        py::arg("vehicle"), py::arg("length_cm"), py::arg("width_cm"), py::arg("orientation"),
        py::arg("margins") = fit::Margins(),
        "Fit verdicts for many bays in one call, in order.");

    m.def(
        "check_facilities",
        [](const fit::Vehicle& vehicle, const std::vector<double>& max_height_cm,
           const fit::Margins& margins) {
            std::vector<fit::FitResult> out;
            out.reserve(max_height_cm.size());
            for (const double height : max_height_cm) {
                fit::FacilityLimits limits;
                // A non-positive height means the operator published none, which is
                // deliberately distinct from unlimited: the default limit is left in
                // place so the verdict comes back UNVERIFIED rather than FITS.
                if (height > 0.0) limits.max_height_cm = height;
                out.push_back(fit::check_facility(vehicle, limits, margins));
            }
            return out;
        },
        py::arg("vehicle"), py::arg("max_height_cm"), py::arg("margins") = fit::Margins(),
        "Fit verdicts for many facilities in one call, in order. A height of 0 or less "
        "means unpublished, not unlimited.");
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

    // -- navigation handoff --------------------------------------------------
    py::class_<nav::NavTarget>(m, "NavTarget")
        .def(py::init<>())
        .def_readwrite("lat", &nav::NavTarget::lat)
        .def_readwrite("lon", &nav::NavTarget::lon)
        .def_readwrite("label", &nav::NavTarget::label)
        .def_readwrite("is_entrance", &nav::NavTarget::is_entrance)
        .def("valid", &nav::NavTarget::valid);

    py::class_<nav::NavOrigin>(m, "NavOrigin")
        .def(py::init<>())
        .def_readwrite("lat", &nav::NavOrigin::lat)
        .def_readwrite("lon", &nav::NavOrigin::lon)
        .def_readwrite("present", &nav::NavOrigin::present)
        .def("valid", &nav::NavOrigin::valid);

    py::class_<nav::NavLink>(m, "NavLink")
        .def_readonly("provider", &nav::NavLink::provider)
        .def_readonly("display_name", &nav::NavLink::display_name)
        .def_readonly("url", &nav::NavLink::url);

    m.def("format_coordinate", &nav::format_coordinate, py::arg("value"),
          "Print a coordinate at full precision, no exponent, locale independent.");
    m.def("build_nav_links", &nav::build_links, py::arg("target"),
          py::arg("origin") = nav::NavOrigin{},
          "Every navigation handoff URL for one exact destination.");

    // --------------------------------------------------------------- legal
    //
    // Where a car may legally stop or park. This sits beside the fit engine rather than
    // inside it: fit answers whether the car goes there, legality answers whether the
    // driver may leave it there, and a space needs both. Every verdict carries the
    // article it came from, so a refusal can be shown to the user in the words of the
    // statute instead of as an unexplained absence.

    py::enum_<legal::Manoeuvre>(m, "Manoeuvre")
        .value("STOPPING", legal::Manoeuvre::Stopping)
        .value("PARKING", legal::Manoeuvre::Parking)
        .def("__str__", [](legal::Manoeuvre v) { return std::string(legal::to_string(v)); });

    py::enum_<legal::LegalVerdict>(m, "LegalVerdict")
        .value("LEGAL", legal::LegalVerdict::Legal)
        .value("PROHIBITED", legal::LegalVerdict::Prohibited)
        .value("CONDITIONAL", legal::LegalVerdict::Conditional)
        .value("UNKNOWN", legal::LegalVerdict::Unknown)
        .def("__str__", [](legal::LegalVerdict v) { return std::string(legal::to_string(v)); });

    py::enum_<legal::AnchorKind>(m, "AnchorKind")
        .value("JUNCTION", legal::AnchorKind::Junction)
        .value("JUNCTION_WITH_CYCLE_PATH", legal::AnchorKind::JunctionWithCyclePath)
        .value("PEDESTRIAN_CROSSING", legal::AnchorKind::PedestrianCrossing)
        .value("LEVEL_CROSSING", legal::AnchorKind::LevelCrossing)
        .value("BUS_STOP_SIGN", legal::AnchorKind::BusStopSign)
        .value("TRAM_STOP", legal::AnchorKind::TramStop)
        .value("FIRE_HYDRANT", legal::AnchorKind::FireHydrant)
        .value("DRIVEWAY", legal::AnchorKind::Driveway)
        .value("BRIDGE", legal::AnchorKind::Bridge)
        .value("UNDERPASS", legal::AnchorKind::Underpass)
        .value("TUNNEL", legal::AnchorKind::Tunnel)
        .value("CYCLE_LANE", legal::AnchorKind::CycleLane)
        .value("FOOTWAY", legal::AnchorKind::Footway)
        .value("EMERGENCY_ACCESS", legal::AnchorKind::EmergencyAccess)
        .value("PUBLIC_ENTRANCE", legal::AnchorKind::PublicEntrance)
        .value("DISABLED_BAY", legal::AnchorKind::DisabledBay)
        .value("LOADING_BAY", legal::AnchorKind::LoadingBay)
        .value("BUS_LANE", legal::AnchorKind::BusLane)
        .value("YELLOW_LINE_SOLID", legal::AnchorKind::YellowLineSolid)
        .value("YELLOW_LINE_BROKEN", legal::AnchorKind::YellowLineBroken)
        .def("__str__", [](legal::AnchorKind v) { return std::string(legal::to_string(v)); });

    py::class_<legal::Context>(m, "LegalContext")
        .def(py::init<>())
        .def_readwrite("built_up", &legal::Context::built_up)
        .def_readwrite("road_has_marked_bays", &legal::Context::road_has_marked_bays)
        .def_readwrite("inside_marked_bay", &legal::Context::inside_marked_bay)
        .def_readwrite("permit_zone_without_permit", &legal::Context::permit_zone_without_permit)
        .def_readwrite("disc_zone", &legal::Context::disc_zone)
        .def_readwrite("anchors_loaded", &legal::Context::anchors_loaded);

    py::class_<legal::LegalFinding>(m, "LegalFinding")
        .def_readonly("verdict", &legal::LegalFinding::verdict)
        .def_readonly("anchor", &legal::LegalFinding::anchor)
        .def_readonly("distance_cm", &legal::LegalFinding::distance_cm)
        .def_readonly("required_cm", &legal::LegalFinding::required_cm)
        .def_readonly("citation", &legal::LegalFinding::citation)
        .def_readonly("reason", &legal::LegalFinding::reason)
        .def_property_readonly(
            "verdict_name",
            [](const legal::LegalFinding& f) { return std::string(legal::to_string(f.verdict)); })
        .def_property_readonly(
            "anchor_name",
            [](const legal::LegalFinding& f) { return std::string(legal::to_string(f.anchor)); })
        .def_property_readonly("allowed",
                               [](const legal::LegalFinding& f) {
                                   // Conditional counts as allowed-with-a-condition, and
                                   // Unknown deliberately does not count as allowed.
                                   return f.verdict == legal::LegalVerdict::Legal ||
                                          f.verdict == legal::LegalVerdict::Conditional;
                               })
        .def("__repr__", [](const legal::LegalFinding& f) {
            return std::string("LegalFinding(") + legal::to_string(f.verdict) + ", " +
                   legal::to_string(f.anchor) + ", " + f.citation + ")";
        });

    py::class_<legal::Rulebook>(m, "Rulebook")
        .def_readonly("country", &legal::Rulebook::country)
        .def_readonly("instrument", &legal::Rulebook::instrument)
        .def_readonly("complete", &legal::Rulebook::complete)
        .def_property_readonly("rule_count",
                               [](const legal::Rulebook& b) { return b.rules.size(); })
        .def_property_readonly("max_distance_cm", [](const legal::Rulebook& b) {
            return legal::max_distance_cm(b);
        })
        .def_property_readonly(
            "anchor_names",
            [](const legal::Rulebook& b) {
                // Every map feature this book depends on. A caller uses it to check that
                // an area's ingest actually looked for all of them: a rule whose anchor
                // was never collected cannot clear a space, only fail to condemn it.
                std::vector<std::string> out;
                for (const auto& rule : b.rules) {
                    std::string name(legal::to_string(rule.anchor));
                    if (std::find(out.begin(), out.end(), name) == out.end()) {
                        out.push_back(name);
                    }
                }
                return out;
            })
        .def_property_readonly(
            "citations",
            [](const legal::Rulebook& b) {
                // Every distinct article the book rests on, for the attribution line.
                std::vector<std::string> out;
                for (const auto& rule : b.rules) {
                    std::string citation(rule.citation);
                    if (std::find(out.begin(), out.end(), citation) == out.end()) {
                        out.push_back(citation);
                    }
                }
                return out;
            })
        .def("__repr__", [](const legal::Rulebook& b) {
            return std::string("Rulebook(") + b.country + ", " +
                   std::to_string(b.rules.size()) + " rules, complete=" +
                   (b.complete ? "True" : "False") + ")";
        });

    py::class_<legal::AnchorIndex>(m, "AnchorIndex")
        .def(py::init<>())
        .def("__len__", &legal::AnchorIndex::size)
        .def("reserve", &legal::AnchorIndex::reserve, py::arg("n"))
        .def("build", &legal::AnchorIndex::build)
        .def("clear", &legal::AnchorIndex::clear)
        .def(
            "add",
            [](legal::AnchorIndex& index, legal::AnchorKind kind, double lat, double lon) {
                index.add(kind, geo::LatLon{lat, lon});
            },
            py::arg("kind"), py::arg("lat"), py::arg("lon"))
        .def(
            "add_many",
            [](legal::AnchorIndex& index,
               const std::vector<std::tuple<legal::AnchorKind, double, double>>& items) {
                index.reserve(index.size() + items.size());
                for (const auto& [kind, lat, lon] : items) {
                    index.add(kind, geo::LatLon{lat, lon});
                }
            },
            py::arg("items"),
            "Bulk insert (kind, lat, lon) triples. One crossing of the boundary instead "
            "of one per row.");

    m.def("rulebook_nl", &legal::nl::rulebook, "RVV 1990, articles 23 to 25.");
    m.def("rulebook_de", &legal::de::rulebook, "StVO paragraph 12 and Zeichen 224.");
    m.def("rulebook_tr", &legal::tr::rulebook, "Karayollari Trafik Kanunu 2918, articles 60-61.");
    m.def("rulebook_fr", &legal::fr::rulebook,
          "France: deliberately empty and marked incomplete, so it answers UNKNOWN.");

    m.def(
        "rulebook_for",
        [](const std::string& country) {
            if (country == "NL") return legal::nl::rulebook();
            if (country == "DE") return legal::de::rulebook();
            if (country == "TR") return legal::tr::rulebook();
            if (country == "FR") return legal::fr::rulebook();
            // An unknown country gets an incomplete book, which answers UNKNOWN rather
            // than allowing everything. Falling back to the Dutch rules here would apply
            // Dutch law in a country that does not have it, quietly and confidently.
            return legal::Rulebook{"??", "no rulebook for this country", {}, false};
        },
        py::arg("country"),
        "The rulebook for an ISO 3166-1 alpha-2 code. Unknown codes get an incomplete "
        "book, never a substitute country's rules.");

    m.def(
        "legal_evaluate",
        [](const legal::Rulebook& book, legal::Manoeuvre manoeuvre,
           const std::vector<std::tuple<legal::AnchorKind, double>>& hits,
           const legal::Context& context) {
            std::vector<legal::AnchorHit> measured;
            measured.reserve(hits.size());
            for (const auto& [kind, distance_cm] : hits) {
                measured.push_back(legal::AnchorHit{kind, distance_cm});
            }
            return legal::evaluate(book, manoeuvre, measured, context);
        },
        py::arg("book"), py::arg("manoeuvre"), py::arg("hits"),
        py::arg("context") = legal::Context{},
        "Judge one point from already-measured (kind, distance_cm) hits.");

    m.def(
        "legal_violations",
        [](const legal::Rulebook& book, legal::Manoeuvre manoeuvre,
           const std::vector<std::tuple<legal::AnchorKind, double>>& hits,
           const legal::Context& context) {
            std::vector<legal::AnchorHit> measured;
            measured.reserve(hits.size());
            for (const auto& [kind, distance_cm] : hits) {
                measured.push_back(legal::AnchorHit{kind, distance_cm});
            }
            return legal::violations(book, manoeuvre, measured, context);
        },
        py::arg("book"), py::arg("manoeuvre"), py::arg("hits"),
        py::arg("context") = legal::Context{},
        "Every rule this point breaks, worst first, not just the leading one.");

    m.def(
        "legal_evaluate_at",
        [](const legal::Rulebook& book, legal::Manoeuvre manoeuvre, legal::AnchorIndex& anchors,
           double lat, double lon, const legal::Context& context) {
            return legal::evaluate_at(book, manoeuvre, anchors, geo::LatLon{lat, lon}, context);
        },
        py::arg("book"), py::arg("manoeuvre"), py::arg("anchors"), py::arg("lat"), py::arg("lon"),
        py::arg("context") = legal::Context{},
        "Judge one point, sweeping the index at the radius the book itself requires.");

    m.def(
        "legal_evaluate_many",
        [](const legal::Rulebook& book, legal::Manoeuvre manoeuvre, legal::AnchorIndex& anchors,
           const std::vector<std::pair<double, double>>& points,
           const std::vector<legal::Context>& contexts, const legal::Context& shared) {
            if (!contexts.empty() && contexts.size() != points.size()) {
                throw std::invalid_argument(
                    "legal_evaluate_many: contexts must be empty or one per point");
            }
            std::vector<geo::LatLon> at;
            at.reserve(points.size());
            for (const auto& [lat, lon] : points) at.push_back(geo::LatLon{lat, lon});
            return legal::evaluate_many(book, manoeuvre, anchors, at, contexts, shared);
        },
        py::arg("book"), py::arg("manoeuvre"), py::arg("anchors"), py::arg("points"),
        py::arg("contexts") = std::vector<legal::Context>{},
        py::arg("shared") = legal::Context{},
        "Judge many (lat, lon) points in one call. 400 candidates against 20k anchors "
        "costs about 7 ms.");

    // -------------------------------------------------------------- vision
    //
    // The uncalibrated kerb-gap finder. It runs on a two-second loop per watched camera
    // rather than once per search, and its occlusion guard is quadratic in the detection
    // count, which is why it is here rather than in the interpreter.
    //
    // Only the uncalibrated path is exposed. The calibrated CurbGapEstimator in gap.hpp
    // needs a validated homography and a surveyed kerb centreline, and it is driven by
    // the C++ vision worker, which has both. Handing Python a half-configured version of
    // it would invite a caller to feed it a made-up scale and get metres back that look
    // like measurements.

    py::class_<vision::UncalibratedGapConfig>(m, "GapConfig")
        .def(py::init<>())
        .def_readwrite("typical_car_width_m", &vision::UncalibratedGapConfig::typical_car_width_m)
        .def_readwrite("min_gap_m", &vision::UncalibratedGapConfig::min_gap_m)
        .def_readwrite("max_gap_m", &vision::UncalibratedGapConfig::max_gap_m)
        .def_readwrite("min_depth_m", &vision::UncalibratedGapConfig::min_depth_m)
        .def_readwrite("min_cars_for_confident_scale",
                       &vision::UncalibratedGapConfig::min_cars_for_confident_scale)
        .def_readwrite("min_car_width_px", &vision::UncalibratedGapConfig::min_car_width_px)
        .def_readwrite("min_band_tolerance_px",
                       &vision::UncalibratedGapConfig::min_band_tolerance_px)
        .def_readwrite("frame_edge_margin_px",
                       &vision::UncalibratedGapConfig::frame_edge_margin_px);

    py::class_<vision::ImageGap>(m, "ImageGap")
        .def_readonly("x1", &vision::ImageGap::x1)
        .def_readonly("y1", &vision::ImageGap::y1)
        .def_readonly("x2", &vision::ImageGap::x2)
        .def_readonly("y2", &vision::ImageGap::y2)
        .def_readonly("length_m", &vision::ImageGap::length_m)
        .def_readonly("depth_m", &vision::ImageGap::depth_m)
        .def("__repr__", [](const vision::ImageGap& g) {
            return "ImageGap(length_m=" + std::to_string(g.length_m) +
                   ", depth_m=" + std::to_string(g.depth_m) + ")";
        });

    py::class_<vision::Scale>(m, "GapScale")
        .def_readonly("pixels_per_metre", &vision::Scale::pixels_per_metre)
        .def_readonly("confident", &vision::Scale::confident)
        .def("usable", &vision::Scale::usable);

    // Boxes cross as flat tuples rather than as objects. A frame is tens of detections
    // and this runs every two seconds per camera, so building a bound object per box
    // would cost more than the whole calculation.
    using BoxTuple = std::tuple<double, double, double, double, bool, bool>;
    const auto to_boxes = [](const std::vector<BoxTuple>& rows) {
        std::vector<vision::ImageBox> boxes;
        boxes.reserve(rows.size());
        for (const auto& [x1, y1, x2, y2, flanking, is_car] : rows) {
            boxes.push_back(vision::ImageBox{x1, y1, x2, y2, flanking, is_car});
        }
        return boxes;
    };

    m.def(
        "estimate_gap_scale",
        [to_boxes](const std::vector<BoxTuple>& rows, const vision::UncalibratedGapConfig& config) {
            return vision::estimate_scale(to_boxes(rows), config);
        },
        py::arg("boxes"), py::arg("config") = vision::UncalibratedGapConfig(),
        "Pixels per metre from the median detected car width. Boxes are "
        "(x1, y1, x2, y2, flanking, is_car) tuples.");

    m.def(
        "kerb_band",
        [to_boxes](const std::vector<BoxTuple>& rows, const vision::UncalibratedGapConfig& config) {
            const auto band = vision::kerb_band(to_boxes(rows), config);
            if (!band.valid) return py::object(py::none());
            return py::object(py::make_tuple(band.low, band.high));
        },
        py::arg("boxes"), py::arg("config") = vision::UncalibratedGapConfig(),
        "The vertical band the parked cars occupy, or None when there are too few.");

    m.def(
        "find_free_spaces",
        [to_boxes](const std::vector<BoxTuple>& rows, double pixels_per_metre,
                   double frame_width, const vision::UncalibratedGapConfig& config) {
            return vision::find_free_spaces(to_boxes(rows), pixels_per_metre, frame_width,
                                            config);
        },
        py::arg("boxes"), py::arg("pixels_per_metre"), py::arg("frame_width"),
        py::arg("config") = vision::UncalibratedGapConfig(),
        "Kerb gaps between consecutive parked vehicles. Lengths are estimates from the "
        "detected-car scale, never measurements.");

    // ------------------------------------------------------------- routing
    //
    // Everything here is bulk. A Dutch city extract is ~188k nodes and several hundred
    // thousand directed edges, so an interface that crossed the boundary once per node
    // would spend more time in pybind than the sweep itself takes. Nodes and edges go
    // across as parallel flat lists, which Python builds with one comprehension each
    // and pybind converts in a single pass.
    //
    // Node ids on this interface are OSM ids, not the dense internal indices, so a
    // caller never has to know that the graph renumbers anything.

    py::enum_<routing::Profile>(m, "Profile")
        .value("CAR", routing::Profile::Car)
        .value("FOOT", routing::Profile::Foot);

    py::class_<routing::Leg>(m, "Leg")
        .def_readonly("ok", &routing::Leg::ok)
        .def_readonly("distance_m", &routing::Leg::distance_m)
        .def_readonly("duration_min", &routing::Leg::duration_min)
        .def_readonly("confidence", &routing::Leg::confidence)
        .def_readonly("path", &routing::Leg::path,
                      "Dense node indices along the route, empty for sweep results.");

    py::class_<routing::RoadGraph>(m, "RoadGraph")
        .def(py::init<>())
        .def("__len__", &routing::RoadGraph::node_count)
        .def("node_count", &routing::RoadGraph::node_count)
        .def("build", &routing::RoadGraph::build,
             "Compress staged edges into CSR. Call once after all add_* calls.")
        .def(
            "add_nodes",
            [](routing::RoadGraph& g, const std::vector<std::int64_t>& ids,
               const std::vector<double>& lats, const std::vector<double>& lons) {
                if (ids.size() != lats.size() || ids.size() != lons.size()) {
                    throw std::invalid_argument("add_nodes: ids, lats and lons must be equal length");
                }
                g.reserve_nodes(g.node_count() + ids.size());
                for (std::size_t i = 0; i < ids.size(); ++i) g.add_node(ids[i], lats[i], lons[i]);
            },
            py::arg("ids"), py::arg("lats"), py::arg("lons"),
            "Bulk add nodes from parallel lists of OSM id, latitude and longitude.")
        .def(
            "add_edges",
            [](routing::RoadGraph& g, routing::Profile profile,
               const std::vector<std::int64_t>& from, const std::vector<std::int64_t>& to,
               const std::vector<double>& length_m, const std::vector<double>& seconds) {
                const std::size_t n = from.size();
                if (to.size() != n || length_m.size() != n || seconds.size() != n) {
                    throw std::invalid_argument("add_edges: all four lists must be equal length");
                }
                g.reserve_edges(profile, n);
                std::size_t skipped = 0;
                for (std::size_t i = 0; i < n; ++i) {
                    const auto a = g.index_of(from[i]);
                    const auto b = g.index_of(to[i]);
                    // An edge naming a node the graph has never seen is dropped rather
                    // than faulted: a bounding-box OSM extract clips ways at the border,
                    // so dangling references are normal, not corruption.
                    if (a == routing::kNoNode || b == routing::kNoNode) {
                        ++skipped;
                        continue;
                    }
                    g.add_edge(profile, a, b, length_m[i], seconds[i]);
                }
                return skipped;
            },
            py::arg("profile"), py::arg("from_ids"), py::arg("to_ids"), py::arg("length_m"),
            py::arg("seconds"),
            "Bulk add edges from parallel lists. Returns how many named an unknown node.")
        .def("index_of", &routing::RoadGraph::index_of, py::arg("external_id"),
             "Dense index for an OSM id, or 2**32-1 if the graph has never seen it.")
        .def("external_id", &routing::RoadGraph::external_id, py::arg("node"))
        .def("position", [](const routing::RoadGraph& g, std::uint32_t node) {
            const auto& p = g.position(node);
            return py::make_tuple(p.lat, p.lon);
        }, py::arg("node"))
        .def("edge_count", [](const routing::RoadGraph& g, routing::Profile profile) {
            return g.adjacency(profile).edge_count();
        }, py::arg("profile"));

    py::class_<routing::RoadRouter>(m, "RoadRouter")
        // keep_alive ties the graph's lifetime to the router's: the router holds a raw
        // pointer, so letting Python collect the graph first would dangle it.
        .def(py::init<const routing::RoadGraph&>(), py::arg("graph"), py::keep_alive<1, 2>())
        .def("largest_component", &routing::RoadRouter::largest_component, py::arg("profile"))
        .def("component_size", &routing::RoadRouter::component_size, py::arg("profile"),
             py::arg("component"))
        .def(
            "components",
            [](routing::RoadRouter& r, routing::Profile profile) {
                return r.components(profile);
            },
            py::arg("profile"),
            "Component id per dense node index. 2**32-1 means the node has no edges here.")
        .def(
            "nearest_node",
            [](routing::RoadRouter& r, double lat, double lon, routing::Profile profile,
               std::uint32_t component) { return r.nearest_node(lat, lon, profile, component); },
            py::arg("lat"), py::arg("lon"), py::arg("profile"),
            py::arg("component") = routing::kNoComponent,
            "Nearest routable dense node index, optionally inside one component.")
        .def(
            "costs_from",
            [](routing::RoadRouter& r, double lat, double lon, routing::Profile profile,
               double max_seconds) {
                const auto table = r.costs_from(lat, lon, profile, max_seconds);
                // Only reached nodes cross the boundary. Returning the dense table would
                // hand Python 188k infinities it has no use for.
                py::dict out;
                for (std::uint32_t n = 0; n < table.seconds.size(); ++n) {
                    if (table.reached(n)) {
                        out[py::int_(n)] = py::make_tuple(table.seconds[n], table.metres[n]);
                    }
                }
                py::object origin = py::none();
                if (table.ok()) origin = py::int_(table.origin);
                return py::make_tuple(out, origin);
            },
            py::arg("lat"), py::arg("lon"), py::arg("profile"), py::arg("max_seconds") = 1500.0,
            "One-to-many sweep. Returns ({node: (seconds, metres)}, origin_node_or_None).")
        .def(
            "many_costs",
            [](routing::RoadRouter& r, double lat, double lon,
               const std::vector<std::pair<double, double>>& targets, routing::Profile profile,
               double max_seconds) {
                std::vector<geo::LatLon> points;
                points.reserve(targets.size());
                for (const auto& [tlat, tlon] : targets) points.push_back(geo::LatLon{tlat, tlon});
                return r.many_costs(lat, lon, points, profile, max_seconds);
            },
            py::arg("lat"), py::arg("lon"), py::arg("targets"), py::arg("profile"),
            py::arg("max_seconds") = 1500.0,
            "Route to many (lat, lon) targets in a single sweep. One Leg per target, in order.")
        .def("route", &routing::RoadRouter::route, py::arg("from_lat"), py::arg("from_lon"),
             py::arg("to_lat"), py::arg("to_lon"), py::arg("profile"),
             "Point to point with the path. Leg.ok is False when no path exists.");
}
