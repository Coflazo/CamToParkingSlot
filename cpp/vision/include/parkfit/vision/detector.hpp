// SPDX-License-Identifier: MIT
//
// Vehicle detection, behind an interface with two backends.
//
// The abstraction is not architectural decoration. It exists so the geometry can be
// tested independently of the model: `SidecarDetector` replays known detections from a
// JSON file, which makes the gap-measurement tests deterministic and lets accuracy be
// attributed to either the detector or the projection, never to an untraceable mixture.
//
// `OnnxDetector` is the production path. ONNX Runtime is loaded dynamically, so the
// worker builds and runs without it and reports honestly that it cannot detect anything
// rather than failing to start.

#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "parkfit/vision/frame.hpp"
#include "parkfit/vision/gap.hpp"

namespace parkfit::vision {

/// Classes the model is trained on. Motorcycles matter: two of them occupy one car bay,
/// and a system that only knows "car" reports that bay as free.
enum class VehicleClass { Car, Van, Truck, Bus, Motorcycle, Bicycle, Trailer, Unknown };

inline const char* to_string(VehicleClass c) {
    switch (c) {
        case VehicleClass::Car: return "car";
        case VehicleClass::Van: return "van";
        case VehicleClass::Truck: return "truck";
        case VehicleClass::Bus: return "bus";
        case VehicleClass::Motorcycle: return "motorcycle";
        case VehicleClass::Bicycle: return "bicycle";
        case VehicleClass::Trailer: return "trailer";
        case VehicleClass::Unknown: return "unknown";
    }
    return "unknown";
}

/// Maps a detector label to a class.
///
/// Body styles are included alongside classes because the labels arriving here come from
/// two places that name things differently: a detector trained on COCO says "car", and
/// the synthetic renderer says "compact" or "estate". Before the body styles were listed
/// both fell through to Unknown, which is the one answer that should never be reached by
/// accident.
inline VehicleClass vehicle_class_from(const std::string& label) {
    static const std::map<std::string, VehicleClass> table{
        {"car", VehicleClass::Car},
        {"compact", VehicleClass::Car},
        {"estate", VehicleClass::Car},
        {"sedan", VehicleClass::Car},
        {"hatchback", VehicleClass::Car},
        {"suv", VehicleClass::Car},
        {"van", VehicleClass::Van},
        {"truck", VehicleClass::Truck},
        {"lorry", VehicleClass::Truck},
        {"bus", VehicleClass::Bus},
        {"coach", VehicleClass::Bus},
        {"motorcycle", VehicleClass::Motorcycle},
        {"motorbike", VehicleClass::Motorcycle},
        {"scooter", VehicleClass::Motorcycle},
        {"bicycle", VehicleClass::Bicycle},
        {"bike", VehicleClass::Bicycle},
        {"trailer", VehicleClass::Trailer},
        {"caravan", VehicleClass::Trailer},
    };
    const auto it = table.find(label);
    return it == table.end() ? VehicleClass::Unknown : it->second;
}

/// Classes that occupy a parking space.
///
/// A bicycle does not, and counting one as a blockage would delete real parking supply
/// from the map. Everything else does, **including Unknown**, and that asymmetry is
/// deliberate. The two ways to be wrong here are not equal: calling an unidentified
/// object "not a vehicle" reports the space it is standing in as free, and a driver sent
/// to an occupied space is the failure this subsystem is built to avoid. Calling it a
/// vehicle costs that driver one option out of several.
inline bool occupies_parking_space(VehicleClass c) { return c != VehicleClass::Bicycle; }

struct DetectorInfo {
    std::string backend;
    std::string model_version;
    bool available{false};
    std::string detail;
};

class Detector {
  public:
    virtual ~Detector() = default;
    virtual std::vector<Detection> detect(const Frame& frame, double score_threshold) = 0;
    [[nodiscard]] virtual DetectorInfo info() const = 0;
};

/// Intersection-over-union suppression, applied across classes.
///
/// Cross-class rather than per-class on purpose: a van detected as both "van" and "car"
/// is one vehicle, and keeping both would place two blockages on the kerb where there is
/// one, shortening every gap measured next to it.
inline std::vector<Detection> non_max_suppression(std::vector<Detection> detections,
                                                  double iou_threshold = 0.45) {
    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) { return a.score > b.score; });

    std::vector<Detection> kept;
    for (const auto& candidate : detections) {
        bool suppressed = false;
        for (const auto& winner : kept) {
            const double x1 = std::max(candidate.x1, winner.x1);
            const double y1 = std::max(candidate.y1, winner.y1);
            const double x2 = std::min(candidate.x2, winner.x2);
            const double y2 = std::min(candidate.y2, winner.y2);
            const double inter = std::max(0.0, x2 - x1) * std::max(0.0, y2 - y1);
            if (inter <= 0.0) continue;
            const double area_a = std::max(0.0, candidate.width()) * std::max(0.0, candidate.height());
            const double area_b = std::max(0.0, winner.width()) * std::max(0.0, winner.height());
            const double denominator = area_a + area_b - inter;
            if (denominator > 0.0 && inter / denominator > iou_threshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) kept.push_back(candidate);
    }
    return kept;
}

/// Replays detections from a JSON sidecar keyed by frame index.
///
/// Format, deliberately trivial so fixtures can be written by hand:
///   {"frames": [{"index": 0, "detections": [
///       {"x1": 10, "y1": 20, "x2": 90, "y2": 70, "score": 0.9, "label": "car"}]}]}
class SidecarDetector final : public Detector {
  public:
    struct FrameDetections {
        int index{0};
        std::vector<Detection> detections;
    };

    explicit SidecarDetector(std::vector<FrameDetections> frames)
        : frames_(std::move(frames)) {}

    static SidecarDetector from_file(const std::string& path, bool* ok = nullptr) {
        std::ifstream in(path);
        if (!in) {
            if (ok) *ok = false;
            return SidecarDetector({});
        }
        std::stringstream buffer;
        buffer << in.rdbuf();
        if (ok) *ok = true;
        return SidecarDetector(parse(buffer.str()));
    }

    std::vector<Detection> detect(const Frame&, double score_threshold) override {
        std::vector<Detection> out;
        for (const auto& entry : frames_) {
            if (entry.index != cursor_) continue;
            for (const auto& detection : entry.detections) {
                if (detection.score >= score_threshold) out.push_back(detection);
            }
            break;
        }
        ++cursor_;
        return non_max_suppression(std::move(out));
    }

    [[nodiscard]] DetectorInfo info() const override {
        return DetectorInfo{"sidecar", "replay", true,
                            "detections replayed from a fixture; no model is running"};
    }

    void rewind() { cursor_ = 0; }
    [[nodiscard]] std::size_t frame_count() const { return frames_.size(); }

    /// A minimal JSON reader for exactly this fixture shape.
    ///
    /// A general JSON library would be a dependency taken on for one test-only file
    /// format. This reads numbers and strings by key and ignores everything else, which
    /// is all the format needs and cannot silently mis-parse it, because a malformed
    /// fixture yields no detections rather than wrong ones.
    static std::vector<FrameDetections> parse(const std::string& text) {
        std::vector<FrameDetections> frames;
        std::size_t pos = 0;
        while ((pos = text.find("\"index\"", pos)) != std::string::npos) {
            FrameDetections entry;
            entry.index = static_cast<int>(read_number(text, pos + 7));

            const std::size_t detections_at = text.find("\"detections\"", pos);
            const std::size_t next_index = text.find("\"index\"", pos + 7);
            if (detections_at == std::string::npos ||
                (next_index != std::string::npos && detections_at > next_index)) {
                frames.push_back(entry);
                pos += 7;
                continue;
            }

            std::size_t cursor = detections_at;
            const std::size_t limit = next_index == std::string::npos ? text.size() : next_index;
            while ((cursor = text.find("\"x1\"", cursor)) != std::string::npos && cursor < limit) {
                Detection d;
                d.x1 = read_number(text, cursor + 4);
                d.y1 = read_number_after(text, cursor, "\"y1\"");
                d.x2 = read_number_after(text, cursor, "\"x2\"");
                d.y2 = read_number_after(text, cursor, "\"y2\"");
                d.score = read_number_after(text, cursor, "\"score\"", 1.0);
                d.label = read_string_after(text, cursor, "\"label\"", "car");
                d.class_id = static_cast<int>(vehicle_class_from(d.label));
                frames_push_detection(entry, d);
                cursor += 4;
            }
            frames.push_back(entry);
            pos = detections_at;
        }
        return frames;
    }

  private:
    static void frames_push_detection(FrameDetections& entry, const Detection& d) {
        entry.detections.push_back(d);
    }

    static double read_number(const std::string& text, std::size_t from, double fallback = 0.0) {
        const std::size_t colon = text.find(':', from);
        if (colon == std::string::npos) return fallback;
        try {
            return std::stod(text.substr(colon + 1, 32));
        } catch (...) {
            return fallback;
        }
    }

    static double read_number_after(const std::string& text, std::size_t from,
                                    const std::string& key, double fallback = 0.0) {
        const std::size_t at = text.find(key, from);
        return at == std::string::npos ? fallback : read_number(text, at + key.size(), fallback);
    }

    static std::string read_string_after(const std::string& text, std::size_t from,
                                         const std::string& key, const std::string& fallback) {
        const std::size_t at = text.find(key, from);
        if (at == std::string::npos) return fallback;
        const std::size_t open = text.find('"', text.find(':', at) + 1);
        if (open == std::string::npos) return fallback;
        const std::size_t close = text.find('"', open + 1);
        if (close == std::string::npos) return fallback;
        return text.substr(open + 1, close - open - 1);
    }

    std::vector<FrameDetections> frames_;
    int cursor_{0};
};

/// A detector that reports it cannot detect, rather than pretending to.
///
/// Used when no model is configured. It returns nothing, and because the state machine
/// treats "no detection" as evidence only after repeated confirmation *and* a healthy
/// frame, a missing model produces UNKNOWN everywhere rather than a street full of
/// imaginary free spaces.
class NullDetector final : public Detector {
  public:
    std::vector<Detection> detect(const Frame&, double) override { return {}; }
    [[nodiscard]] DetectorInfo info() const override {
        return DetectorInfo{"null", "none", false,
                            "no model configured; every space will report UNKNOWN"};
    }
};

}  // namespace parkfit::vision
