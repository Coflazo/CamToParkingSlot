// SPDX-License-Identifier: MIT
//
// Observation publishing.
//
// What leaves this process is the privacy boundary of the whole product. The worker sees
// a public street; what it emits is a number and a timestamp. No pixels, no bounding
// boxes in image space, no vehicle appearance, no plate, no face -- those never leave
// the frame buffer, and the frame buffer is released immediately after processing.
//
// The published record is deliberately auditable rather than minimal. Every observation
// carries its calibration version, model version and frame health, so a wrong answer can
// be traced to the calibration or the model that produced it instead of being an
// anonymous mistake.

#pragma once

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "parkfit/vision/gap.hpp"
#include "parkfit/vision/health.hpp"
#include "parkfit/vision/state_machine.hpp"

namespace parkfit::vision {

/// One published availability event. Mirrors contracts/availability.schema.json.
struct Observation {
    std::string camera_id;
    std::string target_kind;  ///< "bay" or "curb"
    std::string target_id;
    std::string observed_at;  ///< ISO 8601, UTC
    std::string state;        ///< VACANT | OCCUPIED | VACANT_GAP | UNKNOWN

    double gap_length_m{0.0};
    double gap_width_m{0.0};
    double confidence{0.0};

    int calibration_version{0};
    std::string model_version;
    std::string frame_health;
    std::string expires_at;
    std::string reason;
};

/// Escape the handful of characters JSON forbids, and drop control characters.
inline std::string json_escape(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (char c : value) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) >= 0x20) out += c;
                break;
        }
    }
    return out;
}

inline std::string to_json(const Observation& o) {
    std::ostringstream s;
    s << std::fixed << std::setprecision(3);
    s << "{\"camera_id\":\"" << json_escape(o.camera_id) << "\""
      << ",\"target_kind\":\"" << json_escape(o.target_kind) << "\""
      << ",\"target_id\":\"" << json_escape(o.target_id) << "\""
      << ",\"observed_at\":\"" << json_escape(o.observed_at) << "\""
      << ",\"state\":\"" << json_escape(o.state) << "\"";
    if (o.gap_length_m > 0.0) s << ",\"gap_length_m\":" << o.gap_length_m;
    if (o.gap_width_m > 0.0) s << ",\"gap_width_m\":" << o.gap_width_m;
    s << ",\"confidence\":" << o.confidence
      << ",\"calibration_version\":" << o.calibration_version
      << ",\"model_version\":\"" << json_escape(o.model_version) << "\""
      << ",\"frame_health\":\"" << json_escape(o.frame_health) << "\"";
    if (!o.expires_at.empty()) s << ",\"expires_at\":\"" << json_escape(o.expires_at) << "\"";
    if (!o.reason.empty()) s << ",\"reason\":\"" << json_escape(o.reason) << "\"";
    s << "}";
    return s.str();
}

class ObservationPublisher {
  public:
    virtual ~ObservationPublisher() = default;
    virtual void publish(const Observation& observation) = 0;
    virtual void flush() {}
    [[nodiscard]] virtual std::size_t published() const = 0;
};

/// Newline-delimited JSON on stdout, for piping into the ingest worker.
class StdoutPublisher final : public ObservationPublisher {
  public:
    void publish(const Observation& observation) override {
        std::printf("%s\n", to_json(observation).c_str());
        ++count_;
    }
    void flush() override { std::fflush(stdout); }
    [[nodiscard]] std::size_t published() const override { return count_; }

  private:
    std::size_t count_{0};
};

/// Appends newline-delimited JSON to a file, for replay evaluation.
class FilePublisher final : public ObservationPublisher {
  public:
    explicit FilePublisher(const std::string& path) : out_(path, std::ios::app) {}

    void publish(const Observation& observation) override {
        if (!out_) return;
        out_ << to_json(observation) << "\n";
        ++count_;
    }
    void flush() override { out_.flush(); }
    [[nodiscard]] std::size_t published() const override { return count_; }
    [[nodiscard]] bool ok() const { return static_cast<bool>(out_); }

  private:
    std::ofstream out_;
    std::size_t count_{0};
};

/// Keeps observations in memory. Used by tests and the evaluation harness.
class MemoryPublisher final : public ObservationPublisher {
  public:
    void publish(const Observation& observation) override { items_.push_back(observation); }
    [[nodiscard]] std::size_t published() const override { return items_.size(); }
    [[nodiscard]] const std::vector<Observation>& items() const { return items_; }
    void clear() { items_.clear(); }

  private:
    std::vector<Observation> items_;
};

/// Format a UNIX timestamp as ISO 8601 in UTC.
inline std::string iso8601(double unix_seconds) {
    const std::time_t t = static_cast<std::time_t>(unix_seconds);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buffer;
}

/// Build a publishable observation for one marked bay.
inline Observation make_bay_observation(const std::string& camera_id, const std::string& bay_id,
                                        const StateTransition& transition, FrameHealth health,
                                        int calibration_version, const std::string& model_version,
                                        double now_unix, double ttl_s) {
    Observation o;
    o.camera_id = camera_id;
    o.target_kind = "bay";
    o.target_id = bay_id;
    o.observed_at = iso8601(now_unix);
    o.state = to_string(transition.state);
    o.confidence = transition.confidence;
    o.calibration_version = calibration_version;
    o.model_version = model_version;
    o.frame_health = to_string(health);
    o.reason = transition.reason;
    // An availability claim that outlives its own evidence is the failure this whole
    // subsystem exists to avoid, so every record carries its own expiry.
    o.expires_at = iso8601(now_unix + ttl_s);
    return o;
}

/// Build a publishable observation for a measured kerb gap.
inline Observation make_gap_observation(const std::string& camera_id,
                                        const std::string& segment_id, const Gap& gap,
                                        FrameHealth health, int calibration_version,
                                        const std::string& model_version, double now_unix,
                                        double ttl_s) {
    Observation o;
    o.camera_id = camera_id;
    o.target_kind = "curb";
    o.target_id = segment_id;
    o.observed_at = iso8601(now_unix);
    o.state = "VACANT_GAP";
    o.gap_length_m = gap.length_m;
    o.gap_width_m = gap.width_m;
    o.confidence = gap.confidence;
    o.calibration_version = calibration_version;
    o.model_version = model_version;
    o.frame_health = to_string(health);
    o.expires_at = iso8601(now_unix + ttl_s);
    if (gap.occluded_start || gap.occluded_end) {
        o.reason = "gap runs to the edge of the visible segment; length is a lower bound";
    }
    return o;
}

}  // namespace parkfit::vision
