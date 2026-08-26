// SPDX-License-Identifier: MIT
//
// pf_cv_worker: samples one camera, measures occupancy, publishes availability events.
//
// Run:
//   pf_cv_worker --camera cam_017 --url <stream> --calibration calib.json \
//                --segment kerb.json --interval 8 --model curb-gap-0.3.1
//
// Or, for deterministic evaluation with no camera at all:
//   pf_cv_worker --replay fixtures/scene --detections fixtures/scene.json
//
// The worker refuses to start against a live URL unless it is told the feed has been
// cleared for automated processing. That check lives in the Python camera registry,
// which is the only thing that knows a feed's permission status; this flag is how the
// registry communicates its decision, and its absence is a refusal, not a default.

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "parkfit/vision/detector.hpp"
#include "parkfit/vision/onnx_detector.hpp"
#include "parkfit/vision/gap.hpp"
#include "parkfit/vision/health.hpp"
#include "parkfit/vision/homography.hpp"
#include "parkfit/vision/publisher.hpp"
#include "parkfit/vision/source.hpp"
#include "parkfit/vision/state_machine.hpp"

using namespace parkfit::vision;

namespace {

struct Options {
    std::string camera_id{"cam_local"};
    std::string url;
    std::string calibration_path;
    std::string segment_path;
    std::string detections_path;
    std::string onnx_path;
    std::string onnx_library;
    std::string replay_prefix;
    std::string model_version{"none"};
    std::string output_path;
    double interval_s{8.0};
    int width{960};
    int height{540};
    int max_frames{0};  // 0 means run until the source ends
    double score_threshold{0.35};
    bool authorised{false};
    bool verbose{false};
    bool help{false};
};

void print_usage() {
    std::cout << R"(pf_cv_worker - camera occupancy worker

  --camera ID            camera identifier used in published events
  --url URL              stream URL (HLS, RTSP, MJPEG, snapshot or file)
  --authorised           assert this feed is cleared for automated processing
  --calibration FILE     JSON with image/world control point pairs
  --segment FILE         JSON kerb centreline in RD New metres
  --detections FILE      JSON detection sidecar (replay mode, no model needed)
  --onnx FILE            ONNX detector to run; reads FILE.json beside it for the
                         input size and class order
  --onnx-library PATH    explicit ONNX Runtime shared library, if it is not on the
                         search path
  --replay PREFIX        replay frames instead of opening a stream
  --model NAME           model version string recorded in every event
  --interval SECONDS     sampling interval, default 8
  --size WxH             decode size, default 960x540
  --max-frames N         stop after N frames
  --score THRESHOLD      detection score threshold, default 0.35
  --out FILE             append events to a file instead of stdout
  --verbose              log health and geometry to stderr
  -h, --help             this message

Parking changes over minutes, so the default sampling rate is deliberately slow.
Frames are held in memory and released immediately; only occupancy, geometry,
confidence and timestamps are ever published.
)";
}

Options parse_args(int argc, char** argv) {
    Options o;
    const auto value = [&](int& i) -> std::string {
        return (i + 1 < argc) ? argv[++i] : std::string{};
    };
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--camera") o.camera_id = value(i);
        else if (arg == "--url") o.url = value(i);
        else if (arg == "--authorised" || arg == "--authorized") o.authorised = true;
        else if (arg == "--calibration") o.calibration_path = value(i);
        else if (arg == "--segment") o.segment_path = value(i);
        else if (arg == "--detections") o.detections_path = value(i);
        else if (arg == "--replay") o.replay_prefix = value(i);
        else if (arg == "--model") o.model_version = value(i);
        else if (arg == "--onnx") o.onnx_path = value(i);
        else if (arg == "--onnx-library") o.onnx_library = value(i);
        else if (arg == "--out") o.output_path = value(i);
        else if (arg == "--interval") o.interval_s = std::stod(value(i));
        else if (arg == "--score") o.score_threshold = std::stod(value(i));
        else if (arg == "--max-frames") o.max_frames = std::stoi(value(i));
        else if (arg == "--verbose") o.verbose = true;
        else if (arg == "-h" || arg == "--help") o.help = true;
        else if (arg == "--size") {
            const std::string spec = value(i);
            const std::size_t x = spec.find('x');
            if (x != std::string::npos) {
                o.width = std::stoi(spec.substr(0, x));
                o.height = std::stoi(spec.substr(x + 1));
            }
        }
    }
    return o;
}

/// Read numbers out of a small JSON file without taking on a JSON dependency.
std::vector<double> read_numbers(const std::string& text, const std::string& key) {
    std::vector<double> out;
    const std::size_t at = text.find(key);
    if (at == std::string::npos) return out;
    const std::size_t open = text.find('[', at);
    if (open == std::string::npos) return out;

    int depth = 0;
    std::string token;
    for (std::size_t i = open; i < text.size(); ++i) {
        const char c = text[i];
        if (c == '[') { ++depth; continue; }
        if (c == ']') {
            if (!token.empty()) {
                try { out.push_back(std::stod(token)); } catch (...) {}
                token.clear();
            }
            if (--depth == 0) break;
            continue;
        }
        if (c == ',' || std::isspace(static_cast<unsigned char>(c))) {
            if (!token.empty()) {
                try { out.push_back(std::stod(token)); } catch (...) {}
                token.clear();
            }
            continue;
        }
        token += c;
    }
    return out;
}

/// Read a single numeric value for a key.
///
/// Distinct from read_numbers on purpose. read_numbers scans forward for the next `[`,
/// so asking it for a scalar key silently returns the *following* array instead: it read
/// "version": 3 as 102, the first element of image_points, and every published event
/// then carried a calibration version that did not exist.
int read_scalar(const std::string& text, const std::string& key, int fallback) {
    const std::size_t at = text.find(key);
    if (at == std::string::npos) return fallback;
    const std::size_t colon = text.find(':', at + key.size());
    if (colon == std::string::npos) return fallback;
    std::size_t i = colon + 1;
    while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) ++i;
    // A bracket or brace here means the key holds a structure, not a number.
    if (i >= text.size() || text[i] == '[' || text[i] == '{' || text[i] == '"') return fallback;
    try {
        return static_cast<int>(std::stod(text.substr(i, 24)));
    } catch (...) {
        return fallback;
    }
}

std::string read_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) return {};
    std::stringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

bool load_calibration(const std::string& path, Homography& out, int& version,
                      std::string& error) {
    const std::string text = read_file(path);
    if (text.empty()) {
        error = "calibration file could not be read: " + path;
        return false;
    }
    const auto image = read_numbers(text, "\"image_points\"");
    const auto world = read_numbers(text, "\"world_points_rd\"");
    if (image.size() < 8 || world.size() < 8 || image.size() != world.size()) {
        error = "calibration needs at least four matching image and world point pairs";
        return false;
    }

    std::vector<ControlPoint> points;
    for (std::size_t i = 0; i + 1 < image.size(); i += 2) {
        points.push_back(ControlPoint{Point2d{image[i], image[i + 1]},
                                      Point2d{world[i], world[i + 1]}});
    }

    const auto result = calibrate(points);
    if (!result.ok) {
        error = std::string("calibration failed: ") + std::string(result.reason);
        return false;
    }
    // A calibration is only worth using if it demonstrably fits. Publishing metre-level
    // gap measurements from a fit that is half a metre out would be worse than silence.
    if (result.max_error_m > 0.60) {
        error = "calibration reprojection error too large: " +
                std::to_string(result.max_error_m) + " m";
        return false;
    }
    out = result.homography;
    version = read_scalar(text, "\"version\"", 1);
    return true;
}

bool load_segment(const std::string& path, CurbSegment& out, std::string& error) {
    const std::string text = read_file(path);
    if (text.empty()) {
        error = "segment file could not be read: " + path;
        return false;
    }
    const auto centreline = read_numbers(text, "\"centreline_rd\"");
    if (centreline.size() < 4) {
        error = "kerb centreline needs at least two points";
        return false;
    }
    for (std::size_t i = 0; i + 1 < centreline.size(); i += 2) {
        out.centreline.push_back(Point2d{centreline[i], centreline[i + 1]});
    }
    const auto width = read_numbers(text, "\"usable_width_m\"");
    if (!width.empty()) out.usable_width_m = width.front();

    const auto prohibited = read_numbers(text, "\"prohibited\"");
    for (std::size_t i = 0; i + 1 < prohibited.size(); i += 2) {
        out.prohibited.push_back(Interval{prohibited[i], prohibited[i + 1]});
    }

    const std::size_t id_at = text.find("\"id\"");
    if (id_at != std::string::npos) {
        const std::size_t open = text.find('"', text.find(':', id_at) + 1);
        const std::size_t close = open == std::string::npos ? open : text.find('"', open + 1);
        if (open != std::string::npos && close != std::string::npos) {
            out.id = text.substr(open + 1, close - open - 1);
        }
    }
    if (out.id.empty()) out.id = "curb_unknown";
    return true;
}

double now_unix() {
    return std::chrono::duration<double>(
               std::chrono::system_clock::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char** argv) {
    const Options options = parse_args(argc, argv);
    if (options.help) {
        print_usage();
        return 0;
    }

    const bool replaying = !options.replay_prefix.empty() || !options.detections_path.empty();

    if (!options.url.empty() && !options.authorised) {
        // Deliberately a hard stop rather than a warning. Whether a feed may be
        // processed automatically is a licensing question the registry answers, and a
        // worker that guesses at it would make that decision by accident.
        std::cerr << "refusing to open " << options.url << "\n"
                  << "This feed has not been marked as cleared for automated processing.\n"
                  << "Register it first (pf cameras add) and pass --authorised once the\n"
                  << "registry reports a permission status this deployment accepts.\n";
        return 2;
    }
    if (options.url.empty() && !replaying) {
        std::cerr << "nothing to do: pass --url or --replay/--detections\n";
        print_usage();
        return 2;
    }

    Homography homography;
    int calibration_version = 0;
    bool calibrated = false;
    if (!options.calibration_path.empty()) {
        std::string error;
        calibrated = load_calibration(options.calibration_path, homography, calibration_version,
                                      error);
        if (!calibrated) {
            std::cerr << "calibration unusable: " << error << "\n"
                      << "Continuing without geometry: gap lengths cannot be measured and\n"
                      << "every space will be reported UNKNOWN.\n";
        }
    }

    CurbSegment segment;
    bool have_segment = false;
    if (!options.segment_path.empty()) {
        std::string error;
        have_segment = load_segment(options.segment_path, segment, error);
        if (!have_segment) std::cerr << "kerb segment unusable: " << error << "\n";
    }

    std::string model_version = options.model_version;

    std::unique_ptr<Detector> detector;
    if (!options.detections_path.empty()) {
        bool ok = false;
        auto sidecar = std::make_unique<SidecarDetector>(
            SidecarDetector::from_file(options.detections_path, &ok));
        if (!ok) {
            std::cerr << "detection sidecar could not be read: " << options.detections_path
                      << "\n";
            return 2;
        }
        detector = std::move(sidecar);
    } else if (!options.onnx_path.empty()) {
        auto onnx = std::make_unique<OnnxDetector>(options.onnx_path, options.onnx_library);
        if (!onnx->info().available) {
            // Asking for a model and silently getting a detector that sees nothing is how
            // a worker runs for a week publishing UNKNOWN for a camera that was working
            // perfectly. If a model was named, it has to load.
            std::cerr << "onnx detector unavailable: " << onnx->info().detail << "\n";
            std::cerr << "searched for the runtime in:";
            for (const auto& path : OnnxDetector::default_library_candidates()) {
                std::cerr << " " << path;
            }
            std::cerr << "\n";
            return 2;
        }
        // Every published event records which model produced it. When the operator did
        // not name one, take it from the model's own sidecar rather than shipping
        // "none" alongside real detections.
        if (model_version == "none") model_version = onnx->spec().model_version;
        detector = std::move(onnx);
    } else {
        detector = std::make_unique<NullDetector>();
    }

    const DetectorInfo detector_info = detector->info();
    if (!detector_info.available || options.verbose) {
        std::cerr << "detector: " << detector_info.backend << " - " << detector_info.detail
                  << "\n";
    }

    SourceConfig source_config;
    source_config.camera_id = options.camera_id;
    source_config.url = options.url;
    source_config.width = options.width;
    source_config.height = options.height;
    source_config.sample_interval_s = options.interval_s;
    source_config.kind = replaying ? SourceKind::Replay : SourceKind::FfmpegStream;

    std::unique_ptr<FrameSource> source;
    if (replaying) {
        // Two kinds of replay, and the difference matters.
        //
        // Real PPM frames at the prefix drive the *whole* pipeline, model included, which
        // is the only way to check that the C++ preprocessing and decoding agree with the
        // Python ones on actual pixels.
        //
        // No PPMs means the older fixture path: synthetic gradient frames counted off the
        // detection sidecar. That keeps the geometry tests reproducible with no camera, no
        // model and no clock, which is what makes gap error attributable to the geometry
        // rather than to a detector.
        std::vector<Frame> frames = load_ppm_sequence(
            options.replay_prefix, static_cast<std::size_t>(std::max(0, options.max_frames)));

        const auto* sidecar = dynamic_cast<SidecarDetector*>(detector.get());
        const std::size_t count = frames.empty() ? (sidecar ? sidecar->frame_count() : 0) : 0;
        if (!frames.empty() && options.verbose) {
            std::cerr << "replay: " << frames.size() << " frames from " << options.replay_prefix
                      << "*.ppm\n";
        }
        for (std::size_t i = 0; i < count; ++i) {
            Frame f(options.width, options.height, PixelFormat::Gray8);
            // A benign gradient, so the health checker sees a plausible frame while the
            // detections come from the fixture.
            for (int y = 0; y < f.height(); ++y) {
                std::uint8_t* row = f.row(y);
                for (int x = 0; x < f.width(); ++x) {
                    row[x] = static_cast<std::uint8_t>(
                        70 + ((x * 7 + y * 13 + static_cast<int>(i) * 29) % 150));
                }
            }
            frames.push_back(std::move(f));
        }
        source = std::make_unique<ReplaySource>(source_config, std::move(frames));
    } else {
        source = make_source(source_config);
    }

    if (!source->open()) {
        std::cerr << "could not open source\n";
        return 3;
    }

    std::unique_ptr<ObservationPublisher> publisher;
    if (options.output_path.empty()) {
        publisher = std::make_unique<StdoutPublisher>();
    } else {
        publisher = std::make_unique<FilePublisher>(options.output_path);
    }

    FrameHealthChecker health;
    CurbGapEstimator estimator;
    TemporalStateMachine machine;

    const Point2d camera_world =
        have_segment && !segment.centreline.empty()
            ? Point2d{segment.centreline.front().x, segment.centreline.front().y - 8.0}
            : Point2d{0.0, 0.0};

    int processed = 0;
    while (options.max_frames == 0 || processed < options.max_frames) {
        SourceFrame sf = source->next();
        if (!sf.ok) {
            if (options.verbose) std::cerr << "source: " << sf.error << "\n";
            break;
        }

        const HealthReport report = health.check(sf.frame);
        const double timestamp = replaying ? sf.timestamp_s : now_unix();

        if (options.verbose) {
            std::cerr << "frame " << processed << " health=" << to_string(report.state)
                      << " luma=" << report.mean_luma << " sharp=" << report.sharpness << "\n";
        }

        std::vector<Detection> detections;
        if (report.usable()) {
            detections = detector->detect(sf.frame, options.score_threshold);
        }

        if (options.verbose && !detections.empty()) {
            // Boxes in frame coordinates, which is what makes this comparable against the
            // Python decoder on the same file. The two implementations have to agree, and
            // the only way to know they do is to be able to read both.
            for (const auto& d : detections) {
                std::cerr << "  det " << d.label << " score=" << d.score << " box=[" << d.x1
                          << "," << d.y1 << "," << d.x2 << "," << d.y2 << "]\n";
            }
        }

        // The frame has served its purpose. Release it before anything else happens, so
        // imagery of a public street is resident for the shortest possible time.
        sf.frame.release();

        SpaceObservation observation;
        observation.timestamp_s = timestamp;
        observation.health = report.state;
        observation.confidence = report.usable() ? 0.9 : 0.0;
        observation.detection_score = 0.0;
        for (const auto& detection : detections) {
            observation.detection_score = std::max(observation.detection_score, detection.score);
        }

        const StateTransition transition = machine.update(observation);
        if (transition.publishable) {
            publisher->publish(make_bay_observation(
                options.camera_id, have_segment ? segment.id : "bay_default", transition,
                report.state, calibration_version, model_version, timestamp, 45.0));
        }

        if (report.usable() && calibrated && have_segment) {
            const auto gaps = estimator.estimate(segment, homography, detections, camera_world);
            if (gaps.usable) {
                for (const auto& gap : gaps.gaps) {
                    publisher->publish(make_gap_observation(
                        options.camera_id, segment.id, gap, report.state, calibration_version,
                        model_version, timestamp, 45.0));
                }
                if (options.verbose) {
                    std::cerr << "  gaps=" << gaps.gaps.size()
                              << " vehicles=" << gaps.projected_vehicles
                              << " off-kerb=" << gaps.rejected_off_kerb << "\n";
                }
            }
        }

        publisher->flush();
        ++processed;

        if (!replaying) {
            // ffmpeg already paces the stream with its fps filter; this is a guard for
            // sources that deliver faster than asked.
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    source->close();
    if (options.verbose) {
        std::cerr << "processed " << processed << " frames, published "
                  << publisher->published() << " observations\n";
    }
    return 0;
}
