// SPDX-License-Identifier: MIT
//
// Tests for the ONNX detector backend.
//
// The decoder is tested against hand-built tensors rather than against a model, so every
// expected answer here is arithmetic rather than a number copied out of a training run.
// That matters: this decoder has to agree with the Python one in
// `parkfit.ml.train.detector.decode`, and the way that agreement breaks is a rounding or
// an indexing convention drifting on one side only. A test built from a real model's
// output could not tell the difference between the two implementations agreeing and both
// being wrong in the same way.
//
// The loading tests are all negative. A machine running the suite has no reason to have
// ONNX Runtime present, and the property that actually needs guarding is that its absence
// degrades rather than crashes.

#include "test_framework.hpp"

#include <string>
#include <vector>

#include "parkfit/vision/detector.hpp"
#include "parkfit/vision/onnx_detector.hpp"

using namespace parkfit::vision;

namespace {

constexpr int kClasses = 7;
constexpr int kRows = 8;
constexpr int kCols = 12;
constexpr int kStride = 4;

const std::vector<std::string> kNames{"car",        "van",     "truck",
                                      "bus",        "motorcycle", "bicycle",
                                      "trailer"};

/// A blank CenterNet head, sized as above.
struct Head {
    std::vector<float> heatmap = std::vector<float>(kClasses * kRows * kCols, 0.0f);
    std::vector<float> size = std::vector<float>(2 * kRows * kCols, 0.0f);
    std::vector<float> offset = std::vector<float>(2 * kRows * kCols, 0.0f);

    void put(int cls, int y, int x, float score, float w, float h, float ox = 0.0f,
             float oy = 0.0f) {
        const std::size_t plane = static_cast<std::size_t>(kRows) * kCols;
        const std::size_t index = static_cast<std::size_t>(y) * kCols + x;
        heatmap[static_cast<std::size_t>(cls) * plane + index] = score;
        size[index] = w;
        size[plane + index] = h;
        offset[index] = ox;
        offset[plane + index] = oy;
    }

    std::vector<Detection> decode(double threshold = 0.3) const {
        return decode_centernet(heatmap.data(), size.data(), offset.data(), kClasses, kRows,
                                kCols, kStride, threshold, kNames);
    }
};

}  // namespace

TEST_CASE("decode: a single peak becomes one box at the right place") {
    Head head;
    // Cell (3, 5), no sub-pixel offset, so the centre is exactly (5*4, 3*4) = (20, 12).
    head.put(0, 3, 5, 0.9f, 40.0f, 20.0f);

    const auto found = head.decode();
    CHECK_EQ(found.size(), 1u);
    CHECK_NEAR(found[0].x1, 0.0, 1e-6);   // 20 - 40/2
    CHECK_NEAR(found[0].y1, 2.0, 1e-6);   // 12 - 20/2
    CHECK_NEAR(found[0].x2, 40.0, 1e-6);  // 20 + 40/2
    CHECK_NEAR(found[0].y2, 22.0, 1e-6);  // 12 + 20/2
    CHECK_NEAR(found[0].score, 0.9, 1e-6);
    CHECK_EQ(found[0].class_id, 0);
    CHECK_EQ(found[0].label, std::string("car"));
}

TEST_CASE("decode: the sub-pixel offset moves the centre by less than one cell") {
    Head head;
    head.put(0, 3, 5, 0.9f, 40.0f, 20.0f, 0.5f, 0.25f);

    const auto found = head.decode();
    CHECK_EQ(found.size(), 1u);
    // Centre is ((5 + 0.5) * 4, (3 + 0.25) * 4) = (22, 13).
    CHECK_NEAR((found[0].x1 + found[0].x2) / 2.0, 22.0, 1e-6);
    CHECK_NEAR((found[0].y1 + found[0].y2) / 2.0, 13.0, 1e-6);
}

TEST_CASE("decode: a bright neighbourhood yields one detection, not nine") {
    // This is what the local-maximum test exists for. Without it a single vehicle, whose
    // Gaussian lights every cell around its centre, is reported once per cell.
    Head head;
    const std::size_t plane = static_cast<std::size_t>(kRows) * kCols;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            const int y = 3 + dy;
            const int x = 5 + dx;
            const std::size_t index = static_cast<std::size_t>(y) * kCols + x;
            // Falls off from the centre, as a splatted Gaussian does.
            head.heatmap[index] = (dx == 0 && dy == 0) ? 0.95f : 0.6f;
            head.size[index] = 40.0f;
            head.size[plane + index] = 20.0f;
        }
    }

    const auto found = head.decode();
    CHECK_EQ(found.size(), 1u);
    CHECK_NEAR(found[0].score, 0.95, 1e-6);
}

TEST_CASE("decode: two vehicles far apart are both reported") {
    Head head;
    head.put(0, 3, 2, 0.8f, 30.0f, 18.0f);
    head.put(0, 3, 9, 0.7f, 30.0f, 18.0f);

    const auto found = head.decode();
    CHECK_EQ(found.size(), 2u);
    // Sorted by score, strongest first.
    CHECK_NEAR(found[0].score, 0.8, 1e-6);
    CHECK_NEAR(found[1].score, 0.7, 1e-6);
}

TEST_CASE("decode: peaks below the threshold are not reported") {
    Head head;
    head.put(0, 3, 5, 0.25f, 40.0f, 20.0f);

    CHECK_EQ(head.decode(0.3).size(), 0u);
    CHECK_EQ(head.decode(0.2).size(), 1u);
}

TEST_CASE("decode: each channel carries its own class") {
    Head head;
    head.put(1, 2, 3, 0.8f, 30.0f, 20.0f);   // van
    head.put(4, 5, 8, 0.75f, 12.0f, 14.0f);  // motorcycle

    const auto found = head.decode();
    CHECK_EQ(found.size(), 2u);
    CHECK_EQ(found[0].label, std::string("van"));
    CHECK_EQ(found[1].label, std::string("motorcycle"));
}

TEST_CASE("decode: a peak with no size is not a box") {
    // Softplus cannot emit zero, but a corrupt or truncated tensor can, and a zero-area
    // box downstream becomes a zero-length kerb interval that quietly means nothing.
    Head head;
    head.put(0, 3, 5, 0.9f, 0.0f, 20.0f);
    CHECK_EQ(head.decode().size(), 0u);
}

TEST_CASE("decode: null tensors are refused rather than dereferenced") {
    const auto found =
        decode_centernet(nullptr, nullptr, nullptr, kClasses, kRows, kCols, kStride, 0.3, kNames);
    CHECK_EQ(found.size(), 0u);
}

TEST_CASE("decode: a peak in the corner cell still decodes") {
    // The local-maximum scan reads a 3x3 window, so the corners are where an unguarded
    // implementation walks off the array.
    Head head;
    head.put(0, 0, 0, 0.9f, 20.0f, 10.0f);
    head.put(2, kRows - 1, kCols - 1, 0.85f, 20.0f, 10.0f);

    const auto found = head.decode();
    CHECK_EQ(found.size(), 2u);
}

TEST_CASE("decode: more peaks than the cap yields exactly the cap, strongest first") {
    Head head;
    // Every other cell on one row, so none suppresses another.
    int placed = 0;
    for (int x = 0; x < kCols; x += 2) {
        head.put(0, 3, x, 0.5f + 0.01f * x, 20.0f, 10.0f);
        ++placed;
    }
    CHECK(placed > 3);

    const auto found = decode_centernet(head.heatmap.data(), head.size.data(),
                                        head.offset.data(), kClasses, kRows, kCols, kStride, 0.3,
                                        kNames, 3);
    CHECK_EQ(found.size(), 3u);
    CHECK(found[0].score >= found[1].score);
    CHECK(found[1].score >= found[2].score);
}

// ---------------------------------------------------------------------------
// Model spec
// ---------------------------------------------------------------------------
TEST_CASE("spec: a sidecar is parsed") {
    const std::string json = R"({
      "model_version": "curb-detector-0.1.0",
      "input_name": "image",
      "output_names": ["heatmap", "size", "offset"],
      "input_width": 512,
      "input_height": 288,
      "output_stride": 4,
      "class_names": ["car", "van", "truck", "bus", "motorcycle", "bicycle", "trailer"]
    })";

    const OnnxModelSpec spec = OnnxModelSpec::from_json(json);
    CHECK_EQ(spec.model_version, std::string("curb-detector-0.1.0"));
    CHECK_EQ(spec.input_width, 512);
    CHECK_EQ(spec.input_height, 288);
    CHECK_EQ(spec.output_stride, 4);
    CHECK_EQ(spec.input_name, std::string("image"));
    CHECK_EQ(spec.output_names.size(), 3u);
    CHECK_EQ(spec.output_names[0], std::string("heatmap"));
    CHECK_EQ(spec.class_names.size(), 7u);
    CHECK_EQ(spec.class_names[4], std::string("motorcycle"));
}

TEST_CASE("spec: an empty sidecar keeps the documented defaults") {
    const OnnxModelSpec spec = OnnxModelSpec::from_json("");
    CHECK_EQ(spec.input_width, 512);
    CHECK_EQ(spec.input_height, 288);
    CHECK_EQ(spec.output_stride, 4);
    CHECK_EQ(spec.class_names.size(), 7u);
}

TEST_CASE("spec: a sidecar missing a key keeps that default and reads the rest") {
    const OnnxModelSpec spec = OnnxModelSpec::from_json(R"({"input_width": 640})");
    CHECK_EQ(spec.input_width, 640);
    CHECK_EQ(spec.input_height, 288);  // untouched
}

TEST_CASE("spec: the class order matches VehicleClass") {
    // The whole point of fixing the channel order is that no lookup table is needed at
    // the boundary. If these ever diverge, every detection silently changes class.
    const OnnxModelSpec spec;
    CHECK_EQ(vehicle_class_from(spec.class_names[0]), VehicleClass::Car);
    CHECK_EQ(vehicle_class_from(spec.class_names[1]), VehicleClass::Van);
    CHECK_EQ(vehicle_class_from(spec.class_names[2]), VehicleClass::Truck);
    CHECK_EQ(vehicle_class_from(spec.class_names[3]), VehicleClass::Bus);
    CHECK_EQ(vehicle_class_from(spec.class_names[4]), VehicleClass::Motorcycle);
    CHECK_EQ(vehicle_class_from(spec.class_names[5]), VehicleClass::Bicycle);
    CHECK_EQ(vehicle_class_from(spec.class_names[6]), VehicleClass::Trailer);
}

// ---------------------------------------------------------------------------
// Degradation
// ---------------------------------------------------------------------------
TEST_CASE("detector: a missing model reports unavailable instead of throwing") {
    OnnxDetector detector("does/not/exist.onnx");
    const DetectorInfo info = detector.info();
    CHECK_EQ(info.backend, std::string("onnxruntime"));
    CHECK(!info.available);
    CHECK(!info.detail.empty());
}

TEST_CASE("detector: an unavailable detector detects nothing rather than crashing") {
    OnnxDetector detector("does/not/exist.onnx");
    Frame frame(64, 48, PixelFormat::Rgb24);
    CHECK_EQ(detector.detect(frame, 0.3).size(), 0u);
}

TEST_CASE("detector: a named library that does not exist is reported, not guessed around") {
    OnnxDetector detector("does/not/exist.onnx", "definitely-not-onnxruntime.dll");
    CHECK(!detector.info().available);
    // The operator has to be able to see which path was tried.
    CHECK(detector.info().detail.find("definitely-not-onnxruntime.dll") != std::string::npos);
}

TEST_CASE("detector: the search list is non-empty and names a real library file") {
    const auto candidates = OnnxDetector::default_library_candidates();
    CHECK(!candidates.empty());
    bool mentions_runtime = false;
    for (const auto& path : candidates) {
        if (path.find("onnxruntime") != std::string::npos) mentions_runtime = true;
    }
    CHECK(mentions_runtime);
}

// ---------------------------------------------------------------------------
// Class mapping
// ---------------------------------------------------------------------------
TEST_CASE("classes: body styles the renderer emits all map to a real class") {
    // The synthetic renderer names body styles, not vehicle classes. Before this mapping
    // existed, "compact" and "estate" both fell through to Unknown.
    CHECK_EQ(vehicle_class_from("compact"), VehicleClass::Car);
    CHECK_EQ(vehicle_class_from("estate"), VehicleClass::Car);
    CHECK_EQ(vehicle_class_from("sedan"), VehicleClass::Car);
    CHECK_EQ(vehicle_class_from("hatchback"), VehicleClass::Car);
    CHECK_EQ(vehicle_class_from("suv"), VehicleClass::Car);
}

TEST_CASE("classes: an unrecognised object blocks the space rather than freeing it") {
    // The conservative direction. Treating an unidentified object as "not a vehicle"
    // reports the space it is standing in as free, which is a false-free, and the false-
    // free rate is the metric this whole subsystem is built around.
    CHECK(occupies_parking_space(VehicleClass::Unknown));
    CHECK(occupies_parking_space(VehicleClass::Car));
    CHECK(occupies_parking_space(VehicleClass::Motorcycle));
    // A bicycle genuinely does not consume a car bay.
    CHECK(!occupies_parking_space(VehicleClass::Bicycle));
}

PF_TEST_MAIN()
