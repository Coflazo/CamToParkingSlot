// SPDX-License-Identifier: MIT
//
// Tests for the bay occupancy classifier.
//
// The same discipline as test_onnx.cpp: everything that can be checked without a model is
// checked without one. The softmax and the sidecar parser are arithmetic and string
// handling, so their expected answers are derived here rather than copied out of a
// training run, and the loading tests are negative because a machine running this suite
// has no reason to have ONNX Runtime installed.
//
// The property that matters most is the one about lengths. The worker indexes verdicts
// against its own list of bays, so a batch that quietly drops an off-screen crop would
// shift every bay after it and put one bay's answer on another bay's polygon. Several
// tests below exist only to pin that down.

#include "test_framework.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "parkfit/vision/frame.hpp"
#include "parkfit/vision/occupancy.hpp"

using namespace parkfit::vision;

namespace {

/// A frame of a single flat colour, enough to exercise packing and bounds.
Frame solid_frame(int width, int height, std::uint8_t value) {
    Frame frame(width, height, PixelFormat::Rgb24);
    for (int y = 0; y < height; ++y) {
        std::uint8_t* row = frame.row(y);
        for (int x = 0; x < width * 3; ++x) row[x] = value;
    }
    return frame;
}

}  // namespace

// ---------------------------------------------------------------------------
// softmax
// ---------------------------------------------------------------------------
TEST_CASE("softmax: equal logits are exactly a half") {
    CHECK_NEAR(occupied_probability(0.0f, 0.0f), 0.5, 1e-9);
    CHECK_NEAR(occupied_probability(3.5f, 3.5f), 0.5, 1e-9);
}

TEST_CASE("softmax: the answer follows the larger logit") {
    CHECK(occupied_probability(0.0f, 4.0f) > 0.98);
    CHECK(occupied_probability(4.0f, 0.0f) < 0.02);
}

TEST_CASE("softmax: two logits one apart give 1/(1+e^-1) wherever they sit") {
    // Two logits one apart: the answer is 1 / (1 + e^-1), independent of where they sit.
    const double expected = 1.0 / (1.0 + std::exp(-1.0));
    CHECK_NEAR(occupied_probability(0.0f, 1.0f), expected, 1e-9);
    CHECK_NEAR(occupied_probability(100.0f, 101.0f), expected, 1e-6);
}

TEST_CASE("softmax: logits large enough to overflow exp do not produce NaN") {
    // exp(400) is infinity in a double. Subtracting the max before exponentiating is the
    // whole reason this does not return NaN.
    const double high = occupied_probability(-400.0f, 400.0f);
    const double low = occupied_probability(400.0f, -400.0f);
    CHECK(high == high);  // not NaN
    CHECK(low == low);
    CHECK_NEAR(high, 1.0, 1e-9);
    CHECK_NEAR(low, 0.0, 1e-9);
}

// ---------------------------------------------------------------------------
// sidecar parsing
// ---------------------------------------------------------------------------
TEST_CASE("spec: an absent sidecar falls back to the shipped model contract") {
    const OccupancySpec spec = OccupancySpec::from_json("");
    CHECK_EQ(spec.input_width, 96);
    CHECK_EQ(spec.input_height, 96);
    CHECK_EQ(spec.input_name, std::string("patch"));
    CHECK_EQ(spec.output_name, std::string("logits"));
    CHECK_NEAR(spec.operating_threshold, 0.5, 1e-12);
}

TEST_CASE("spec: a real sidecar is read field for field") {
    const std::string json = R"({
      "model_version": "bay-occupancy-1.0.0",
      "input_name": "patch",
      "output_names": ["logits"],
      "input_width": 96,
      "input_height": 96,
      "class_names": ["free", "occupied"],
      "operating_threshold": 0.1
    })";
    const OccupancySpec spec = OccupancySpec::from_json(json);
    CHECK_EQ(spec.model_version, std::string("bay-occupancy-1.0.0"));
    CHECK_EQ(spec.output_name, std::string("logits"));
    CHECK_EQ(spec.class_names.size(), static_cast<std::size_t>(2));
    CHECK_EQ(spec.class_names[1], std::string("occupied"));
    CHECK_NEAR(spec.operating_threshold, 0.1, 1e-12);
}

TEST_CASE("spec: a hand written sidecar may name the output as a scalar") {
    const OccupancySpec spec = OccupancySpec::from_json(R"({"output_name": "scores"})");
    CHECK_EQ(spec.output_name, std::string("scores"));
}

TEST_CASE("spec: a threshold outside (0,1) is corrupt and falls back to 0.5") {
    // A corrupt sidecar would otherwise make every bay free, or every bay occupied,
    // silently and everywhere.
    CHECK_NEAR(OccupancySpec::from_json(R"({"operating_threshold": 0})").operating_threshold,
                   0.5, 1e-12);
    CHECK_NEAR(OccupancySpec::from_json(R"({"operating_threshold": 1})").operating_threshold,
                   0.5, 1e-12);
    CHECK_NEAR(OccupancySpec::from_json(R"({"operating_threshold": 7.5})").operating_threshold,
                   0.5, 1e-12);
}

TEST_CASE("spec: a partial sidecar keeps the default input size") {
    const OccupancySpec spec = OccupancySpec::from_json(R"({"model_version": "x"})");
    CHECK_EQ(spec.input_width, 96);
    CHECK_EQ(spec.input_height, 96);
}

// ---------------------------------------------------------------------------
// crops
// ---------------------------------------------------------------------------
TEST_CASE("crop: a bay needs positive extent to be worth a forward pass") {
    CHECK((BayCrop{0, 0, 10, 10}).valid());
    CHECK(!(BayCrop{0, 0, 0, 10}).valid());
    CHECK(!(BayCrop{0, 0, 10, 0}).valid());
    CHECK(!(BayCrop{0, 0, -4, 10}).valid());
}

// ---------------------------------------------------------------------------
// missing runtime
// ---------------------------------------------------------------------------
TEST_CASE("loading: a missing model reports unavailable rather than throwing") {
    BayOccupancyClassifier classifier("no/such/occupancy.onnx", "no/such/onnxruntime.dll");
    const DetectorInfo info = classifier.info();
    CHECK(!info.available);
    CHECK_EQ(info.backend, std::string("onnxruntime"));
    CHECK(!info.detail.empty());
}

TEST_CASE("batch: an unavailable model answers Unknown once per bay, not zero times") {
    BayOccupancyClassifier classifier("no/such/occupancy.onnx", "no/such/onnxruntime.dll");
    const Frame frame = solid_frame(64, 64, 128);

    const std::vector<BayCrop> crops{{0, 0, 16, 16}, {16, 16, 16, 16}, {32, 32, 16, 16}};
    const std::vector<BayVerdict> verdicts = classifier.classify_batch(frame, crops);

    // Length, not emptiness: the caller indexes these against its own bay list.
    CHECK_EQ(verdicts.size(), crops.size());
    for (const BayVerdict& verdict : verdicts) {
        CHECK(verdict.state == BayState::Unknown);
        CHECK(!verdict.is_occupied());
    }
}

TEST_CASE("batch: the single-bay convenience call is Unknown too") {
    BayOccupancyClassifier classifier("no/such/occupancy.onnx", "no/such/onnxruntime.dll");
    const Frame frame = solid_frame(32, 32, 200);
    const BayVerdict verdict = classifier.classify(frame, BayCrop{0, 0, 16, 16});
    CHECK(verdict.state == BayState::Unknown);
}

TEST_CASE("batch: no bays in means no verdicts out") {
    BayOccupancyClassifier classifier("no/such/occupancy.onnx", "no/such/onnxruntime.dll");
    const Frame frame = solid_frame(32, 32, 10);
    CHECK(classifier.classify_batch(frame, {}).empty());
}

TEST_CASE("loading: the sidecar is read even when the model will not load") {
    // spec() has to describe what this classifier would have expected, so an operator
    // debugging a missing runtime can still see the contract it was built against.
    BayOccupancyClassifier classifier("no/such/occupancy.onnx", "no/such/onnxruntime.dll");
    CHECK_EQ(classifier.spec().input_width, 96);
    CHECK_EQ(classifier.spec().input_name, std::string("patch"));
}

TEST_CASE("batch: an empty frame is Unknown rather than a read past the end") {
    BayOccupancyClassifier classifier("no/such/occupancy.onnx", "no/such/onnxruntime.dll");
    const Frame empty;
    const std::vector<BayVerdict> verdicts =
        classifier.classify_batch(empty, {BayCrop{0, 0, 8, 8}});
    CHECK_EQ(verdicts.size(), static_cast<std::size_t>(1));
    CHECK(verdicts[0].state == BayState::Unknown);
}

// ---------------------------------------------------------------------------
// With a real model
// ---------------------------------------------------------------------------
// These only run when PARKFIT_OCCUPANCY_MODEL points at an exported classifier and ONNX
// Runtime is loadable. Skipping is the right behaviour rather than a gap: CI has neither,
// and a test that fails for want of a 6 MB binary tells you nothing about the code.
namespace {

const char* model_from_env() {
#ifdef _MSC_VER
    // getenv is deprecated under MSVC's secure CRT warnings, and _dupenv_s leaks unless
    // freed; a static duplicate is simpler and this runs once.
    static std::string cached;
    static bool looked = false;
    if (!looked) {
        looked = true;
        char* value = nullptr;
        std::size_t length = 0;
        if (_dupenv_s(&value, &length, "PARKFIT_OCCUPANCY_MODEL") == 0 && value) {
            cached.assign(value);
            free(value);
        }
    }
    return cached.empty() ? nullptr : cached.c_str();
#else
    return std::getenv("PARKFIT_OCCUPANCY_MODEL");
#endif
}

}  // namespace

TEST_CASE("model: a real export loads and answers every bay in the batch") {
    const char* path = model_from_env();
    if (path == nullptr) return;

    BayOccupancyClassifier classifier(path);
    // Say out loud whether the model opened. A silent skip here would look identical to a
    // pass, and pointing the suite at a model you believe in and learning nothing is
    // exactly the false green this project spends its time avoiding.
    std::printf("       [model] available=%d  %s\n", classifier.info().available ? 1 : 0,
                classifier.info().detail.c_str());
    if (!classifier.info().available) return;  // no runtime on this machine

    const Frame frame = solid_frame(320, 240, 90);
    const std::vector<BayCrop> crops{
        {0, 0, 64, 64}, {64, 0, 64, 64}, {128, 40, 80, 60}, {200, 100, 100, 100}};
    const std::vector<BayVerdict> verdicts = classifier.classify_batch(frame, crops);

    CHECK_EQ(verdicts.size(), crops.size());
    for (const BayVerdict& verdict : verdicts) {
        CHECK(verdict.state != BayState::Unknown);
        CHECK(verdict.occupied_probability >= 0.0);
        CHECK(verdict.occupied_probability <= 1.0);
    }
}

TEST_CASE("model: one bay alone agrees with the same bay inside a batch") {
    // Batching is an optimisation, so it must not change an answer. If packing ever gets
    // an offset wrong this is the test that notices, because bay three in a batch of four
    // would start returning bay two's probability.
    const char* path = model_from_env();
    if (path == nullptr) return;

    BayOccupancyClassifier classifier(path);
    if (!classifier.info().available) return;

    Frame frame(256, 256, PixelFormat::Rgb24);
    for (int y = 0; y < 256; ++y) {
        std::uint8_t* row = frame.row(y);
        for (int x = 0; x < 256; ++x) {
            row[x * 3 + 0] = static_cast<std::uint8_t>(x);
            row[x * 3 + 1] = static_cast<std::uint8_t>(y);
            row[x * 3 + 2] = static_cast<std::uint8_t>((x + y) / 2);
        }
    }

    const std::vector<BayCrop> crops{
        {0, 0, 96, 96}, {96, 0, 96, 96}, {0, 96, 96, 96}, {96, 96, 96, 96}};
    const std::vector<BayVerdict> batched = classifier.classify_batch(frame, crops);
    CHECK_EQ(batched.size(), crops.size());

    for (std::size_t i = 0; i < crops.size(); ++i) {
        const BayVerdict alone = classifier.classify(frame, crops[i]);
        CHECK_NEAR(alone.occupied_probability, batched[i].occupied_probability, 1e-6);
    }

    // Printed so the same frame can be pushed through the Python side and the two
    // preprocessing paths compared by eye or by script. A cross-language model is only
    // as good as its resampling agreeing, and that is not something the C++ can check
    // about itself.
    std::printf("       [parity] gradient256 crops 96x96:");
    for (const BayVerdict& verdict : batched) {
        std::printf(" %.6f", verdict.occupied_probability);
    }
    std::printf("\n");
}

TEST_CASE("model: an off-screen bay stays Unknown while its neighbours are answered") {
    const char* path = model_from_env();
    if (path == nullptr) return;

    BayOccupancyClassifier classifier(path);
    if (!classifier.info().available) return;

    const Frame frame = solid_frame(200, 200, 140);
    const std::vector<BayCrop> crops{
        {0, 0, 64, 64},        // on the frame
        {900, 900, 64, 64},    // off the right and bottom
        {100, 100, 64, 64},    // on the frame
    };
    const std::vector<BayVerdict> verdicts = classifier.classify_batch(frame, crops);

    CHECK_EQ(verdicts.size(), static_cast<std::size_t>(3));
    CHECK(verdicts[0].state != BayState::Unknown);
    CHECK(verdicts[1].state == BayState::Unknown);
    CHECK(verdicts[2].state != BayState::Unknown);
}

TEST_CASE("verdict: the default is Unknown, which is what claims nothing") {
    const BayVerdict verdict;
    CHECK(verdict.state == BayState::Unknown);
    CHECK_NEAR(verdict.occupied_probability, 0.0, 1e-12);
    CHECK(!verdict.is_occupied());
}

PF_TEST_MAIN()
