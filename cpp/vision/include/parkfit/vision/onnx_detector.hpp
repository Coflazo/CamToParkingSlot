// SPDX-License-Identifier: MIT
//
// The production detector: a CenterNet graph run through ONNX Runtime.
//
// Two decisions shape this file.
//
// **ONNX Runtime is loaded at runtime, not linked.** The library is opened with
// LoadLibrary/dlopen and reached through its C API, so parkfit_vision has no link-time
// dependency on it and the worker builds on a machine that has never heard of it. A
// deployment without the library gets a detector that reports `available == false` and
// detects nothing, which the state machine already treats as UNKNOWN. That is the
// correct answer, and it is a great deal better than a worker that will not start.
//
// **No ONNX Runtime type appears here.** The whole C API lives behind a pimpl in the
// .cpp, because its header is four hundred kilobytes and including it from a header
// would push that through every translation unit that wants to know what a Detector is.
//
// The model contract is small and is read from a JSON sidecar beside the .onnx file
// rather than compiled in: input size, output stride, and the class order. Retraining at
// a different input size therefore cannot silently start feeding the model wrongly
// scaled images.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "parkfit/vision/detector.hpp"
#include "parkfit/vision/frame.hpp"

namespace parkfit::vision {

/// What the graph expects, read from `<model>.json`.
struct OnnxModelSpec {
    int input_width{512};
    int input_height{288};
    int output_stride{4};
    std::string input_name{"image"};
    std::vector<std::string> output_names{"heatmap", "size", "offset"};
    std::vector<std::string> class_names{"car", "van",       "truck",  "bus",
                                         "motorcycle", "bicycle", "trailer"};
    std::string model_version{"unknown"};

    /// Parse a sidecar. Missing fields keep their defaults, which match the shipped
    /// model, so an absent sidecar degrades to a documented assumption rather than a
    /// crash.
    static OnnxModelSpec from_json(const std::string& text);
};

/// Runs the exported detector. Construction never throws; ask `info().available`.
class OnnxDetector final : public Detector {
  public:
    /// `library` may be empty, in which case a short list of usual locations is tried,
    /// including the onnxruntime shipped inside a Python virtual environment.
    explicit OnnxDetector(const std::string& model_path, const std::string& library = "");
    ~OnnxDetector() override;

    OnnxDetector(const OnnxDetector&) = delete;
    OnnxDetector& operator=(const OnnxDetector&) = delete;

    std::vector<Detection> detect(const Frame& frame, double score_threshold) override;
    [[nodiscard]] DetectorInfo info() const override;

    [[nodiscard]] const OnnxModelSpec& spec() const;

    /// Where the loader will look when no explicit path is given. Exposed so the worker
    /// can print it when the library is missing, rather than leaving the operator to
    /// guess what it searched.
    static std::vector<std::string> default_library_candidates();

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Decode a CenterNet head into boxes. Split out from the detector so it can be tested
/// against the Python reference on hand-built tensors, with no model or runtime present.
///
/// `heatmap` is [classes][rows][cols], `size` and `offset` are [2][rows][cols].
std::vector<Detection> decode_centernet(const float* heatmap, const float* size,
                                        const float* offset, int classes, int rows, int cols,
                                        int stride, double score_threshold,
                                        const std::vector<std::string>& class_names,
                                        int max_detections = 64);

}  // namespace parkfit::vision
