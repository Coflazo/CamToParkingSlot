// SPDX-License-Identifier: MIT
//
// Bay occupancy: is this known parking space occupied right now?
//
// This is the question the product actually asks, and it is deliberately narrower than
// detection. Amsterdam publishes 210,247 parking bays as surveyed polygons, so where
// every bay is is known before a single pixel is read. Finding vehicles in a street scene
// is the hard version of the problem and the detector generalised badly across cameras;
// classifying a crop whose corners are already known transfers far better, because the
// model is answering one bounded question instead of searching an image.
//
// **This runs in C++ because it is on the clock.** Training happens in Python, where an
// extra minute costs nothing. Inference happens once per bay per frame, and a camera
// overlooking a street can see a few hundred bays, so the whole set has to clear in the
// gap between two samples. Crops are packed into one batched tensor and pushed through a
// single Run call rather than looped one at a time, which is most of the difference.
//
// The model is the same ONNX Runtime arrangement as the detector: loaded at runtime
// rather than linked, reached through the C API, with no ONNX Runtime type in this
// header. A deployment without the library gets `info().available == false` and every
// bay comes back Unknown, which the state machine already treats as "do not claim
// anything", and which is the correct answer rather than a worker that will not start.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "parkfit/vision/detector.hpp"
#include "parkfit/vision/frame.hpp"

namespace parkfit::vision {

/// What the classifier graph expects, read from `<model>.json`.
struct OccupancySpec {
    int input_width{96};
    int input_height{96};
    std::string input_name{"patch"};
    std::string output_name{"logits"};
    std::vector<std::string> class_names{"free", "occupied"};

    /// The point the model was actually tuned at, chosen on validation to hold the
    /// false-free rate under target. It travels in the sidecar rather than living as a
    /// constant here, because shipping weights without their operating point leaves the
    /// worker guessing at 0.5, which is not what anyone measured.
    double operating_threshold{0.5};

    std::string model_version{"unknown"};

    /// Parse a sidecar. Missing fields keep their defaults, which match the shipped
    /// model, so an absent sidecar degrades to a documented assumption rather than a
    /// crash.
    static OccupancySpec from_json(const std::string& text);
};

/// A rectangle in frame pixels. The worker derives these from bay polygons through the
/// homography it already holds, so nothing here has to know about geography.
struct BayCrop {
    int x{0};
    int y{0};
    int width{0};
    int height{0};

    [[nodiscard]] bool valid() const { return width > 0 && height > 0; }
};

/// What the classifier decided about one bay.
enum class BayState { Unknown, Free, Occupied };

struct BayVerdict {
    BayState state{BayState::Unknown};

    /// P(occupied). Reported even when the state is Unknown, so a caller can see how
    /// close a refusal was rather than only that it happened.
    double occupied_probability{0.0};

    [[nodiscard]] bool is_occupied() const { return state == BayState::Occupied; }
};

/// Softmax over the two logits, returning P(occupied).
///
/// Free rather than a member so it can be tested without a model, a runtime, or a file
/// on disk. Numerically stable: the max is subtracted before exponentiating, because a
/// confident graph emits logits large enough to overflow a float otherwise.
double occupied_probability(float logit_free, float logit_occupied);

/// Runs the exported occupancy classifier. Construction never throws; ask
/// `info().available`.
class BayOccupancyClassifier final {
  public:
    /// `library` may be empty, in which case the same locations the detector searches are
    /// tried, including the onnxruntime shipped inside a Python virtual environment.
    explicit BayOccupancyClassifier(const std::string& model_path,
                                    const std::string& library = "");
    ~BayOccupancyClassifier();

    BayOccupancyClassifier(const BayOccupancyClassifier&) = delete;
    BayOccupancyClassifier& operator=(const BayOccupancyClassifier&) = delete;

    /// Classify one bay. Convenience over the batch call, which is what the worker uses.
    [[nodiscard]] BayVerdict classify(const Frame& frame, const BayCrop& crop);

    /// Classify every bay in one pass.
    ///
    /// Returns one verdict per crop, in order, always the same length as `crops`. A crop
    /// that falls outside the frame comes back Unknown rather than being dropped, because
    /// the caller is indexing these against its own bay list and a short answer would
    /// silently shift every bay after it.
    [[nodiscard]] std::vector<BayVerdict> classify_batch(const Frame& frame,
                                                         const std::vector<BayCrop>& crops);

    [[nodiscard]] DetectorInfo info() const;
    [[nodiscard]] const OccupancySpec& spec() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace parkfit::vision
