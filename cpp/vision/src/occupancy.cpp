// SPDX-License-Identifier: MIT
//
// Bay occupancy classifier. See occupancy.hpp for why this runs in C++ and why the
// runtime is loaded rather than linked.

#include "parkfit/vision/occupancy.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

#include "onnx_session.hpp"
#include "parkfit/vision/onnx_detector.hpp"

namespace parkfit::vision {

using detail::read_double;
using detail::read_file;
using detail::read_int;
using detail::read_string;
using detail::read_string_array;

// ---------------------------------------------------------------------------
OccupancySpec OccupancySpec::from_json(const std::string& text) {
    OccupancySpec spec;
    if (text.empty()) return spec;

    read_int(text, "input_width", spec.input_width);
    read_int(text, "input_height", spec.input_height);
    read_string(text, "input_name", spec.input_name);
    read_string(text, "model_version", spec.model_version);
    read_double(text, "operating_threshold", spec.operating_threshold);
    read_string_array(text, "class_names", spec.class_names);

    // The exporter writes output_names as an array of one, because the detector's sidecar
    // has three and keeping the two files the same shape is worth more than saving a
    // bracket. Fall back to the scalar key so a hand-written sidecar also works.
    std::vector<std::string> outputs;
    if (read_string_array(text, "output_names", outputs) && !outputs.empty()) {
        spec.output_name = outputs.front();
    } else {
        read_string(text, "output_name", spec.output_name);
    }

    // A threshold outside (0, 1) is a corrupt sidecar rather than an unusual choice, and
    // honouring it would silently make every bay free or every bay occupied.
    if (!(spec.operating_threshold > 0.0 && spec.operating_threshold < 1.0)) {
        spec.operating_threshold = 0.5;
    }
    return spec;
}

// ---------------------------------------------------------------------------
double occupied_probability(float logit_free, float logit_occupied) {
    const float top = std::max(logit_free, logit_occupied);
    const double a = std::exp(static_cast<double>(logit_free - top));
    const double b = std::exp(static_cast<double>(logit_occupied - top));
    const double total = a + b;
    if (!(total > 0.0)) return 0.0;
    return b / total;
}

// ---------------------------------------------------------------------------
struct BayOccupancyClassifier::Impl {
    detail::OnnxSession ort;
    OccupancySpec spec;

    /// Scratch for the batched NCHW input, reused across frames for the same reason the
    /// detector reuses its own: a worker runs for months and reallocating per frame buys
    /// nothing.
    std::vector<float> input;

    bool start(const std::string& model_path, const std::string& requested_library) {
        std::vector<std::string> candidates;
        if (!requested_library.empty()) {
            candidates.push_back(requested_library);
        } else {
            candidates = OnnxDetector::default_library_candidates();
        }
        return ort.start(model_path, candidates);
    }

    /// Write one crop into the batch at `slot`, resampled to the model input.
    ///
    /// Nearest neighbour, matching the training pipeline. Interpolating here and not
    /// there would hand the model soft edges it never saw, which is a distribution shift
    /// introduced by the deployment rather than by the world.
    void pack(const Frame& frame, const BayCrop& crop, std::size_t slot, int in_w, int in_h) {
        const std::size_t plane = static_cast<std::size_t>(in_w) * in_h;
        const std::size_t base = slot * 3 * plane;
        const int channels = frame.channels();
        const bool bgr = frame.format() == PixelFormat::Bgr24;

        const double scale_x = static_cast<double>(crop.width) / in_w;
        const double scale_y = static_cast<double>(crop.height) / in_h;

        for (int y = 0; y < in_h; ++y) {
            int src_y = crop.y + static_cast<int>(static_cast<double>(y) * scale_y);
            src_y = std::clamp(src_y, 0, frame.height() - 1);
            const std::uint8_t* src_row = frame.row(src_y);

            for (int x = 0; x < in_w; ++x) {
                int src_x = crop.x + static_cast<int>(static_cast<double>(x) * scale_x);
                src_x = std::clamp(src_x, 0, frame.width() - 1);
                const std::uint8_t* p = src_row + static_cast<std::size_t>(src_x) * channels;

                float r;
                float g;
                float b;
                if (channels == 1) {
                    r = g = b = static_cast<float>(p[0]);
                } else if (bgr) {
                    b = static_cast<float>(p[0]);
                    g = static_cast<float>(p[1]);
                    r = static_cast<float>(p[2]);
                } else {
                    r = static_cast<float>(p[0]);
                    g = static_cast<float>(p[1]);
                    b = static_cast<float>(p[2]);
                }

                const std::size_t index = static_cast<std::size_t>(y) * in_w + x;
                input[base + index] = r / 255.0f;
                input[base + plane + index] = g / 255.0f;
                input[base + 2 * plane + index] = b / 255.0f;
            }
        }
    }
};

// ---------------------------------------------------------------------------
BayOccupancyClassifier::BayOccupancyClassifier(const std::string& model_path,
                                               const std::string& library)
    : impl_(std::make_unique<Impl>()) {
    // The sidecar is read first and unconditionally: even when the model fails to load,
    // spec() should describe what this classifier would have expected.
    const std::string sidecar_path = [&] {
        const std::size_t dot = model_path.find_last_of('.');
        return dot == std::string::npos ? model_path + ".json" : model_path.substr(0, dot) + ".json";
    }();
    impl_->spec = OccupancySpec::from_json(read_file(sidecar_path));

    if (!impl_->start(model_path, library)) {
        impl_->ort.available = false;
    }
}

BayOccupancyClassifier::~BayOccupancyClassifier() = default;

const OccupancySpec& BayOccupancyClassifier::spec() const { return impl_->spec; }

DetectorInfo BayOccupancyClassifier::info() const {
    DetectorInfo out;
    out.backend = "onnxruntime";
    out.model_version = impl_->spec.model_version;
    out.available = impl_->ort.available;
    out.detail = impl_->ort.detail;
    return out;
}

BayVerdict BayOccupancyClassifier::classify(const Frame& frame, const BayCrop& crop) {
    const auto verdicts = classify_batch(frame, {crop});
    return verdicts.empty() ? BayVerdict{} : verdicts.front();
}

std::vector<BayVerdict> BayOccupancyClassifier::classify_batch(
    const Frame& frame, const std::vector<BayCrop>& crops) {
    // One Unknown per crop up front. Every early return below then keeps the promise
    // that the answer is the same length as the question, which is what lets the caller
    // index verdicts against its own bay list.
    std::vector<BayVerdict> verdicts(crops.size());
    if (!impl_->ort.available || crops.empty()) return verdicts;
    if (frame.width() <= 0 || frame.height() <= 0) return verdicts;

    // Only crops that actually land on the frame are worth a forward pass. The rest stay
    // Unknown, and `slots` remembers where each packed crop belongs.
    std::vector<std::size_t> slots;
    slots.reserve(crops.size());
    for (std::size_t i = 0; i < crops.size(); ++i) {
        const BayCrop& crop = crops[i];
        if (!crop.valid()) continue;
        if (crop.x >= frame.width() || crop.y >= frame.height()) continue;
        if (crop.x + crop.width <= 0 || crop.y + crop.height <= 0) continue;
        slots.push_back(i);
    }
    if (slots.empty()) return verdicts;

    const int in_w = impl_->spec.input_width;
    const int in_h = impl_->spec.input_height;
    if (in_w <= 0 || in_h <= 0) return verdicts;

    const std::size_t plane = static_cast<std::size_t>(in_w) * in_h;
    impl_->input.assign(slots.size() * 3 * plane, 0.0f);
    for (std::size_t slot = 0; slot < slots.size(); ++slot) {
        impl_->pack(frame, crops[slots[slot]], slot, in_w, in_h);
    }

    const std::array<std::int64_t, 4> shape{static_cast<std::int64_t>(slots.size()), 3, in_h,
                                            in_w};
    OrtValue* input_tensor = nullptr;
    if (impl_->ort.failed(
            impl_->ort.api->CreateTensorWithDataAsOrtValue(
                impl_->ort.memory, impl_->input.data(), impl_->input.size() * sizeof(float),
                shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor),
            "CreateTensorWithDataAsOrtValue")) {
        return verdicts;
    }

    const char* input_names[] = {impl_->spec.input_name.c_str()};
    const char* output_names[] = {impl_->spec.output_name.c_str()};
    OrtValue* output = nullptr;

    OrtStatus* status = impl_->ort.api->Run(impl_->ort.session, nullptr, input_names,
                                            &input_tensor, 1, output_names, 1, &output);
    impl_->ort.api->ReleaseValue(input_tensor);
    if (impl_->ort.failed(status, "Run")) {
        if (output) impl_->ort.api->ReleaseValue(output);
        return verdicts;
    }

    // [batch, 2]. Anything else means the sidecar and the graph disagree, and guessing
    // which to believe would put a wrong verdict on a real bay.
    std::size_t rows = 0;
    std::size_t columns = 0;
    {
        OrtTensorTypeAndShapeInfo* info = nullptr;
        if (!impl_->ort.failed(impl_->ort.api->GetTensorTypeAndShape(output, &info),
                               "GetTensorTypeAndShape")) {
            std::size_t dim_count = 0;
            impl_->ort.api->GetDimensionsCount(info, &dim_count);
            std::vector<std::int64_t> dims(dim_count, 0);
            impl_->ort.api->GetDimensions(info, dims.data(), dim_count);
            if (dim_count == 2) {
                rows = static_cast<std::size_t>(std::max<std::int64_t>(0, dims[0]));
                columns = static_cast<std::size_t>(std::max<std::int64_t>(0, dims[1]));
            }
            impl_->ort.api->ReleaseTensorTypeAndShapeInfo(info);
        }
    }

    if (columns == 2 && rows == slots.size()) {
        float* logits = nullptr;
        impl_->ort.api->GetTensorMutableData(output, reinterpret_cast<void**>(&logits));
        if (logits != nullptr) {
            const double threshold = impl_->spec.operating_threshold;
            for (std::size_t slot = 0; slot < slots.size(); ++slot) {
                const double probability =
                    occupied_probability(logits[slot * 2], logits[slot * 2 + 1]);
                BayVerdict& verdict = verdicts[slots[slot]];
                verdict.occupied_probability = probability;
                verdict.state = probability >= threshold ? BayState::Occupied : BayState::Free;
            }
        }
    } else {
        impl_->ort.detail = "occupancy graph returned an unexpected output shape";
    }

    impl_->ort.api->ReleaseValue(output);
    return verdicts;
}

}  // namespace parkfit::vision
