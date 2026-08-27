// SPDX-License-Identifier: MIT
//
// ONNX Runtime detector backend. See onnx_detector.hpp for why the library is loaded at
// runtime rather than linked, and why no ONNX Runtime type escapes this file.

#include "parkfit/vision/onnx_detector.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

#include "onnx_session.hpp"

namespace parkfit::vision {

// The JSON readers and the dynamic loader now live in onnx_session.hpp, shared with
// the occupancy classifier. Pulled in by name so the call sites below read unchanged.
using detail::read_file;
using detail::read_int;
using detail::read_string;
using detail::read_string_array;

// ---------------------------------------------------------------------------
OnnxModelSpec OnnxModelSpec::from_json(const std::string& text) {
    OnnxModelSpec spec;
    if (text.empty()) return spec;
    read_int(text, "input_width", spec.input_width);
    read_int(text, "input_height", spec.input_height);
    read_int(text, "output_stride", spec.output_stride);
    read_string(text, "input_name", spec.input_name);
    read_string(text, "model_version", spec.model_version);
    read_string_array(text, "output_names", spec.output_names);
    read_string_array(text, "class_names", spec.class_names);
    return spec;
}

std::vector<std::string> OnnxDetector::default_library_candidates() {
    // Order matters, and getting it wrong is quiet. A bare library name is resolved by
    // the OS loader through the system search path, which on a developer machine happily
    // finds some other application's copy: this box had a 1.17 runtime on PATH while the
    // project's own virtual environment held the 1.29 the build was configured against,
    // and the loader took the old one and refused the model. The environment we control
    // is therefore tried first and the bare name is the last resort rather than the
    // first guess.
    return {
#ifdef _WIN32
        ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll",
        "../.venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll",
        "../../.venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll",
        "onnxruntime.dll",
#else
        ".venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so",
        "../.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so",
        "libonnxruntime.so",
        "libonnxruntime.so.1",
#endif
    };
}

// ---------------------------------------------------------------------------
struct OnnxDetector::Impl {
    detail::OnnxSession ort;
    OnnxModelSpec spec;

    /// Scratch buffer for the NCHW input, reused across frames. A worker samples a frame
    /// every few seconds for months; allocating half a megabyte each time is free in
    /// wall-clock terms and needless in every other.
    std::vector<float> input;

    /// Open the model, defaulting the library search to the usual locations.
    bool start(const std::string& model_path, const std::string& requested_library) {
        std::vector<std::string> candidates;
        if (!requested_library.empty()) {
            candidates.push_back(requested_library);
        } else {
            candidates = OnnxDetector::default_library_candidates();
        }
        return ort.start(model_path, candidates);
    }
};

// ---------------------------------------------------------------------------
OnnxDetector::OnnxDetector(const std::string& model_path, const std::string& library)
    : impl_(std::make_unique<Impl>()) {
    // The sidecar is read first and unconditionally: even when the model fails to load,
    // spec() should describe what this detector would have expected.
    const std::string sidecar_path = [&] {
        const std::size_t dot = model_path.find_last_of('.');
        return dot == std::string::npos ? model_path + ".json" : model_path.substr(0, dot) + ".json";
    }();
    impl_->spec = OnnxModelSpec::from_json(read_file(sidecar_path));

    if (!impl_->start(model_path, library)) {
        impl_->ort.available = false;
    }
}

OnnxDetector::~OnnxDetector() = default;

const OnnxModelSpec& OnnxDetector::spec() const { return impl_->spec; }

DetectorInfo OnnxDetector::info() const {
    DetectorInfo out;
    out.backend = "onnxruntime";
    out.model_version = impl_->spec.model_version;
    out.available = impl_->ort.available;
    out.detail = impl_->ort.detail;
    return out;
}

std::vector<Detection> OnnxDetector::detect(const Frame& frame, double score_threshold) {
    if (!impl_->ort.available || frame.width() <= 0 || frame.height() <= 0) return {};

    const int in_w = impl_->spec.input_width;
    const int in_h = impl_->spec.input_height;
    const std::size_t elements = static_cast<std::size_t>(3) * in_w * in_h;
    impl_->input.assign(elements, 0.0f);

    // Nearest neighbour, matching the training pipeline exactly. Interpolating here and
    // not there would hand the model gradients along every edge that it never saw during
    // training, which is a distribution shift introduced by the deployment rather than by
    // the world.
    const double scale_x = static_cast<double>(frame.width()) / in_w;
    const double scale_y = static_cast<double>(frame.height()) / in_h;
    const int channels = frame.channels();
    const bool bgr = frame.format() == PixelFormat::Bgr24;

    for (int y = 0; y < in_h; ++y) {
        const int src_y =
            std::min(frame.height() - 1, static_cast<int>(static_cast<double>(y) * scale_y));
        const std::uint8_t* src_row = frame.row(src_y);
        for (int x = 0; x < in_w; ++x) {
            const int src_x =
                std::min(frame.width() - 1, static_cast<int>(static_cast<double>(x) * scale_x));
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

            const std::size_t plane = static_cast<std::size_t>(in_w) * in_h;
            const std::size_t index = static_cast<std::size_t>(y) * in_w + x;
            impl_->input[index] = r / 255.0f;
            impl_->input[plane + index] = g / 255.0f;
            impl_->input[2 * plane + index] = b / 255.0f;
        }
    }

    const std::array<std::int64_t, 4> shape{1, 3, in_h, in_w};
    OrtValue* input_tensor = nullptr;
    if (impl_->ort.failed(impl_->ort.api->CreateTensorWithDataAsOrtValue(
                          impl_->ort.memory, impl_->input.data(), impl_->input.size() * sizeof(float),
                          shape.data(), shape.size(),
                          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor),
                      "CreateTensorWithDataAsOrtValue")) {
        return {};
    }

    std::vector<const char*> input_names{impl_->spec.input_name.c_str()};
    std::vector<const char*> output_names;
    output_names.reserve(impl_->spec.output_names.size());
    for (const auto& name : impl_->spec.output_names) output_names.push_back(name.c_str());

    std::vector<OrtValue*> outputs(output_names.size(), nullptr);
    OrtStatus* status =
        impl_->ort.api->Run(impl_->ort.session, nullptr, input_names.data(), &input_tensor, 1,
                        output_names.data(), output_names.size(), outputs.data());
    impl_->ort.api->ReleaseValue(input_tensor);
    if (impl_->ort.failed(status, "Run")) {
        for (auto* value : outputs) {
            if (value) impl_->ort.api->ReleaseValue(value);
        }
        return {};
    }

    // Shape comes from the heatmap: [1, classes, rows, cols].
    int classes = 0;
    int rows = 0;
    int cols = 0;
    {
        OrtTensorTypeAndShapeInfo* info = nullptr;
        if (!impl_->ort.failed(impl_->ort.api->GetTensorTypeAndShape(outputs[0], &info),
                           "GetTensorTypeAndShape")) {
            std::size_t dim_count = 0;
            impl_->ort.api->GetDimensionsCount(info, &dim_count);
            std::vector<std::int64_t> dims(dim_count, 0);
            impl_->ort.api->GetDimensions(info, dims.data(), dim_count);
            if (dim_count == 4) {
                classes = static_cast<int>(dims[1]);
                rows = static_cast<int>(dims[2]);
                cols = static_cast<int>(dims[3]);
            }
            impl_->ort.api->ReleaseTensorTypeAndShapeInfo(info);
        }
    }

    std::vector<Detection> detections;
    if (classes > 0 && rows > 0 && cols > 0) {
        float* heatmap = nullptr;
        float* size = nullptr;
        float* offset = nullptr;
        impl_->ort.api->GetTensorMutableData(outputs[0], reinterpret_cast<void**>(&heatmap));
        impl_->ort.api->GetTensorMutableData(outputs[1], reinterpret_cast<void**>(&size));
        impl_->ort.api->GetTensorMutableData(outputs[2], reinterpret_cast<void**>(&offset));

        if (heatmap && size && offset) {
            detections = decode_centernet(heatmap, size, offset, classes, rows, cols,
                                          impl_->spec.output_stride, score_threshold,
                                          impl_->spec.class_names);
            // Boxes come back in model-input pixels. The caller measures geometry against
            // the frame it supplied, so they are scaled back before anyone sees them.
            for (auto& d : detections) {
                d.x1 *= scale_x;
                d.x2 *= scale_x;
                d.y1 *= scale_y;
                d.y2 *= scale_y;
            }
        }
    }

    for (auto* value : outputs) {
        if (value) impl_->ort.api->ReleaseValue(value);
    }
    return detections;
}

// ---------------------------------------------------------------------------
std::vector<Detection> decode_centernet(const float* heatmap, const float* size,
                                        const float* offset, int classes, int rows, int cols,
                                        int stride, double score_threshold,
                                        const std::vector<std::string>& class_names,
                                        int max_detections) {
    std::vector<Detection> detections;
    if (!heatmap || !size || !offset || classes <= 0 || rows <= 0 || cols <= 0) return detections;

    const std::size_t plane = static_cast<std::size_t>(rows) * cols;
    const auto at = [cols](const float* base, int y, int x) {
        return base[static_cast<std::size_t>(y) * cols + x];
    };

    for (int c = 0; c < classes; ++c) {
        const float* channel = heatmap + static_cast<std::size_t>(c) * plane;
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                const float score = at(channel, y, x);
                if (score < static_cast<float>(score_threshold)) continue;

                // A 3x3 local maximum. One vehicle lights a small neighbourhood, so
                // taking every cell over the threshold would report the same car up to
                // nine times and leave non-maximum suppression to clean up a mess that
                // never needed making.
                bool peak = true;
                for (int dy = -1; dy <= 1 && peak; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        const int ny = y + dy;
                        const int nx = x + dx;
                        if (ny < 0 || nx < 0 || ny >= rows || nx >= cols) continue;
                        if (at(channel, ny, nx) > score) {
                            peak = false;
                            break;
                        }
                    }
                }
                if (!peak) continue;

                const double w = at(size, y, x);
                const double h = at(size + plane, y, x);
                if (w <= 0.0 || h <= 0.0) continue;

                const double cx = (x + at(offset, y, x)) * stride;
                const double cy = (y + at(offset + plane, y, x)) * stride;

                Detection d;
                d.x1 = cx - w * 0.5;
                d.y1 = cy - h * 0.5;
                d.x2 = cx + w * 0.5;
                d.y2 = cy + h * 0.5;
                d.score = score;
                d.class_id = c;
                d.label = c < static_cast<int>(class_names.size()) ? class_names[c] : "unknown";
                detections.push_back(d);
            }
        }
    }

    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) { return a.score > b.score; });
    if (static_cast<int>(detections.size()) > max_detections) {
        detections.resize(static_cast<std::size_t>(max_detections));
    }
    return detections;
}

}  // namespace parkfit::vision
