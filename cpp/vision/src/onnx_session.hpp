// SPDX-License-Identifier: MIT
//
// Shared ONNX Runtime plumbing: library loading, session creation, and the small JSON
// reader the model sidecars are written in.
//
// This is a private header. It lives under src/ rather than include/ and is never
// installed, because it includes onnxruntime_c_api.h, and that header is four hundred
// kilobytes. Anything that merely wants to know what a Detector is should not pay for it.
//
// It exists because there are now two models. The detector answers "what vehicles are in
// this frame" and the occupancy classifier answers "is this known bay occupied", which
// are different questions with different graphs, but identical requirements for getting
// ONNX Runtime open in the first place. Copying a hundred lines of dlopen and error
// handling into the second one would have meant fixing every future loader bug twice.
//
// Everything here is `inline` on purpose: two translation units include it, and
// non-inline definitions in a header are a duplicate symbol at link time.

#pragma once

#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

#include "onnxruntime_c_api.h"

namespace parkfit::vision::detail {

// ---------------------------------------------------------------------------
// A very small JSON reader.
//
// It only ever reads sidecars this project writes, so it does not need to be a real
// parser. It does need to fail loudly rather than return a plausible wrong number, so
// every reader reports whether it found the key instead of silently handing back a
// default the caller cannot distinguish from a real value.
// ---------------------------------------------------------------------------
inline std::size_t find_key(const std::string& text, const std::string& key) {
    const std::string quoted = "\"" + key + "\"";
    const std::size_t at = text.find(quoted);
    if (at == std::string::npos) return std::string::npos;
    const std::size_t colon = text.find(':', at + quoted.size());
    return colon == std::string::npos ? std::string::npos : colon + 1;
}

inline bool read_int(const std::string& text, const std::string& key, int& out) {
    const std::size_t at = find_key(text, key);
    if (at == std::string::npos) return false;
    try {
        out = std::stoi(text.substr(at));
    } catch (...) {
        return false;
    }
    return true;
}

inline bool read_double(const std::string& text, const std::string& key, double& out) {
    const std::size_t at = find_key(text, key);
    if (at == std::string::npos) return false;
    try {
        out = std::stod(text.substr(at));
    } catch (...) {
        return false;
    }
    return true;
}

inline bool read_string(const std::string& text, const std::string& key, std::string& out) {
    const std::size_t at = find_key(text, key);
    if (at == std::string::npos) return false;
    const std::size_t open = text.find('"', at);
    if (open == std::string::npos) return false;
    const std::size_t close = text.find('"', open + 1);
    if (close == std::string::npos) return false;
    out = text.substr(open + 1, close - open - 1);
    return true;
}

inline bool read_string_array(const std::string& text, const std::string& key,
                              std::vector<std::string>& out) {
    const std::size_t at = find_key(text, key);
    if (at == std::string::npos) return false;
    const std::size_t open = text.find('[', at);
    if (open == std::string::npos) return false;
    const std::size_t close = text.find(']', open);
    if (close == std::string::npos) return false;

    std::vector<std::string> values;
    std::size_t cursor = open + 1;
    while (cursor < close) {
        const std::size_t quote = text.find('"', cursor);
        if (quote == std::string::npos || quote > close) break;
        const std::size_t end = text.find('"', quote + 1);
        if (end == std::string::npos || end > close) break;
        values.push_back(text.substr(quote + 1, end - quote - 1));
        cursor = end + 1;
    }
    if (values.empty()) return false;
    out = std::move(values);
    return true;
}

inline std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

// ---------------------------------------------------------------------------
// Dynamic library handling
// ---------------------------------------------------------------------------
#ifdef _WIN32
using LibHandle = HMODULE;
inline LibHandle open_library(const std::string& path) { return ::LoadLibraryA(path.c_str()); }
inline void* find_symbol(LibHandle h, const char* name) {
    return reinterpret_cast<void*>(::GetProcAddress(h, name));
}
inline void close_library(LibHandle h) {
    if (h) ::FreeLibrary(h);
}
inline std::wstring widen(const std::string& text) {
    if (text.empty()) return {};
    const int needed =
        ::MultiByteToWideChar(CP_UTF8, 0, text.c_str(), static_cast<int>(text.size()), nullptr, 0);
    std::wstring out(static_cast<std::size_t>(needed), L'\0');
    ::MultiByteToWideChar(CP_UTF8, 0, text.c_str(), static_cast<int>(text.size()), out.data(),
                          needed);
    return out;
}
#else
using LibHandle = void*;
inline LibHandle open_library(const std::string& path) {
    return ::dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
}
inline void* find_symbol(LibHandle h, const char* name) { return ::dlsym(h, name); }
inline void close_library(LibHandle h) {
    if (h) ::dlclose(h);
}
#endif

/// An open ONNX Runtime session, or an honest account of why there is not one.
///
/// Construction never throws and `start` never throws. A deployment without the runtime
/// gets `available == false` and a `detail` string naming every path that was tried,
/// which is a far better failure than a worker that will not start.
struct OnnxSession {
    LibHandle library{};
    const OrtApi* api{nullptr};
    OrtEnv* env{nullptr};
    OrtSession* session{nullptr};
    OrtSessionOptions* options{nullptr};
    OrtMemoryInfo* memory{nullptr};

    bool available{false};
    std::string detail;
    std::string library_path;

    OnnxSession() = default;
    OnnxSession(const OnnxSession&) = delete;
    OnnxSession& operator=(const OnnxSession&) = delete;

    ~OnnxSession() {
        if (api) {
            if (memory) api->ReleaseMemoryInfo(memory);
            if (session) api->ReleaseSession(session);
            if (options) api->ReleaseSessionOptions(options);
            if (env) api->ReleaseEnv(env);
        }
        close_library(library);
    }

    /// Turn an OrtStatus into a message and release it. Returns true when there was an
    /// error, so call sites read as `if (failed(status, "doing the thing")) return;`.
    bool failed(OrtStatus* status, const char* what) {
        if (status == nullptr) return false;
        detail = std::string(what) + ": " + api->GetErrorMessage(status);
        api->ReleaseStatus(status);
        return true;
    }

    /// Try one candidate all the way through to a usable OrtApi.
    ///
    /// Opening the library is not the same as being able to use it. A machine can carry
    /// several ONNX Runtimes, and one that opens but is older than the headers this was
    /// built against is a dead end rather than an answer, so a failure here closes the
    /// handle and lets the caller move on to the next path.
    bool try_candidate(const std::string& path) {
        library = open_library(path);
        if (!library) return false;
        library_path = path;

        auto* get_base =
            reinterpret_cast<const OrtApiBase* (*)()>(find_symbol(library, "OrtGetApiBase"));
        if (get_base == nullptr) {
            detail = path + " does not export OrtGetApiBase; it is not ONNX Runtime";
            close_library(library);
            library = {};
            return false;
        }
        const OrtApiBase* base = get_base();
        if (base == nullptr) {
            detail = path + ": OrtGetApiBase returned null";
            close_library(library);
            library = {};
            return false;
        }
        api = base->GetApi(ORT_API_VERSION);
        if (api == nullptr) {
            detail = path + ": built against API version " + std::to_string(ORT_API_VERSION) +
                     ", library reports " + base->GetVersionString();
            close_library(library);
            library = {};
            return false;
        }
        return true;
    }

    bool load_library(const std::vector<std::string>& candidates) {
        std::vector<std::string> reasons;
        for (const auto& path : candidates) {
            if (try_candidate(path)) return true;
            if (!detail.empty()) reasons.push_back(detail);
        }
        detail = "no usable ONNX Runtime; tried";
        for (const auto& path : candidates) detail += " " + path;
        for (const auto& reason : reasons) detail += " | " + reason;
        return false;
    }

    bool start(const std::string& model_path, const std::vector<std::string>& candidates) {
        if (!load_library(candidates)) return false;

        if (failed(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "parkfit", &env), "CreateEnv")) {
            return false;
        }
        if (failed(api->CreateSessionOptions(&options), "CreateSessionOptions")) return false;
        // One thread. The worker samples at a fraction of a frame per second, so latency
        // is irrelevant, and an intra-op pool sized to the core count is memory spent to
        // no purpose on a box that may be running several workers.
        api->SetIntraOpNumThreads(options, 1);
        api->SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL);

#ifdef _WIN32
        const std::wstring wide = widen(model_path);
        OrtStatus* status = api->CreateSession(env, wide.c_str(), options, &session);
#else
        OrtStatus* status = api->CreateSession(env, model_path.c_str(), options, &session);
#endif
        if (failed(status, "CreateSession")) return false;

        if (failed(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory),
                   "CreateCpuMemoryInfo")) {
            return false;
        }

        available = true;
        detail = "loaded " + model_path + " via " + library_path;
        return true;
    }
};

}  // namespace parkfit::vision::detail
