// SPDX-License-Identifier: MIT
//
// A deliberately tiny test harness.
//
// This exists instead of Catch2 or GoogleTest so the C++ tree builds with nothing but
// a compiler, no FetchContent, no network at configure time, no vendored megabytes.
// It covers exactly what these tests need: named cases, file-and-line failure reporting,
// tolerance comparison for floating point, and a CTest-compatible exit code.

#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

namespace pftest {

struct Case {
    std::string name;
    std::function<void()> fn;
};

inline std::vector<Case>& registry() {
    static std::vector<Case> cases;
    return cases;
}

inline int& failure_count() {
    static int n = 0;
    return n;
}

inline std::string& current_case() {
    static std::string s;
    return s;
}

struct Registrar {
    Registrar(const char* name, std::function<void()> fn) {
        registry().push_back(Case{name, std::move(fn)});
    }
};

inline void report_failure(const char* file, int line, const std::string& message) {
    ++failure_count();
    std::fprintf(stderr, "  FAIL  [%s]\n        %s:%d\n        %s\n", current_case().c_str(), file,
                 line, message.c_str());
}

inline bool nearly_equal(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

inline int run_all() {
    int failed_cases = 0;
    for (auto& c : registry()) {
        current_case() = c.name;
        const int before = failure_count();
        try {
            c.fn();
        } catch (const std::exception& e) {
            report_failure(__FILE__, __LINE__, std::string("uncaught exception: ") + e.what());
        } catch (...) {
            report_failure(__FILE__, __LINE__, "uncaught unknown exception");
        }
        const bool ok = failure_count() == before;
        if (!ok) ++failed_cases;
        std::printf("%-6s %s\n", ok ? "ok" : "FAILED", c.name.c_str());
    }
    std::printf("\n%zu cases, %d failed assertions, %d failed cases\n", registry().size(),
                failure_count(), failed_cases);
    return failed_cases == 0 ? 0 : 1;
}

}  // namespace pftest

#define PF_CONCAT_INNER(a, b) a##b
#define PF_CONCAT(a, b) PF_CONCAT_INNER(a, b)

#define TEST_CASE(name)                                                       \
    static void PF_CONCAT(pf_case_, __LINE__)();                              \
    static ::pftest::Registrar PF_CONCAT(pf_reg_, __LINE__)(                   \
        name, []() { PF_CONCAT(pf_case_, __LINE__)(); });                      \
    static void PF_CONCAT(pf_case_, __LINE__)()

#define CHECK(expr)                                                           \
    do {                                                                      \
        if (!(expr)) ::pftest::report_failure(__FILE__, __LINE__, "CHECK(" #expr ") is false"); \
    } while (0)

#define CHECK_EQ(a, b)                                                        \
    do {                                                                      \
        auto pf_a = (a);                                                      \
        auto pf_b = (b);                                                      \
        if (!(pf_a == pf_b)) {                                                \
            ::pftest::report_failure(__FILE__, __LINE__,                       \
                                     "CHECK_EQ(" #a ", " #b "): values differ"); \
        }                                                                     \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                 \
    do {                                                                      \
        const double pf_a = static_cast<double>(a);                           \
        const double pf_b = static_cast<double>(b);                           \
        const double pf_t = static_cast<double>(tol);                         \
        if (!::pftest::nearly_equal(pf_a, pf_b, pf_t)) {                      \
            char pf_buf[256];                                                 \
            std::snprintf(pf_buf, sizeof(pf_buf),                             \
                          "CHECK_NEAR(" #a ", " #b ") -> %.6f vs %.6f (tol %.6f, delta %.6f)", \
                          pf_a, pf_b, pf_t, std::fabs(pf_a - pf_b));          \
            ::pftest::report_failure(__FILE__, __LINE__, pf_buf);             \
        }                                                                     \
    } while (0)

#define PF_TEST_MAIN() \
    int main() { return ::pftest::run_all(); }
