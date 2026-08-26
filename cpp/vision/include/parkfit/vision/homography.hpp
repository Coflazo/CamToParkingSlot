// SPDX-License-Identifier: MIT
//
// Image-to-ground homography.
//
// This is what makes a camera measurement mean anything. A bounding box in pixels tells
// you nothing about whether a car fits: perspective makes a five-metre gap at the far
// end of a street occupy fewer pixels than a two-metre gap near the camera. Without a
// calibrated mapping to the ground plane, "the gap looks about this wide" is not a
// measurement, it is a guess with a number attached.
//
// Solved with the normalised Direct Linear Transform. Normalisation is not optional:
// raw pixel coordinates give the DLT design matrix a condition number in the millions,
// and the resulting homography is numerically worthless. Hartley's isotropic scaling --
// centre the points, scale so the mean distance from the origin is sqrt(2) -- fixes it.
//
// World coordinates are RD New metres, which is exactly what Amsterdam publishes its
// parking-bay corners in. Those corners are surveyed reference points, freely available,
// in the same metric frame the geometry needs -- so calibrating a camera that overlooks
// marked bays needs no field survey at all.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <vector>

namespace parkfit::vision {

struct Point2d {
    double x{};
    double y{};
};

/// One image-to-world correspondence.
struct ControlPoint {
    Point2d image;  ///< pixels
    Point2d world;  ///< RD New metres
};

/// A 3x3 projective transform, row-major.
struct Homography {
    std::array<double, 9> m{{1, 0, 0, 0, 1, 0, 0, 0, 1}};

    [[nodiscard]] Point2d apply(const Point2d& p) const {
        const double w = m[6] * p.x + m[7] * p.y + m[8];
        if (std::abs(w) < 1e-12) return Point2d{0.0, 0.0};
        return Point2d{(m[0] * p.x + m[1] * p.y + m[2]) / w,
                       (m[3] * p.x + m[4] * p.y + m[5]) / w};
    }

    [[nodiscard]] bool valid() const {
        for (double v : m) {
            if (!std::isfinite(v)) return false;
        }
        return std::abs(m[0] * m[4] - m[1] * m[3]) > 1e-12;
    }

    /// Inverse transform, for projecting world geometry back into the image.
    [[nodiscard]] Homography inverse() const {
        const double a = m[0], b = m[1], c = m[2];
        const double d = m[3], e = m[4], f = m[5];
        const double g = m[6], h = m[7], i = m[8];

        const double A = e * i - f * h;
        const double B = -(d * i - f * g);
        const double C = d * h - e * g;
        const double det = a * A + b * B + c * C;

        Homography out;
        if (std::abs(det) < 1e-15) return out;
        const double inv = 1.0 / det;
        out.m = {A * inv,                 -(b * i - c * h) * inv,  (b * f - c * e) * inv,
                 B * inv,                 (a * i - c * g) * inv,   -(a * f - c * d) * inv,
                 C * inv,                 -(a * h - b * g) * inv,  (a * e - b * d) * inv};
        return out;
    }
};

namespace detail {

/// Hartley normalisation: translate to the centroid, scale so mean radius is sqrt(2).
struct Normalisation {
    double scale{1.0};
    double cx{0.0};
    double cy{0.0};

    [[nodiscard]] Point2d apply(const Point2d& p) const {
        return Point2d{(p.x - cx) * scale, (p.y - cy) * scale};
    }
};

inline Normalisation normalisation_for(const std::vector<Point2d>& points) {
    Normalisation n;
    if (points.empty()) return n;
    for (const auto& p : points) {
        n.cx += p.x;
        n.cy += p.y;
    }
    n.cx /= static_cast<double>(points.size());
    n.cy /= static_cast<double>(points.size());

    double mean_distance = 0.0;
    for (const auto& p : points) {
        mean_distance += std::hypot(p.x - n.cx, p.y - n.cy);
    }
    mean_distance /= static_cast<double>(points.size());
    n.scale = mean_distance > 1e-12 ? std::sqrt(2.0) / mean_distance : 1.0;
    return n;
}

/// Smallest-singular-vector of A^T A, by Jacobi eigenvalue iteration.
///
/// A full SVD would be the textbook route, but the matrix here is a fixed 9x9 symmetric
/// one and Jacobi converges on it in a handful of sweeps. That keeps the whole vision
/// module free of a linear-algebra dependency.
inline std::array<double, 9> smallest_eigenvector_9(std::array<double, 81> a) {
    std::array<double, 81> v{};
    for (int i = 0; i < 9; ++i) v[static_cast<std::size_t>(i) * 9 + i] = 1.0;

    for (int sweep = 0; sweep < 60; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < 9; ++p) {
            for (int q = p + 1; q < 9; ++q) {
                off += a[static_cast<std::size_t>(p) * 9 + q] * a[static_cast<std::size_t>(p) * 9 + q];
            }
        }
        if (off < 1e-22) break;

        for (int p = 0; p < 9; ++p) {
            for (int q = p + 1; q < 9; ++q) {
                const double apq = a[static_cast<std::size_t>(p) * 9 + q];
                if (std::abs(apq) < 1e-18) continue;
                const double app = a[static_cast<std::size_t>(p) * 9 + p];
                const double aqq = a[static_cast<std::size_t>(q) * 9 + q];
                const double theta = (aqq - app) / (2.0 * apq);
                const double t = (theta >= 0.0 ? 1.0 : -1.0) /
                                 (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;

                for (int k = 0; k < 9; ++k) {
                    const double akp = a[static_cast<std::size_t>(k) * 9 + p];
                    const double akq = a[static_cast<std::size_t>(k) * 9 + q];
                    a[static_cast<std::size_t>(k) * 9 + p] = c * akp - s * akq;
                    a[static_cast<std::size_t>(k) * 9 + q] = s * akp + c * akq;
                }
                for (int k = 0; k < 9; ++k) {
                    const double apk = a[static_cast<std::size_t>(p) * 9 + k];
                    const double aqk = a[static_cast<std::size_t>(q) * 9 + k];
                    a[static_cast<std::size_t>(p) * 9 + k] = c * apk - s * aqk;
                    a[static_cast<std::size_t>(q) * 9 + k] = s * apk + c * aqk;
                }
                for (int k = 0; k < 9; ++k) {
                    const double vkp = v[static_cast<std::size_t>(k) * 9 + p];
                    const double vkq = v[static_cast<std::size_t>(k) * 9 + q];
                    v[static_cast<std::size_t>(k) * 9 + p] = c * vkp - s * vkq;
                    v[static_cast<std::size_t>(k) * 9 + q] = s * vkp + c * vkq;
                }
            }
        }
    }

    int smallest = 0;
    double best = std::numeric_limits<double>::max();
    for (int i = 0; i < 9; ++i) {
        const double lambda = a[static_cast<std::size_t>(i) * 9 + i];
        if (lambda < best) {
            best = lambda;
            smallest = i;
        }
    }

    std::array<double, 9> out{};
    for (int i = 0; i < 9; ++i) {
        out[static_cast<std::size_t>(i)] = v[static_cast<std::size_t>(i) * 9 + smallest];
    }
    return out;
}

}  // namespace detail

/// Fit a homography to correspondences by normalised DLT. Needs at least four.
inline bool solve_homography(const std::vector<ControlPoint>& points, Homography& out) {
    if (points.size() < 4) return false;

    std::vector<Point2d> image_points;
    std::vector<Point2d> world_points;
    image_points.reserve(points.size());
    world_points.reserve(points.size());
    for (const auto& cp : points) {
        image_points.push_back(cp.image);
        world_points.push_back(cp.world);
    }

    const auto ni = detail::normalisation_for(image_points);
    const auto nw = detail::normalisation_for(world_points);

    // Accumulate A^T A directly: 2 rows per correspondence, 9 columns, so the normal
    // matrix is a fixed 9x9 regardless of how many points there are.
    std::array<double, 81> ata{};
    for (const auto& cp : points) {
        const Point2d p = ni.apply(cp.image);
        const Point2d q = nw.apply(cp.world);

        const std::array<double, 9> r1{-p.x, -p.y, -1.0, 0.0, 0.0, 0.0,
                                       q.x * p.x, q.x * p.y, q.x};
        const std::array<double, 9> r2{0.0, 0.0, 0.0, -p.x, -p.y, -1.0,
                                       q.y * p.x, q.y * p.y, q.y};
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                ata[static_cast<std::size_t>(i) * 9 + j] +=
                    r1[static_cast<std::size_t>(i)] * r1[static_cast<std::size_t>(j)] +
                    r2[static_cast<std::size_t>(i)] * r2[static_cast<std::size_t>(j)];
            }
        }
    }

    const std::array<double, 9> h = detail::smallest_eigenvector_9(ata);

    // Undo the normalisation: H = Tw^-1 * Hn * Ti
    const double si = ni.scale;
    const double sw = nw.scale;
    const std::array<double, 9> ti{si, 0, -si * ni.cx, 0, si, -si * ni.cy, 0, 0, 1};
    const std::array<double, 9> tw_inv{1.0 / sw, 0, nw.cx, 0, 1.0 / sw, nw.cy, 0, 0, 1};

    const auto multiply = [](const std::array<double, 9>& a, const std::array<double, 9>& b) {
        std::array<double, 9> r{};
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                double sum = 0.0;
                for (int k = 0; k < 3; ++k) {
                    sum += a[static_cast<std::size_t>(i) * 3 + k] *
                           b[static_cast<std::size_t>(k) * 3 + j];
                }
                r[static_cast<std::size_t>(i) * 3 + j] = sum;
            }
        }
        return r;
    };

    std::array<double, 9> result = multiply(tw_inv, multiply(h, ti));
    if (std::abs(result[8]) > 1e-15) {
        for (double& value : result) value /= result[8];
    }

    out.m = result;
    return out.valid();
}

/// Root-mean-square reprojection error, in world metres.
inline double reprojection_error_m(const Homography& h, const std::vector<ControlPoint>& points) {
    if (points.empty()) return 0.0;
    double sum_sq = 0.0;
    for (const auto& cp : points) {
        const Point2d projected = h.apply(cp.image);
        const double dx = projected.x - cp.world.x;
        const double dy = projected.y - cp.world.y;
        sum_sq += dx * dx + dy * dy;
    }
    return std::sqrt(sum_sq / static_cast<double>(points.size()));
}

inline double max_reprojection_error_m(const Homography& h,
                                       const std::vector<ControlPoint>& points) {
    double worst = 0.0;
    for (const auto& cp : points) {
        const Point2d projected = h.apply(cp.image);
        worst = std::max(worst, std::hypot(projected.x - cp.world.x, projected.y - cp.world.y));
    }
    return worst;
}

struct CalibrationResult {
    Homography homography;
    double rms_error_m{0.0};
    double max_error_m{0.0};
    std::vector<std::size_t> inliers;
    bool ok{false};
    std::string_view reason;
};

/// Fit with RANSAC, then refit on the inliers.
///
/// Control points are clicked by a human on a video still, and a human will occasionally
/// click the wrong drain cover. One bad correspondence drags a least-squares fit across
/// the whole image, so the outlier has to be rejected rather than averaged.
inline CalibrationResult calibrate(const std::vector<ControlPoint>& points,
                                   double inlier_threshold_m = 0.35,
                                   int iterations = 400,
                                   std::uint32_t seed = 12345) {
    CalibrationResult result;
    if (points.size() < 4) {
        result.reason = "at least four control points are required";
        return result;
    }

    if (points.size() == 4) {
        // With the minimum set there is nothing to vote against; fit and report honestly.
        if (!solve_homography(points, result.homography)) {
            result.reason = "degenerate control points";
            return result;
        }
        result.rms_error_m = reprojection_error_m(result.homography, points);
        result.max_error_m = max_reprojection_error_m(result.homography, points);
        result.inliers = {0, 1, 2, 3};
        result.ok = true;
        return result;
    }

    std::mt19937 rng(seed);
    std::uniform_int_distribution<std::size_t> pick(0, points.size() - 1);

    std::vector<std::size_t> best_inliers;
    Homography best;

    for (int iteration = 0; iteration < iterations; ++iteration) {
        std::array<std::size_t, 4> sample{};
        for (int i = 0; i < 4; ++i) {
            bool unique = false;
            while (!unique) {
                sample[static_cast<std::size_t>(i)] = pick(rng);
                unique = true;
                for (int j = 0; j < i; ++j) {
                    if (sample[static_cast<std::size_t>(j)] == sample[static_cast<std::size_t>(i)]) {
                        unique = false;
                        break;
                    }
                }
            }
        }

        std::vector<ControlPoint> subset;
        subset.reserve(4);
        for (std::size_t index : sample) subset.push_back(points[index]);

        Homography candidate;
        if (!solve_homography(subset, candidate)) continue;

        std::vector<std::size_t> inliers;
        for (std::size_t i = 0; i < points.size(); ++i) {
            const Point2d projected = candidate.apply(points[i].image);
            const double error =
                std::hypot(projected.x - points[i].world.x, projected.y - points[i].world.y);
            if (error <= inlier_threshold_m) inliers.push_back(i);
        }
        if (inliers.size() > best_inliers.size()) {
            best_inliers = std::move(inliers);
            best = candidate;
        }
    }

    if (best_inliers.size() < 4) {
        result.reason = "no consistent set of four control points was found";
        return result;
    }

    std::vector<ControlPoint> inlier_points;
    inlier_points.reserve(best_inliers.size());
    for (std::size_t index : best_inliers) inlier_points.push_back(points[index]);

    Homography refined;
    if (solve_homography(inlier_points, refined)) {
        best = refined;
    }

    result.homography = best;
    result.inliers = best_inliers;
    result.rms_error_m = reprojection_error_m(best, inlier_points);
    result.max_error_m = max_reprojection_error_m(best, inlier_points);
    result.ok = true;
    return result;
}

/// Tracks whether the camera still sees what it saw when it was calibrated.
///
/// A camera that has been knocked, re-aimed or blown by wind still produces perfectly
/// sharp frames of entirely the wrong place, and every calibration built for it is now
/// wrong. Reprojecting the original control points and measuring how far they have moved
/// is the cheapest way to notice.
class PoseValidator {
  public:
    PoseValidator() = default;
    explicit PoseValidator(std::vector<Point2d> reference) : reference_(std::move(reference)) {}

    /// Mean displacement, in pixels, between the reference points and the current ones.
    /// Returns a negative value when there is nothing to compare against.
    [[nodiscard]] double drift(const std::vector<Point2d>& observed) const {
        if (reference_.empty() || observed.size() != reference_.size()) return -1.0;
        double sum = 0.0;
        for (std::size_t i = 0; i < observed.size(); ++i) {
            sum += std::hypot(observed[i].x - reference_[i].x, observed[i].y - reference_[i].y);
        }
        return sum / static_cast<double>(observed.size());
    }

    [[nodiscard]] bool has_reference() const { return !reference_.empty(); }
    void set_reference(std::vector<Point2d> reference) { reference_ = std::move(reference); }

  private:
    std::vector<Point2d> reference_;
};

}  // namespace parkfit::vision
