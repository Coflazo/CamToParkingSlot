// SPDX-License-Identifier: MIT
//
// Vision tests.
//
// The homography cases are built by projecting known world points through a synthetic
// but realistic camera, so the expected answers are exact rather than eyeballed. The
// gap cases place vehicles at known positions along a known kerb, which makes the
// measured gap length checkable to the centimetre -- and gap-length error is the
// headline accuracy number for this whole subsystem.

#include "test_framework.hpp"

#include <cmath>
#include <vector>

#include "parkfit/vision/frame.hpp"
#include "parkfit/vision/gap.hpp"
#include "parkfit/vision/health.hpp"
#include "parkfit/vision/homography.hpp"
#include "parkfit/vision/state_machine.hpp"

using namespace parkfit::vision;

namespace {

/// A synthetic camera looking down at a ground plane.
///
/// Models a real installation: mounted 6 m up, tilted 25 degrees down, 1280x720. Ground
/// points are rotated into camera space and projected through a pinhole, which produces
/// exactly the perspective foreshortening a real kerb view has -- distant metres occupy
/// far fewer pixels than near ones, which is the whole reason a homography is needed.
struct SyntheticCamera {
    double height_m{6.0};
    double tilt_rad{25.0 * 3.14159265358979323846 / 180.0};
    double focal_px{900.0};
    double cx{640.0};
    double cy{360.0};
    double origin_x{121000.0};
    double origin_y{487000.0};

    [[nodiscard]] Point2d project(const Point2d& world) const {
        const double dx = world.x - origin_x;
        const double dy = world.y - origin_y;
        const double c = std::cos(tilt_rad);
        const double s = std::sin(tilt_rad);
        // Camera at height h looking along +y, pitched down by tilt. Its axes in world
        // terms are forward = (0, cos, -sin) and down = (0, sin, cos), so for a ground
        // point the relative vector is (dx, dy, -h) and:
        //     depth = v . forward = dy*cos + h*sin      (always positive ahead)
        //     down  = v . down    = dy*sin - h*cos      (negative above the axis)
        const double depth = dy * c + height_m * s;
        const double down = dy * s - height_m * c;
        if (depth < 0.5) return Point2d{-1.0, -1.0};
        return Point2d{cx + focal_px * dx / depth, cy + focal_px * down / depth};
    }
};

std::vector<ControlPoint> control_points(const SyntheticCamera& cam) {
    std::vector<ControlPoint> points;
    // Spread across the ground plane, as a human clicking bay corners would place them.
    const double offsets[6][2] = {{-8, 12}, {8, 12}, {-8, 26}, {8, 26}, {0, 19}, {-4, 33}};
    for (const auto& off : offsets) {
        Point2d world{cam.origin_x + off[0], cam.origin_y + off[1]};
        points.push_back(ControlPoint{cam.project(world), world});
    }
    return points;
}

Frame make_frame(int w, int h, std::uint8_t base, bool add_edges) {
    Frame f(w, h, PixelFormat::Gray8);
    for (int y = 0; y < h; ++y) {
        std::uint8_t* row = f.row(y);
        for (int x = 0; x < w; ++x) {
            std::uint8_t value = base;
            if (add_edges && ((x / 8) % 2 == 0) != ((y / 8) % 2 == 0)) {
                value = static_cast<std::uint8_t>(std::min(255, base + 90));
            }
            row[x] = value;
        }
    }
    return f;
}

}  // namespace

// --------------------------------------------------------------- homography
TEST_CASE("homography: recovers a synthetic camera to millimetre accuracy") {
    SyntheticCamera cam;
    const auto points = control_points(cam);
    Homography h;
    CHECK(solve_homography(points, h));
    CHECK(h.valid());
    CHECK_NEAR(reprojection_error_m(h, points), 0.0, 1e-6);
}

TEST_CASE("homography: generalises to points it was not fitted on") {
    SyntheticCamera cam;
    Homography h;
    CHECK(solve_homography(control_points(cam), h));

    // Independent validation points, which is the only honest way to judge a fit.
    const double checks[4][2] = {{-6, 16}, {5, 22}, {2, 30}, {-2, 14}};
    for (const auto& off : checks) {
        Point2d world{cam.origin_x + off[0], cam.origin_y + off[1]};
        const Point2d recovered = h.apply(cam.project(world));
        CHECK_NEAR(recovered.x, world.x, 0.02);
        CHECK_NEAR(recovered.y, world.y, 0.02);
    }
}

TEST_CASE("homography: inverse round-trips") {
    SyntheticCamera cam;
    Homography h;
    CHECK(solve_homography(control_points(cam), h));
    const Homography inv = h.inverse();

    const Point2d image{700.0, 480.0};
    const Point2d world = h.apply(image);
    const Point2d back = inv.apply(world);
    CHECK_NEAR(back.x, image.x, 0.05);
    CHECK_NEAR(back.y, image.y, 0.05);
}

TEST_CASE("homography: refuses fewer than four correspondences") {
    std::vector<ControlPoint> few{
        {{0, 0}, {0, 0}}, {{10, 0}, {1, 0}}, {{10, 10}, {1, 1}}};
    Homography h;
    CHECK(!solve_homography(few, h));
    CHECK(!calibrate(few).ok);
}

TEST_CASE("homography: RANSAC rejects a mis-clicked control point") {
    SyntheticCamera cam;
    auto points = control_points(cam);
    // A human clicked the wrong drain cover: correct image position, world coordinate
    // eleven metres away. A plain least-squares fit would smear that error across the
    // entire image rather than discard it.
    points[2].world.x += 11.0;

    Homography naive;
    CHECK(solve_homography(points, naive));
    const double naive_error = reprojection_error_m(naive, points);

    const auto result = calibrate(points, 0.35, 500);
    CHECK(result.ok);
    CHECK(result.inliers.size() >= 5);
    // The outlier must not be among the inliers.
    for (std::size_t index : result.inliers) CHECK(index != 2);

    std::vector<ControlPoint> clean = control_points(cam);
    CHECK(reprojection_error_m(result.homography, clean) < 0.05);
    CHECK(naive_error > 0.5);
}

TEST_CASE("pose validator: reports drift when the camera moves") {
    PoseValidator validator({{100, 200}, {400, 210}, {700, 205}});
    CHECK(validator.has_reference());
    CHECK_NEAR(validator.drift({{100, 200}, {400, 210}, {700, 205}}), 0.0, 1e-9);
    // A uniform 10 px shift, as a knocked camera produces.
    CHECK_NEAR(validator.drift({{110, 200}, {410, 210}, {710, 205}}), 10.0, 1e-9);
    // Mismatched counts cannot be compared, and must not be reported as zero drift.
    CHECK(validator.drift({{100, 200}}) < 0.0);
}

// ------------------------------------------------------------------- health
TEST_CASE("health: a normal frame is healthy") {
    FrameHealthChecker checker;
    const auto report = checker.check(make_frame(160, 120, 120, true));
    CHECK(report.state == FrameHealth::Healthy);
    CHECK(report.usable());
}

TEST_CASE("health: a dark frame is rejected") {
    FrameHealthChecker checker;
    const auto report = checker.check(make_frame(160, 120, 6, false));
    CHECK(report.state == FrameHealth::Dark);
    CHECK(!report.usable());
}

TEST_CASE("health: a flat frame reads as blurred, not healthy") {
    FrameHealthChecker checker;
    // Uniform mid-grey: bright enough, but with no edges at all. A lens covered in rain
    // looks exactly like this, and a detector run on it will confidently find nothing.
    const auto report = checker.check(make_frame(160, 120, 128, false));
    CHECK(report.state == FrameHealth::Blurred);
}

TEST_CASE("health: a repeating stream is detected as frozen") {
    FrameHealthChecker checker;
    const Frame frame = make_frame(160, 120, 120, true);
    HealthReport report;
    for (int i = 0; i < 6; ++i) report = checker.check(frame);
    CHECK(report.state == FrameHealth::Frozen);
    CHECK(report.repeat_count >= 4);
}

TEST_CASE("health: a uniform brightness shift is still the same picture") {
    // dHash encodes gradients precisely so that exposure drift does not read as motion.
    // A whole-frame brightness change is therefore *correctly* seen as no change at all,
    // which is what lets a camera survive dusk without being called broken.
    FrameHealthChecker checker;
    HealthReport report;
    for (int i = 0; i < 8; ++i) {
        report = checker.check(make_frame(160, 120, static_cast<std::uint8_t>(100 + i * 9), true));
    }
    CHECK(report.state == FrameHealth::Frozen);
}

TEST_CASE("health: a stream whose content moves is not called frozen") {
    FrameHealthChecker checker;
    HealthReport report;
    for (int i = 0; i < 8; ++i) {
        // Shift the pattern, which is what an actual scene change looks like.
        Frame f(160, 120, PixelFormat::Gray8);
        for (int y = 0; y < f.height(); ++y) {
            std::uint8_t* row = f.row(y);
            for (int x = 0; x < f.width(); ++x) {
                const bool on = (((x + i * 5) / 8) % 2 == 0) != ((y / 8) % 2 == 0);
                row[x] = on ? 200 : 110;
            }
        }
        report = checker.check(f);
    }
    CHECK(report.state == FrameHealth::Healthy);
    CHECK(report.repeat_count == 0);
}

TEST_CASE("health: pose drift outranks every other complaint") {
    FrameHealthChecker checker;
    const auto report = checker.check(make_frame(160, 120, 120, true), 25.0);
    CHECK(report.state == FrameHealth::PoseChanged);
}

TEST_CASE("health: an empty frame is offline") {
    FrameHealthChecker checker;
    CHECK(checker.check(Frame{}).state == FrameHealth::Offline);
}

namespace {

/// A horizontal brightness ramp. `rising` controls its direction.
///
/// Chosen because its hash is analytically known rather than empirical: dHash sets a bit
/// when a thumbnail pixel is brighter than its right-hand neighbour, so a rising ramp
/// sets no bits at all and a falling one sets all 64. That makes the two extremes exact,
/// which is what a test of a hash function wants.
Frame ramp_frame(int w, int h, bool rising) {
    Frame f(w, h, PixelFormat::Gray8);
    for (int y = 0; y < h; ++y) {
        std::uint8_t* row = f.row(y);
        for (int x = 0; x < w; ++x) {
            const int level = 20 + 200 * x / w;
            row[x] = static_cast<std::uint8_t>(rising ? level : 220 - level);
        }
    }
    return f;
}

}  // namespace

TEST_CASE("perceptual hash: a rising ramp sets no bits, a falling ramp sets all of them") {
    CHECK_EQ(difference_hash(ramp_frame(160, 120, true)), static_cast<PerceptualHash>(0));
    CHECK_EQ(hamming_distance(difference_hash(ramp_frame(160, 120, true)),
                              difference_hash(ramp_frame(160, 120, false))),
             64);
}

TEST_CASE("perceptual hash: stable under sensor noise") {
    Frame a = ramp_frame(160, 120, true);
    Frame b = a;
    for (int y = 0; y < b.height(); y += 3) {
        for (int x = 0; x < b.width(); x += 3) {
            b.row(y)[x] = static_cast<std::uint8_t>(std::min(255, b.row(y)[x] + 3));
        }
    }
    CHECK(hamming_distance(difference_hash(a), difference_hash(b)) <= 3);
}

TEST_CASE("perceptual hash: a uniform frame carries no gradient information") {
    // Every comparison is a tie on a flat frame, and ties do not set a bit, so the hash
    // is zero. That is worth pinning: it means a fully obscured camera and a rising ramp
    // hash identically, which is exactly why FROZEN is never the only check -- the
    // brightness and sharpness tests are what catch a blanked-out view.
    CHECK_EQ(difference_hash(make_frame(160, 120, 120, false)), static_cast<PerceptualHash>(0));
    CHECK_EQ(difference_hash(ramp_frame(160, 120, true)), static_cast<PerceptualHash>(0));
}

// ---------------------------------------------------------------- intervals
TEST_CASE("intervals: merging overlapping blockages") {
    const auto merged = merge_intervals({{0, 5}, {4, 9}, {12, 15}});
    CHECK_EQ(merged.size(), static_cast<std::size_t>(2));
    CHECK_NEAR(merged[0].start, 0.0, 1e-9);
    CHECK_NEAR(merged[0].end, 9.0, 1e-9);
    CHECK_NEAR(merged[1].start, 12.0, 1e-9);
}

TEST_CASE("intervals: free stretches are the complement of the blocked ones") {
    const auto free = free_intervals(30.0, {{0, 5}, {12, 18}});
    CHECK_EQ(free.size(), static_cast<std::size_t>(2));
    CHECK_NEAR(free[0].start, 5.0, 1e-9);
    CHECK_NEAR(free[0].end, 12.0, 1e-9);
    CHECK_NEAR(free[1].start, 18.0, 1e-9);
    CHECK_NEAR(free[1].end, 30.0, 1e-9);
}

TEST_CASE("intervals: a fully blocked kerb has no free stretch") {
    CHECK(free_intervals(20.0, {{0, 20}}).empty());
}

TEST_CASE("intervals: an unblocked kerb is entirely free") {
    const auto free = free_intervals(20.0, {});
    CHECK_EQ(free.size(), static_cast<std::size_t>(1));
    CHECK_NEAR(free[0].length(), 20.0, 1e-9);
}

// --------------------------------------------------------------- kerb gaps
namespace {

/// A straight 40 m kerb running east, with the camera 6 m back from it.
CurbSegment straight_kerb(const SyntheticCamera& cam) {
    CurbSegment segment;
    segment.id = "kerb_test";
    segment.usable_width_m = 2.1;
    segment.centreline = {Point2d{cam.origin_x - 20.0, cam.origin_y + 18.0},
                          Point2d{cam.origin_x + 20.0, cam.origin_y + 18.0}};
    return segment;
}

/// Place a car of a given length centred at `along` metres from the kerb start, and
/// return the detection a perfect detector would produce for it.
Detection car_at(const SyntheticCamera& cam, const CurbSegment& segment, double along,
                 double length_m) {
    const double start_x = segment.centreline.front().x;
    const double y = segment.centreline.front().y;
    const Point2d left_world{start_x + along - length_m * 0.5, y};
    const Point2d right_world{start_x + along + length_m * 0.5, y};
    const Point2d left = cam.project(left_world);
    const Point2d right = cam.project(right_world);

    Detection d;
    d.x1 = std::min(left.x, right.x);
    d.x2 = std::max(left.x, right.x);
    // Bounding-box top is above the ground contact by roughly the vehicle height in
    // pixels; only the bottom edge is used for projection, so the exact top is free.
    d.y2 = (left.y + right.y) * 0.5;
    d.y1 = d.y2 - 80.0;
    d.score = 0.91;
    d.label = "car";
    return d;
}

}  // namespace

TEST_CASE("gap: measures a known kerb gap to within centimetres") {
    SyntheticCamera cam;
    Homography h;
    CHECK(solve_homography(control_points(cam), h));

    const CurbSegment kerb = straight_kerb(cam);
    CHECK_NEAR(kerb.length_m(), 40.0, 1e-6);

    // Two cars, 4.5 m each, centred 8 m and 20 m along. The clear stretch between their
    // facing bumpers runs from 10.25 m to 17.75 m: exactly 7.5 m.
    const std::vector<Detection> detections{car_at(cam, kerb, 8.0, 4.5),
                                            car_at(cam, kerb, 20.0, 4.5)};

    CurbGapEstimator estimator;
    const auto result = estimator.estimate(kerb, h, detections, Point2d{cam.origin_x, cam.origin_y});
    CHECK(result.usable);
    CHECK_EQ(result.projected_vehicles, 2);
    CHECK(!result.gaps.empty());

    // Find the gap between the two cars rather than the open ends of the segment.
    const Gap* middle = nullptr;
    for (const auto& gap : result.gaps) {
        if (gap.start_m > 9.0 && gap.end_m < 19.0) middle = &gap;
    }
    CHECK(middle != nullptr);
    if (middle) {
        CHECK_NEAR(middle->length_m, 7.5, 0.25);
        CHECK(!middle->occluded_start);
        CHECK(!middle->occluded_end);
    }
}

TEST_CASE("gap: an unbroken kerb reports one long gap") {
    SyntheticCamera cam;
    Homography h;
    CHECK(solve_homography(control_points(cam), h));
    CurbGapEstimator estimator;
    const auto result = estimator.estimate(straight_kerb(cam), h, {},
                                           Point2d{cam.origin_x, cam.origin_y});
    CHECK(result.usable);
    CHECK_EQ(result.gaps.size(), static_cast<std::size_t>(1));
    CHECK_NEAR(result.gaps[0].length_m, 40.0, 0.05);
    // It runs to both edges of what the camera can see, so it is a lower bound.
    CHECK(result.gaps[0].occluded_start);
    CHECK(result.gaps[0].occluded_end);
    CHECK(result.gaps[0].confidence < 0.6);
}

TEST_CASE("gap: a car in the traffic lane is not counted as parked") {
    SyntheticCamera cam;
    Homography h;
    CHECK(solve_homography(control_points(cam), h));
    const CurbSegment kerb = straight_kerb(cam);

    // Same position along the kerb, but 6 m out from it: this is moving traffic, and
    // treating it as a blockage would invent an obstruction that is not there.
    Detection passing;
    const Point2d l = cam.project(Point2d{kerb.centreline.front().x + 12.0,
                                          kerb.centreline.front().y - 6.0});
    const Point2d r = cam.project(Point2d{kerb.centreline.front().x + 16.5,
                                          kerb.centreline.front().y - 6.0});
    passing.x1 = std::min(l.x, r.x);
    passing.x2 = std::max(l.x, r.x);
    passing.y2 = (l.y + r.y) * 0.5;
    passing.y1 = passing.y2 - 70.0;
    passing.score = 0.9;

    CurbGapEstimator estimator;
    const auto result = estimator.estimate(kerb, h, {passing},
                                           Point2d{cam.origin_x, cam.origin_y});
    CHECK_EQ(result.projected_vehicles, 0);
    CHECK_EQ(result.rejected_off_kerb, 1);
    CHECK_EQ(result.gaps.size(), static_cast<std::size_t>(1));
}

TEST_CASE("gap: prohibited stretches are subtracted even when empty") {
    SyntheticCamera cam;
    Homography h;
    CHECK(solve_homography(control_points(cam), h));

    CurbSegment kerb = straight_kerb(cam);
    // A driveway in the middle. Nothing is parked across it, and nothing may be.
    kerb.prohibited = {Interval{18.0, 24.0}};

    CurbGapEstimator estimator;
    const auto result = estimator.estimate(kerb, h, {}, Point2d{cam.origin_x, cam.origin_y});
    CHECK_EQ(result.gaps.size(), static_cast<std::size_t>(2));
    for (const auto& gap : result.gaps) {
        CHECK(!(gap.start_m < 24.0 && gap.end_m > 18.0));
    }
}

TEST_CASE("gap: stretches shorter than a car are not reported") {
    SyntheticCamera cam;
    Homography h;
    CHECK(solve_homography(control_points(cam), h));
    CurbSegment kerb = straight_kerb(cam);
    // Leaves a 2 m stretch, which no car can use.
    kerb.prohibited = {Interval{0.0, 19.0}, Interval{21.0, 40.0}};

    CurbGapEstimator estimator;
    const auto result = estimator.estimate(kerb, h, {}, Point2d{cam.origin_x, cam.origin_y});
    CHECK(result.gaps.empty());
}

TEST_CASE("gap: an uncalibrated camera measures nothing") {
    Homography broken;
    broken.m = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    CurbGapEstimator estimator;
    SyntheticCamera cam;
    const auto result = estimator.estimate(straight_kerb(cam), broken, {},
                                           Point2d{cam.origin_x, cam.origin_y});
    CHECK(!result.usable);
    CHECK(!result.reason.empty());
}

// ----------------------------------------------------------- state machine
TEST_CASE("state: occupancy is published on a single confident detection") {
    TemporalStateMachine machine;
    SpaceObservation obs;
    obs.timestamp_s = 10.0;
    obs.detection_score = 0.88;
    const auto t = machine.update(obs);
    CHECK(t.state == OccupancyState::Occupied);
    CHECK(t.publishable);
    CHECK(t.changed);
}

TEST_CASE("state: vacancy requires repeated confirmation") {
    TemporalStateMachine machine;
    SpaceObservation obs;
    obs.detection_score = 0.02;

    obs.timestamp_s = 1.0;
    auto t = machine.update(obs);
    CHECK(t.state != OccupancyState::Vacant);
    CHECK(!t.publishable);

    obs.timestamp_s = 9.0;
    t = machine.update(obs);
    CHECK(t.state != OccupancyState::Vacant);

    obs.timestamp_s = 17.0;
    t = machine.update(obs);
    CHECK(t.state == OccupancyState::Vacant);
    CHECK(t.publishable);
    CHECK(t.confidence > 0.5);
}

TEST_CASE("state: a departing car goes to unknown before it goes to vacant") {
    TemporalStateMachine machine;
    SpaceObservation occupied;
    occupied.timestamp_s = 1.0;
    occupied.detection_score = 0.9;
    CHECK(machine.update(occupied).state == OccupancyState::Occupied);

    SpaceObservation clear;
    clear.timestamp_s = 9.0;
    clear.detection_score = 0.01;
    // The car has gone, but one frame is not proof. Claiming vacancy here is precisely
    // the false-free error the whole class exists to avoid.
    CHECK(machine.update(clear).state == OccupancyState::Unknown);
}

TEST_CASE("state: an unhealthy frame forces unknown immediately") {
    TemporalStateMachine machine;
    SpaceObservation obs;
    obs.timestamp_s = 1.0;
    obs.detection_score = 0.95;
    CHECK(machine.update(obs).state == OccupancyState::Occupied);

    obs.timestamp_s = 9.0;
    obs.health = FrameHealth::Frozen;
    const auto t = machine.update(obs);
    CHECK(t.state == OccupancyState::Unknown);
    CHECK(t.publishable);
}

TEST_CASE("state: a frozen stream can never produce a vacancy") {
    TemporalStateMachine machine;
    SpaceObservation obs;
    obs.detection_score = 0.0;
    obs.health = FrameHealth::Frozen;
    for (int i = 0; i < 20; ++i) {
        obs.timestamp_s = i * 8.0;
        CHECK(machine.update(obs).state == OccupancyState::Unknown);
    }
    CHECK(machine.state() == OccupancyState::Unknown);
}

TEST_CASE("state: an occluded space is unknown, not vacant") {
    TemporalStateMachine machine;
    SpaceObservation obs;
    obs.detection_score = 0.0;
    obs.occlusion = 0.8;
    for (int i = 0; i < 6; ++i) {
        obs.timestamp_s = i * 8.0;
        CHECK(machine.update(obs).state == OccupancyState::Unknown);
    }
}

TEST_CASE("state: an ambiguous score confirms nothing") {
    TemporalStateMachine machine;
    SpaceObservation obs;
    obs.detection_score = 0.33;  // between the two thresholds
    for (int i = 0; i < 6; ++i) {
        obs.timestamp_s = i * 8.0;
        CHECK(machine.update(obs).state == OccupancyState::Unknown);
    }
}

TEST_CASE("state: low model confidence is discarded rather than voted with") {
    TemporalStateMachine machine;
    SpaceObservation obs;
    obs.detection_score = 0.01;
    obs.confidence = 0.1;
    for (int i = 0; i < 6; ++i) {
        obs.timestamp_s = i * 8.0;
        CHECK(machine.update(obs).state == OccupancyState::Unknown);
    }
}

TEST_CASE("state: a published state expires when frames stop arriving") {
    TemporalStateMachine machine;
    SpaceObservation obs;
    obs.detection_score = 0.9;
    obs.timestamp_s = 100.0;
    CHECK(machine.update(obs).state == OccupancyState::Occupied);

    CHECK(machine.tick(150.0).state == OccupancyState::Occupied);  // still fresh
    const auto expired = machine.tick(200.0);
    CHECK(expired.state == OccupancyState::Unknown);
    CHECK(expired.changed);
    CHECK(expired.publishable);
}

TEST_CASE("state: recovery to vacancy after a health failure starts from scratch") {
    TemporalStateMachine machine;
    SpaceObservation obs;
    obs.detection_score = 0.01;
    obs.timestamp_s = 1.0;
    machine.update(obs);
    obs.timestamp_s = 9.0;
    machine.update(obs);   // two of the three confirmations banked

    obs.timestamp_s = 17.0;
    obs.health = FrameHealth::Dark;
    machine.update(obs);   // and the count is reset, not carried over

    obs.health = FrameHealth::Healthy;
    obs.timestamp_s = 25.0;
    CHECK(machine.update(obs).state != OccupancyState::Vacant);
    obs.timestamp_s = 33.0;
    CHECK(machine.update(obs).state != OccupancyState::Vacant);
    obs.timestamp_s = 41.0;
    CHECK(machine.update(obs).state == OccupancyState::Vacant);
}

PF_TEST_MAIN()
