// SPDX-License-Identifier: MIT
//
// Temporal occupancy state.
//
// This class exists to make one error rare at the cost of making another common, and
// the asymmetry is the entire point.
//
// Telling a driver a space is occupied when it is free costs them one option out of ten.
// Telling them a space is free when it is occupied costs them the trip: they drive
// across a city, find a car in it, and stop believing the app. The two errors are not
// remotely equal, so the transitions are not symmetric either:
//
//   * OCCUPIED is published immediately on a single confident detection.
//   * VACANT requires several consecutive clean observations.
//   * UNKNOWN is published the moment anything is wrong, and is never held back.
//
// UNKNOWN is a first-class answer, not a failure. A camera that has frozen, gone dark
// or been knocked out of alignment should say so, and the ranking will fall back to a
// predictive estimate. Silently continuing to publish the last good state is how a
// vision system tells a confident lie.

#pragma once

#include <algorithm>
#include <cstdint>
#include <string>

#include "parkfit/vision/health.hpp"

namespace parkfit::vision {

enum class OccupancyState : std::uint8_t {
    Unknown,
    Occupied,
    Vacant,
};

inline const char* to_string(OccupancyState s) {
    switch (s) {
        case OccupancyState::Unknown: return "UNKNOWN";
        case OccupancyState::Occupied: return "OCCUPIED";
        case OccupancyState::Vacant: return "VACANT";
    }
    return "UNKNOWN";
}

struct StateMachineConfig {
    /// Consecutive vacant observations before vacancy is published. Three at one frame
    /// per eight seconds is roughly twenty-four seconds of agreement -- long enough to
    /// ride out a single bad inference, short enough that a space freed at the kerb is
    /// still there when the driver is told about it.
    int vacant_confirmations{3};

    /// A single detection above this score is enough to publish OCCUPIED, because that
    /// direction is the safe one.
    double occupied_min_score{0.45};

    /// Vacancy needs a *clean* frame: not merely no detection, but no detection that
    /// was anywhere near being one. A borderline blob in the bay is evidence against
    /// vacancy even when it falls below the detection threshold.
    double vacant_max_score{0.22};

    /// Below this the classifier is not confident enough for its answer to count either
    /// way, and the observation is discarded rather than voted with.
    double min_usable_confidence{0.30};

    /// Once published, a state persists for this long without new evidence. After that
    /// the answer decays to UNKNOWN rather than growing stale in place.
    double state_ttl_s{75.0};
};

/// One observation of a single space.
struct SpaceObservation {
    double timestamp_s{0.0};
    /// Strongest detection score overlapping the space. Zero means nothing was seen.
    double detection_score{0.0};
    /// How much the classifier trusts its own reading of this frame.
    double confidence{1.0};
    FrameHealth health{FrameHealth::Healthy};
    /// Fraction of the space hidden by something in front of it. A space behind a van
    /// cannot be called vacant, however empty the visible part looks.
    double occlusion{0.0};
};

struct StateTransition {
    OccupancyState state{OccupancyState::Unknown};
    bool changed{false};
    bool publishable{false};
    double confidence{0.0};
    int consecutive_vacant{0};
    std::string reason;
};

/// Per-space temporal filter.
class TemporalStateMachine {
  public:
    explicit TemporalStateMachine(StateMachineConfig config = {}) : config_(config) {}

    StateTransition update(const SpaceObservation& observation) {
        StateTransition transition;
        const OccupancyState previous = state_;

        // Anything wrong with the frame ends the discussion. A detector run on a frozen
        // or dark image produces a confident answer about a picture that no longer
        // describes the street.
        if (observation.health != FrameHealth::Healthy) {
            reset_to_unknown();
            transition.state = state_;
            transition.changed = previous != state_;
            transition.publishable = true;
            transition.reason = std::string("frame not usable: ") + to_string(observation.health);
            last_update_s_ = observation.timestamp_s;
            return transition;
        }

        if (observation.confidence < config_.min_usable_confidence) {
            reset_to_unknown();
            transition.state = state_;
            transition.changed = previous != state_;
            transition.publishable = true;
            transition.reason = "model confidence too low to count";
            last_update_s_ = observation.timestamp_s;
            return transition;
        }

        // A heavily occluded space is unknown, not vacant. Whatever is behind the van
        // is exactly what we cannot see.
        if (observation.occlusion > 0.5) {
            reset_to_unknown();
            transition.state = state_;
            transition.changed = previous != state_;
            transition.publishable = true;
            transition.reason = "space is occluded";
            last_update_s_ = observation.timestamp_s;
            return transition;
        }

        if (observation.detection_score >= config_.occupied_min_score) {
            consecutive_vacant_ = 0;
            state_ = OccupancyState::Occupied;
            transition.state = state_;
            transition.changed = previous != state_;
            transition.publishable = true;
            transition.confidence = std::min(0.98, observation.detection_score);
            transition.reason = "vehicle detected";
            last_update_s_ = observation.timestamp_s;
            return transition;
        }

        if (observation.detection_score <= config_.vacant_max_score) {
            ++consecutive_vacant_;
            if (consecutive_vacant_ >= config_.vacant_confirmations) {
                state_ = OccupancyState::Vacant;
                transition.confidence =
                    std::min(0.95, 0.6 + 0.1 * (consecutive_vacant_ - config_.vacant_confirmations)
                                       + 0.25 * observation.confidence);
                transition.reason = "space clear across consecutive observations";
                transition.publishable = true;
            } else {
                // Not yet confirmed. Say nothing rather than publish a maybe.
                transition.reason = "clear, awaiting confirmation";
                transition.publishable = false;
                if (state_ == OccupancyState::Occupied) {
                    // A car has just left. Until vacancy is confirmed the honest answer
                    // is that we no longer know, not that the space is still taken.
                    state_ = OccupancyState::Unknown;
                }
            }
            transition.state = state_;
            transition.changed = previous != state_;
            transition.consecutive_vacant = consecutive_vacant_;
            last_update_s_ = observation.timestamp_s;
            return transition;
        }

        // Between the two thresholds: something is there, but not clearly enough to
        // call it a vehicle and far too much to call the space clear.
        consecutive_vacant_ = 0;
        state_ = OccupancyState::Unknown;
        transition.state = state_;
        transition.changed = previous != state_;
        transition.publishable = true;
        transition.reason = "ambiguous: neither clearly occupied nor clearly clear";
        last_update_s_ = observation.timestamp_s;
        return transition;
    }

    /// Decay to UNKNOWN when no fresh observation has arrived in time.
    ///
    /// Called on a timer, not on a frame, because the dangerous case is precisely the
    /// one where frames have stopped arriving and nothing is calling `update`.
    StateTransition tick(double now_s) {
        StateTransition transition;
        transition.state = state_;
        if (state_ == OccupancyState::Unknown) return transition;
        if (now_s - last_update_s_ <= config_.state_ttl_s) return transition;

        const OccupancyState previous = state_;
        reset_to_unknown();
        transition.state = state_;
        transition.changed = previous != state_;
        transition.publishable = true;
        transition.reason = "no fresh observation; state has expired";
        return transition;
    }

    [[nodiscard]] OccupancyState state() const { return state_; }
    [[nodiscard]] int consecutive_vacant() const { return consecutive_vacant_; }
    [[nodiscard]] double last_update_s() const { return last_update_s_; }

    void reset() {
        state_ = OccupancyState::Unknown;
        consecutive_vacant_ = 0;
        last_update_s_ = 0.0;
    }

  private:
    void reset_to_unknown() {
        state_ = OccupancyState::Unknown;
        consecutive_vacant_ = 0;
    }

    StateMachineConfig config_;
    OccupancyState state_{OccupancyState::Unknown};
    int consecutive_vacant_{0};
    double last_update_s_{0.0};
};

}  // namespace parkfit::vision
