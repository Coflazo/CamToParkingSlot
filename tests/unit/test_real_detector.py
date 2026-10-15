"""The wall-clock learning-rate schedule of the real-frame detector.

A run with a time budget cannot know its step count up front, so the schedule takes the
fraction of the budget already spent instead. It has to keep OneCycleLR's shape, or a
budgeted run would not be comparable to the step-counted runs before it.
"""

from __future__ import annotations

import math

import pytest

from parkfit.ml.train.real_detector import one_cycle_lr


def test_starts_low_peaks_at_the_end_of_warmup_and_ends_near_zero():
    assert one_cycle_lr(0.0, 2e-3) == pytest.approx(2e-3 / 25)
    assert one_cycle_lr(0.3, 2e-3) == pytest.approx(2e-3)
    assert one_cycle_lr(1.0, 2e-3) == pytest.approx(2e-3 / 25 / 1e4)


def test_rises_through_warmup_and_falls_after_it():
    rising = [one_cycle_lr(f / 100, 1.0) for f in range(0, 31)]
    falling = [one_cycle_lr(f / 100, 1.0) for f in range(30, 101)]
    assert rising == sorted(rising)
    assert falling == sorted(falling, reverse=True)


def test_matches_torch_one_cycle_cosine_halfway_down():
    # OneCycleLR's cosine annealing at 50 % of the decay phase sits midway between the
    # peak and the floor.
    peak, floor = 1.0, 1.0 / 25 / 1e4
    fraction = 0.3 + 0.7 * 0.5
    assert one_cycle_lr(fraction, peak) == pytest.approx(
        floor + (peak - floor) * (1 + math.cos(math.pi * 0.5)) / 2
    )


def test_fractions_outside_the_budget_are_clamped():
    assert one_cycle_lr(-0.5, 1.0) == one_cycle_lr(0.0, 1.0)
    assert one_cycle_lr(1.7, 1.0) == one_cycle_lr(1.0, 1.0)
