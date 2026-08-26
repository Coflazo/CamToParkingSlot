"""A generative model of Dutch urban parking demand.

This exists to bootstrap the learned occupancy model. The system has been ingesting live
data for one day, which is not enough history to fit anything: a model of "how full is
this street at 18:00 on a Friday" needs Fridays, plural.

So this module *invents* history, from a demand model built out of how Dutch city
parking actually behaves rather than out of noise. Two archetypes drive it:

**Residential.** Occupancy peaks overnight, when residents are home and their cars are
parked, and troughs mid-morning after the commute out. Amsterdam's permit zones look
like this: a canal street at 03:00 has no free space at any price.

**Destination.** Occupancy peaks between the lunch hour and late evening and collapses
overnight. Museum quarter, Leidseplein, the station garages.

Every real street is a blend. The blend weight moves with distance from the centre:
inner streets carry heavy destination demand *on top of* an already-high residential
baseline, which is exactly why they are the hardest place to park.

**On circularity.** The learned model must not simply rediscover the function written
here, or the evaluation would be measuring a tautology. Two things prevent that:

* The features the model sees (:mod:`parkfit.prediction.features`) are derived from what
  the *database* knows, coordinates, capacity, tariff, bay geometry. They do not
  include the archetype weight, the amplitudes, or the phase offsets used below.
* The model trains on sampled binary observations, not on the underlying rate. It has to
  recover a probability from Bernoulli draws, which is a real estimation problem.

What that earns is a defensible claim: the model recovers latent demand structure it
cannot see, better than a flat prior does. It is *not* a claim about real Amsterdam
occupancy. That needs real history, and is marked as such in TODO.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

#: Dam square. Distance from here is the strongest single predictor of parking pressure
#: in the Amsterdam data, which is unsurprising and still worth encoding explicitly.
CITY_CENTRE_LAT = 52.3730
CITY_CENTRE_LON = 4.8926


@dataclass(frozen=True)
class DemandProfile:
    """Latent parameters for one parking target. Never exposed to the model."""

    residential_weight: float
    #: Occupancy this target settles at with no time-of-day effect at all.
    baseline: float
    #: How hard the diurnal cycle swings it.
    amplitude: float
    #: Per-target idiosyncrasy: a courtyard nobody knows about, a street beside a school.
    quirk: float
    #: Turnover: how fast a freed space is retaken, in events per minute at full demand.
    churn_per_min: float


def _residential_curve(hour: float) -> float:
    """Occupancy shape for a street whose cars belong to the people living on it.

    Peaks at 03:00, troughs at 11:00. Modelled as a single cosine because the underlying
    driver, people leaving for work and coming back, genuinely is one daily cycle.
    """
    return math.cos(2.0 * math.pi * (hour - 3.0) / 24.0)


def _destination_curve(hour: float) -> float:
    """Occupancy shape for a street whose cars belong to visitors.

    Two humps: a daytime one centred on 14:00 for shops and offices, and an evening one
    centred on 20:00 for restaurants and theatres. The evening hump is narrower and
    slightly taller, which is what makes 20:00 on a Saturday the worst time to arrive.
    """
    day = math.exp(-(((hour - 14.0) / 4.2) ** 2))
    evening = math.exp(-(((hour - 20.0) / 2.6) ** 2))
    return 2.0 * (0.62 * day + 0.72 * evening) - 1.0


def _weekday_offset(weekday: int, hour: float) -> float:
    """How much busier or quieter this weekday is, in log-odds.

    Saturday afternoon and evening are the peak of the week in a Dutch city centre.
    Sunday morning is the trough. Monday to Thursday are flat and unremarkable, which is
    why they are not special-cased.
    """
    if weekday == 5:  # Saturday
        return 0.42 if 11.0 <= hour <= 23.0 else -0.10
    if weekday == 6:  # Sunday
        return -0.55 if hour < 12.0 else 0.08
    if weekday == 4 and hour >= 17.0:  # Friday evening
        return 0.30
    return 0.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(min(1.0, h)))


def profile_for(
    lat: float,
    lon: float,
    *,
    is_facility: bool,
    metered: bool,
    capacity: int | None,
    seed_value: int,
) -> DemandProfile:
    """Derive the latent demand parameters for one target.

    Deterministic in ``seed_value`` so that regenerating history reproduces it exactly.
    That matters: an evaluation you cannot reproduce is an anecdote.
    """
    km = haversine_km(lat, lon, CITY_CENTRE_LAT, CITY_CENTRE_LON)

    # Destination pull decays with a ~2.4 km scale. At the Dam it dominates; in
    # Nieuw-West almost everything parked on the street lives there.
    destination_pull = math.exp(-km / 2.4)
    residential_weight = 1.0 - 0.72 * destination_pull

    # A deterministic per-target wobble, drawn from the id rather than from a PRNG so it
    # survives a regeneration with a different global seed.
    quirk = ((seed_value * 2654435761) % 1000) / 1000.0 - 0.5

    if is_facility:
        # Garages hold many interchangeable spaces, so "has a free space" is far more
        # often true than for any one named kerb bay. Big garages are freer still.
        size_relief = 0.0 if not capacity else min(0.18, 0.02 * math.log1p(capacity))
        baseline = 0.55 + 0.18 * destination_pull - size_relief
        amplitude = 1.05 + 0.45 * destination_pull
        residential_weight = min(residential_weight, 0.35)
        churn = 0.05 + 0.10 * destination_pull
    else:
        # A metered bay is contested by definition; the meter exists because demand
        # exceeds supply. An unmetered bay in the same street sits noticeably freer.
        baseline = (0.68 if metered else 0.56) + 0.12 * destination_pull
        amplitude = (0.85 if metered else 0.75) + 0.45 * destination_pull
        churn = 0.03 + 0.16 * destination_pull

    return DemandProfile(
        residential_weight=max(0.0, min(1.0, residential_weight)),
        baseline=max(0.05, min(0.95, baseline + 0.06 * quirk)),
        # Amplitude is in log-odds units, matching how occupancy_rate composes effects.
        amplitude=max(0.10, amplitude + 0.20 * quirk),
        quirk=quirk,
        churn_per_min=max(0.005, churn),
    )


def _logit(p: float) -> float:
    p = min(0.999, max(0.001, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def occupancy_rate(profile: DemandProfile, weekday: int, minute_of_day: int) -> float:
    """Latent probability that this target is occupied at this moment.

    This is the ground truth the learned model is trying to recover. It is never stored
    and never becomes a feature; only Bernoulli draws from it are.

    Effects compose additively **in log-odds**, not multiplicatively on the probability.
    That is not a stylistic choice. A centre bay whose baseline is 0.90 and whose evening
    peak multiplies by 1.10 lands at 0.99, and so does its afternoon, and so does the
    Saturday version of both, everything clips against 1.0 and the diurnal signal the
    model is supposed to learn disappears into a flat line. In log-odds the same effects
    stay ordered and bounded without a clamp doing the work.
    """
    hour = minute_of_day / 60.0
    shape = profile.residential_weight * _residential_curve(hour) + (
        1.0 - profile.residential_weight
    ) * _destination_curve(hour)
    x = _logit(profile.baseline) + profile.amplitude * shape + _weekday_offset(weekday, hour)
    return max(0.01, min(0.99, _sigmoid(x)))


@lru_cache(maxsize=1)
def _shape_and_offsets() -> tuple[np.ndarray, np.ndarray]:
    """The parts of the curve that do not depend on the target: shapes and weekday offsets.

    Cached once for the process. Neither term involves a profile, so recomputing them per
    target, 10,080 evaluations each time, would dominate the simulation.
    """
    hour = np.arange(1440, dtype=np.float64) / 60.0
    residential = np.cos(2.0 * np.pi * (hour - 3.0) / 24.0)
    day = np.exp(-(((hour - 14.0) / 4.2) ** 2))
    evening = np.exp(-(((hour - 20.0) / 2.6) ** 2))
    destination = 2.0 * (0.62 * day + 0.72 * evening) - 1.0

    offsets = np.array(
        [[_weekday_offset(weekday, float(h)) for h in hour] for weekday in range(7)],
        dtype=np.float64,
    )
    return np.stack([residential, destination]), offsets


def occupancy_table(profile: DemandProfile) -> np.ndarray:
    """Occupancy for every ``(weekday, minute of day)``, as a 7x1440 array.

    A numpy restatement of :func:`occupancy_rate`. Simulating one target for three weeks
    at one-minute resolution needs 30,240 evaluations, and doing that through the scalar
    path for a few hundred targets costs more than the entire Markov walk it feeds.

    The duplication is deliberate but not unsupervised: ``test_prediction`` asserts the two
    agree to 1e-12 over the whole grid, the same way the RD contract test pins the C++
    polynomial to pyproj.
    """
    curves, offsets = _shape_and_offsets()
    shape = profile.residential_weight * curves[0] + (1.0 - profile.residential_weight) * curves[1]
    x = _logit(profile.baseline) + profile.amplitude * shape[None, :] + offsets
    return np.clip(1.0 / (1.0 + np.exp(-x)), 0.01, 0.99)


def lambda_table(profile: DemandProfile) -> np.ndarray:
    """Take rate for every ``(weekday, minute of day)``. Mirrors :func:`vacancy_lambda`."""
    occ = occupancy_table(profile)
    return profile.churn_per_min * (0.15 + occ**2)


def vacancy_lambda(profile: DemandProfile, weekday: int, minute_of_day: int) -> float:
    """Rate at which a currently-free space is taken, per minute.

    Tied to occupancy rather than independent of it, because the two are the same
    phenomenon seen from different sides: a street is full *because* every freed space is
    immediately retaken. A space on a 95%-occupied street survives a couple of minutes;
    the same space at 04:00 on a 40%-occupied street survives most of an hour.
    """
    occ = occupancy_rate(profile, weekday, minute_of_day)
    # Squared, so the last few percent of occupancy hurt disproportionately. That is the
    # observed behaviour: the difference between 85% and 97% full is not 12% worse, it is
    # the difference between "circle once" and "give up".
    return profile.churn_per_min * (0.15 + occ**2)
