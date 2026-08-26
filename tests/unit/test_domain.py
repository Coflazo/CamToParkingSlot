"""Domain logic: fit, restrictions, evidence resolution, pricing, deduplication.

These are the rules that decide what a driver is told. The negative cases matter most:
a search that occasionally misses a valid garage is annoying, but one that offers an
illegal bay costs a fine and one that offers a space too small costs the trip.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from parkfit.domain.dedupe import merge_duplicates
from parkfit.domain.evidence import ResolvedAvailability, describe_freshness, resolve_availability
from parkfit.domain.pricing import estimate_prices
from parkfit.domain.restrictions import evaluate_restrictions
from parkfit.domain.vehicle import ACCESSORY_HEIGHT_CM, confirm_dimensions, normalise_plate
from parkfit.storage.models import (
    AvailabilityObservation,
    EvidenceSource,
    OccupancyState,
    ParkingRestriction,
)


class TestPlateNormalisation:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("XT-994-N", "XT994N"), ("xt994n", "XT994N"), ("XT 994 N", "XT994N"), ("", "")],
    )
    def test_separators_are_stripped(self, raw, expected):
        assert normalise_plate(raw) == expected


class TestVehicleConfirmation:
    def test_height_confirmation_clears_the_flag(self, polo):
        polo.height_cm = 0.0
        polo.height_confirmed = False
        polo.unconfirmed_fields = ["height_cm", "width_with_mirrors_cm"]

        confirm_dimensions(polo, height_cm=145.1)
        assert polo.height_confirmed
        assert "height_cm" not in polo.unconfirmed_fields

    def test_roof_accessory_raises_the_effective_height(self, polo):
        confirm_dimensions(polo, height_cm=145.1, accessory="roof_box")
        assert polo.height_with_accessories_cm == pytest.approx(
            145.1 + ACCESSORY_HEIGHT_CM["roof_box"]
        )
        assert polo.has_roof_box

    def test_effective_width_falls_back_to_a_mirror_allowance(self, polo):
        polo.width_with_mirrors_cm = 0.0
        assert polo.effective_width_cm == pytest.approx(175.1 + 36.0)

    def test_ready_for_search_needs_length_and_some_width(self, polo):
        assert polo.ready_for_search
        polo.length_cm = 0.0
        assert not polo.ready_for_search


@pytest.mark.native
class TestVehicleFit:
    """Exercised through the native module, which is what the search path uses."""

    @staticmethod
    def _native():
        from parkfit.native import native

        if native is None:
            pytest.skip("native module not built")
        return native

    def test_a_normal_car_clears_a_two_metre_barrier(self, polo):
        n = self._native()
        limits = n.FacilityLimits()
        limits.max_height_cm = 200.0
        result = n.check_facility(polo.to_native(), limits, n.Margins())
        assert result.verdict_name == "FITS"

    def test_a_roof_rack_is_what_fails_the_barrier(self, tall_van):
        n = self._native()
        limits = n.FacilityLimits()
        limits.max_height_cm = 210.0
        assert n.check_facility(tall_van.to_native(), limits, n.Margins()).verdict_name == (
            "DOES_NOT_FIT"
        )

        tall_van.height_with_accessories_cm = 0.0  # rack removed
        assert n.check_facility(tall_van.to_native(), limits, n.Margins()).acceptable

    def test_an_unpublished_barrier_height_is_unverified_never_fits(self, tall_van):
        """Two thirds of RDW facilities publish no height. UNVERIFIED is the common case."""
        n = self._native()
        result = n.check_facility(tall_van.to_native(), n.FacilityLimits(), n.Margins())
        assert result.verdict_name == "UNVERIFIED"
        assert "facility_max_height" in list(result.unverified_dimensions)

    def test_a_polo_fits_the_median_amsterdam_kerb_bay(self, polo):
        """The median parallel bay in central Amsterdam is 1.96 m wide.

        Judging it with perpendicular door clearance rejected an ordinary hatchback by
        four centimetres, which deleted most of the on-street supply from every search.
        """
        n = self._native()
        result = n.check_bay(polo.to_native(), 551.0, 196.0, n.BayOrientation.PARALLEL,
                             n.Margins())
        assert result.acceptable, result.verdict_name

    def test_orientation_changes_the_verdict_for_one_rectangle(self, polo):
        n = self._native()
        perpendicular = n.check_bay(polo.to_native(), 440.0, 250.0,
                                    n.BayOrientation.PERPENDICULAR, n.Margins())
        parallel = n.check_bay(polo.to_native(), 440.0, 250.0, n.BayOrientation.PARALLEL,
                               n.Margins())
        assert perpendicular.acceptable
        assert parallel.verdict_name == "DOES_NOT_FIT"

    def test_safety_floors_cannot_be_tuned_away(self, tall_van):
        n = self._native()
        margins = n.Margins()
        margins.parallel_front_cm = 0.0
        margins.parallel_rear_cm = -100.0
        margins.vertical_cm = 0.0
        clamped = margins.clamped()
        assert clamped.parallel_front_cm > 0.0
        assert clamped.vertical_cm > 0.0
        # And a van still cannot be squeezed into a compact bay.
        assert n.check_bay(tall_van.to_native(), 460.0, 200.0, n.BayOrientation.PARALLEL,
                           margins).verdict_name == "DOES_NOT_FIT"

    def test_mirrors_govern_apertures_bodywork_governs_paint(self, polo):
        n = self._native()
        gate = n.FacilityLimits()
        gate.max_height_cm = 250.0
        gate.max_width_cm = 220.0
        assert n.check_facility(polo.to_native(), gate, n.Margins()).verdict_name == (
            "DOES_NOT_FIT"
        )
        assert n.check_bay(polo.to_native(), 600.0, 220.0, n.BayOrientation.PERPENDICULAR,
                           n.Margins()).acceptable


class TestRestrictions:
    def _rule(self, target_id, **kwargs):
        defaults = {
            "target_kind": "bay", "target_id": target_id, "rule_type": "test",
            "weekday_mask": 0b1111111, "start_minute": 0, "end_minute": 1440,
            "source_name": "test",
        }
        defaults.update(kwargs)
        return ParkingRestriction(**defaults)

    def test_a_permit_bay_is_refused_to_a_visitor(self, session, polo):
        session.add(self._rule(1, permit_required=True, rule_type="permit_only"))
        session.flush()
        verdicts = evaluate_restrictions(
            session, [("bay", 1)],
            arrival=datetime.now(UTC), departure=datetime.now(UTC) + timedelta(hours=2),
            vehicle=polo,
        )
        assert not verdicts[("bay", 1)].allowed
        assert verdicts[("bay", 1)].requires_permit

    def test_a_disabled_bay_is_refused_unless_asked_for(self, session, polo):
        session.add(self._rule(2, disabled_only=True, rule_type="disabled_only"))
        session.flush()
        now = datetime.now(UTC)
        refused = evaluate_restrictions(session, [("bay", 2)], arrival=now,
                                        departure=now + timedelta(hours=1), vehicle=polo)
        allowed = evaluate_restrictions(session, [("bay", 2)], arrival=now,
                                        departure=now + timedelta(hours=1), vehicle=polo,
                                        needs_disabled_bay=True)
        assert not refused[("bay", 2)].allowed
        assert allowed[("bay", 2)].allowed

    def test_a_charging_bay_is_refused_to_a_petrol_car(self, session, polo):
        session.add(self._rule(3, ev_only=True, rule_type="ev_charging_only"))
        session.flush()
        now = datetime.now(UTC)
        verdicts = evaluate_restrictions(session, [("bay", 3)], arrival=now,
                                         departure=now + timedelta(hours=1), vehicle=polo)
        assert not verdicts[("bay", 3)].allowed

    def test_an_electric_car_may_use_a_charging_bay_only_while_charging(self, session, polo):
        session.add(self._rule(4, ev_only=True, rule_type="ev_charging_only"))
        session.flush()
        polo.is_ev = True
        now = datetime.now(UTC)
        parking_only = evaluate_restrictions(session, [("bay", 4)], arrival=now,
                                             departure=now + timedelta(hours=1), vehicle=polo)
        charging = evaluate_restrictions(session, [("bay", 4)], arrival=now,
                                         departure=now + timedelta(hours=1), vehicle=polo,
                                         needs_ev_charging=True)
        assert not parking_only[("bay", 4)].allowed
        assert charging[("bay", 4)].allowed

    def test_a_rule_outside_the_stay_does_not_apply(self, session, polo):
        # Forbidden 02:00 to 04:00 only; the visit is in the afternoon.
        session.add(self._rule(5, forbids_parking=True, start_minute=120, end_minute=240))
        session.flush()
        arrival = datetime(2026, 6, 3, 14, 0, tzinfo=UTC)
        verdicts = evaluate_restrictions(session, [("bay", 5)], arrival=arrival,
                                         departure=arrival + timedelta(hours=2), vehicle=polo)
        assert verdicts[("bay", 5)].allowed

    def test_a_rule_on_another_weekday_does_not_apply(self, session, polo):
        # Monday only. 2026-06-03 is a Wednesday.
        session.add(self._rule(6, forbids_parking=True, weekday_mask=0b0000001))
        session.flush()
        arrival = datetime(2026, 6, 3, 10, 0, tzinfo=UTC)
        verdicts = evaluate_restrictions(session, [("bay", 6)], arrival=arrival,
                                         departure=arrival + timedelta(hours=1), vehicle=polo)
        assert verdicts[("bay", 6)].allowed

    def test_an_overnight_stay_is_judged_against_every_day_it_spans(self, session, polo):
        """Friday evening to Saturday morning is governed by both days."""
        # Saturday only, all day.
        session.add(self._rule(7, forbids_parking=True, weekday_mask=1 << 5))
        session.flush()
        friday_evening = datetime(2026, 6, 5, 20, 0, tzinfo=UTC)  # Friday
        verdicts = evaluate_restrictions(session, [("bay", 7)], arrival=friday_evening,
                                         departure=friday_evening + timedelta(hours=14),
                                         vehicle=polo)
        assert not verdicts[("bay", 7)].allowed

    def test_a_maximum_stay_shorter_than_the_visit_is_refused(self, session, polo):
        session.add(self._rule(8, max_duration_minutes=60, rule_type="max_duration"))
        session.flush()
        now = datetime.now(UTC)
        short = evaluate_restrictions(session, [("bay", 8)], arrival=now,
                                      departure=now + timedelta(minutes=30), vehicle=polo)
        long = evaluate_restrictions(session, [("bay", 8)], arrival=now,
                                     departure=now + timedelta(minutes=180), vehicle=polo)
        assert short[("bay", 8)].allowed
        assert short[("bay", 8)].warnings
        assert not long[("bay", 8)].allowed

    def test_a_bay_with_no_rules_is_allowed(self, session, polo):
        now = datetime.now(UTC)
        verdicts = evaluate_restrictions(session, [("bay", 99)], arrival=now,
                                         departure=now + timedelta(hours=1), vehicle=polo)
        assert verdicts[("bay", 99)].allowed


class TestEvidenceResolution:
    def test_the_strongest_source_wins(self, session, recent_observation):
        session.add(recent_observation("facility", 1, evidence=EvidenceSource.STATIC_DATABASE,
                                       state=OccupancyState.VACANT, vacant=50))
        session.add(recent_observation("facility", 1, evidence=EvidenceSource.OPERATOR_FEED,
                                       state=OccupancyState.OCCUPIED, vacant=0))
        session.flush()
        resolved = resolve_availability(session, [("facility", 1)])[("facility", 1)]
        assert resolved.evidence is EvidenceSource.OPERATOR_FEED
        assert resolved.state is OccupancyState.OCCUPIED
        assert resolved.conflicting_sources == 1

    def test_the_newest_wins_within_one_source(self, session, recent_observation):
        session.add(recent_observation("facility", 2, age_s=600, vacant=3))
        session.add(recent_observation("facility", 2, age_s=10, vacant=41))
        session.flush()
        resolved = resolve_availability(session, [("facility", 2)])[("facility", 2)]
        assert resolved.vacant_spaces == 41

    def test_a_live_source_goes_stale_and_says_so(self, session, recent_observation):
        session.add(recent_observation("facility", 3, age_s=900))
        session.flush()
        resolved = resolve_availability(session, [("facility", 3)], stale_after_s=300)[
            ("facility", 3)
        ]
        assert resolved.stale
        assert "stale" in describe_freshness(resolved)

    def test_static_data_is_never_called_stale(self, session, recent_observation):
        """Static information is not stale at five minutes old; it is simply static."""
        # Within the six-hour lookback, so the observation is actually found. Beyond it
        # the resolver correctly reports no data at all, which is a different case.
        session.add(recent_observation("facility", 4, age_s=3 * 3600,
                                       evidence=EvidenceSource.STATIC_DATABASE))
        session.flush()
        resolved = resolve_availability(session, [("facility", 4)])[("facility", 4)]
        assert not resolved.stale
        assert describe_freshness(resolved) == "static information"

    def test_an_unobserved_target_gets_a_prior_not_a_zero(self, session):
        """Unknown is not the same as full. Zero would rank a car park we simply have no
        reading for below one we know to be full, which is backwards."""
        resolved = resolve_availability(session, [("facility", 7)])[("facility", 7)]
        assert resolved.state is OccupancyState.UNKNOWN
        assert resolved.probability_available > 0.3

    def test_priors_differ_between_a_kerb_bay_and_a_garage(self):
        garage = ResolvedAvailability(
            target_kind="facility", target_id=1, state=OccupancyState.UNKNOWN,
            evidence=EvidenceSource.STATIC_DATABASE, observed_at=None, age_s=float("inf"),
            confidence=0.0,
        )
        metered_bay = ResolvedAvailability(
            target_kind="bay", target_id=2, state=OccupancyState.UNKNOWN,
            evidence=EvidenceSource.STATIC_DATABASE, observed_at=None, age_s=float("inf"),
            confidence=0.0, metered=True,
        )
        free_bay = ResolvedAvailability(
            target_kind="bay", target_id=3, state=OccupancyState.UNKNOWN,
            evidence=EvidenceSource.STATIC_DATABASE, observed_at=None, age_s=float("inf"),
            confidence=0.0, metered=False,
        )
        # A garage has many interchangeable spaces; one named kerb bay does not.
        assert garage.prior > free_bay.prior > metered_bay.prior

    def test_an_occupied_reading_is_near_zero(self, session, recent_observation):
        session.add(recent_observation("facility", 5, state=OccupancyState.OCCUPIED, vacant=0))
        session.flush()
        resolved = resolve_availability(session, [("facility", 5)])[("facility", 5)]
        assert resolved.probability_available < 0.1

    def test_more_free_spaces_means_higher_confidence(self, session, recent_observation):
        session.add(recent_observation("facility", 10, vacant=1))
        session.add(recent_observation("facility", 11, vacant=40))
        session.flush()
        resolved = resolve_availability(session, [("facility", 10), ("facility", 11)])
        assert (
            resolved[("facility", 11)].probability_available
            > resolved[("facility", 10)].probability_available
        )

    def test_batching_resolves_many_targets_in_one_call(self, session, recent_observation):
        for i in range(20, 40):
            session.add(recent_observation("facility", i, vacant=i))
        session.flush()
        keys = [("facility", i) for i in range(20, 40)]
        resolved = resolve_availability(session, keys)
        assert len(resolved) == 20


class TestPricing:
    def test_an_unmetered_bay_is_not_presented_as_free_parking(self, session, seeded_bays):
        """NIET FISCAAL usually means permit-controlled rather than genuinely free, and
        the regime data does not always say which."""
        free_bay = next(b for b in seeded_bays if not b.fiscal)
        prices = estimate_prices(session, [(("bay", free_bay.id), "on_street_bay")],
                                 arrival=datetime.now(UTC), duration_minutes=120)
        price, note = prices[("bay", free_bay.id)]
        assert price == 0.0
        assert "check the signs" in note

    def test_a_metered_amsterdam_bay_uses_the_city_rate(self, session, seeded_bays):
        bay = seeded_bays[0]
        prices = estimate_prices(session, [(("bay", bay.id), "on_street_bay")],
                                 arrival=datetime.now(UTC), duration_minutes=120)
        price, note = prices[("bay", bay.id)]
        assert price == pytest.approx(15.0, abs=0.01)  # 7.50/hour, two hours
        assert "Amsterdam" in note

    def test_every_price_carries_a_provenance_note(self, session, seeded_facilities):
        targets = [(("facility", f.id), f.kind) for f in seeded_facilities]
        prices = estimate_prices(session, targets, arrival=datetime.now(UTC),
                                 duration_minutes=90)
        assert len(prices) == len(seeded_facilities)
        for _price, note in prices.values():
            assert note

    def test_price_scales_with_duration(self, session, seeded_facilities):
        facility = seeded_facilities[0]
        short = estimate_prices(session, [(("facility", facility.id), "garage")],
                                arrival=datetime.now(UTC), duration_minutes=60)
        long = estimate_prices(session, [(("facility", facility.id), "garage")],
                               arrival=datetime.now(UTC), duration_minutes=240)
        assert long[("facility", facility.id)][0] > short[("facility", facility.id)][0]


class TestDeduplication:
    class _Candidate:
        def __init__(self, key, kind, name, lat, lon, capacity, height, source):
            self.key, self.kind, self.name = key, kind, name
            self.lat, self.lon = lat, lon
            self.capacity, self.max_height_cm, self.source_name = capacity, height, source

    def test_the_same_garage_from_two_sources_is_merged(self):
        rdw = self._Candidate(("facility", 1), "garage", "Garage The Bank (Amsterdam)",
                              52.36620, 4.89860, 110, 210.0, "RDW-NPR")
        osm = self._Candidate(("facility", 2), "garage", "The Bank",
                              52.36625, 4.89855, None, None, "OpenStreetMap")
        merged = merge_duplicates([rdw, osm])
        assert len(merged) == 1
        # The union of what is known, not merely the better record.
        assert merged[0].max_height_cm == 210.0
        assert merged[0].capacity == 110

    def test_genuinely_different_garages_are_kept(self):
        a = self._Candidate(("facility", 1), "garage", "Garage The Bank", 52.36620, 4.89860,
                            110, 210.0, "RDW-NPR")
        b = self._Candidate(("facility", 2), "garage", "Parkeergarage Rokin", 52.3700, 4.8920,
                            None, None, "OpenStreetMap")
        assert len(merge_duplicates([a, b])) == 2

    def test_bays_are_never_merged(self):
        """Individual marked bays sit metres apart and are distinct spaces; collapsing
        them would delete real parking supply."""
        bays = [
            self._Candidate(("bay", i), "on_street_bay", "Street bay",
                            52.3690 + i * 1e-5, 4.9010, None, None, "Amsterdam-Parkeervakken")
            for i in range(5)
        ]
        assert len(merge_duplicates(bays)) == 5
