"""The search engine: destination and vehicle in, ranked parking out.

The whole product converges here. A search runs eleven steps:

1. Geocode the destination (OSM points of interest, then PDOK).
2. Retrieve candidates within an expanding radius, facilities and marked bays.
3. Drop candidates that are illegal for this driver at this time.
4. Drop candidates this vehicle physically cannot use.
5. Route the drive leg, one sweep for all candidates.
6. Route the walk leg, one sweep for all candidates.
7. Resolve current availability by source priority.
8. Estimate price for the intended duration.
9. Score by expected total inconvenience.
10. Diversify across streets and facilities.
11. Record what was recommended, so the next search can avoid herding.

Order matters. Legality and fit run *before* routing because they are cheap and remove
most candidates, and routing is the expensive step. Availability is resolved after fit
because there is no point asking whether a space is free in a garage the vehicle cannot
enter.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from parkfit.config import Settings, get_settings
from parkfit.domain.dedupe import merge_duplicates
from parkfit.domain.evidence import (
    ResolvedAvailability,
    describe_freshness,
    resolve_availability,
)
from parkfit.domain.pricing import estimate_prices
from parkfit.domain.restrictions import RestrictionVerdict, evaluate_restrictions
from parkfit.domain.vehicle import VehicleProfile
from parkfit.native import native
from parkfit.prediction.features import load_statics as load_target_statics
from parkfit.prediction.model import get_model as get_occupancy_model
from parkfit.routing.provider import (
    Profile,
    RouteResult,
    RoutingService,
    get_routing_service,
)
from parkfit.services.candidate_index import get_candidate_index
from parkfit.services.geocoding import Destination, HybridGeocoder
from parkfit.services.ledger import LedgerEntry, get_ledger
from parkfit.services.legality import LegalVerdict, get_legality_service
from parkfit.storage.models import (
    EvidenceSource,
    FacilityKind,
    OccupancyState,
    ParkingBay,
    ParkingFacility,
    SegmentDynamics,
)

log = logging.getLogger(__name__)


@dataclass
class SearchPreferences:
    max_walk_minutes: float = 12.0
    prefer_covered: bool = False
    prefer_cheapest: bool = False
    needs_ev_charging: bool = False
    needs_disabled_bay: bool = False
    include_on_street: bool = True
    value_of_time_eur_per_min: float = 0.20


@dataclass
class SearchRequest:
    destination: str
    vehicle: VehicleProfile
    origin_lat: float | None = None
    origin_lon: float | None = None
    arrival_time: datetime | None = None
    duration_minutes: int = 120
    preferences: SearchPreferences = field(default_factory=SearchPreferences)
    user_id: int | None = None
    city_hint: str | None = None
    #: ISO 3166-1 alpha-2. Chooses the national geocoder and, through the candidates
    #: it returns, the legal rulebook. Left as None the geocoder tries everything,
    #: which is right when the country genuinely is not known yet.
    country: str | None = None


@dataclass
class Candidate:
    """One parking option, carrying everything needed to explain its ranking."""

    key: tuple[str, int]
    kind: str
    name: str
    lat: float
    lon: float
    group_key: str
    is_exact_space: bool

    fit_verdict: str = "UNVERIFIED"
    fit_slack_cm: float = 0.0
    fit_binding: str | None = None
    fit_unverified: list[str] = field(default_factory=list)

    drive: RouteResult | None = None
    walk: RouteResult | None = None
    availability: ResolvedAvailability | None = None
    restriction: RestrictionVerdict | None = None
    #: Whether road law allows parking here, with the article it turns on. Distinct
    #: from `restriction`, which is the sign and time regime on the bay itself: this
    #: is the statutory setback from junctions, crossings, hydrants and the rest.
    legal: LegalVerdict | None = None

    price_eur: float = 0.0
    price_note: str = ""
    lambda_per_min: float = 0.0

    generalised_cost: float = 0.0
    probability_at_eta: float = 0.0
    confidence_label: str = ""
    expired: bool = False

    capacity: int | None = None
    max_height_cm: float | None = None
    bay_length_cm: float = 0.0
    bay_width_cm: float = 0.0
    orientation: str = ""
    fill_ratio: float = 1.0
    metered: bool = True
    source_name: str = ""
    #: ISO 3166-1 alpha-2, taken from the facility record rather than guessed from
    #: the coordinates. It picks the legal rulebook, and a wrong guess would apply
    #: one country's road law to another country's streets.
    country: str = "NL"


@dataclass
class SearchResponse:
    search_id: str
    destination: Destination | None
    results: list[Candidate]
    considered: int = 0
    merged_duplicates: int = 0
    rejected_illegal: int = 0
    #: Refused by a statutory setback rather than by a sign or a time regime.
    rejected_setback: int = 0
    #: Kept, but with legality unproven because no anchors cover this area.
    legality_unknown: int = 0
    rejected_fit: int = 0
    rejected_walk: int = 0
    radius_m: float = 0.0
    routing_provider: str = ""
    warnings: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0


class SearchEngine:
    """Orchestrates a full parking search."""

    def __init__(
        self,
        session: Session,
        *,
        settings: Settings | None = None,
        routing: RoutingService | None = None,
        geocoder: HybridGeocoder | None = None,
    ):
        self.session = session
        self.settings = settings or get_settings()
        self._routing = routing
        self._owns_routing = routing is None
        self._geocoder = geocoder
        self._owns_geocoder = geocoder is None

    @property
    def routing(self) -> RoutingService:
        if self._routing is None:
            # Shared per process: the road graph costs about a second to load and is
            # immutable once built, so every request rebuilding it was pure waste.
            self._routing = get_routing_service()
            self._owns_routing = False
        return self._routing

    @property
    def geocoder(self) -> HybridGeocoder:
        if self._geocoder is None:
            self._geocoder = HybridGeocoder(self.session)
        return self._geocoder

    def close(self) -> None:
        if self._owns_routing and self._routing is not None:
            self._routing.close()
        if self._owns_geocoder and self._geocoder is not None:
            self._geocoder.close()

    # main entry point ---------------------------------------------------
    def search(self, request: SearchRequest) -> SearchResponse:
        started = datetime.now(UTC)
        search_id = uuid.uuid4().hex[:16]
        response = SearchResponse(search_id=search_id, destination=None, results=[])

        destination = self.geocoder.geocode_one(
            request.destination, city=request.city_hint, country=request.country
        )
        if destination is None:
            response.warnings.append(f"could not locate {request.destination!r}")
            return self._finish(response, started)
        response.destination = destination

        if not destination.is_precise:
            response.warnings.append(
                f"{destination.label!r} was matched loosely; results may be centred on the "
                "wrong part of town"
            )

        arrival = request.arrival_time or datetime.now(UTC)
        origin_lat = request.origin_lat if request.origin_lat is not None else destination.lat
        origin_lon = request.origin_lon if request.origin_lon is not None else destination.lon

        candidates, radius = self._collect_candidates(destination, request)
        before_merge = len(candidates)
        candidates = merge_duplicates(candidates)
        response.merged_duplicates = before_merge - len(candidates)
        response.considered = len(candidates)
        response.radius_m = radius
        if not candidates:
            response.warnings.append("no parking found near this destination")
            return self._finish(response, started)

        candidates = self._filter_legal(candidates, request, arrival, response)
        candidates = self._filter_fit(candidates, request, response)
        if not candidates:
            response.warnings.append(
                "every nearby option was ruled out for this vehicle; try a larger search "
                "radius or check the confirmed dimensions"
            )
            return self._finish(response, started)

        self._route_legs(candidates, destination, origin_lat, origin_lon, request, response)
        candidates = self._filter_walk(candidates, request, response)
        if not candidates:
            response.warnings.append(
                f"nothing within a {request.preferences.max_walk_minutes:.0f} minute walk"
            )
            return self._finish(response, started)

        self._attach_availability(candidates, arrival)
        self._attach_price(candidates, request, arrival)
        self._attach_dynamics(candidates, arrival)
        response.results = self._rank(candidates, request, search_id)
        self._record_recommendations(response.results, search_id, request)

        response.routing_provider = self.routing.active_provider
        return self._finish(response, started)

    @staticmethod
    def _finish(response: SearchResponse, started: datetime) -> SearchResponse:
        response.elapsed_ms = (datetime.now(UTC) - started).total_seconds() * 1000.0
        return response

    # 2. candidate retrieval --------------------------------------------
    def _collect_candidates(
        self, destination: Destination, request: SearchRequest
    ) -> tuple[list[Candidate], float]:
        """Gather candidates, widening the radius until there are enough.

        Expanding rather than fixed: a fixed radius returns nothing at all outside the
        dense city centres, which is exactly where a driver most needs a suggestion.
        """
        radius = self.settings.default_search_radius_m
        candidates: list[Candidate] = []
        while radius <= self.settings.max_search_radius_m:
            candidates = self._candidates_within(destination, request, radius)
            if len(candidates) >= 12:
                break
            radius *= 1.8
        return candidates[: self.settings.max_candidates], radius

    def _candidates_within(
        self, destination: Destination, request: SearchRequest, radius_m: float
    ) -> list[Candidate]:
        """Find candidates by spatial index, then load only those rows.

        Radius search is answered by the in-memory grid rather than by SQL. A bbox
        predicate can only range-scan on the leading index column, so the database ends
        up filtering tens of thousands of bays by longitude, 200 ms with a warm page
        cache, four seconds with a cold one, and every fresh connection is cold.
        """
        index = get_candidate_index()
        index.ensure_built(self.session)

        facility_hits = index.facilities_within(
            destination.lat, destination.lon, radius_m, self.settings.max_facility_candidates
        )
        bay_hits = (
            index.bays_within(
                destination.lat, destination.lon, radius_m, self.settings.max_bay_candidates
            )
            if request.preferences.include_on_street
            else []
        )
        if not facility_hits and not bay_hits:
            return []

        out: list[Candidate] = []

        if facility_hits:
            rows = {
                f.id: f
                for f in self.session.execute(
                    select(ParkingFacility).where(
                        ParkingFacility.id.in_([h.target_id for h in facility_hits])
                    )
                ).scalars()
            }
            for hit in facility_hits:
                facility = rows.get(hit.target_id)
                if facility is None or not facility.active:
                    continue
                out.append(
                    Candidate(
                        key=("facility", facility.id),
                        kind=facility.kind,
                        name=facility.name or "Parking",
                        lat=facility.lat,
                        lon=facility.lon,
                        group_key=f"facility:{facility.id}",
                        is_exact_space=False,
                        capacity=facility.capacity,
                        max_height_cm=facility.max_vehicle_height_cm,
                        source_name=facility.source_name,
                        country=(facility.country or "NL"),
                    )
                )

        if bay_hits:
            rows = {
                b.id: b
                for b in self.session.execute(
                    select(ParkingBay).where(ParkingBay.id.in_([h.target_id for h in bay_hits]))
                ).scalars()
            }
            for hit in bay_hits:
                bay = rows.get(hit.target_id)
                if bay is None:
                    continue
                out.append(
                    Candidate(
                        key=("bay", bay.id),
                        kind="on_street_bay",
                        name=f"{bay.street or 'Street'} bay",
                        lat=bay.lat,
                        lon=bay.lon,
                        # Bays on one street share a failure mode: if the street is full
                        # they are all wrong together, so they compete for one slot.
                        group_key=f"street:{bay.street or bay.neighbourhood_code or bay.id}",
                        is_exact_space=True,
                        bay_length_cm=bay.length_cm,
                        bay_width_cm=bay.width_cm,
                        orientation=bay.orientation,
                        fill_ratio=bay.fill_ratio,
                        metered=bay.fiscal,
                        source_name=bay.source_name,
                    )
                )
        return out

    # 3. legality --------------------------------------------------------
    def _filter_legal(
        self,
        candidates: list[Candidate],
        request: SearchRequest,
        arrival: datetime,
        response: SearchResponse,
    ) -> list[Candidate]:
        departure = arrival + timedelta(minutes=request.duration_minutes)
        verdicts = evaluate_restrictions(
            self.session,
            [c.key for c in candidates],
            arrival=arrival,
            departure=departure,
            vehicle=request.vehicle,
            needs_ev_charging=request.preferences.needs_ev_charging,
            needs_disabled_bay=request.preferences.needs_disabled_bay,
        )
        kept: list[Candidate] = []
        for candidate in candidates:
            verdict = verdicts.get(candidate.key)
            candidate.restriction = verdict
            if verdict is not None and not verdict.allowed:
                response.rejected_illegal += 1
                continue
            kept.append(candidate)

        return self._filter_setbacks(kept, response)

    def _filter_setbacks(
        self, candidates: list[Candidate], response: SearchResponse
    ) -> list[Candidate]:
        """Drop candidates that road law forbids on distance grounds.

        This is a different question from the one above. ``evaluate_restrictions`` reads
        the sign and time regime recorded against the bay: permit hours, loading windows,
        disc zones. This reads the statute itself, and the things it measures from are
        map features rather than anything written on the bay: a hydrant, a crossing, a
        junction. A bay can be perfectly in order on its own record and still be five
        metres from something the law says to stay away from.

        A bay whose polygon the municipality surveyed and published is *not* re-judged on
        setbacks. Amsterdam does not paint a bay inside a bus stop, and second-guessing
        the surveyor with an OSM-derived anchor would throw away good spaces on the
        strength of worse data. The setback rules earn their keep on the candidates that
        are not surveyed bays: camera-derived gaps and OSM-inferred kerb space, which is
        exactly where nobody has checked.
        """
        legality = get_legality_service()
        subjects = [c for c in candidates if not c.is_exact_space]
        if not subjects:
            return candidates

        # Grouped by country, because each country has its own book and a book is chosen
        # per call. A search normally sits in one city and so one group, but a border
        # town is a real place and judging Aachen under Dutch law would be wrong in a way
        # nobody would notice from the output.
        by_country: dict[str, list[Candidate]] = {}
        for candidate in subjects:
            by_country.setdefault((candidate.country or "NL").upper(), []).append(candidate)

        kept: list[Candidate] = []
        refused: dict[tuple[str, int], LegalVerdict] = {}

        for country, group in by_country.items():
            verdicts = legality.evaluate([(c.lat, c.lon) for c in group], country=country)
            for candidate, verdict in zip(group, verdicts, strict=True):
                candidate.legal = verdict
                if verdict.is_unknown:
                    response.legality_unknown += 1
                elif not verdict.allowed:
                    refused[candidate.key] = verdict

        for candidate in candidates:
            verdict = refused.get(candidate.key)
            if verdict is not None:
                response.rejected_setback += 1
                log.debug(
                    "%s refused: %s (%.1f m, needs %.1f m) %s",
                    candidate.name,
                    verdict.anchor,
                    verdict.distance_cm / 100.0,
                    verdict.required_cm / 100.0,
                    verdict.citation,
                )
                continue
            kept.append(candidate)
        return kept

    # 4. physical fit ----------------------------------------------------
    def _filter_fit(
        self, candidates: list[Candidate], request: SearchRequest, response: SearchResponse
    ) -> list[Candidate]:
        if native is None:
            for candidate in candidates:
                candidate.fit_verdict = "UNVERIFIED"
                candidate.fit_unverified = ["native_module_unavailable"]
            return candidates

        vehicle = request.vehicle.to_native()
        margins = native.Margins()
        margins.tight_threshold_cm = 15.0

        # Batched, in two groups. One call each instead of one per candidate: the check
        # itself is a handful of comparisons, so crossing the language boundary for it
        # several hundred times a search costs more than the arithmetic does.
        facilities = [c for c in candidates if c.key[0] == "facility"]
        bays = [c for c in candidates if c.key[0] != "facility"]

        results: dict[tuple[str, int], object] = {}
        if facilities:
            # A missing height is passed as 0, which the native side reads as
            # "unpublished" and answers UNVERIFIED, never as unlimited.
            heights = [c.max_height_cm or 0.0 for c in facilities]
            for candidate, result in zip(
                facilities, native.check_facilities(vehicle, heights, margins), strict=True
            ):
                results[candidate.key] = result
        if bays:
            for candidate, result in zip(
                bays,
                native.check_bays(
                    vehicle,
                    [c.bay_length_cm for c in bays],
                    [c.bay_width_cm for c in bays],
                    # The normalised value goes straight across. It used to be translated
                    # back into Dutch to satisfy a Dutch-only parser, which meant a
                    # Turkish bay had to pretend to be Dutch to be measured.
                    [c.orientation for c in bays],
                    margins,
                ),
                strict=True,
            ):
                results[candidate.key] = result

        kept: list[Candidate] = []
        for candidate in candidates:
            result = results[candidate.key]
            candidate.fit_verdict = result.verdict_name
            candidate.fit_slack_cm = result.min_slack_cm
            candidate.fit_unverified = list(result.unverified_dimensions)
            binding = result.binding_constraint
            candidate.fit_binding = binding.constraint if binding is not None else None

            if result.verdict == native.Verdict.DOES_NOT_FIT:
                response.rejected_fit += 1
                continue
            kept.append(candidate)
        return kept

    # 5 and 6. routing ---------------------------------------------------
    def _route_legs(
        self,
        candidates: list[Candidate],
        destination: Destination,
        origin_lat: float,
        origin_lon: float,
        request: SearchRequest,
        response: SearchResponse,
    ) -> None:
        targets = [(c.lat, c.lon) for c in candidates]
        # One sweep per leg rather than one route per candidate. The alternative is
        # several hundred A* runs, which costs seconds rather than milliseconds.
        drives = self.routing.many_routes(
            origin_lat, origin_lon, targets, Profile.CAR, max_seconds=1800
        )
        walks = self.routing.many_routes(
            destination.lat,
            destination.lon,
            targets,
            Profile.FOOT,
            max_seconds=max(300.0, request.preferences.max_walk_minutes * 60.0 * 1.5),
        )
        for candidate, drive, walk in zip(candidates, drives, walks, strict=True):
            candidate.drive = drive
            candidate.walk = walk
        if any(d.is_estimate for d in drives):
            response.warnings.append(
                "some drive times are straight-line estimates rather than routed"
            )

    def _filter_walk(
        self, candidates: list[Candidate], request: SearchRequest, response: SearchResponse
    ) -> list[Candidate]:
        limit = request.preferences.max_walk_minutes
        kept = []
        for candidate in candidates:
            walk_min = candidate.walk.duration_min if candidate.walk else 0.0
            if walk_min > limit:
                response.rejected_walk += 1
                continue
            kept.append(candidate)
        # Never return nothing purely because of a walking preference: a slightly longer
        # walk is a far better answer than an empty result page.
        if not kept and candidates:
            candidates.sort(key=lambda c: c.walk.duration_min if c.walk else 1e9)
            response.warnings.append(
                "nothing within the preferred walking time; showing the closest options"
            )
            return candidates[:5]
        return kept

    # 7 to 8. availability and price ------------------------------------
    def _attach_availability(self, candidates: list[Candidate], arrival: datetime) -> None:
        resolved = resolve_availability(
            self.session,
            [c.key for c in candidates],
            now=arrival,
            stale_after_s=self.settings.stale_after_s,
        )
        for candidate in candidates:
            availability = resolved.get(candidate.key)
            if availability is not None and candidate.key[0] == "bay":
                # Whether the bay is metered halves or doubles the prior, so it has to
                # travel with the resolved state rather than be guessed at scoring time.
                availability = replace(availability, metered=candidate.metered)
            candidate.availability = availability

        self._attach_model_prior(candidates, arrival)

    def _attach_model_prior(self, candidates: list[Candidate], arrival: datetime) -> None:
        """Replace the flat base rate with a learned one, where nothing live is known.

        Only candidates with no usable observation are touched. A target a sensor saw
        thirty seconds ago does not need a prediction, and letting a model overwrite a
        measurement is exactly the failure the evidence ordering exists to prevent.
        """
        model = get_occupancy_model()
        if not model.available:
            return

        needs_prior = [
            c
            for c in candidates
            if c.availability is not None
            and (c.availability.stale or c.availability.state is OccupancyState.UNKNOWN)
        ]
        if not needs_prior:
            return

        statics = load_target_statics(self.session, [c.key for c in needs_prior])
        pairs: list[tuple[object, datetime]] = []
        matched: list[Candidate] = []
        for candidate in needs_prior:
            static = statics.get(candidate.key)
            if static is not None:
                pairs.append((static, arrival))
                matched.append(candidate)

        occupied = model.probability_occupied(pairs)
        if occupied is None:
            return

        for candidate, p_occupied in zip(matched, occupied, strict=True):
            availability = candidate.availability
            if availability is None:
                continue
            candidate.availability = replace(
                availability,
                model_prior=float(1.0 - p_occupied),
                # A prediction outranks the static register and nothing else. Recording it
                # here is what makes the response say "modelled" rather than implying the
                # number came from somewhere it did not.
                evidence=max(availability.evidence, EvidenceSource.PREDICTIVE_MODEL),
            )

    def _attach_price(
        self, candidates: list[Candidate], request: SearchRequest, arrival: datetime
    ) -> None:
        # One batched pass rather than a query per candidate. The per-candidate version
        # issued 456 round-trips for a single search and cost more than the routing did.
        prices = estimate_prices(
            self.session,
            [(c.key, c.kind) for c in candidates],
            arrival=arrival,
            duration_minutes=request.duration_minutes,
        )
        for candidate in candidates:
            price, note = prices.get(candidate.key, (0.0, "unknown"))
            candidate.price_eur = price
            candidate.price_note = note

    def _attach_dynamics(self, candidates: list[Candidate], arrival: datetime) -> None:
        """Attach the per-segment vacancy decay rate for this weekday and time bucket."""
        weekday = arrival.weekday()
        bucket = (arrival.hour * 60 + arrival.minute) // 15
        rows = (
            self.session.execute(
                select(SegmentDynamics).where(
                    SegmentDynamics.target_id.in_([c.key[1] for c in candidates]),
                    SegmentDynamics.weekday == weekday,
                    SegmentDynamics.quarter_hour == bucket,
                )
            )
            .scalars()
            .all()
        )
        by_key = {(r.target_kind, r.target_id): r for r in rows}
        for candidate in candidates:
            row = by_key.get(candidate.key)
            if row is not None:
                candidate.lambda_per_min = row.lambda_per_min
            else:
                # No history yet. A single identified kerb space is volatile by nature;
                # a garage with many interchangeable spaces is not, so they cannot share
                # a default. These are replaced by learned rates as history accumulates.
                candidate.lambda_per_min = 0.12 if candidate.is_exact_space else 0.01

    # 9 and 10. scoring --------------------------------------------------
    def _rank(
        self, candidates: list[Candidate], request: SearchRequest, search_id: str
    ) -> list[Candidate]:
        if native is None:
            candidates.sort(
                key=lambda c: (
                    (c.drive.duration_min if c.drive else 0)
                    + (c.walk.duration_min if c.walk else 0)
                )
            )
            return candidates[: self.settings.max_results]

        config = native.ScoringConfig()
        config.value_of_time_eur_per_min = request.preferences.value_of_time_eur_per_min
        config.stale_after_s = self.settings.stale_after_s
        config.exact_space_ttl_s = self.settings.exact_space_ttl_s

        herding = self._recent_recommendation_counts([c.key for c in candidates])
        by_id: dict[str, Candidate] = {}
        native_candidates = []

        for candidate in candidates:
            cid = f"{candidate.key[0]}:{candidate.key[1]}"
            by_id[cid] = candidate
            nc = native.Candidate()
            nc.id = cid
            nc.group_key = candidate.group_key
            nc.drive_time_min = candidate.drive.duration_min if candidate.drive else 0.0
            nc.walk_time_min = candidate.walk.duration_min if candidate.walk else 0.0
            nc.price_eur = candidate.price_eur
            availability = candidate.availability
            nc.p_available_now = availability.probability_available if availability else 0.5
            nc.lambda_per_min = candidate.lambda_per_min
            nc.eta_min = candidate.drive.duration_min if candidate.drive else 0.0
            observed = (
                availability is not None
                and availability.observed_at is not None
                and availability.age_s != float("inf")
            )
            nc.observation_age_s = min(availability.age_s, 1e6) if observed else 0.0
            # Never observed means this is the model talking, and it should be labelled
            # as such. Presenting a prior as a stale live reading would be a lie about
            # provenance, and it would also trip the exact-space time-to-live.
            nc.evidence = _native_evidence(
                availability.evidence if observed else EvidenceSource.PREDICTIVE_MODEL
            )
            nc.fit_verdict = _native_verdict(candidate.fit_verdict)
            nc.is_exact_space = candidate.is_exact_space
            nc.recent_recommendation_count = herding.get(candidate.key, 0)
            native_candidates.append(nc)

        scored = native.rank_and_diversify(
            native_candidates,
            config,
            self.settings.max_results,
            self.settings.max_results_per_group,
        )

        out: list[Candidate] = []
        for entry in scored:
            candidate = by_id.get(entry.id)
            if candidate is None:
                continue
            candidate.generalised_cost = entry.generalised_cost
            candidate.probability_at_eta = entry.p_available_at_eta
            candidate.confidence_label = entry.confidence_name
            candidate.expired = entry.expired
            out.append(candidate)
        return out

    def _recent_recommendation_counts(
        self, keys: list[tuple[str, int]]
    ) -> dict[tuple[str, int], int]:
        """How many live recommendations already point at each target.

        Read from the in-memory ledger, not the database. The ledger sees a
        recommendation issued microseconds ago, which a query over committed rows does
        not, and noticing near-simultaneous requests is the entire purpose of the
        signal.
        """
        return get_ledger().counts(keys)

    # 11. record ---------------------------------------------------------
    def _record_recommendations(
        self, results: list[Candidate], search_id: str, request: SearchRequest
    ) -> None:
        """Record what was recommended, in memory.

        Deliberately not a database write. Persisting ten rows synchronously cost
        several seconds on the first write of a process and put that entirely on the
        user, for bookkeeping the response does not depend on. The background flusher
        writes them out instead.
        """
        now = time.time()
        entries = []
        for rank_index, candidate in enumerate(results):
            # An exact space expires quickly; a facility recommendation stays useful
            # for a while, because one car taking a space does not fill the garage.
            ttl = self.settings.exact_space_ttl_s if candidate.is_exact_space else 600.0
            entries.append(
                LedgerEntry(
                    target_kind=candidate.key[0],
                    target_id=candidate.key[1],
                    search_id=search_id,
                    user_id=request.user_id,
                    rank=rank_index,
                    generalised_cost=candidate.generalised_cost,
                    probability_at_eta=candidate.probability_at_eta,
                    confidence_label=candidate.confidence_label,
                    fit_verdict=candidate.fit_verdict,
                    created_at=now,
                    expires_at=now + ttl,
                )
            )
        get_ledger().record(entries)


def _dutch_orientation(value: str) -> str:
    return {
        "parallel": "Langs",
        "perpendicular": "Haaks",
        "angled": "Visgraat",
    }.get(value, "")


def _native_verdict(name: str):
    return {
        "FITS": native.Verdict.FITS,
        "TIGHT_FIT": native.Verdict.TIGHT_FIT,
        "DOES_NOT_FIT": native.Verdict.DOES_NOT_FIT,
        "UNVERIFIED": native.Verdict.UNVERIFIED,
    }.get(name, native.Verdict.UNVERIFIED)


def _native_evidence(source: EvidenceSource):
    return {
        EvidenceSource.OSM_ONLY: native.EvidenceSource.OSM_ONLY,
        EvidenceSource.STATIC_DATABASE: native.EvidenceSource.STATIC_DATABASE,
        EvidenceSource.PREDICTIVE_MODEL: native.EvidenceSource.PREDICTIVE_MODEL,
        EvidenceSource.USER_CONFIRMATION: native.EvidenceSource.USER_CONFIRMATION,
        EvidenceSource.MUNICIPAL_SENSOR: native.EvidenceSource.MUNICIPAL_SENSOR,
        EvidenceSource.CAMERA_OBSERVATION: native.EvidenceSource.CAMERA_OBSERVATION,
        EvidenceSource.OPERATOR_FEED: native.EvidenceSource.OPERATOR_FEED,
    }.get(source, native.EvidenceSource.STATIC_DATABASE)


def summarise(candidate: Candidate) -> str:
    """One-line description of a result, for the CLI and for logs."""
    availability = candidate.availability
    freshness = describe_freshness(availability) if availability else "no live data"
    drive = f"{candidate.drive.duration_min:.0f} min drive" if candidate.drive else "?"
    walk = f"{candidate.walk.duration_min:.0f} min walk" if candidate.walk else "?"
    return (
        f"{candidate.name[:38]:<38} {drive:>13} {walk:>13} "
        f"EUR {candidate.price_eur:5.2f}  {candidate.fit_verdict:<12} "
        f"P {candidate.probability_at_eta:.2f}  {freshness}"
    )


FACILITY_KINDS_COVERED = {FacilityKind.GARAGE.value}
