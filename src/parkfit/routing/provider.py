"""Routing providers, arranged as a degradation chain.

A parking search needs two legs: drive from the current position to the parking
*entrance*, then walk from the parking *exit* to the destination. Routing to a facility
centroid instead of its entrance is a classic parking-app failure -- an Amsterdam garage
centroid can sit inside a block, across a canal, or in a pedestrian zone.

Three providers, tried in order:

1. :class:`NativeGraphProvider` -- A* over a cached OpenStreetMap road graph. No
   containers, no services, correct one-way and turn handling.
2. :class:`OsrmProvider` -- a running OSRM instance, when ``OSRM_URL`` is set.
3. :class:`HaversineProvider` -- straight-line distance times a detour factor.

The chain always terminates in something that answers. A parking app that returns no
results because a routing container is down is worse than one that says "about 8
minutes" with lower confidence -- and each result carries which provider produced it,
so the ranking can discount accordingly.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import httpx

from parkfit.config import Settings, get_settings
from parkfit.geo.rd import haversine_m

log = logging.getLogger(__name__)


class Profile(str, Enum):
    CAR = "car"
    FOOT = "foot"


class RoutingUnavailable(RuntimeError):
    """Raised by a provider that cannot answer, so the chain moves on."""


@dataclass(frozen=True)
class RouteResult:
    distance_m: float
    duration_min: float
    profile: Profile
    provider: str
    #: ``[[lon, lat], ...]`` when the provider returns geometry.
    geometry: list[list[float]] = field(default_factory=list)
    #: How much to trust the duration. A straight-line estimate is not a route.
    confidence: float = 1.0

    @property
    def is_estimate(self) -> bool:
        return self.confidence < 0.75


class RoutingProvider(ABC):
    name: str = "abstract"

    @abstractmethod
    def route(
        self, from_lat: float, from_lon: float, to_lat: float, to_lon: float, profile: Profile
    ) -> RouteResult: ...

    def available(self) -> bool:
        return True

    def close(self) -> None:  # pragma: no cover - most providers hold nothing
        return None


class HaversineProvider(RoutingProvider):
    """Straight-line distance scaled by a detour factor.

    Always available and always approximate. The detour factor converts crow-flight
    distance into plausible street distance; 1.42 suits Amsterdam, where canals force
    longer diversions than a typical European grid. Walking uses a lower factor because
    pedestrians cross bridges and squares a car cannot.
    """

    name = "haversine"

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def route(
        self, from_lat: float, from_lon: float, to_lat: float, to_lon: float, profile: Profile
    ) -> RouteResult:
        straight = haversine_m(from_lat, from_lon, to_lat, to_lon)
        if profile is Profile.CAR:
            factor = self.settings.detour_factor
            speed = self.settings.drive_speed_kmh
        else:
            factor = 1.0 + (self.settings.detour_factor - 1.0) * 0.55
            speed = self.settings.walk_speed_kmh

        distance = straight * factor
        duration = (distance / 1000.0) / max(1e-6, speed) * 60.0
        # A short hop is dominated by fixed costs -- parking, doors, the barrier -- that
        # a distance-over-speed model cannot see, so it never reports under a minute.
        duration = max(duration, 0.5)
        return RouteResult(
            distance_m=distance,
            duration_min=duration,
            profile=profile,
            provider=self.name,
            confidence=0.55,
        )


class OsrmProvider(RoutingProvider):
    """HTTP client for an OSRM routing server."""

    name = "osrm"
    PROFILE_PATH = {Profile.CAR: "driving", Profile.FOOT: "foot"}

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._client: httpx.Client | None = None

    def available(self) -> bool:
        return bool(self.settings.osrm_url)

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=10.0, headers={"User-Agent": self.settings.user_agent}
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def route(
        self, from_lat: float, from_lon: float, to_lat: float, to_lon: float, profile: Profile
    ) -> RouteResult:
        if not self.settings.osrm_url:
            raise RoutingUnavailable("OSRM_URL is not configured")
        path = self.PROFILE_PATH[profile]
        url = (
            f"{self.settings.osrm_url.rstrip('/')}/route/v1/{path}/"
            f"{from_lon},{from_lat};{to_lon},{to_lat}"
        )
        try:
            response = self.client.get(
                url, params={"overview": "simplified", "geometries": "geojson"}
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RoutingUnavailable(f"OSRM request failed: {exc}") from exc

        routes = payload.get("routes") or []
        if not routes:
            raise RoutingUnavailable("OSRM returned no route")
        best = routes[0]
        geometry = (best.get("geometry") or {}).get("coordinates") or []
        return RouteResult(
            distance_m=float(best.get("distance", 0.0)),
            duration_min=float(best.get("duration", 0.0)) / 60.0,
            profile=profile,
            provider=self.name,
            geometry=[[float(c[0]), float(c[1])] for c in geometry],
            confidence=0.95,
        )


class RoutingService:
    """Tries each provider in turn and returns the first real answer."""

    def __init__(self, providers: list[RoutingProvider] | None = None,
                 settings: Settings | None = None):
        self.settings = settings or get_settings()
        if providers is None:
            from parkfit.routing.graph import NativeGraphProvider

            providers = [
                NativeGraphProvider(self.settings),
                OsrmProvider(self.settings),
                HaversineProvider(self.settings),
            ]
        self.providers = providers

    def route(
        self, from_lat: float, from_lon: float, to_lat: float, to_lon: float, profile: Profile
    ) -> RouteResult:
        errors: list[str] = []
        for provider in self.providers:
            if not provider.available():
                continue
            try:
                return provider.route(from_lat, from_lon, to_lat, to_lon, profile)
            except RoutingUnavailable as exc:
                errors.append(f"{provider.name}: {exc}")
            except Exception as exc:  # noqa: BLE001 - a provider fault must not end a search
                log.warning("routing provider %s raised: %s", provider.name, exc)
                errors.append(f"{provider.name}: {exc}")

        # The chain is built so this cannot normally happen: HaversineProvider needs
        # nothing but arithmetic. Reaching here means every provider was excluded.
        log.error("all routing providers failed: %s", "; ".join(errors))
        return HaversineProvider(self.settings).route(
            from_lat, from_lon, to_lat, to_lon, profile
        )

    def many_routes(
        self,
        from_lat: float,
        from_lon: float,
        targets: list[tuple[float, float]],
        profile: Profile,
        *,
        max_seconds: float = 1500.0,
    ) -> list[RouteResult]:
        """Route from one point to many. Falls back per target, never wholesale.

        The graph provider answers all targets in a single sweep when it can. Any
        target it cannot reach inside the budget is filled in by the straight-line
        estimate, so one unreachable candidate does not degrade the whole result set to
        estimates.
        """
        fallback = HaversineProvider(self.settings)
        results: list[RouteResult | None] = [None] * len(targets)

        for provider in self.providers:
            if not provider.available() or not hasattr(provider, "_ensure_router"):
                continue
            try:
                router = provider._ensure_router()  # noqa: SLF001 - same package
                results = router.many_costs(
                    from_lat, from_lon, targets, profile, max_seconds=max_seconds
                )
                break
            except RoutingUnavailable:
                continue
            except Exception as exc:  # noqa: BLE001
                log.warning("one-to-many routing failed: %s", exc)
                continue

        return [
            r if r is not None else fallback.route(from_lat, from_lon, t[0], t[1], profile)
            for r, t in zip(results, targets, strict=True)
        ]

    def close(self) -> None:
        for provider in self.providers:
            provider.close()

    @property
    def active_provider(self) -> str:
        for provider in self.providers:
            if provider.available():
                return provider.name
        return "none"
