"""Request and response models.

Every availability claim the API makes carries four things beside the number itself:
where it came from, when it was observed, how confident we are, and what wording the
client is permitted to show. That is not decoration; it is the difference between a
product that says "47 spaces, updated 23 seconds ago" and one that says "47 spaces" and
quietly means "47 spaces at some point this afternoon".
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class ApiModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
class RegisterRequest(ApiModel):
    email: EmailStr
    password: str = Field(min_length=10, max_length=200)
    display_name: str | None = Field(default=None, max_length=120)


class LoginRequest(ApiModel):
    email: EmailStr
    password: str


class TokenResponse(ApiModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int


class UserResponse(ApiModel):
    id: int
    email: str
    display_name: str | None
    max_walk_minutes: int
    prefer_covered: bool
    prefer_cheapest: bool
    value_of_time_eur_per_min: float
    store_search_history: bool


class UserPreferencesUpdate(ApiModel):
    max_walk_minutes: int | None = Field(default=None, ge=1, le=60)
    prefer_covered: bool | None = None
    prefer_cheapest: bool | None = None
    value_of_time_eur_per_min: float | None = Field(default=None, ge=0.0, le=5.0)
    store_search_history: bool | None = None


# ---------------------------------------------------------------------------
# Vehicles
# ---------------------------------------------------------------------------
class PlateLookupRequest(ApiModel):
    plate: str = Field(min_length=4, max_length=12)

    @field_validator("plate")
    @classmethod
    def strip_separators(cls, value: str) -> str:
        return value.replace("-", "").replace(" ", "").upper()


class PlateLookupResponse(ApiModel):
    """What RDW knows, and explicitly what it does not.

    ``unconfirmed_fields`` always contains ``height_cm``: the register does not publish
    vehicle height, and height is the dimension a barrier actually stops. The client is
    expected to ask for these rather than accept the record as complete.
    """

    found: bool
    make: str | None = None
    model: str | None = None
    length_cm: float = 0.0
    body_width_cm: float = 0.0
    width_with_mirrors_cm: float = 0.0
    weight_kg: float = 0.0
    fuel_type: str | None = None
    is_ev: bool = False
    unconfirmed_fields: list[str] = Field(default_factory=list)
    note: str = ""


class VehicleCreate(ApiModel):
    nickname: str = Field(min_length=1, max_length=80)
    make: str | None = None
    model: str | None = None
    length_cm: float = Field(gt=0, le=2500)
    body_width_cm: float = Field(gt=0, le=400)
    width_with_mirrors_cm: float = Field(default=0.0, ge=0, le=450)
    height_cm: float = Field(gt=0, le=500)
    height_with_accessories_cm: float = Field(default=0.0, ge=0, le=600)
    weight_kg: float = Field(default=0.0, ge=0, le=40000)
    is_ev: bool = False
    charging_connector: str | None = None
    has_trailer: bool = False
    has_roof_box: bool = False
    extra_parallel_clearance_cm: float = Field(default=0.0, ge=0, le=200)


class VehicleUpdate(ApiModel):
    nickname: str | None = None
    height_cm: float | None = Field(default=None, gt=0, le=500)
    height_with_accessories_cm: float | None = Field(default=None, ge=0, le=600)
    width_with_mirrors_cm: float | None = Field(default=None, ge=0, le=450)
    has_trailer: bool | None = None
    has_roof_box: bool | None = None
    extra_parallel_clearance_cm: float | None = Field(default=None, ge=0, le=200)


class VehicleResponse(ApiModel):
    id: int
    nickname: str
    make: str | None
    model: str | None
    length_cm: float
    body_width_cm: float
    width_with_mirrors_cm: float
    height_cm: float
    height_with_accessories_cm: float
    weight_kg: float
    is_ev: bool
    has_trailer: bool
    has_roof_box: bool
    extra_parallel_clearance_cm: float
    height_confirmed: bool
    width_confirmed: bool


# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------
class GeocodeResult(ApiModel):
    label: str
    lat: float
    lon: float
    kind: str
    source: str
    confidence: float
    city: str | None = None


class GeocodeResponse(ApiModel):
    query: str
    results: list[GeocodeResult]


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
class SearchPreferencesIn(ApiModel):
    max_walk_minutes: float = Field(default=12.0, ge=1, le=60)
    prefer_covered: bool = False
    prefer_cheapest: bool = False
    needs_ev_charging: bool = False
    needs_disabled_bay: bool = False
    include_on_street: bool = True
    value_of_time_eur_per_min: float = Field(default=0.20, ge=0.0, le=5.0)


class SearchCreate(ApiModel):
    destination: str = Field(min_length=2, max_length=200)
    vehicle_id: int | None = None
    origin_lat: float | None = Field(default=None, ge=-90, le=90)
    origin_lon: float | None = Field(default=None, ge=-180, le=180)
    arrival_time: datetime | None = None
    expected_duration_minutes: int = Field(default=120, ge=5, le=60 * 24 * 7)
    city_hint: str | None = None
    preferences: SearchPreferencesIn = Field(default_factory=SearchPreferencesIn)


class FitDetail(ApiModel):
    verdict: str
    slack_cm: float
    binding_constraint: str | None = None
    unverified: list[str] = Field(default_factory=list)
    explanation: str = ""


class LegDetail(ApiModel):
    distance_m: float
    duration_min: float
    provider: str
    is_estimate: bool
    geometry: list[list[float]] = Field(default_factory=list)


class EvidenceDetail(ApiModel):
    """Where an availability claim came from, and how old it is."""

    source: str
    observed_at: datetime | None
    age_seconds: float | None
    freshness: str
    confidence_label: str
    stale: bool
    conflicting_sources: int = 0
    vacant_spaces: int | None = None
    total_spaces: int | None = None


class RecommendationResponse(ApiModel):
    id: str
    kind: str
    name: str
    lat: float
    lon: float
    rank: int
    generalised_cost_eur: float
    probability_at_arrival: float
    price_eur: float
    price_note: str
    drive: LegDetail | None = None
    walk: LegDetail | None = None
    fit: FitDetail
    evidence: EvidenceDetail
    capacity: int | None = None
    max_height_cm: float | None = None
    bay_length_cm: float = 0.0
    bay_width_cm: float = 0.0
    orientation: str = ""
    restriction_warnings: list[str] = Field(default_factory=list)
    is_exact_space: bool = False
    expires_at: datetime | None = None


class SearchResponse(ApiModel):
    search_id: str
    destination: GeocodeResult | None
    results: list[RecommendationResponse]
    considered: int
    merged_duplicates: int
    rejected_illegal: int
    rejected_fit: int
    rejected_walk: int
    radius_m: float
    routing_provider: str
    warnings: list[str] = Field(default_factory=list)
    elapsed_ms: float


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------
class UserConfirmationRequest(ApiModel):
    target_kind: Literal["facility", "bay", "curb"]
    target_id: int
    outcome: Literal["parked", "was_occupied", "not_found", "illegal"]
    note: str | None = Field(default=None, max_length=500)
    recommendation_id: int | None = None


class ObservationResponse(ApiModel):
    accepted: bool
    message: str


# ---------------------------------------------------------------------------
# Facilities
# ---------------------------------------------------------------------------
class FacilityDetail(ApiModel):
    id: int
    name: str
    kind: str
    lat: float | None
    lon: float | None
    city: str | None
    capacity: int | None
    charging_capacity: int | None
    max_vehicle_height_cm: float | None
    source_name: str
    evidence: EvidenceDetail | None = None
    attribution: str | None = None


# ---------------------------------------------------------------------------
# Admin: cameras
# ---------------------------------------------------------------------------
class CameraCreate(ApiModel):
    camera_id: str = Field(min_length=2, max_length=80)
    stream_url: str
    stream_type: Literal["hls", "mjpeg", "rtsp", "snapshot", "file"]
    public_page_url: str | None = None
    owner: str | None = None
    operator: str | None = None
    lat: float | None = None
    lon: float | None = None
    heading_deg: float | None = None
    fixed_mount: bool = True
    #: Setting this to owner_attested is an assertion that you hold the rights to
    #: process this feed. It is recorded, and it is what production mode requires.
    permission_status: Literal[
        "authorised", "owner_attested", "robots_ok", "unverified", "blocked"
    ] = "unverified"
    licence_url: str | None = None
    agreement_reference: str | None = None
    sample_interval_s: float = Field(default=8.0, ge=1.0, le=300.0)
    notes: str | None = None


class CameraResponse(ApiModel):
    id: int
    camera_id: str
    owner: str | None
    stream_type: str | None
    lat: float | None
    lon: float | None
    permission_status: str
    enabled: bool
    technical_status: str
    last_frame_at: datetime | None
    may_process: bool = False
    blocking_reason: str | None = None


class CalibrationCreate(ApiModel):
    #: Image-space and world-space point pairs. World coordinates are RD New metres,
    #: which is what Amsterdam bay corners are already published in.
    image_points: list[list[float]] = Field(min_length=4)
    world_points_rd: list[list[float]] = Field(min_length=4)
    visible_bay_ids: list[str] = Field(default_factory=list)
    privacy_mask: list[list[list[float]]] = Field(default_factory=list)
    notes: str | None = None


class CalibrationResponse(ApiModel):
    id: int
    version: int
    reprojection_error_m: float
    max_error_m: float
    valid: bool
    created_at: datetime
    message: str = ""


class DataQualityResponse(ApiModel):
    id: int
    detected_at: datetime
    source_name: str
    severity: str
    kind: str
    detail: str
    resolved: bool


class HealthResponse(ApiModel):
    status: str
    version: str
    native_module: bool
    native_version: str | None
    database: str
    routing_provider: str
    facilities: int
    bays: int
    points_of_interest: int
    live_observations_last_hour: int
