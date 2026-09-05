"""Database schema.

Two decisions shape this file.

**Geometry is stored portably, not in PostGIS types.** Points live as indexed
``lat``/``lon`` floats and polygons as GeoJSON text, so the identical schema runs on
SQLite and PostgreSQL. Radius search does not need a database index at all; it is
answered by the C++ spatial grid, which sweeps 250k bays in under 100 microseconds.
PostGIS therefore becomes an optional accelerator for ad-hoc spatial SQL rather than a
hard dependency that stops the product running on a laptop.

**Observations are append-only.** ``AvailabilityObservation`` is never updated or
deleted. When several sources disagree about the same facility we keep every claim and
resolve the current state on read, ordered by :class:`EvidenceSource`. Overwriting would
destroy the audit trail that lets us answer the only question that matters after a bad
recommendation: which source told us that, and when.
"""

from __future__ import annotations

import enum
from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class EvidenceSource(enum.IntEnum):
    """Ordered weakest to strongest. The integer value *is* the priority.

    Mirrors ``parkfit::rank::EvidenceSource`` in the C++ core; the two must stay in
    step, which the contract tests enforce.
    """

    OSM_ONLY = 0
    STATIC_DATABASE = 1
    PREDICTIVE_MODEL = 2
    USER_CONFIRMATION = 3
    MUNICIPAL_SENSOR = 4
    CAMERA_OBSERVATION = 5
    OPERATOR_FEED = 6


class FacilityKind(enum.StrEnum):
    GARAGE = "garage"
    SURFACE_LOT = "surface_lot"
    PARK_AND_RIDE = "park_and_ride"
    ON_STREET_ZONE = "on_street_zone"
    TRUCK_PARKING = "truck_parking"
    UNKNOWN = "unknown"


class BayOrientation(enum.StrEnum):
    """Matches the Amsterdam ``parkeervakken`` ``type`` field."""

    PARALLEL = "parallel"  # Langs
    PERPENDICULAR = "perpendicular"  # Haaks
    ANGLED = "angled"  # Visgraat
    UNKNOWN = "unknown"


class OccupancyState(enum.StrEnum):
    VACANT = "vacant"
    OCCUPIED = "occupied"
    VACANT_GAP = "vacant_gap"
    UNKNOWN = "unknown"


class CameraPermission(enum.StrEnum):
    """Whether a feed may be processed automatically.

    ``ROBOTS_OK`` means the host permits crawling and no prohibition was found, which is
    enough for local research but not for a production deployment. The distinction is
    enforced by :mod:`parkfit.cameras.registry`, not by convention.
    """

    AUTHORISED = "authorised"  # written agreement or explicit open licence
    OWNER_ATTESTED = "owner_attested"  # operator attests they hold the rights
    ROBOTS_OK = "robots_ok"  # crawlable, terms unverified: dev use only
    UNVERIFIED = "unverified"
    BLOCKED = "blocked"  # robots.txt or terms forbid automated access


class FrameHealth(enum.StrEnum):
    HEALTHY = "healthy"
    DARK = "dark"
    BLURRED = "blurred"
    OBSTRUCTED = "obstructed"
    FROZEN = "frozen"
    POSE_CHANGED = "pose_changed"
    OFFLINE = "offline"


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
class SourceLicence(Base):
    """Machine-readable registry of every upstream dataset and its reuse terms.

    Recording this is not paperwork. OpenStreetMap is ODbL and carries share-alike
    obligations; RDW vehicle data is public domain; individual camera feeds are neither.
    A product that mixes them has to know which is which before it publishes anything.
    """

    __tablename__ = "source_licences"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_name: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    dataset_url: Mapped[str] = mapped_column(Text)
    licence: Mapped[str] = mapped_column(String(120))
    licence_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    attribution_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    commercial_use: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    share_alike: Mapped[bool] = mapped_column(Boolean, default=False)
    refresh_frequency: Mapped[str | None] = mapped_column(String(60), nullable=True)
    data_contact: Mapped[str | None] = mapped_column(String(200), nullable=True)
    schema_version: Mapped[str | None] = mapped_column(String(40), nullable=True)
    last_reviewed: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class ProvenanceMixin:
    """Every ingested row records where it came from and when."""

    source_name: Mapped[str] = mapped_column(String(120), index=True, default="")
    source_record_id: Mapped[str | None] = mapped_column(String(160), nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    source_updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


# ---------------------------------------------------------------------------
# Parking supply
# ---------------------------------------------------------------------------
class AreaManager(Base, ProvenanceMixin):
    """RDW *gebiedsbeheerder*: the municipality or operator running an area."""

    __tablename__ = "area_managers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    area_manager_id: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    valid_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Populated from the RDW index dataset: where this operator publishes live data.
    static_data_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    dynamic_data_url: Mapped[str | None] = mapped_column(Text, nullable=True)


class ParkingFacility(Base, ProvenanceMixin):
    """A garage, surface lot, park-and-ride site or on-street tariff zone."""

    __tablename__ = "parking_facilities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    external_id: Mapped[str] = mapped_column(String(160), index=True)
    area_manager_id: Mapped[str | None] = mapped_column(String(40), index=True, nullable=True)

    name: Mapped[str] = mapped_column(String(300), default="")
    kind: Mapped[str] = mapped_column(String(30), default=FacilityKind.UNKNOWN.value, index=True)

    lat: Mapped[float | None] = mapped_column(Float, nullable=True, index=True)
    lon: Mapped[float | None] = mapped_column(Float, nullable=True, index=True)
    geometry_geojson: Mapped[str | None] = mapped_column(Text, nullable=True)

    street: Mapped[str | None] = mapped_column(String(200), nullable=True)
    house_number: Mapped[str | None] = mapped_column(String(40), nullable=True)
    postcode: Mapped[str | None] = mapped_column(String(20), nullable=True)
    city: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    province: Mapped[str | None] = mapped_column(String(80), nullable=True)

    capacity: Mapped[int | None] = mapped_column(Integer, nullable=True)
    charging_capacity: Mapped[int | None] = mapped_column(Integer, nullable=True)
    disabled_capacity: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # A NULL limit means "not published", which is deliberately distinct from "unlimited".
    # The fit engine reports UNVERIFIED rather than inventing a value.
    max_vehicle_height_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_vehicle_width_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_vehicle_length_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_vehicle_weight_kg: Mapped[float | None] = mapped_column(Float, nullable=True)

    limited_access: Mapped[bool] = mapped_column(Boolean, default=False)
    open_all_year: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    exit_possible_all_day: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    # The `_eur` in these two names is historical, from when the pilot was Dutch-only.
    # The actual unit is whatever `currency` says, and a reader must not assume euros:
    # an Istanbul lot stores its hourly rate here in lira. Renaming them would rename a
    # feature the occupancy model was trained on, which is not worth doing for cosmetics.
    tariff_eur_per_hour: Mapped[float | None] = mapped_column(Float, nullable=True)
    tariff_day_max_eur: Mapped[float | None] = mapped_column(Float, nullable=True)
    tariff_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: ISO 4217 for the two tariff columns above. Prices are never converted between
    #: currencies: a search happens in one city, so every candidate it compares shares a
    #: currency, and inventing an exchange rate to rank across them would be a made-up
    #: number in the middle of the ranking.
    currency: Mapped[str] = mapped_column(String(3), default="EUR")
    #: ISO 3166-1 alpha-2. Chooses the legal rulebook and the clearance standard, and it
    #: is stored rather than inferred from coordinates so a wrong guess cannot apply one
    #: country's road law to another's streets.
    country: Mapped[str] = mapped_column(String(2), default="NL", index=True)

    geocode_precision: Mapped[str | None] = mapped_column(String(30), nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)

    entrances: Mapped[list[FacilityEntrance]] = relationship(
        back_populates="facility", cascade="all, delete-orphan"
    )
    opening_hours: Mapped[list[OpeningHours]] = relationship(
        back_populates="facility", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("source_name", "external_id", name="uq_facility_source_external"),
        Index("ix_facility_bbox", "lat", "lon"),
        Index("ix_facility_active_kind", "active", "kind"),
    )


class FacilityEntrance(Base):
    """Where a car actually enters.

    Routing to a facility centroid is a classic parking-app bug: an Amsterdam garage
    centroid can sit inside a block, on the far side of a canal, or in a pedestrian
    zone. The drive leg must target the entrance, and the walk leg starts from the exit.
    """

    __tablename__ = "facility_entrances"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    facility_id: Mapped[int] = mapped_column(
        ForeignKey("parking_facilities.id", ondelete="CASCADE"), index=True
    )
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    is_entry: Mapped[bool] = mapped_column(Boolean, default=True)
    is_exit: Mapped[bool] = mapped_column(Boolean, default=True)
    label: Mapped[str | None] = mapped_column(String(120), nullable=True)

    facility: Mapped[ParkingFacility] = relationship(back_populates="entrances")


class OpeningHours(Base):
    """One access window, from the RDW PARKING TOEGANG dataset."""

    __tablename__ = "opening_hours"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    facility_id: Mapped[int] = mapped_column(
        ForeignKey("parking_facilities.id", ondelete="CASCADE"), index=True
    )
    weekday: Mapped[int] = mapped_column(Integer)  # 0 = Monday
    # Minutes past midnight. An end past 1440 encodes a window running into the next
    # day, so overnight parking arithmetic stays a comparison instead of a special case.
    open_minute: Mapped[int] = mapped_column(Integer)
    close_minute: Mapped[int] = mapped_column(Integer)

    facility: Mapped[ParkingFacility] = relationship(back_populates="opening_hours")


class ParkingBay(Base, ProvenanceMixin):
    """A single marked parking bay.

    Sourced from the Amsterdam ``parkeervakken`` dataset, which publishes an exact
    polygon per bay in RD New. Dimensions are computed once at ingest by the C++
    rotating-calipers routine and stored, so a search never re-derives geometry.
    """

    __tablename__ = "parking_bays"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    external_id: Mapped[str] = mapped_column(String(80), index=True)

    lat: Mapped[float] = mapped_column(Float, index=True)
    lon: Mapped[float] = mapped_column(Float, index=True)
    # Kept in RD metres: lengths measured here are true metres, whereas measuring in
    # WGS84 would fold in datum offset and cosine-latitude distortion.
    geometry_rd_json: Mapped[str] = mapped_column(Text)

    street: Mapped[str | None] = mapped_column(String(200), nullable=True, index=True)
    neighbourhood_code: Mapped[str | None] = mapped_column(String(20), nullable=True)

    orientation: Mapped[str] = mapped_column(
        String(20), default=BayOrientation.UNKNOWN.value, index=True
    )
    # Conservative usable dimensions (area / extent). See geo.shapes.measure_bay:
    # the enclosing rectangle overstates a trapezoidal bay, in the optimistic
    # direction, for the number that decides whether a car fits.
    length_cm: Mapped[float] = mapped_column(Float, default=0.0)
    width_cm: Mapped[float] = mapped_column(Float, default=0.0)
    # Maximum extent, kept for display and for auditing the difference.
    max_length_cm: Mapped[float] = mapped_column(Float, default=0.0)
    max_width_cm: Mapped[float] = mapped_column(Float, default=0.0)
    # How rectangular the bay is. Below ~0.85 the two measures diverge enough that the
    # fit verdict carries less confidence.
    fill_ratio: Mapped[float] = mapped_column(Float, default=1.0)
    angle_rad: Mapped[float] = mapped_column(Float, default=0.0)
    bay_count: Mapped[int] = mapped_column(Integer, default=1)

    fiscal: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    sign_code: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)
    regimes_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    curb_segment_id: Mapped[int | None] = mapped_column(
        ForeignKey("curb_segments.id", ondelete="SET NULL"), nullable=True, index=True
    )

    __table_args__ = (
        UniqueConstraint("source_name", "external_id", name="uq_bay_source_external"),
        Index("ix_bay_bbox", "lat", "lon"),
    )


class PointOfInterest(Base, ProvenanceMixin):
    """A named place a user might type as a destination.

    This table exists because the official Dutch geocoder cannot answer the question
    users actually ask. PDOK indexes the address register; searching it for
    "Rembrandthuis" returns nothing at all, while "Jodenbreestraat 4" returns an exact
    match. People do not know the address of the museum they are driving to, so the
    OpenStreetMap point-of-interest layer is indexed here and searched first.
    """

    __tablename__ = "points_of_interest"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    external_id: Mapped[str] = mapped_column(String(80), index=True)

    name: Mapped[str] = mapped_column(String(300), index=True)
    #: Lower-cased, punctuation-stripped form used for matching.
    normalised_name: Mapped[str] = mapped_column(String(300), index=True)
    category: Mapped[str] = mapped_column(String(60), index=True, default="place")

    lat: Mapped[float] = mapped_column(Float, index=True)
    lon: Mapped[float] = mapped_column(Float, index=True)
    #: ISO 3166-1 alpha-2 for the area this point was ingested from.
    #:
    #: Without it the index leaks across borders in a way that looks like a match. The
    #: table held only Dutch places, a search for "Tour Eiffel" scored the word "Tour"
    #: against an Amsterdam boat-tour company, and the product answered a French query
    #: with a canal in the Netherlands at 0.42 confidence. A wrong answer carrying a
    #: plausible score is worse than no answer at all.
    country: Mapped[str] = mapped_column(String(2), default="NL", index=True)

    city: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    street: Mapped[str | None] = mapped_column(String(200), nullable=True)
    house_number: Mapped[str | None] = mapped_column(String(40), nullable=True)
    postcode: Mapped[str | None] = mapped_column(String(20), nullable=True)
    #: Alternate names (name:en, name:nl, alt_name, short_name) as a JSON array.
    aliases_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: Rough prominence, used to break ties between same-named places.
    importance: Mapped[float] = mapped_column(Float, default=0.5)

    __table_args__ = (
        UniqueConstraint("source_name", "external_id", name="uq_poi_source_external"),
        Index("ix_poi_name_city", "normalised_name", "city"),
    )


class CurbSegment(Base, ProvenanceMixin):
    """A stretch of legally parkable kerb, as a centreline.

    Kerb gaps are measured along this line, which is why it is stored as an ordered
    polyline in RD metres rather than as an area.
    """

    __tablename__ = "curb_segments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    external_id: Mapped[str] = mapped_column(String(80), index=True)
    street: Mapped[str | None] = mapped_column(String(200), nullable=True, index=True)

    lat: Mapped[float] = mapped_column(Float, index=True)
    lon: Mapped[float] = mapped_column(Float, index=True)
    centreline_rd_json: Mapped[str] = mapped_column(Text)
    length_m: Mapped[float] = mapped_column(Float, default=0.0)
    usable_width_m: Mapped[float | None] = mapped_column(Float, nullable=True)

    side: Mapped[str | None] = mapped_column(String(10), nullable=True)  # left / right
    orientation: Mapped[str] = mapped_column(String(20), default=BayOrientation.PARALLEL.value)


class ParkingRestriction(Base, ProvenanceMixin):
    """A time- or vehicle-dependent rule attached to a facility, bay or kerb segment.

    Deliberately generic: the same table carries permit zones, disabled-only bays,
    loading windows, market-day closures and temporary roadworks, because a driver
    experiences all of them identically, as "you may not park here right now".
    """

    __tablename__ = "parking_restrictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    target_kind: Mapped[str] = mapped_column(String(20), index=True)  # facility|bay|curb
    target_id: Mapped[int] = mapped_column(Integer, index=True)

    rule_type: Mapped[str] = mapped_column(String(40), index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    weekday_mask: Mapped[int] = mapped_column(Integer, default=0b1111111)
    start_minute: Mapped[int] = mapped_column(Integer, default=0)
    end_minute: Mapped[int] = mapped_column(Integer, default=1440)
    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    valid_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    max_duration_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    permit_required: Mapped[bool] = mapped_column(Boolean, default=False)
    disabled_only: Mapped[bool] = mapped_column(Boolean, default=False)
    ev_only: Mapped[bool] = mapped_column(Boolean, default=False)
    forbids_parking: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = (Index("ix_restriction_target", "target_kind", "target_id"),)


# ---------------------------------------------------------------------------
# Users and vehicles
# ---------------------------------------------------------------------------
class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    display_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    max_walk_minutes: Mapped[int] = mapped_column(Integer, default=12)
    prefer_covered: Mapped[bool] = mapped_column(Boolean, default=False)
    prefer_cheapest: Mapped[bool] = mapped_column(Boolean, default=False)
    value_of_time_eur_per_min: Mapped[float] = mapped_column(Float, default=0.20)

    # Opt-in, defaulting to off. Destination history is unusually revealing, it maps
    # where somebody goes and when, so it is never collected by default.
    store_search_history: Mapped[bool] = mapped_column(Boolean, default=False)

    vehicles: Mapped[list[Vehicle]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class Vehicle(Base):
    """A confirmed vehicle profile.

    The licence plate is *not* stored by default. It is used once to query RDW and then
    discarded, because the dimensions are the only thing the product needs and a plate
    is a direct identifier for a person.
    """

    __tablename__ = "vehicles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)

    nickname: Mapped[str] = mapped_column(String(80), default="")
    make: Mapped[str | None] = mapped_column(String(80), nullable=True)
    model: Mapped[str | None] = mapped_column(String(120), nullable=True)

    length_cm: Mapped[float] = mapped_column(Float, default=0.0)
    body_width_cm: Mapped[float] = mapped_column(Float, default=0.0)
    width_with_mirrors_cm: Mapped[float] = mapped_column(Float, default=0.0)
    height_cm: Mapped[float] = mapped_column(Float, default=0.0)
    height_with_accessories_cm: Mapped[float] = mapped_column(Float, default=0.0)
    weight_kg: Mapped[float] = mapped_column(Float, default=0.0)

    length_confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    width_confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    height_confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    weight_confirmed: Mapped[bool] = mapped_column(Boolean, default=False)

    fuel_type: Mapped[str | None] = mapped_column(String(40), nullable=True)
    emission_class: Mapped[str | None] = mapped_column(String(20), nullable=True)
    is_ev: Mapped[bool] = mapped_column(Boolean, default=False)
    charging_connector: Mapped[str | None] = mapped_column(String(40), nullable=True)
    has_trailer: Mapped[bool] = mapped_column(Boolean, default=False)
    has_roof_box: Mapped[bool] = mapped_column(Boolean, default=False)

    extra_parallel_clearance_cm: Mapped[float] = mapped_column(Float, default=0.0)

    # Only populated when the user opts in for plate-based payment integrations.
    plate_encrypted: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    user: Mapped[User] = relationship(back_populates="vehicles")


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------
class CameraSource(Base, ProvenanceMixin):
    """A registered camera feed.

    No worker may open a stream that is not represented here with an acceptable
    ``permission_status``. Being visible on a web page is not the same as being
    licensed for automated analysis, and this table is where that difference is
    recorded rather than assumed.
    """

    __tablename__ = "camera_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    camera_id: Mapped[str] = mapped_column(String(80), unique=True, index=True)

    owner: Mapped[str | None] = mapped_column(String(200), nullable=True)
    operator: Mapped[str | None] = mapped_column(String(200), nullable=True)
    public_page_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    stream_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    stream_type: Mapped[str | None] = mapped_column(
        String(30), nullable=True
    )  # hls|mjpeg|rtsp|snapshot|file

    lat: Mapped[float | None] = mapped_column(Float, nullable=True, index=True)
    lon: Mapped[float | None] = mapped_column(Float, nullable=True, index=True)
    heading_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    field_of_view_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    fixed_mount: Mapped[bool] = mapped_column(Boolean, default=True)

    permission_status: Mapped[str] = mapped_column(
        String(30), default=CameraPermission.UNVERIFIED.value, index=True
    )
    commercial_reuse_allowed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    automated_processing_allowed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    licence_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    agreement_reference: Mapped[str | None] = mapped_column(String(200), nullable=True)
    robots_allowed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    terms_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    privacy_contact: Mapped[str | None] = mapped_column(String(200), nullable=True)
    retention_rules: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_legal_review: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    sample_interval_s: Mapped[float] = mapped_column(Float, default=8.0)
    technical_status: Mapped[str] = mapped_column(String(30), default=FrameHealth.OFFLINE.value)
    last_frame_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    calibrations: Mapped[list[CameraCalibration]] = relationship(
        back_populates="camera", cascade="all, delete-orphan"
    )


class CameraCalibration(Base):
    """An image-to-world homography plus the evidence that it is still valid."""

    __tablename__ = "camera_calibrations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    camera_source_id: Mapped[int] = mapped_column(
        ForeignKey("camera_sources.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer, default=1)

    homography_json: Mapped[str] = mapped_column(Text)
    control_points_json: Mapped[str] = mapped_column(Text)
    reprojection_error_m: Mapped[float] = mapped_column(Float, default=0.0)
    max_error_m: Mapped[float] = mapped_column(Float, default=0.0)

    # Reference keypoints used to detect that the camera has been moved or knocked.
    pose_reference_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    visible_bay_ids_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    privacy_mask_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    valid: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    camera: Mapped[CameraSource] = relationship(back_populates="calibrations")

    __table_args__ = (
        UniqueConstraint("camera_source_id", "version", name="uq_calibration_version"),
    )


# ---------------------------------------------------------------------------
# Observations and predictions
# ---------------------------------------------------------------------------
class AvailabilityObservation(Base):
    """Append-only record of what some source claimed, and when.

    Never updated. Never deleted outside retention policy. Conflicts between sources
    are resolved on read by :data:`evidence_source` priority, so the history of what we
    were told stays intact.
    """

    __tablename__ = "availability_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    target_kind: Mapped[str] = mapped_column(String(20), index=True)  # facility|bay|curb
    target_id: Mapped[int] = mapped_column(Integer, index=True)

    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    evidence_source: Mapped[int] = mapped_column(Integer, index=True)
    state: Mapped[str] = mapped_column(String(20), default=OccupancyState.UNKNOWN.value)

    vacant_spaces: Mapped[int | None] = mapped_column(Integer, nullable=True)
    occupied_spaces: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_spaces: Mapped[int | None] = mapped_column(Integer, nullable=True)
    occupancy_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)

    gap_length_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    gap_width_m: Mapped[float | None] = mapped_column(Float, nullable=True)

    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    camera_id: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    calibration_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(60), nullable=True)
    frame_health: Mapped[str | None] = mapped_column(String(30), nullable=True)
    source_name: Mapped[str] = mapped_column(String(120), default="", index=True)

    __table_args__ = (
        Index("ix_obs_target_time", "target_kind", "target_id", "observed_at"),
        Index("ix_obs_source_time", "evidence_source", "observed_at"),
    )


class AvailabilityPrediction(Base):
    """Modelled probability that a target is free at a future moment."""

    __tablename__ = "availability_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    target_kind: Mapped[str] = mapped_column(String(20), index=True)
    target_id: Mapped[int] = mapped_column(Integer, index=True)

    prediction_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    probability_available: Mapped[float] = mapped_column(Float, default=0.0)
    lambda_per_min: Mapped[float] = mapped_column(Float, default=0.0)
    model_version: Mapped[str | None] = mapped_column(String(60), nullable=True)

    __table_args__ = (Index("ix_pred_target_time", "target_id", "prediction_time"),)


class SegmentDynamics(Base):
    """Estimated vacancy decay rate for one street segment in one time bucket.

    Lambda is the rate at which a visible free space disappears. It varies enormously:
    a canal-side street at 18:00 on a Friday empties in under two minutes, the same
    street at 04:00 holds a space for hours. Storing it per segment, weekday and
    15-minute bucket is what makes the exponential survival model defensible.
    """

    __tablename__ = "segment_dynamics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    target_kind: Mapped[str] = mapped_column(String(20), index=True)
    target_id: Mapped[int] = mapped_column(Integer, index=True)

    weekday: Mapped[int] = mapped_column(Integer)
    quarter_hour: Mapped[int] = mapped_column(Integer)  # 0..95

    lambda_per_min: Mapped[float] = mapped_column(Float, default=0.0)
    base_occupancy: Mapped[float] = mapped_column(Float, default=0.0)
    sample_count: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        UniqueConstraint(
            "target_kind", "target_id", "weekday", "quarter_hour", name="uq_segment_bucket"
        ),
    )


class Recommendation(Base):
    """A short-lived record of what we told one user.

    Its purpose is anti-herding: before offering an exact space we count how many other
    live recommendations already point at it and decay its probability accordingly.
    Without this the app manufactures its own congestion by sending every driver in the
    neighbourhood to the single space it happens to be able to see.
    """

    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    search_id: Mapped[str] = mapped_column(String(64), index=True)
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)

    target_kind: Mapped[str] = mapped_column(String(20))
    target_id: Mapped[int] = mapped_column(Integer, index=True)
    rank: Mapped[int] = mapped_column(Integer, default=0)

    generalised_cost: Mapped[float] = mapped_column(Float, default=0.0)
    probability_at_eta: Mapped[float] = mapped_column(Float, default=0.0)
    confidence_label: Mapped[str] = mapped_column(String(40), default="")
    fit_verdict: Mapped[str] = mapped_column(String(20), default="")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)

    __table_args__ = (Index("ix_recommendation_active", "target_id", "expires_at"),)


class UserConfirmation(Base):
    """Ground truth from the only sensor that is always present: the driver."""

    __tablename__ = "user_confirmations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    target_kind: Mapped[str] = mapped_column(String(20))
    target_id: Mapped[int] = mapped_column(Integer, index=True)

    outcome: Mapped[str] = mapped_column(String(30))  # parked|was_occupied|not_found|illegal
    reported_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    recommendation_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)


class DataQualityIncident(Base):
    """A logged disagreement, staleness event or upstream schema change."""

    __tablename__ = "data_quality_incidents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    source_name: Mapped[str] = mapped_column(String(120), index=True)
    severity: Mapped[str] = mapped_column(String(20), default="warning")
    kind: Mapped[str] = mapped_column(String(60), index=True)
    detail: Mapped[str] = mapped_column(Text)
    target_kind: Mapped[str | None] = mapped_column(String(20), nullable=True)
    target_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
