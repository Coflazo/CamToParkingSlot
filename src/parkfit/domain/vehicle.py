"""Vehicle profiles and the RDW licence-plate lookup.

RDW publishes the national vehicle register openly, and a plate lookup returns make,
model, length, width, mass and fuel type. It does **not** reliably publish height, which
is the single most important dimension for parking: a height barrier is the constraint
that physically stops a vehicle. So the flow is lookup, then *confirm*: the user is
asked for height, mirror width and anything on the roof, and only confirmed values are
treated as known.

The plate itself is discarded after the lookup. It is a direct identifier for a person
and the product only needs the dimensions; keeping it would create a licence-plate
database as a side effect of a parking search.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from parkfit.ingest.base import BaseAdapter, IngestResult, SourceMeta, parse_float

log = logging.getLogger(__name__)

RDW_VEHICLE_DATASET = "m9d7-ebf2"
_PLATE_CLEAN = re.compile(r"[^A-Z0-9]")

#: Typical extra height of common roof accessories, in centimetres. Offered as defaults
#: in the confirmation step; the user can always override.
ACCESSORY_HEIGHT_CM = {
    "roof_box": 40.0,
    "roof_rack": 12.0,
    "bicycle_carrier_roof": 55.0,
    "none": 0.0,
}


def normalise_plate(plate: str) -> str:
    """Strip separators and upper-case. Dutch plates are written ``XT-994-N``."""
    return _PLATE_CLEAN.sub("", (plate or "").upper())


@dataclass
class VehicleProfile:
    """A vehicle as the fit engine sees it. All lengths in centimetres."""

    id: str = ""
    nickname: str = ""
    make: str | None = None
    model: str | None = None

    length_cm: float = 0.0
    body_width_cm: float = 0.0
    width_with_mirrors_cm: float = 0.0
    height_cm: float = 0.0
    height_with_accessories_cm: float = 0.0
    weight_kg: float = 0.0

    length_confirmed: bool = False
    width_confirmed: bool = False
    height_confirmed: bool = False
    weight_confirmed: bool = False

    fuel_type: str | None = None
    emission_class: str | None = None
    is_ev: bool = False
    charging_connector: str | None = None
    has_trailer: bool = False
    has_roof_box: bool = False

    extra_parallel_clearance_cm: float = 0.0
    unconfirmed_fields: list[str] = field(default_factory=list)

    def to_native(self) -> Any:
        """Convert to the C++ vehicle struct, or a stand-in when unbuilt."""
        from parkfit.native import native

        if native is None:
            return self
        v = native.Vehicle()
        v.id = self.id
        v.nickname = self.nickname
        v.length_cm = self.length_cm
        v.body_width_cm = self.body_width_cm
        v.width_with_mirrors_cm = self.width_with_mirrors_cm
        v.height_cm = self.height_cm
        v.height_with_accessories_cm = self.height_with_accessories_cm
        v.weight_kg = self.weight_kg
        v.is_ev = self.is_ev
        v.has_trailer = self.has_trailer
        v.has_roof_box = self.has_roof_box
        v.extra_parallel_clearance_cm = self.extra_parallel_clearance_cm
        provenance = native.DimensionProvenance()
        provenance.length_confirmed = self.length_confirmed
        provenance.width_confirmed = self.width_confirmed
        provenance.height_confirmed = self.height_confirmed
        provenance.weight_confirmed = self.weight_confirmed
        v.provenance = provenance
        return v

    @property
    def effective_height_cm(self) -> float:
        return self.height_with_accessories_cm or self.height_cm

    @property
    def effective_width_cm(self) -> float:
        if self.width_with_mirrors_cm > 0:
            return self.width_with_mirrors_cm
        return self.body_width_cm + 36.0 if self.body_width_cm > 0 else 0.0

    @property
    def ready_for_search(self) -> bool:
        """Enough is known to give a meaningful answer rather than a shrug."""
        return self.length_cm > 0 and (self.body_width_cm > 0 or self.width_with_mirrors_cm > 0)


class RdwVehicleClient(BaseAdapter):
    """Looks up a vehicle in the RDW open register by licence plate."""

    meta = SourceMeta(
        name="RDW-Voertuigen",
        url="https://opendata.rdw.nl/resource/m9d7-ebf2.json",
        licence="CC0-1.0 / Public Domain",
        licence_url="https://www.rdw.nl/over-rdw/dienstverlening/open-data",
        attribution="Vehicle data: RDW",
        commercial_use=True,
        share_alike=False,
        refresh="daily",
        notes="Height is not reliably published and must be confirmed by the owner.",
    )

    def lookup(self, plate: str) -> VehicleProfile | None:
        """Resolve a plate to a partially-filled profile.

        Returns ``None`` when the plate is unknown. Every dimension RDW omits is listed
        in :attr:`VehicleProfile.unconfirmed_fields` so the UI can ask for exactly what
        is missing instead of re-asking for everything.
        """
        cleaned = normalise_plate(plate)
        if not cleaned or len(cleaned) > 8:
            return None

        rows = self.fetch_json(
            f"{self.settings.rdw_base_url}/{RDW_VEHICLE_DATASET}.json", {"kenteken": cleaned}
        )
        if not rows:
            return None
        row = rows[0]

        # RDW publishes these in centimetres.
        length = parse_float(row.get("lengte"))
        width = parse_float(row.get("breedte"))
        mass = parse_float(row.get("massa_ledig_voertuig"))
        fuel = row.get("brandstof_omschrijving") or row.get("aandrijving")

        profile = VehicleProfile(
            nickname=str(row.get("handelsbenaming") or cleaned).title()[:80],
            make=(row.get("merk") or "").title() or None,
            model=(row.get("handelsbenaming") or "").title() or None,
            length_cm=length or 0.0,
            body_width_cm=width or 0.0,
            weight_kg=mass or 0.0,
            fuel_type=str(fuel) if fuel else None,
            emission_class=row.get("europese_voertuigcategorie"),
            length_confirmed=bool(length),
            width_confirmed=bool(width),
            weight_confirmed=bool(mass),
        )
        profile.is_ev = bool(fuel and "elektr" in str(fuel).lower())

        if width:
            # Mirrors add roughly 18 cm per side on a passenger car. Offered as a
            # starting point, flagged unconfirmed, because mirror span varies far more
            # between models than bodywork does.
            profile.width_with_mirrors_cm = width + 36.0

        missing = []
        if not length:
            missing.append("length_cm")
        if not width:
            missing.append("body_width_cm")
        # Height is essentially never present in this dataset, and it is the dimension
        # that decides whether a car clears a barrier, so it is always asked for.
        missing.append("height_cm")
        missing.append("width_with_mirrors_cm")
        profile.unconfirmed_fields = missing
        return profile

    def run(self, **kwargs: Any) -> IngestResult:
        """Plate lookup is on demand; there is nothing to bulk-ingest."""
        result = IngestResult(source=self.meta.name)
        result.skipped = 1
        return result


def confirm_dimensions(
    profile: VehicleProfile,
    *,
    height_cm: float | None = None,
    width_with_mirrors_cm: float | None = None,
    accessory: str | None = None,
    accessory_height_cm: float | None = None,
    has_trailer: bool | None = None,
    extra_parallel_clearance_cm: float | None = None,
) -> VehicleProfile:
    """Apply owner-confirmed dimensions and clear the corresponding flags."""
    if height_cm and height_cm > 0:
        profile.height_cm = height_cm
        profile.height_confirmed = True
        profile.unconfirmed_fields = [f for f in profile.unconfirmed_fields if f != "height_cm"]

    if width_with_mirrors_cm and width_with_mirrors_cm > 0:
        profile.width_with_mirrors_cm = width_with_mirrors_cm
        profile.width_confirmed = True
        profile.unconfirmed_fields = [
            f for f in profile.unconfirmed_fields if f != "width_with_mirrors_cm"
        ]

    extra = accessory_height_cm
    if extra is None and accessory:
        extra = ACCESSORY_HEIGHT_CM.get(accessory, 0.0)
    if extra is not None and profile.height_cm > 0:
        profile.height_with_accessories_cm = profile.height_cm + max(0.0, extra)
        profile.has_roof_box = extra > 0
    elif profile.height_cm > 0 and profile.height_with_accessories_cm <= 0:
        profile.height_with_accessories_cm = profile.height_cm

    if has_trailer is not None:
        profile.has_trailer = has_trailer
    if extra_parallel_clearance_cm is not None:
        profile.extra_parallel_clearance_cm = max(0.0, extra_parallel_clearance_cm)
    return profile
