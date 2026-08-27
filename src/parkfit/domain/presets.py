"""A test fleet, with real dimensions from the Dutch vehicle register.

Every number here was read out of RDW's ``Gekentekende voertuigen`` dataset for an
actually-registered example of that model. Nothing is from a manufacturer brochure and
nothing is estimated, because the whole point of the fit engine is that a four-centimetre
error decides whether a car goes in a bay, and a brochure figure rounded to the nearest
five centimetres is not good enough to test that.

**Body types are the register's own, not marketing's.** RDW classifies by ``inrichting``,
and it has no category for "SUV". A BMW X5 and a Range Rover Evoque are both filed as
``stationwagen``, alongside a Skoda Octavia estate. That is not a mistake in the data; it
is what the register says, and the fit engine cares about the measurements rather than
what a showroom calls the shape. The ``segment`` field below carries the everyday word so
a person can find the car they mean.

**Width is bodywork.** RDW's ``breedte`` excludes mirrors, which matches how the fit
engine uses it: mirrors are checked against apertures, bodywork against painted lines.
Mirror span is added at 36 cm and flagged unconfirmed, because it varies far more between
models than bodywork does.

**Height is the vehicle, empty.** Anything on the roof is never in the register and is
exactly what turns a car that clears a barrier into one that does not.
"""

from __future__ import annotations

from dataclasses import dataclass

from parkfit.domain.vehicle import VehicleProfile


@dataclass(frozen=True)
class VehiclePreset:
    """One real vehicle, as the register describes it."""

    key: str
    label: str
    #: The everyday word: hatchback, sedan, estate, SUV, MPV, van, city car.
    segment: str
    #: RDW's own ``inrichting`` classification, kept because it differs from `segment`.
    rdw_body_type: str
    make: str
    model: str
    length_cm: float
    body_width_cm: float
    height_cm: float
    weight_kg: float
    is_ev: bool = False

    @property
    def width_with_mirrors_cm(self) -> float:
        """Bodywork plus roughly 18 cm of mirror per side."""
        return self.body_width_cm + 36.0

    def to_profile(self) -> VehicleProfile:
        profile = VehicleProfile(
            id=self.key,
            nickname=self.label,
            make=self.make,
            model=self.model,
            length_cm=self.length_cm,
            body_width_cm=self.body_width_cm,
            width_with_mirrors_cm=self.width_with_mirrors_cm,
            height_cm=self.height_cm,
            height_with_accessories_cm=self.height_cm,
            weight_kg=self.weight_kg,
            length_confirmed=True,
            width_confirmed=False,  # mirror span is inferred, not registered
            height_confirmed=True,
            weight_confirmed=True,
        )
        profile.is_ev = self.is_ev
        return profile


#: The fleet. One or more per segment, spanning a Toyota Aygo at 3.70 m to a Mercedes
#: Sprinter at 6.97 m, which is the range the fit engine has to get right.
PRESETS: tuple[VehiclePreset, ...] = (
    VehiclePreset(
        key="aygo",
        label="Toyota Aygo X",
        segment="city car",
        rdw_body_type="hatchback",
        make="Toyota",
        model="Aygo X",
        length_cm=370,
        body_width_cm=174,
        height_cm=153,
        weight_kg=1020,
    ),
    VehiclePreset(
        key="fiat500",
        label="Fiat 500",
        segment="city car",
        rdw_body_type="hatchback",
        make="Fiat",
        model="500",
        length_cm=363,
        body_width_cm=168,
        height_cm=153,
        weight_kg=1400,
    ),
    VehiclePreset(
        key="polo",
        label="Volkswagen Polo",
        segment="hatchback",
        rdw_body_type="hatchback",
        make="Volkswagen",
        model="Polo",
        length_cm=407,
        body_width_cm=175,
        height_cm=144,
        weight_kg=1196,
    ),
    VehiclePreset(
        key="zoe",
        label="Renault Zoe",
        segment="hatchback (electric)",
        rdw_body_type="hatchback",
        make="Renault",
        model="Zoe",
        length_cm=409,
        body_width_cm=173,
        height_cm=156,
        weight_kg=1577,
        is_ev=True,
    ),
    VehiclePreset(
        key="s60",
        label="Volvo S60",
        segment="sedan",
        rdw_body_type="sedan",
        make="Volvo",
        model="S60",
        length_cm=460,
        body_width_cm=180,
        height_cm=143,
        weight_kg=1566,
    ),
    VehiclePreset(
        key="a4avant",
        label="Audi A4 Avant",
        segment="estate",
        rdw_body_type="stationwagen",
        make="Audi",
        model="A4 Avant",
        length_cm=474,
        body_width_cm=184,
        height_cm=139,
        weight_kg=1570,
    ),
    VehiclePreset(
        key="octavia",
        label="Skoda Octavia Combi",
        segment="estate",
        rdw_body_type="stationwagen",
        make="Skoda",
        model="Octavia",
        length_cm=469,
        body_width_cm=183,
        height_cm=151,
        weight_kg=1350,
    ),
    VehiclePreset(
        key="v70",
        label="Volvo V70",
        segment="estate",
        rdw_body_type="stationwagen",
        make="Volvo",
        model="V70",
        length_cm=481,
        body_width_cm=186,
        height_cm=155,
        weight_kg=1668,
    ),
    VehiclePreset(
        key="evoque",
        label="Range Rover Evoque",
        segment="SUV",
        rdw_body_type="stationwagen",
        make="Land Rover",
        model="Range Rover Evoque",
        length_cm=437,
        body_width_cm=190,
        height_cm=165,
        weight_kg=2192,
    ),
    VehiclePreset(
        key="x5",
        label="BMW X5 xDrive45e",
        segment="large SUV",
        rdw_body_type="stationwagen",
        make="BMW",
        model="X5 xDrive45e",
        length_cm=492,
        body_width_cm=200,
        height_cm=175,
        weight_kg=2510,
    ),
    VehiclePreset(
        key="modely",
        label="Tesla Model Y",
        segment="SUV (electric)",
        rdw_body_type="MPV",
        make="Tesla",
        model="Model Y",
        length_cm=475,
        body_width_cm=192,
        height_cm=162,
        weight_kg=2054,
        is_ev=True,
    ),
    VehiclePreset(
        key="touran",
        label="Volkswagen Touran",
        segment="MPV",
        rdw_body_type="MPV",
        make="Volkswagen",
        model="Touran",
        length_cm=453,
        body_width_cm=183,
        height_cm=161,
        weight_kg=1464,
    ),
    VehiclePreset(
        key="transit",
        label="Ford Transit camper",
        segment="van",
        rdw_body_type="kampeerwagen",
        make="Ford",
        model="Transit",
        length_cm=670,
        body_width_cm=206,
        height_cm=277,
        weight_kg=2893,
    ),
    VehiclePreset(
        key="sprinter",
        label="Mercedes-Benz Sprinter",
        segment="large van",
        rdw_body_type="voor rolstoelen",
        make="Mercedes-Benz",
        model="Sprinter",
        length_cm=697,
        body_width_cm=202,
        height_cm=262,
        weight_kg=2797,
    ),
)

BY_KEY = {preset.key: preset for preset in PRESETS}


def get(key: str) -> VehiclePreset | None:
    return BY_KEY.get(key.lower())


def by_segment() -> dict[str, list[VehiclePreset]]:
    grouped: dict[str, list[VehiclePreset]] = {}
    for preset in PRESETS:
        grouped.setdefault(preset.segment, []).append(preset)
    return grouped
