"""Shared test fixtures.

Every test runs against a fresh temporary SQLite database. The real one holds 210,000
Amsterdam bays and a road graph, which makes it both slow to work against and dangerous
to mutate; tests that quietly depend on production data pass on this machine and fail
everywhere else.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

# Point every setting at the temporary tree before anything imports the config.
os.environ.setdefault("PARKFIT_ENVIRONMENT", "dev")


@pytest.fixture(scope="session", autouse=True)
def _isolate_settings() -> Iterator[None]:
    """Redirect the database and cache into a scratch directory for the whole run.

    Deliberately a repo-local directory rather than ``tmp_path_factory``: the system
    temp directory is not writable on every machine this has to run on, and a test suite
    that cannot start is worse than one that leaves a folder behind. It is gitignored
    and wiped at the start of each run.
    """
    data_dir = Path(__file__).resolve().parent.parent / "data" / "test-run"
    if data_dir.exists():
        shutil.rmtree(data_dir, ignore_errors=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    os.environ["PARKFIT_DATA_DIR"] = str(data_dir)
    os.environ["PARKFIT_DATABASE_URL"] = f"sqlite+aiosqlite:///{(data_dir / 'test.db').as_posix()}"

    from parkfit.config import get_settings
    from parkfit.storage.session import reset_engines

    get_settings.cache_clear()
    reset_engines()
    yield
    reset_engines()


@pytest.fixture
def scratch_dir(_isolate_settings) -> Iterator[Path]:
    """A writable scratch directory, standing in for ``tmp_path``.

    Same reason as ``_isolate_settings``: pytest's own temp factory cannot create its
    root on this machine, so a test that asks for ``tmp_path`` errors before it runs.
    """
    root = Path(os.environ["PARKFIT_DATA_DIR"]) / "scratch"
    root.mkdir(parents=True, exist_ok=True)
    yield root


@pytest.fixture
def session(_isolate_settings) -> Iterator:
    """A clean database for one test."""
    from parkfit.storage.session import create_all, drop_all, session_scope

    drop_all()
    create_all()
    with session_scope() as s:
        yield s


@pytest.fixture
def polo():
    """A Volkswagen Polo, as RDW returns it for a real plate."""
    from parkfit.domain.vehicle import VehicleProfile

    return VehicleProfile(
        id="veh_polo",
        nickname="Polo",
        make="Volkswagen",
        model="Polo",
        length_cm=405.3,
        body_width_cm=175.1,
        width_with_mirrors_cm=194.0,
        height_cm=145.1,
        height_with_accessories_cm=145.1,
        weight_kg=1105.0,
        length_confirmed=True,
        width_confirmed=True,
        height_confirmed=True,
        weight_confirmed=True,
    )


@pytest.fixture
def tall_van():
    """A van with a roof rack: the case height filtering exists for."""
    from parkfit.domain.vehicle import VehicleProfile

    return VehicleProfile(
        id="veh_van",
        nickname="Transporter",
        length_cm=590.0,
        body_width_cm=190.4,
        width_with_mirrors_cm=246.0,
        height_cm=199.0,
        height_with_accessories_cm=232.0,
        weight_kg=2000.0,
        length_confirmed=True,
        width_confirmed=True,
        height_confirmed=True,
    )


@pytest.fixture
def amsterdam_bay_ring() -> list[list[float]]:
    """A real Amsterdam bay polygon, verbatim from the parkeervakken API.

    Bay 110675492544 on Abidjanweg. A trapezoid, which is why it exercises the
    measurement path that an enclosing rectangle gets wrong.
    """
    return [
        [110677.64, 492542.17],
        [110672.28, 492543.33],
        [110670.76, 492545.68],
        [110678.06, 492544.12],
    ]


@pytest.fixture
def prinsengracht_ring() -> list[list[float]]:
    """A skewed Amsterdam canal bay: a 48-degree parallelogram, 5.66 by 2.61 m."""
    return [
        [120629.3, 487100.4],
        [120633.2, 487096.3],
        [120633.0, 487093.7],
        [120629.1, 487097.8],
    ]


@pytest.fixture
def seeded_facilities(session):
    """Three facilities near the Rembrandt House, with differing data completeness."""
    from parkfit.storage.models import ParkingFacility

    rows = [
        ParkingFacility(
            source_name="RDW-NPR",
            external_id="2460:363_BANK",
            name="Garage The Bank (Amsterdam)",
            kind="garage",
            lat=52.36620,
            lon=4.89860,
            city="Amsterdam",
            capacity=110,
            max_vehicle_height_cm=210.0,
            active=True,
        ),
        ParkingFacility(
            source_name="RDW-NPR",
            external_id="2448:363_BIJ",
            name="Garage De Bijenkorf (Amsterdam)",
            kind="garage",
            lat=52.37383,
            lon=4.89518,
            city="Amsterdam",
            capacity=None,
            max_vehicle_height_cm=None,
            active=True,  # height not published
        ),
        ParkingFacility(
            source_name="RDW-NPR",
            external_id="2448:363_LOW",
            name="Garage Laag (Amsterdam)",
            kind="garage",
            lat=52.36700,
            lon=4.90000,
            city="Amsterdam",
            capacity=60,
            max_vehicle_height_cm=180.0,
            active=True,  # too low for a van
        ),
    ]
    session.add_all(rows)
    # Commit rather than flush: the API opens its own sessions, and rows still inside
    # this test's transaction are invisible to them.
    session.commit()
    return rows


@pytest.fixture
def seeded_bays(session):
    """Marked bays spanning the layouts and sizes that matter for fit."""
    import json

    from parkfit.storage.models import ParkingBay

    def bay(
        external_id, street, orientation, length_cm, width_cm, fiscal=True, lat=52.3690, lon=4.9010
    ):
        return ParkingBay(
            source_name="Amsterdam-Parkeervakken",
            external_id=external_id,
            street=street,
            orientation=orientation,
            length_cm=length_cm,
            width_cm=width_cm,
            max_length_cm=length_cm,
            max_width_cm=width_cm,
            fill_ratio=0.98,
            fiscal=fiscal,
            lat=lat,
            lon=lon,
            geometry_rd_json=json.dumps([[0, 0], [1, 0], [1, 1], [0, 1]]),
        )

    rows = [
        bay("bay_langs_ok", "Jodenbreestraat", "parallel", 580.0, 205.0),
        bay("bay_langs_narrow", "Jodenbreestraat", "parallel", 580.0, 178.0),
        bay("bay_langs_short", "Waterlooplein", "parallel", 430.0, 210.0),
        bay("bay_haaks_ok", "Waterlooplein", "perpendicular", 500.0, 250.0),
        bay("bay_haaks_narrow", "Waterlooplein", "perpendicular", 500.0, 198.0),
        bay("bay_free", "Nieuwe Herengracht", "parallel", 600.0, 220.0, fiscal=False),
    ]
    session.add_all(rows)
    # Commit rather than flush: the API opens its own sessions, and rows still inside
    # this test's transaction are invisible to them.
    session.commit()
    return rows


@pytest.fixture
def recent_observation():
    """Build an availability observation at a chosen age."""
    from parkfit.storage.models import AvailabilityObservation, EvidenceSource, OccupancyState

    def make(
        target_kind: str,
        target_id: int,
        *,
        age_s: float = 10.0,
        evidence: EvidenceSource = EvidenceSource.OPERATOR_FEED,
        state: OccupancyState = OccupancyState.VACANT,
        vacant: int | None = 12,
    ):
        observed = datetime.now(UTC) - timedelta(seconds=age_s)
        return AvailabilityObservation(
            target_kind=target_kind,
            target_id=target_id,
            observed_at=observed.replace(tzinfo=None),
            evidence_source=int(evidence),
            state=state.value,
            vacant_spaces=vacant,
            occupied_spaces=None,
            total_spaces=None,
            confidence=0.95,
            source_name="test",
        )

    return make
