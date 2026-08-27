"""FastAPI application.

Mounts the versioned API and exposes a health endpoint that reports what is actually
loaded, whether the native module is present, which routing provider is live, how much
data is in the database, rather than a bare "ok". A parking service that answers
"healthy" while its road graph failed to load is worse than one that admits it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import func, select

from parkfit import __version__
from parkfit.api.routers import auth, parking, search, vehicles
from parkfit.api.schemas import HealthResponse
from parkfit.config import get_settings
from parkfit.native import HAS_NATIVE, native_version
from parkfit.services.ledger import flush_ledger
from parkfit.storage.models import (
    AvailabilityObservation,
    ParkingBay,
    ParkingFacility,
    PointOfInterest,
)
from parkfit.storage.session import checkpoint, create_all, session_scope

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    settings = get_settings()
    create_all()
    # Fold any leftover write-ahead log back in before serving. Otherwise the first
    # request unlucky enough to cross the autocheckpoint threshold pays for it.
    stats = checkpoint(analyze=False)
    if stats.get("checkpoint_ms"):
        log.info("WAL checkpoint: %s ms", stats["checkpoint_ms"])
    log.info(
        "CamToParkingSlot %s starting | database=%s | native=%s",
        __version__,
        "postgres" if settings.is_postgres else "sqlite",
        HAS_NATIVE,
    )
    if not HAS_NATIVE:
        log.warning(
            "native module not built: vehicle fit and ranking fall back to Python. "
            "Run tasks.ps1 build."
        )
    flusher = asyncio.create_task(_flush_recommendations_periodically())
    try:
        yield
    finally:
        flusher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await flusher
        # One last drain so a clean shutdown does not lose the tail.
        await asyncio.to_thread(flush_ledger)
        log.info("CamToParkingSlot shutting down")


async def _flush_recommendations_periodically(interval_s: float = 20.0) -> None:
    """Persist buffered recommendations off the request path."""
    while True:
        try:
            await asyncio.sleep(interval_s)
            written = await asyncio.to_thread(flush_ledger)
            if written:
                log.debug("flushed %d recommendations", written)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("recommendation flush failed: %s", exc)


app = FastAPI(
    title="CamToParkingSlot",
    version=__version__,
    description=(
        "Vehicle-aware parking search for the Netherlands.\n\n"
        "Every availability claim carries its source, its observation time and a "
        "confidence label. The product promise is not that a space will always be "
        "found; it is that what is shown is what is known."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # The progressive web app runs on a separate dev origin. A production deployment
    # should narrow this to the real front-end origin.
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

v1 = APIRouter(prefix="/v1")
v1.include_router(auth.router)
v1.include_router(vehicles.router)
v1.include_router(search.router)
v1.include_router(parking.router)

try:
    from parkfit.api.routers import admin

    v1.include_router(admin.router)
except ImportError:  # pragma: no cover - admin router is optional
    log.info("admin router not available")

app.include_router(v1)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Report what is actually loaded, not merely that the process is up."""
    settings = get_settings()
    one_hour_ago = (datetime.now(UTC) - timedelta(hours=1)).replace(tzinfo=None)

    with session_scope() as session:
        facilities = (
            session.execute(
                select(func.count()).select_from(ParkingFacility).where(ParkingFacility.active)
            ).scalar()
            or 0
        )
        bays = session.execute(select(func.count()).select_from(ParkingBay)).scalar() or 0
        pois = session.execute(select(func.count()).select_from(PointOfInterest)).scalar() or 0
        live = (
            session.execute(
                select(func.count())
                .select_from(AvailabilityObservation)
                .where(AvailabilityObservation.observed_at >= one_hour_ago)
            ).scalar()
            or 0
        )

    from parkfit.routing.provider import RoutingService

    routing = RoutingService(settings=settings)
    provider = routing.active_provider
    routing.close()

    return HealthResponse(
        status="ok",
        version=__version__,
        native_module=HAS_NATIVE,
        native_version=native_version(),
        database="postgres" if settings.is_postgres else "sqlite",
        routing_provider=provider,
        facilities=facilities,
        bays=bays,
        points_of_interest=pois,
        live_observations_last_hour=live,
    )


@app.get("/", include_in_schema=False)
def index() -> JSONResponse:
    return JSONResponse(
        {
            "name": "CamToParkingSlot",
            "version": __version__,
            "docs": "/docs",
            "health": "/health",
            "api": "/v1",
            "attribution": [
                "Parking register: RDW / Nationaal Parkeer Register",
                "Live occupancy: Nationaal Dataportaal Wegverkeer (NDW)",
                "Parking bays: Gemeente Amsterdam",
                "Geocoding: PDOK Locatieserver (Kadaster / BZK)",
                "Map data: (c) OpenStreetMap contributors, ODbL",
            ],
        }
    )
