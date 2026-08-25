"""Runtime configuration.

Everything is overridable by environment variable or a local ``.env``. The defaults are
chosen so that a fresh clone runs with no configuration at all: SQLite on disk, the
native routing engine, and the camera registry in research mode.
"""

from __future__ import annotations

import enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"


class Environment(str, enum.Enum):
    DEV = "dev"
    PROD = "prod"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PARKFIT_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    environment: Environment = Environment.DEV
    debug: bool = True

    # --- storage -----------------------------------------------------------
    # A Postgres URL switches the dialect layer to PostGIS automatically; nothing else
    # in the application needs to know which engine it is talking to.
    database_url: str = Field(default="")
    data_dir: Path = DEFAULT_DATA_DIR

    # --- upstream data sources --------------------------------------------
    rdw_base_url: str = "https://opendata.rdw.nl/resource"
    rdw_app_token: str | None = None
    ndw_base_url: str = "https://opendata.ndw.nu"
    pdok_base_url: str = "https://api.pdok.nl/bzk/locatieserver/search/v3_1"
    amsterdam_base_url: str = "https://api.data.amsterdam.nl/v1"
    overpass_url: str = "https://overpass-api.de/api/interpreter"

    # Overpass asks that automated clients identify themselves; some mirrors reject
    # requests without a User-Agent outright, which is why this is not optional.
    user_agent: str = "ParkFitNL/0.1 (+https://github.com/Coflazo/CamToParkingSlot)"
    http_timeout_s: float = 45.0
    http_max_retries: int = 3

    # --- routing -----------------------------------------------------------
    osrm_url: str | None = None
    # Straight-line distance times this approximates real street distance. 1.35 is the
    # usual figure for dense European grids; Amsterdam runs a little higher because of
    # the canals, which is exactly why the native graph router is preferred when built.
    detour_factor: float = 1.42
    drive_speed_kmh: float = 24.0
    walk_speed_kmh: float = 4.8

    # --- search ------------------------------------------------------------
    default_search_radius_m: float = 800.0
    max_search_radius_m: float = 5000.0
    max_candidates: int = 400
    max_results: int = 10
    max_results_per_group: int = 2

    # --- freshness ---------------------------------------------------------
    stale_after_s: float = 300.0
    exact_space_ttl_s: float = 45.0

    # --- cameras -----------------------------------------------------------
    # In dev, a feed whose host permits crawling may be processed locally for research.
    # In prod, only an explicit authorisation or owner attestation will do. This is the
    # single switch that separates "trying it on my laptop" from "running a service".
    camera_allow_robots_ok: bool = True
    camera_frame_interval_s: float = 8.0
    camera_persist_frames: bool = False

    # --- auth --------------------------------------------------------------
    jwt_secret: str = "dev-secret-change-me"
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60 * 24 * 14

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return f"sqlite+aiosqlite:///{(self.data_dir / 'parkfit.db').as_posix()}"

    @property
    def sync_database_url(self) -> str:
        """The same database, through a synchronous driver (Alembic, CLI, ingest)."""
        url = self.resolved_database_url
        return url.replace("+aiosqlite", "").replace("+asyncpg", "+psycopg")

    @property
    def is_postgres(self) -> bool:
        return self.resolved_database_url.startswith(("postgresql", "postgres"))

    @property
    def is_production(self) -> bool:
        return self.environment is Environment.PROD

    def cache_dir(self, name: str) -> Path:
        p = self.data_dir / "cache" / name
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
