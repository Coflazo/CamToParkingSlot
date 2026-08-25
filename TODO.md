# ParkFit NL — Build Status

Live tracker. Updated as each phase lands.

**Legend:** `[x]` done and verified · `[~]` in progress · `[ ]` not started

---

## Progress

| Phase | Subsystem | Status |
|-------|-----------|--------|
| P0 | Foundation — repo, venv, CMake, task runner | `[x]` done |
| P1 | C++ core — geo, index, fit, ranking | `[x]` done |
| P2 | Data ingestion + storage | `[ ]` |
| P3 | Search, routing, API | `[ ]` |
| P4 | Availability prediction | `[ ]` |
| P5 | C++ vision worker | `[ ]` |
| P6 | Camera registry + source auditor | `[ ]` |
| P7 | ML pipeline | `[ ]` |
| P8 | Web PWA | `[ ]` |
| P9 | Testing + evaluation harness | `[ ]` |
| P10 | Docs, ops, delivery | `[ ]` |

---

## P0 — Foundation `[x]`

- [x] Repository tree + `.gitignore` + `git init` → `Coflazo/CamToParkingSlot`
- [x] `pyproject.toml` with core / ml / audit / pg / dev extras
- [x] `uv` virtualenv, Python 3.12.10
- [x] Dependencies: fastapi, sqlalchemy, shapely, pyproj, torch 2.13 CPU, onnxruntime 1.29, OpenCV 5.0, LightGBM 4.7
- [x] `CMakeLists.txt` — auto-discovers MSVC 14.43 + bundled CMake + Ninja from VS Build Tools
- [x] `tasks.ps1` runner (setup / build / test / ingest / serve / web / eval / cv / audit)
- [x] PowerShell 5.1 native-stderr handling so CMake warnings do not abort the runner

## P1 — C++ core `[x]` — 63/63 tests pass

- [x] `geo/rd.hpp` — RD New (EPSG:28992) ↔ WGS84, **validated against pyproj: 0.23 m, round-trip 1.5 cm**
- [x] `geo/primitives.hpp` — haversine, bearing, offset, bbox, point-to-segment
- [x] `geo/polygon.hpp` — area, centroid, containment, convex hull, **rotating-calipers min-area rectangle**
- [x] `fit/vehicle.hpp` — vehicle profile with per-dimension provenance
- [x] `fit/vehicle_fit.hpp` — facility / bay / kerb-gap fit with clamped safety floors
- [x] `rank/score.hpp` — survival decay, anti-herding, generalised cost, diversification
- [x] `index/grid.hpp` — uniform spatial grid, **93 µs/query over 250k entries**
- [x] Zero-dependency test harness (`cpp/tests/test_framework.hpp`) — no Catch2 download needed

### Corrections found by the tests

| Finding | Impact |
|---|---|
| **Mirrors vs painted lines** — `check_bay` required bays wider than the *mirror* span. Mirrors legitimately overhang paint into neighbouring airspace. | Would have deleted most Dutch on-street supply from every search. Split into aperture width (mirrors) vs bay width (bodywork). |
| **Failure penalty too low** (9 min) | Ranking gambled on 4 %-probability kerb spaces. Raised to 14 min, matching observed European city-centre search times. |
| Two fabricated test constants (Amsterdam bay lat/lon, Amsterdam–Rotterdam distance) | Replaced with pyproj/Geod ground truth. |

## P2 — Data ingestion + storage `[ ]`

- [ ] SQLAlchemy 2.0 models + Alembic migrations (SQLite default, PostGIS dialect)
- [ ] RDW adapter — 8 verified Socrata datasets
- [ ] NDW adapter — live DATEX II truck-parking occupancy, emission zones, traffic signs, roadworks
- [ ] Amsterdam adapter — `parkeervakken` bay polygons (RD → WGS84)
- [ ] PDOK + OSM/Overpass adapters
- [ ] Provenance and licence tracking on every record
- [ ] Source-priority conflict resolver

## P3 — Search, routing, API `[ ]`
## P4 — Availability prediction `[ ]`
## P5 — C++ vision worker `[ ]`
## P6 — Camera registry + source auditor `[ ]`
## P7 — ML pipeline `[ ]`
## P8 — Web PWA `[ ]`
## P9 — Testing + evaluation `[ ]`
## P10 — Docs, ops, delivery `[ ]`

---

## Verified live data sources

| Source | What it gives | Status |
|---|---|---|
| `api.data.amsterdam.nl/v1/parkeervakken/` | Every Amsterdam bay as an RD polygon, Langs/Haaks, eType, regimes | reachable |
| `opendata.ndw.nu/Truckparking_Parking_Status.xml` | Live DATEX II v3 vacant-space counts, ~1 min refresh | reachable |
| RDW `t5pc-eb34` | 237 geolocated garages | reachable |
| RDW `b3us-f26s` | 3137 spec rows incl. `maximumvehicleheight` | reachable |
| RDW `m9d7-ebf2` | Vehicle lookup by licence plate | reachable |
| PDOK Locatieserver | Dutch address geocoding | reachable |
| Overpass (GET + UA header) | OSM parking + POIs | reachable |

---

## Known constraints

- **Docker daemon is down** and there is no Ubuntu WSL distro → PostGIS/OSRM are the optional upgrade path, never the critical path.
- **No licensed live Dutch kerb-camera feed exists.** Registry defaults to permissive for local research; production requires explicit authorisation.
- **No anti-bot evasion.** The auditor honours robots.txt. Verified: `skylinewebcams.com` and `livetraffic.eu` allow all crawlers; `worldcams.tv` disallows `/player`, `/ajax/`, `/go` → auto-marked `BLOCKED`.
