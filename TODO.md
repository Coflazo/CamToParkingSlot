# ParkFit NL — Build Status

Live tracker. Updated as each phase lands.

**Legend:** `[x]` done and verified · `[~]` in progress · `[ ]` not started

---

## Progress

| Phase | Subsystem | Status |
|-------|-----------|--------|
| P0 | Foundation — repo, venv, CMake, task runner | `[x]` |
| P1 | C++ core — geo, index, fit, ranking | `[x]` |
| P2 | Data ingestion + storage | `[x]` |
| P3 | Search, routing, REST API | `[x]` |
| P4 | Availability prediction | `[~]` priors + λ decay live; learned model pending |
| P5 | C++ vision worker | `[x]` |
| P6 | Camera registry + source auditor | `[x]` |
| P7 | ML pipeline | `[ ]` |
| P8 | Web PWA | `[ ]` |
| P9 | Python test suite + evaluation harness | `[ ]` |
| P10 | Docs, ops, delivery | `[ ]` |

---

## What runs today

```powershell
.\tasks.ps1 setup      # venv, dependencies, CMake configure
.\tasks.ps1 build      # C++ core, vision, pybind11 module  -> 101 tests pass
uv run pf status       # what is loaded and how much data is present
uv run pf ingest all   # RDW + NDW + Amsterdam + OSM
uv run pf search "Rembrandt House Museum" --duration 150
.\tasks.ps1 serve      # API on :8000, docs at /docs
```

**Measured:** search warm median **146 ms**, p90 292 ms (target was p95 < 500 ms).
Kerb-gap measurement error on the replay fixture: **worst 1.0 cm** (target 25 cm MAE).

---

## P0 — Foundation `[x]`

- [x] Repo, `.gitignore`, git remote `Coflazo/CamToParkingSlot`
- [x] `uv` env, Python 3.12.10, all dependencies incl. torch CPU / onnxruntime / OpenCV
- [x] CMake auto-discovers MSVC 14.43 + bundled CMake + Ninja from VS Build Tools
- [x] `tasks.ps1` runner, hardened against PowerShell 5.1 native-stderr behaviour

## P1 — C++ core `[x]`

- [x] `geo/rd.hpp` — RD New ↔ WGS84, **validated against pyproj: 0.23 m, round-trip 1.5 cm**
- [x] `geo/polygon.hpp` — convex hull, rotating-calipers rectangle, quadrilateral bay measurement
- [x] `fit/` — facility / bay / kerb-gap fit with clamped safety floors
- [x] `rank/score.hpp` — survival decay, anti-herding, generalised cost, diversification
- [x] `index/grid.hpp` — uniform spatial grid, **93 µs/query over 250k entries**
- [x] pybind11 bindings — **bit-for-bit parity** with the Python implementations
- [x] Zero-dependency test harness (no Catch2 download needed)

## P2 — Data ingestion + storage `[x]`

- [x] SQLAlchemy 2.0 schema, portable SQLite / PostgreSQL, append-only observations
- [x] **RDW** — 8 Socrata tables joined; 4,871 facilities
- [x] **NDW** — live DATEX II occupancy, with validation (the feed is genuinely corrupt in places)
- [x] **Amsterdam** — **210,247 bay polygons** with layout, sign codes, time regimes
- [x] **OSM** — 4,863 car parks, 1,348 points of interest, 188,715-node road graph
- [x] **PDOK** — address geocoding
- [x] Provenance + licence registry on every record; source-priority conflict resolution

## P3 — Search, routing, API `[x]`

- [x] Hybrid geocoder — **18/18 on real destination names**
- [x] Routing: native A* + one-to-many Dijkstra → OSRM → straight-line
- [x] Search engine, 11 ordered steps, cross-source dedup, restriction evaluation
- [x] Batched pricing with provenance notes
- [x] FastAPI: 12 endpoints, JWT + Argon2id, SSE availability stream
- [x] In-memory recommendation ledger for anti-herding

## P4 — Availability prediction `[~]`

- [x] Exponential survival decay `P(eta) = P(now)·e^(−λt)`
- [x] Per-target-type base rates (0.15 metered bay, 0.62 facility)
- [x] `SegmentDynamics` table for learned per-segment, per-weekday, per-quarter-hour λ
- [ ] LightGBM occupancy model + synthetic history generator to bootstrap it

## P5 — C++ vision worker `[x]` — 37 tests

- [x] `frame.hpp` — owned buffer, explicit release
- [x] `health.hpp` — brightness, Laplacian sharpness, clipping, 64-bit gradient freeze hash
- [x] `homography.hpp` — normalised DLT + Jacobi + RANSAC; recovers a synthetic camera to millimetres
- [x] `gap.hpp` — ground-contact projection, kerb interval arithmetic, occlusion-aware confidence
- [x] `state_machine.hpp` — asymmetric transitions guarding the false-free rate
- [x] `source.hpp` — ffmpeg subprocess (HLS/RTSP/MJPEG/file) + deterministic replay
- [x] `detector.hpp` — sidecar backend for reproducible tests, null backend that admits it cannot detect
- [x] `publisher.hpp` — the privacy boundary: no pixels leave the process
- [x] `pf_cv_worker` — refuses an unauthorised live URL as a hard stop

## P6 — Camera registry + auditor `[x]`

- [x] Registry with a two-sided permission gate (dev accepts `robots_ok`, prod does not)
- [x] Ownership attestation requiring an agreement reference
- [x] Auditor: robots.txt honoured, headless rendering, media-request observation
- [x] `pf cameras add / list / enable / audit`

### What the audit actually found

Commercial webcam aggregators **do not expose stream URLs to a well-behaved client**.
Their players resolve manifests at runtime through endpoints their own robots files
frequently disallow. Rendering the page and observing its media requests — which is
reading, not circumvention — still yields nothing for `skylinewebcams.com`,
`livetraffic.eu` or `worldcams.tv`.

That is the finding, not a gap to route around. It is also why the product does not
depend on harvested feeds. The working path for a camera you hold rights to:

```
pf cameras add --id cam_017 --url <stream> --type hls --attest "permission ref 2026-014"
pf cameras enable cam_017
```

Verified robots verdicts (2026-08-26): `skylinewebcams.com` and `livetraffic.eu` allow
all agents; `worldcams.tv` disallows `/player`, `/ajax/`, `/go`, `/list/`.

## P7 — ML pipeline `[ ]`

- [ ] Synthetic parking-scene generator with exact ground-truth gap lengths
- [ ] Occupancy classifier + lightweight detector, PyTorch → ONNX
- [ ] ONNX Runtime C++ backend wired into `pf_cv_worker`

## P8 — Web PWA `[ ]`
## P9 — Python tests + `pf eval` metric table `[ ]`
## P10 — Docs, DPIA template, SBOM, compose files `[ ]`

---

## Corrections found by measurement

Each of these was a real defect caught by running the system, not by reading it.

| Finding | Why it mattered |
|---|---|
| **Parallel bays judged with perpendicular physics** | Rejected an ordinary Polo from the median 1.96 m Amsterdam kerb bay by 4 cm. 1846 of 2427 rejections near one destination were width. |
| **Enclosing rectangle used for skewed bays** | A Prinsengracht parallelogram (5.66 × 2.61 m at 48°) measured 7.40 × 1.89 m — neither dimension real. Fixed: quadrilaterals measured by their own edges. Polo usable bays 40% → **59.6%**. |
| **Marked bays given open-gap clearances** | 50 cm front + rear is manoeuvring room for reversing between two cars, not for a bay you park into. |
| **httpx `params={}` wipes a URL query** | Every "next page" fetch silently re-read page 1. Logs said 4000 fetched, 1000 created. |
| **One transaction across a 200k-row ingest** | A 403 at the pagination ceiling rolled back everything. Logs said "200,000 processed"; the database was empty. |
| **DATEX II parsed by subtree search** | Returned a sub-area's vacancy as the whole site's — a record has four `parkingNumberOfVacantSpaces` elements. |
| **NDW feed genuinely corrupt** | One site: capacity 210, reporting 1146 vacant and **−1046 occupied**. Now validated, with contradictions resolving pessimistically. |
| **Forward reachability ≠ connectivity** | One-way streets make the car graph directed; A* found no path between two nodes labelled the same component. Replaced with Tarjan SCC. |
| **Nearest-node snapping ignored connectivity** | Origin landed on an 11-node service island. |
| **SQL bbox scan for radius search** | 200 ms warm, 4 s cold. Moved to the C++ grid, as the architecture always specified. |
| **Recommendations written inside the request** | Every search blocked on a write it did not need. |
| **Unobserved bay treated as expired observation** | Probability driven to zero — yet those bays still ranked #1, because a free space with no chance still beat a paid one on price. |
| **PDOK scored without a relevance term** | "Van Gogh Allee" in Rhoon outranked the Van Gogh Museum. |
| **Overpass selectors omitted `relation`** | Van Gogh Museum, Johan Cruijff ArenA and Ziggo Dome were all missing — the largest venues are multipolygons. |
| **`read_numbers` returned the next array for a scalar key** | Every published event carried calibration version 102 for a calibration whose version is 3. |

---

## Known constraints

- **Docker daemon is down**, no Ubuntu WSL distro → PostGIS/OSRM are the optional upgrade path, never the critical path.
- **No licensed live Dutch kerb-camera feed exists.** See the audit finding above.
- **No anti-bot evasion.** The auditor honours robots.txt and stops where access is refused.
- Amsterdam's API caps plain pagination at 200,000 rows (403 at page 101); the full city needs the per-neighbourhood partitioned ingest.
