# ParkFit NL

Find parking in the Netherlands that **actually fits your vehicle**, ranked by what the
trip will really cost you, with the evidence behind every claim.

You enter a destination and which of your cars you are driving. You get back parking
options filtered by whether the vehicle physically fits and is legally allowed there,
ranked by expected total inconvenience, drive time, walk time, price, and the risk that
the space is gone when you arrive, with each result showing where its information came
from and how old it is.

---

## The promise: stated precisely

This does **not** promise to always find you a free kerbside space. Nothing can: there is
no nationwide network of kerb-facing Dutch camera feeds available for reuse, and any
product built on the assumption that there is would be built on nothing.

What it promises is narrower and actually keepable:

> We rank the parking most likely to work for *your specific vehicle*, and we show you
> the fit, the price, the travel time, the freshness of the data and our confidence,
> so you can judge for yourself.

Concretely, that means the system will tell you it **does not know** rather than guess.
Of the 370 geolocated facilities in the national register, only 165 publish a height
limit, so `UNVERIFIED` is the common answer to "does my van fit", and the app says so
instead of quietly rounding it to yes.

---

## Quick start

Everything runs on a stock Windows machine. No Docker, no WSL, no cloud account.

```powershell
.\tasks.ps1 setup        # virtualenv, dependencies, CMake configure
.\tasks.ps1 build        # C++ core, vision worker, Python extension
.\tasks.ps1 test         # 101 C++ tests, 102 Python tests

uv run pf ingest all     # RDW, NDW, Amsterdam and OpenStreetMap
uv run pf ingest roads   # cache the routable road graph

uv run pf search "Rembrandt House Museum" --duration 150
.\tasks.ps1 serve        # API on :8000, interactive docs at /docs
.\tasks.ps1 web          # progressive web app on :5173
```

`.\tasks.ps1 build` finds MSVC, CMake and Ninja inside Visual Studio Build Tools, so
there is nothing else to install.

---

## What it measures: and what those numbers are

Run `uv run pf evaluate` to reproduce this table.

| Metric | Measured | Target | |
|---|---:|---:|---|
| **False-free rate** | 0.65 % | ≤ 2 % | pass |
| Vacant precision | 99.04 % | ≥ 98 % | pass |
| Vacant recall | 91.94 % | ≥ 90 % | pass |
| False "fits" rate | 0.00 % | ≤ 2 % | pass |
| Gap-length mean absolute error | 0.001 m | ≤ 0.25 m | pass |
| Gap-length 95th percentile error | 0.002 m | ≤ 0.50 m | pass |
| Search latency, p95 | 182 ms | ≤ 500 ms | pass |

**The false-free rate is the number that matters.** It is how often the system calls a
space free when it is not. Overall accuracy hides it entirely: a detector that reports
everything occupied and one that reports everything free score identically on accuracy,
and one of them is useless while the other is actively harmful.

Gap measurement is evaluated against synthetic scenes whose gap lengths are known to the
millimetre by construction, because ground truth otherwise means standing in a Dutch
street with a tape measure in six lighting conditions. That gives a floor that can be
regression-tested; real footage raises the ceiling.

---

## How it works

```
Progressive web app  (Vite · TypeScript · MapLibre)
        │  HTTP / server-sent events
FastAPI  ── search · ranking · vehicles · geocoding · camera admin
        │
        ├─ parkfit_native   (pybind11 → C++)
        │     RD↔WGS84 · spatial grid · vehicle fit · generalised-cost ranking
        ├─ SQLite  │  PostgreSQL + PostGIS  (auto-detected)
        └─ ingest workers → RDW · NDW · Amsterdam · PDOK · OpenStreetMap
                                   ▲
pf_cv_worker (C++) ── ffmpeg → frame health → homography → detect → temporal filter
                       → availability events
```

Python owns orchestration, I/O and anything that talks to a network. C++ owns the
arithmetic that runs per candidate on every search: coordinate transforms, the radius
sweep over 210,000 bays, vehicle fit and the ranking. That split is not decoration, it
is what removes the PostGIS and OSRM dependencies and lets the whole product run on a
laptop with nothing configured.

### Where the data comes from

| Source | What it gives | Licence |
|---|---|---|
| [Amsterdam `parkeervakken`](https://api.data.amsterdam.nl/v1/parkeervakken/) | **210,247 bay polygons** in RD New, with layout, sign code and time regimes | CC-BY-4.0 |
| [RDW open data](https://opendata.rdw.nl/) | National parking register: garages, park-and-ride, capacities, height limits, and vehicle lookup by plate | CC0 |
| [NDW](https://opendata.ndw.nu/) | Live DATEX II parking occupancy, environmental zones, roadworks | CC0 |
| [PDOK Locatieserver](https://www.pdok.nl/) | Dutch address geocoding | CC0 |
| [OpenStreetMap](https://www.openstreetmap.org/) | Points of interest, car parks outside the register, road network | **ODbL**: share-alike, attribution required |

The ODbL row is why every source carries its licence in the database rather than being
assumed equivalent. Attribution travels with the data.

---

## Three findings that shaped the design

**Amsterdam publishes every parking bay as an exact polygon.** This is the single most
valuable dataset in the project, and it changes what computer vision has to do. Without
it, a camera must answer "where is a legal space and how long is it", the research-grade
problem. With it, geometry is a solved data problem and vision is left with the tractable
question: *is this known bay occupied right now?* The bay corners double as surveyed
calibration points, in the same metric frame the geometry needs.

**The official Dutch geocoder cannot find the destinations people type.** Searching PDOK
for "Rembrandthuis" returns **zero** results; "Jodenbreestraat 4" returns an exact match.
PDOK indexes the address register, not places. A driver setting off for the Rembrandt
House does not know its street number, so the geocoder searches an OpenStreetMap
point-of-interest index first and falls back to PDOK. It resolves 18 of 18 real
destination names.

**There is no harvestable camera network.** Commercial webcam aggregators do not expose
stream URLs to a well-behaved client; their players resolve manifests at runtime through
endpoints their own robots files frequently disallow. The auditor honours robots.txt,
renders client-side pages and observes the media requests they make, reading, not
circumvention, and still finds nothing for the major aggregators. That is a finding, not
a gap to route around. The working path for a camera you hold rights to is two commands:

```
pf cameras add --id cam_017 --url <stream> --type hls --attest "permission ref 2026-014"
pf cameras enable cam_017
```

---

## Vehicle fit

The engine models the constraints separately because they are physically different:

| Constraint | Compared against | Why |
|---|---|---|
| Garage height barrier | Height **including roof box** | The rack is what hits the barrier, not the roof |
| Garage / aperture width | Width **across mirrors** | A wall does not yield |
| Marked bay width | **Bodywork** width | Paint is not a wall; mirrors overhang into neighbouring airspace |
| Parallel kerb bay | Bodywork + ~5 cm | No car beside you: one flank is pavement, the other the traffic lane |
| Perpendicular bay | Bodywork + 25 cm | A car each side, so you need door-opening room |
| Open kerb gap | Length + 50 cm each end | Manoeuvring room to reverse in between two cars |

Getting this wrong is not academic. Charging mirror width against a painted bay rejected
an ordinary Volkswagen Polo from the median 1.96 m Amsterdam kerb bay by four
centimetres, which silently deleted most of the city's on-street supply from every
search. Clearance floors are hard-coded and cannot be tuned away to manufacture results.

---

## Privacy

The camera subsystem is built so that what leaves the process is a number and a
timestamp. No pixels, no image-space boxes, no vehicle appearance, no plates, no faces.
Frames are released immediately after processing, before anything else runs.

These controls stay regardless of whether the project is commercial. GDPR has no
non-commercial exemption, and the household exemption explicitly excludes monitoring
public space, so the same rules apply to a hobby project and a service.

- No facial recognition, no licence-plate recognition, no demographic analysis
- Frames processed in memory and discarded; nothing written to disk in normal operation
- Only occupancy, geometry, confidence and timestamps persisted
- A licence plate is used once for the RDW lookup and then discarded, only the
  dimensions are kept
- Destination history is opt-in and off by default: it maps where somebody goes and when
- A worker refuses to open a feed the registry has not cleared, as a hard stop rather
  than a warning

See [`docs/privacy/`](docs/privacy/) for the DPIA template and the data-protection notes.

---

## Repository layout

```
cpp/          C++ core (geo, spatial index, fit, ranking) and vision worker
src/parkfit/  FastAPI service, ingest adapters, search engine, camera registry, ML
web/          Progressive web app
tests/        101 C++ cases, 102 Python cases, replay fixtures
docs/         Architecture, privacy, data sources, camera registry
scripts/      Maintenance utilities
```

`TODO.md` tracks build status per phase, including a table of every defect found by
measurement rather than by reading.

---

## Licence and attribution

MIT for this code. The data carries its own terms:

- Parking register: RDW / Nationaal Parkeer Register (CC0)
- Live occupancy: Nationaal Dataportaal Wegverkeer (CC0)
- Parking bays: Gemeente Amsterdam (CC-BY-4.0)
- Geocoding: PDOK Locatieserver, Kadaster / BZK (CC0)
- Map data: © OpenStreetMap contributors, **ODbL**
