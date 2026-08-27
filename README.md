# CamToParkingSlot

Dutch parking search that answers one question properly: **will my car actually fit in that
space, and will it still be there when I arrive?**

Enter a destination and which car you are driving. You get parking ranked by what it really
costs you, drive time plus walk time plus price plus the risk of arriving to find it gone,
filtered to spaces your vehicle physically fits. Every claim carries its source and its age.

---

## Results

Measured by `pf evaluate`, not asserted. Full report in `docs/architecture/evaluation.json`.

| Metric | Measured | Target | |
|---|---:|---:|:--|
| False-free rate | **0.56 %** | ≤ 2 % | pass |
| Vacant precision | **99.13 %** | ≥ 98 % | pass |
| Vacant recall | **93.39 %** | ≥ 90 % | pass |
| False "it fits" rate | **0.00 %** | ≤ 2 % | pass |
| Gap-length MAE | **0.001 m** | ≤ 0.25 m | pass |
| Gap-length p95 error | **0.003 m** | ≤ 0.50 m | pass |
| Search latency p95 | **170 ms** | ≤ 500 ms | pass |

Over 3,000 fit trials, 4,000 frame samples, 60 rendered scenes and 12 search runs.

**False-free is the metric that matters.** It counts how often the system says a space is
free when it is not. Overall accuracy hides it: a detector that reports every space
occupied scores well on accuracy and is useless, while one that occasionally invents a free
space sends someone across a city for nothing. Ten of 1,225 truly-vacant trials were called
wrong in the unsafe direction.

### Vehicle detector

| | |
|---|---:|
| F1 | **0.994** |
| Precision | 0.996 |
| Recall | 0.992 |
| Box corner MAE | **0.81 px** |
| Parameters | 322,331 |

F1 by lighting condition: day 1.000, overcast 1.000, rain 1.000, dusk 1.000, glare 0.988,
night 0.974.

### Occupancy model

Brier score, lower is better. Split by target **and** by time, never at random.

| Held out | Rows | Model | Flat prior | Per kind | Per target | AUC |
|---|---:|---:|---:|---:|---:|---:|
| unseen time | 30,704 | **0.2005** | 0.2689 | 0.2192 | 0.2075 | 0.717 |
| unseen targets | 38,304 | **0.2127** | 0.2362 | 0.2255 | n/a | 0.648 |

The per-target column is the one worth reading. It is the best possible single number for
that specific bay, so beating it requires predicting time-of-day structure that no constant
can express.

---

## Three findings that shaped the build

**Amsterdam publishes every parking bay as a surveyed polygon.** 210,247 of them, with
layout, sign code and time regimes. That demotes computer vision from "find a legal space
and measure it", which is a research problem, to "is this known bay occupied", which is
tractable. Bay corners double as surveyed calibration points for the cameras, in the same
metric frame.

**The national address geocoder cannot find "Rembrandthuis".** PDOK indexes the address
register, not places, and returns zero results for the museum's name while nailing
"Jodenbreestraat 4". A parking app whose users type destinations needs a hybrid: OSM points
of interest first, addresses second. 18 of 18 real destination names now resolve correctly.

**Sparse polling destroys a decay-rate estimate.** A free space on a busy centre street has
a mean dwell of about five minutes. Fifteen-minute polling therefore recovers 70 % of the
true rate and thirty-minute polling 52 %, because turnovers that start and finish inside one
sample are invisible. That is a property of the feed, not the estimator, and it is why
municipal bay sensors report every minute.

---

## Get it

You need **git**, **Python 3.12+**, **Node 20+**, **ffmpeg**, and a C++20 compiler
(MSVC Build Tools on Windows, gcc or clang elsewhere). CMake and Ninja come bundled with
the VS Build Tools; on Linux and macOS install them from your package manager.

```bash
git clone https://github.com/Coflazo/CamToParkingSlot.git
cd CamToParkingSlot
```

Dependencies are managed with [uv](https://docs.astral.sh/uv/). Install it once:

```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Run it

The commands are the same everywhere. Only the way you invoke the task runner differs.

### Windows, PowerShell

```powershell
.\tasks.ps1 setup
.\tasks.ps1 build
.\tasks.ps1 test
```

### Windows, cmd.exe

```bat
powershell -ExecutionPolicy Bypass -File tasks.ps1 setup
powershell -ExecutionPolicy Bypass -File tasks.ps1 build
powershell -ExecutionPolicy Bypass -File tasks.ps1 test
```

### Windows, Git Bash or WSL

```bash
powershell.exe -ExecutionPolicy Bypass -File tasks.ps1 setup
powershell.exe -ExecutionPolicy Bypass -File tasks.ps1 build
```

### macOS and Linux

There is no PowerShell dependency; run the three steps directly.

```bash
uv sync --all-extras
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
uv run pytest tests -q
```

### Then, on any platform

```bash
uv run pf ingest all                                  # pull the open data
uv run pf search "Rembrandt House Museum" --duration 120
uv run pf evaluate                                    # the metric table above
uv run pf status                                      # what is loaded
```

The web app needs two terminals, because each command blocks:

```bash
# terminal 1
uv run uvicorn parkfit.api.app:app --host 127.0.0.1 --port 8000

# terminal 2
cd web && npm install && npm run dev
```

Then open **http://127.0.0.1:5173**. API docs are at **http://127.0.0.1:8000/docs**.

On Windows you can use `.\tasks.ps1 serve` and `.\tasks.ps1 web` instead.

The first search takes about four seconds while the 188,715-node road graph and the
spatial index load. Every search after that is around 200 ms.

### Machine learning

Either from the command line, or step by step with charts:

```bash
uv run pf detect all                # dataset, detector training, ONNX export
uv run pf predict all               # occupancy history, decay rates, model
uv run pf cameras discover          # every mapped camera in the Netherlands
uv run jupyter lab notebooks/       # the same pipelines, visual, one step at a time
```

No Docker required. PostGIS, Redis and OSRM are an optional upgrade path, never the
critical path.

---

## How it is put together

```
Web app  (Vite, TypeScript, MapLibre)
    |
FastAPI  search, ranking, vehicles, geocoding, availability stream
    |
    |-- parkfit_native (pybind11 -> C++)
    |      geo, spatial index, routing, fit engine, ranking, navigation links
    |-- SQLite by default, PostgreSQL and PostGIS when DATABASE_URL says so
    |-- ingest workers: RDW, NDW, PDOK, OSM, Amsterdam
                              ^
pf_cv_worker (C++)  ffmpeg -> health -> homography -> ONNX -> state machine
```

C++ handles everything on the hot path. The spatial grid answers a radius query over
250,000 bays in 93 µs; the same work as a SQL bounding-box scan took 200 ms warm and four
seconds cold.

**The fit engine is the product.** A bay is measured from its own polygon by pairing
opposite edges, not by its enclosing rectangle: a Prinsengracht bay 5.66 by 2.61 m at 48°
has an enclosing rectangle of 7.40 by 1.89 m, which matches neither dimension. Fixing that
took usable bays for a VW Polo from 40 % to 59.6 %.

Mirrors and bodywork are tracked separately, because mirrors overhang painted lines into
airspace that is nobody's bay. Parallel kerb bays get NEN 2443 clearances rather than
perpendicular ones: the median Amsterdam kerb bay is 1.96 m and a Polo is 1.75 m wide, so
demanding 25 cm of lateral margin rejects an ordinary car by four centimetres.

---

## Evidence, not guesses

Every response says where a number came from and how old it is. Sources are ranked and
never overwrite each other: operator feed, then camera, then municipal sensor, then user
report, then model, then the static register. All observations are kept; disagreement is
resolved on read.

A live source whose last observation is older than the staleness window stops being a live
source and is presented as stale. Showing a five-minute-old count as though it were current
is how a parking app teaches people not to trust it.

---

## Take me there

Tapping a result hands the space to Google Maps, Apple Maps, Waze, Yandex, OpenStreetMap or
whatever the device registers for `geo:` URIs, as **coordinates, never an address**. A
street string gets re-geocoded by the receiving app against its own database and lands
somewhere near, not somewhere exact. Coordinates go over at seven decimal places, about a
centimetre, so the format is never what limits accuracy.

For a car park the destination is the entrance where one is recorded, and the interface says
which point it is. A driver routed to a garage centroid arrives inside a building outline
and still has to find the ramp.

---

## What this does not claim

**The occupancy model is trained on simulated history.** The system has been ingesting live
data for a day, which is not enough to fit anything: a model of "how full is this street at
18:00 on a Friday" needs Fridays, plural. What the numbers above measure is whether the
model recovers latent demand structure it cannot see, which is a real estimation problem.
It is not a claim about real Amsterdam occupancy.

**There is no licensed live Dutch kerb-camera feed.** Commercial webcam aggregators do not
expose stream URLs to a well-behaved client; their players resolve manifests at runtime
through endpoints their own robots files disallow. Rendering the page and observing its
media requests, which is reading rather than circumvention, still yields nothing. That is
the finding, not a gap to route around, and it is why the product does not depend on
harvested feeds.

**The camera registry refuses by default.** A feed that has not been assessed does not run.
Production accepts only an explicit authorisation or an owner attestation; "the robots file
allowed it" is not a licence. Frames are processed in memory and discarded, and only
occupancy, geometry, confidence and timestamps are ever published. No face or plate
recognition, anywhere.

---

## Stack

C++20 for geometry, indexing, routing, fit, ranking and the vision worker. Python 3.12 with
FastAPI, SQLAlchemy 2.0 and pybind11. PyTorch to ONNX Runtime for detection, LightGBM for
occupancy. TypeScript and MapLibre for the web app. No OpenCV in the C++ path: the
homography, the Jacobi eigensolver, the RANSAC and the perceptual hashing are all here and
all tested.

314 tests. Data from RDW, NDW, PDOK, OpenStreetMap and the City of Amsterdam, all open,
each with its licence recorded in `docs/data_sources/sources.md`.
