# CamToParkingSlot

### Every parking app tells you a space exists. This one draws your car in it.

![A Volvo S60 drawn to scale in a Waterlooplein bay, with 36 cm to spare and the tightest clearance labelled at the width](docs/images/fit.png)

That is a real Amsterdam bay and a real Volvo S60, both at their published dimensions,
both from Dutch open data. The 36 cm is subtraction. Bay and car are drawn at one scale,
so the picture cannot flatter the fit.

Width is what is tight here. The bay is 2.21 m, the S60 is 1.80 m across the bodywork, and
a parallel bay asks for only 5 cm of lateral margin in total, which leaves 36 cm. Length is
nowhere near as close, at 54 cm clear off each end.

---

## Results

I got these by running `pf evaluate`. Full output in `docs/architecture/evaluation.json`.

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

The one I watch is the false-free rate, which counts how often the system says a space is
free when it is not. Accuracy on its own hides that. A detector that calls every space
occupied scores well on accuracy and is useless, and one that invents a free space now and
then sends someone across a city for nothing. 10 of 1,225 truly vacant trials came out
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

Brier score, lower is better. Split by target and by time, never at random.

| Held out | Rows | Model | Flat prior | Per kind | Per target | AUC |
|---|---:|---:|---:|---:|---:|---:|
| unseen time | 30,704 | **0.2005** | 0.2689 | 0.2192 | 0.2075 | 0.717 |
| unseen targets | 38,304 | **0.2127** | 0.2362 | 0.2255 | n/a | 0.648 |

The per-target column is the one to compare against. It is the best a single constant can
do for that specific bay, so beating it means the model picked up time-of-day structure
that no constant can express.

---

## Three things I did not expect

Amsterdam publishes every parking bay as a surveyed polygon. All 210,247 of them, with the
layout, the sign code and the time regimes. That changed the plan. I had assumed the hard
part was finding a legal space and measuring it with computer vision, which is a research
problem. With the polygons in hand, the question is only whether a known bay is occupied,
which is doable. The bay corners are surveyed points in the same metric frame too, so the
cameras can be calibrated against them.

The national address geocoder cannot find "Rembrandthuis". PDOK indexes the address
register rather than places, so it returns nothing for the museum's name and gets
"Jodenbreestraat 4" exactly right. People type destinations, so the geocoder has to be
hybrid: OSM points of interest first, addresses second. 18 of 18 real destination names
resolve now.

Polling slowly wrecks the decay-rate estimate. A free space on a busy centre street lasts
about five minutes on average. Poll every fifteen minutes and you recover 70 % of the true
rate, every thirty and you get 52 %, because a turnover that starts and finishes between
two samples never happened as far as you can tell. That is the feed's fault rather than
the estimator's, and it is why municipal bay sensors report every minute.

---

## The interface

![The opening screen: "Your car. That bay. Measured." in display type over a dimmed map of Amsterdam](docs/images/hero.png)

![The working state: search console, ranked results with evidence badges and fit diagrams, over a live map](docs/images/app.png)

You type where you are going and pick which car. Results are ranked by what parking
actually costs you, meaning drive time plus walk time plus price plus the chance it is
gone when you arrive, and filtered to spaces the car fits.

The line under the search box reads "308 considered within 800 m, ruled out: 103 too large
for your vehicle, 87 not permitted". Switch the car to a Sprinter and the kerb bays vanish,
because a 5.7 m bay cannot take a 7 m van.

<img src="docs/images/mobile.png" alt="The same search on a phone: the console stacks, the status pill drops, and the results keep their fit diagrams" width="300">

I checked it at 320, 390, 412, 768, 1024, 1440, 1920 and 2560 pixels wide, plus landscape
phones. Three things actually broke. At 320 px the page overflowed, because the status pill
and the Vehicles button will not share a row. The search field collapsed to 50 px beside
its own submit button. And a 568 px-tall viewport could not hold the headline and the
console at once, so on short screens the supporting copy goes and the console stays.

---

## Get it

You need git, Python 3.12+, Node 20+, ffmpeg, and a C++20 compiler (MSVC Build Tools on
Windows, gcc or clang elsewhere). CMake and Ninja come bundled with the VS Build Tools; on
Linux and macOS install them from your package manager.

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

The steps are the same everywhere. Only the way you invoke the task runner differs.

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

No PowerShell needed, so run the steps directly.

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

Then open http://127.0.0.1:5173. API docs are at http://127.0.0.1:8000/docs.

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

No Docker required. PostGIS, Redis and OSRM are an optional upgrade path.

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

The spatial grid answers a radius query over 250,000 bays in 93 µs. I measured the same
query as a SQL bounding-box scan at 200 ms warm and about four seconds cold, which is why
the grid is there.

Bay size comes from the polygon itself, by pairing opposite edges. The enclosing rectangle
does not work: a Prinsengracht bay that is 5.66 by 2.61 m sits at 48°, and its enclosing
rectangle is 7.40 by 1.89 m, which matches neither dimension. Switching to edge pairs took
usable bays for a VW Polo from 40 % to 59.6 %.

Mirrors and bodywork are tracked separately, because mirrors hang over the painted line
into airspace that is nobody's bay. Parallel kerb bays get NEN 2443 parallel clearances
instead of perpendicular ones. The median Amsterdam kerb bay is 1.96 m and a Polo is 1.75 m
wide, so asking for 25 cm of lateral margin rejects an ordinary car by four centimetres.

---

## Where every number comes from

Every response carries its source and how old it is. Sources are ranked and never overwrite
each other: operator feed, then camera, then municipal sensor, then user report, then
model, then the static register. Every observation is kept, and disagreements get resolved
on read.

If a live source's last observation is older than the staleness window, it stops counting
as live and gets labelled stale. I would rather show a stale label than a confident wrong
number.

---

## Take me there

![The handoff sheet showing the exact coordinate and six navigation apps](docs/images/handoff.png)

Tapping a result hands the space over to Google Maps, Apple Maps, Waze, Yandex,
OpenStreetMap or whatever the device registers for `geo:` URIs, as coordinates rather than
an address. A street string gets re-geocoded by the receiving app against its own database,
which lands you near the place instead of on it. Coordinates go over at seven decimal
places, about a centimetre, so the format is never the limiting factor.

For a car park the destination is the entrance where one is recorded, and the interface
says which point it used. Routing to a garage centroid drops you inside a building outline
and you still have to find the ramp.

---

## What does not work yet

The occupancy model is trained on simulated history. The system has been ingesting live
data for about a day, and you cannot fit "how full is this street at 18:00 on a Friday"
with one Friday. What the Brier scores above measure is whether the model recovers demand
structure it cannot see directly, which is a real estimation problem, but it is not a claim
about real Amsterdam occupancy.

The detector does not work on real camera imagery. It scores F1 0.994 on rendered scenes
and falls over on a live frame: put through the Amsterdam Beursplein camera it reported two
motorcycles, one in tree canopy and one on empty pavement, and missed both police vans in
the shot. That is the sim-to-real gap, and it is what training on flat-shaded rendered
boxes gets you. The pipeline works end to end on live video. The model needs real labelled
imagery.

Camera coverage is a map rather than a network. 12,221 mapped camera locations come in from
OpenStreetMap with real coordinates, operator and direction, and nearly all of them are
private CCTV over a shop doorway. They carry no stream URL and permission `UNVERIFIED`,
which is what the registry gate refuses to run. Four cameras are published live by their
operators and can actually be opened: Amsterdam Damrak/Beursplein, Stationseiland and Dam
Square, plus Zaanse Schans. I checked the commercial webcam aggregators too, and they do
not hand a stream URL to a well-behaved client, because their players resolve manifests
through endpoints their own robots files disallow.

The registry refuses by default. A feed nobody has assessed does not run. Production only
accepts an explicit authorisation or an owner attestation, and "the robots file allowed it"
is not a licence. Frames are processed in memory and thrown away, and only occupancy,
geometry, confidence and timestamps ever get published. No face or plate recognition,
anywhere.

---

## Why I piloted in NL

The reason is data, not need. Amsterdam publishes all 210,247 parking bays as surveyed
polygons with sign codes and time regimes. RDW publishes registered dimensions per plate,
including height for 4,862,118 passenger cars. NDW publishes live occupancy. Almost nowhere
else can you check a specific car against a specific bay without doing the survey yourself
first.

So the Netherlands is where the idea is provable, and the table below is where it would
matter more. Amsterdam sits at roughly one car per on-street bay. Istanbul sits at five and
a half.

### Cars per parking space, biggest city of each country

Ranked on registered vehicles divided by parking spaces open to the public in that
country's largest city. Higher is worse. Every row says what its denominator actually
counts, because that varies by city and it is the part that makes these numbers hard to
compare. Sources are numbered against the reference list at the end.

| # | Country or territory | City | Vehicles | Spaces | Vehicles per space | What the denominator counts | Src |
|--:|---|---|--:|--:|--:|---|:--|
| 1 | Turkey | Istanbul | 6,292,611 | 1,143,937 | **5.50** | All non-residential parking: municipal, hospital, shopping centre, private operators | [1] |
| 2 | United Kingdom | London | ~3,100,000 | ~1,000,000 | **~3.1** | Regulated on-street only, so this is an overstatement | [2] |
| 3 | France | Paris | 617,000 | 220,000 | **2.80** | On-street plus public car parks | [3] |
| 4 | China | Beijing | 4,400,000 | 1,930,000 | **2.28** | All parking counted in the THUPDi survey | [4] |
| 5 | Russia | Moscow | ~4,000,000 | 1,900,000 | **~2.11** | Registered parking spaces citywide | [5] |
| 6 | India | Mumbai (south) | 354,000 | 190,000 to 220,000 | **~1.73** | Estimated total capacity, south Mumbai only | [6] |
| 7 | **Netherlands** | **Amsterdam** | **220,000** | **210,247** | **1.05** | **On-street bays, from the city's own polygon register** | **[7][8]** |
| 8 | Germany | Berlin | 1,240,000 | 1,276,312 | **0.97** | On-street, counted by survey vehicle | [9] |
| 9 | South Korea | Seoul | not published | not published | **0.83** | Official parking supply rate of 120.7 % | [10] |
| 10 | Japan | Tokyo (23 wards) | not published | not published | **<1** | Spaces rose 24 % while registrations fell 9 % over a decade | [11] |

London ranks second on a denominator that counts only regulated kerbside, so its true
figure is lower than Paris's. I left it in the position the published numbers give it and
flagged it rather than quietly adjusting it.

### The other ten, where the ratio is not published

I could not find a city-wide vehicles-per-space figure for these from a source I trust, so
they are listed alphabetically with what is actually documented. Filling the gap with an
estimate would defeat the point of the table.

| Country or territory | City | What is documented | Src |
|---|---|---|:--|
| Brazil | São Paulo | Parking minimums abolished citywide in 2014; fleet about 6 million | [12] |
| Egypt | Cairo | Chronic shortage described in the planning literature, no city inventory published | [13] |
| Hong Kong | Hong Kong | Over 800,000 spaces as of June 2025; the Bureau reports the ratio improving without publishing it | [14] |
| Italy | Rome | 26 metered spaces per 1,000 residents; Milan short 89,000 daytime spaces | [15][16] |
| Mexico | Mexico City | 73 % of drivers had abandoned a parking search in the previous 12 months | [17] |
| Nigeria | Lagos | Office buildings documented as under-provisioned, no city inventory published | [18] |
| Poland | Warsaw | Over 2 million registered cars; up to 17,000 on-street spaces projected lost by 2040 | [19] |
| Spain | Madrid | Up to 41,000 on-street spaces projected lost to growing car sizes by 2040 | [19] |
| United States | New York | 107 hours per driver per year spent searching, the highest INRIX measured anywhere | [20] |
| Vietnam | Ho Chi Minh City | Built parking land is 2.69 ha against the 550 ha planned for static traffic | [21] |

One thing that showed up across nearly every source: cars are getting wider faster than
kerbs are. New cars gain about 1.2 cm of length a year, and Transport & Environment project
European cities losing 8.5 % to 14 % of on-street parking by 2040 for that reason alone
[19]. A bay that fitted the car parked in it in 2005 does not necessarily fit its
replacement, which is the entire premise of this project.

Next stages go to the countries at the top of the first table. Turkey first, since Istanbul
is both the worst ratio I found and a city with an open municipal parking dataset, then the
UK and France.

### References

[1] Anadolu Ajansı. (2026). *İstanbul'da her 6 araca yaklaşık bir park yeri düşüyor*. https://www.aa.com.tr/tr/gundem/istanbulda-her-6-araca-yaklasik-bir-park-yeri-dusuyor/3890310

[2] Centre for London. (n.d.). *Car ownership, use and parking in London*. https://centreforlondon.org/reader/parking-kerbside-mangement/chapter-1/

[3] Paris ZigZag. (n.d.). *Les chiffres fous des transports parisiens*. https://www.pariszigzag.fr/paris-au-quotidien/les-chiffres-fous-des-transports-parisiens/

[4] CGTN. (n.d.). *Report shows extreme shortage of parking spaces in China's megacities*. https://news.cgtn.com/news/3d67544e7a49444e/share.html

[5] Global Highways. (n.d.). *Huge programme to develop parking infrastructure in Moscow being introduced*. https://www.globalhighways.com/news/huge-programme-develop-parking-infrastructure-moscow-being-introduced

[6] Government of Maharashtra. (2025). *Maharashtra proof of parking policy: Concept note and discussion paper*. https://cdnbbsr.s3waas.gov.in/s3a012869311d64a44b5a0d567cd20de04/uploads/2025/05/20250501170274283.pdf

[7] Gemeente Amsterdam, Onderzoek en Statistiek. (2024). *Verkeer in cijfers 2024*. https://onderzoek.amsterdam.nl/artikel/verkeer-in-cijfers-2024

[8] Gemeente Amsterdam. (2026). *Parkeervakken* [Data set]. https://api.data.amsterdam.nl/v1/parkeervakken/

[9] Tagesspiegel. (n.d.). *Autostellplätze auf den Straßen gezählt: In Berlin gibt es mehr Parkplätze als Autos*. https://www.tagesspiegel.de/berlin/autostellplatze-auf-den-strassen-gezahlt-in-berlin-gibt-es-mehr-parkplatze-als-autos-12663416.html

[10] Seoul Metropolitan Government. (n.d.). *주차 관련 통계* [Parking statistics]. https://news.seoul.go.kr/traffic/archives/314

[11] Inquirer Business. (n.d.). *Rules for Tokyo parking lots to ease as car ownership falls*. https://business.inquirer.net/286011/rules-for-tokyo-parking-lots-to-ease-as-car-ownership-falls

[12] Parking Reform Atlas. (n.d.). *São Paulo parking minimums abolition*. https://www.parkingreformatlas.org/parking-reform-cases-1/s%C3%A3o-paulo-parking-minimums-abolition

[13] Ibrahim, A. (2022). *Toward solving the car parking issue for Egyptian cities*. https://www.researchgate.net/publication/360604045_Toward_Solving_the_Car_Parking_Issue_For_Egyptian_Cities

[14] Transport and Logistics Bureau. (2025, June 11). *LCQ6: Supply of car parking spaces*. https://www.tlb.gov.hk/eng/legislative/transport/replies/2025/20250611c.html

[15] News Minimalist. (n.d.). *Rome struggles with too many cars and too few parking spaces*. https://www.newsminimalist.com/articles/rome-struggles-with-too-many-cars-and-too-few-parking-spaces-ad71fe0b

[16] Mobility Management Consulting. (2023). *Il futuro piano parcheggi di Milano*. https://www.mobilitymanagement.consulting/il-futuro-piano-parcheggi-di-milano/

[17] Wikipedia contributors. (n.d.). *Parking in Mexico City*. https://en.wikipedia.org/wiki/Parking_in_Mexico_City

[18] Estate Intel. (n.d.). *Analysis of parking provision in Lagos' office buildings*. https://estateintel.com/news/lagos-parking-nightmare-whats-next

[19] Transport & Environment, & Clean Cities. (2026). *'Carspreading' to wipe out up to 14% of on-street parking in European cities*. https://cleancitiescampaign.org/carspreading-to-wipe-out-up-to-14-of-on-street-parking-in-european-cities-study/

[20] INRIX. (2017). *Searching for parking costs Americans $73 billion a year*. https://inrix.com/press-releases/parking-pain-us/

[21] VietnamNet. (n.d.). *HCMC faces tremendous pressure on parking spaces*. https://vietnamnet.vn/en/hcmc-faces-tremendous-pressure-on-parking-spaces-2077838.html

---

## Stack

C++20 for the geometry, spatial index, routing, fit engine, ranking and the vision worker.
Python 3.12 with FastAPI, SQLAlchemy 2.0 and pybind11. PyTorch exported to ONNX Runtime for
detection, LightGBM for occupancy. TypeScript and MapLibre on the web side. There is no
OpenCV in the C++ path: the homography, the Jacobi eigensolver, the RANSAC and the
perceptual hashing are written here and tested here.

314 tests. The data comes from RDW, NDW, PDOK, OpenStreetMap and the City of Amsterdam,
all open, each with its licence written down in `docs/data_sources/sources.md`.
