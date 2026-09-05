# Data sources

Every source, what it gives, what its licence requires, and whether it may be used
commercially. The licence column is not paperwork: OpenStreetMap is share-alike and the
government sources are not, so a product that mixes them has to know which is which
before it publishes anything.

The machine-readable version lives in the `source_licences` table, written by each
adapter on its first run.

Two columns matter more than they look. **Commercial** is `?` where the licence exists
but does not clearly answer the question, and `?` is not a synonym for yes. **Attribution
required** carries the exact string that has to appear, because an attribution that has
been paraphrased is not the one the licence asked for.

---

## Netherlands

Verified reachable 2026-08-26, re-verified 2026-09-05.

| Source | Dataset | Coverage | Licence | Commercial | Attribution required |
|---|---|---|---|---|---|
| Gemeente Amsterdam | `parkeervakken` | **210,247 bay polygons**, EPSG:28992 | CC-BY-4.0 | yes | "Parking bays: Gemeente Amsterdam" |
| RDW | `8u4d-s4q7` GEBIED | 14,748 parking areas | CC0 | yes | none (given anyway) |
| RDW | `b3us-f26s` SPECIFICATIES | 3,137 rows; **only 909 publish a height** | CC0 | yes | none |
| RDW | `t5pc-eb34` / `6wzd-evwu` | 237 garages, 134 P+R, geolocated | CC0 | yes | none |
| RDW | `mz4f-59fw` / `edv8-qiyg` / `2uc2-nnv3` / `f6v7-gjpa` | usage, access, operators, feed index | CC0 | yes | none |
| RDW | `m9d7-ebf2` | vehicle lookup by plate | CC0 | yes | none |
| NDW | `Truckparking_Parking_Status.xml` | live DATEX II v3 occupancy | CC0 | yes | none |
| NDW | `Truckparking_Parking_Table.xml` | static records, capacities | CC0 | yes | none |
| NDW | `emissiezones.xml.gz` | environmental zones | CC0 | yes | none |
| NDW | `verkeersborden_actueel_beeld_wgs84` | **nationwide traffic signs, 248 MB, not yet ingested** | CC0 | yes | none |
| PDOK | Locatieserver v3.1 | Dutch address geocoding | CC0 | yes | "Geocoding: PDOK Locatieserver (Kadaster / BZK)" |

## Türkiye

Verified live 2026-09-05.

| Source | Dataset | Coverage | Licence | Commercial | Attribution required |
|---|---|---|---|---|---|
| İBB / İSPARK | `api.ibb.gov.tr/ispark/Park` | **248 sites, 80,987 spaces, live free-space counts**, 34 districts, 51 on-street | İBB Open Data Licence | **?** | "İstanbul Büyükşehir Belediyesi (İBB) / İSPARK" |
| İBB / İSPARK | `/ParkDetay?id=` | WKT polygons, tariff ladders, addresses | İBB Open Data Licence | **?** | same |
| İBB | `tkmservices/api/TrafficData/v1/TrafficIndex` | live city traffic index | İBB Open Data Licence | **?** | same |

**The İBB licence is not a standard SPDX identifier**, and it does not plainly grant
commercial use, so `commercial_use` is recorded as unknown rather than assumed. Read
<https://data.ibb.gov.tr/license> before any commercial deployment.

**İSPARK publishes no cameras.** Established by probing, not by assumption: every
camera-shaped path under `/ispark` returns the same generic 500; `data.ibb.gov.tr`
returns zero datasets for `kamera`, `camera`, `cctv` and `mobese`; and the tkmservices
help catalogue has no camera model at all, since every camera-shaped `modelName` returns
the identical "not found" page while a real model returns a real one. Any Istanbul camera
layer needs a source that has not been found yet.

## Germany and France

Plumbing complete, data not yet ingested.

| Source | Dataset | Status | Licence | Commercial |
|---|---|---|---|---|
| Autobahn GmbH | `verkehr.autobahn.de/o/autobahn/{road}/services/parking_lorry` | **verified live, keyless**, PKW + LKW counts, 108 roads | Datenlizenz Deutschland | yes |
| Autobahn GmbH | `.../services/webcam` | **verified empty on every road sampled; deprecated** | | |
| Hamburg | `api.hamburg.de/datasets/v1/parkhaeuser` | verified reachable | dl-de/by-2-0 | yes |
| transport.data.gouv.fr | national access point | **verified live**, 782 datasets, 55 parking or road | mixed, per dataset | per dataset |
| transport.data.gouv.fr | Base nationale des lieux de stationnement hors voirie | CSV, national off-street | Licence Ouverte | yes |
| transport.data.gouv.fr | Parkings Indigo, Parkings Saemes | APDS JSON, NeTEx, CSV | per operator | per operator |

## Global

| Source | Used for | Licence | Commercial | Attribution required |
|---|---|---|---|---|
| OpenStreetMap (Overpass) | POIs, car parks, road network, **legal anchors** | **ODbL-1.0** | yes, with share-alike | "© OpenStreetMap contributors" |
| Nominatim | geocoding outside the Netherlands | **ODbL-1.0** | yes, with share-alike | "Geocoding: © OpenStreetMap contributors, via Nominatim" |

**Nominatim's usage policy is a constraint, not a courtesy.** The public instance asks
for at most one request per second, an identifying User-Agent, and no bulk geocoding. All
three are enforced in `parkfit/ingest/nominatim.py`: requests serialise behind a shared
lock with a real sleep, the User-Agent names the project, and `run()` raises rather than
offering a bulk path. A product that ignored that would deserve to be blocked.

## Legal texts

The rulebooks in `cpp/core/include/parkfit/legal/` are transcribed from primary sources,
never from summaries. Each rule carries its article, and the article is what a refusal
shows the driver.

| Country | Instrument | Source | Status |
|---|---|---|---|
| NL | RVV 1990, art. 23 to 25 | `wetten.overheid.nl/BWBR0004825` | transcribed, 13 rules |
| DE | StVO §12, and Anlage 2 zu §41 Zeichen 224 | `gesetze-im-internet.de/stvo_2013` | transcribed, 8 rules |
| TR | Karayolları Trafik Kanunu 2918, md. 60 and 61 | `mevzuat.gov.tr` official PDF | transcribed, 19 rules |
| FR | Code de la route R417-9 to R417-13 | Legifrance | **not transcribed: Cloudflare-walled** |

Reading the primary text rather than a summary changed three numbers that would otherwise
have shipped as confident, wrong citations:

* the Dutch bus-stop setback is **12 m**, not the German 15 m that every summary gives by
  analogy;
* Turkey's articles are **60 and 61**, not 61 and 62;
* Germany's 15 m bus-stop rule lives in **Anlage 2, Zeichen 224**, not in §12, so citing
  §12 for it would put the wrong article in front of a user.

It also preserved a real difference instead of flattening it. A fire hydrant is protected
in Istanbul (`KTK 2918 md. 61(d)`) and not in Amsterdam or Berlin, because RVV 23 and 24
do not mention hydrants and StVO §12 protects marked fire-brigade *access ways* rather
than hydrants. Making the three agree would mean inventing law.

**France is deliberately empty and flagged incomplete**, so it answers `UNKNOWN` rather
than `LEGAL`. An empty rule table breaks no rules, so without the flag the product would
have declared every space in Paris legal on the strength of no legal work at all. Finish
it from the DILA LEGI open-data dump at `echanges.dila.gouv.fr` (section
`LEGISCTA000006177136`) or the PISTE API, not from a driving-school page.

---

## Things worth knowing before relying on them

Each of these cost real debugging time. They are recorded so the next person does not
have to rediscover them.

**RDW publishes `0` for an unknown height limit**, and 2,225 of 3,137 specification rows
do. Zero maps to `NULL`, never to unlimited, treating it as unlimited would route a van
into a 2.0 m barrier. Several geolocated Amsterdam garages have no specification row at
all, so `UNVERIFIED` is the ordinary answer rather than the edge case.

**The NDW live feed is genuinely corrupt in places.** Observed in a single sample: a site
with capacity 210 reporting 1,146 vacant spaces and **-1,046 occupied**; another
reporting 0 vacant, 24 occupied and 100 % occupancy while its own status field said
`spacesAvailable`. Every value is range-checked, contradictions resolve to the pessimistic
reading, and each rejection is recorded as a `DataQualityIncident`.

**DATEX II nests occupancy.** A `parkingRecordStatus` carries a site-level
`parkingOccupancy` *and* a `groupOfParkingSpacesStatus` per sub-area. A real record has
four `parkingNumberOfVacantSpaces` elements reading 8, 4, 4 and 0, and only the first
describes the car park. The parser navigates by direct child and never by subtree search.

**Capacity is split across overlapping groups.** Truck-Inn Nobis, whose real capacity is
210, publishes groups of 0, 210 and 210. Summing them claims 420 spaces; the maximum is
right.

**Amsterdam caps plain pagination at 200,000 rows.** Page 101 returns 403 Forbidden. The
full city needs the per-neighbourhood partitioned ingest, which `pf ingest amsterdam`
uses automatically once it hits the ceiling.

**PDOK cannot find places.** Searching for "Rembrandthuis" returns zero results;
"Jodenbreestraat 4" returns an exact match. It indexes the address register, not points
of interest. Those come from OpenStreetMap.

**Overpass needs GET, not POST.** A POST with a raw body returns 406. It also rejects
clients that do not identify themselves, so a User-Agent is mandatory rather than polite.
And the selector matters: `node[...]` plus `way[...]` misses every multipolygon relation,
which is what the Van Gogh Museum, Johan Cruijff ArenA and Ziggo Dome all are. Use
`nwr[...]`.

**Overpass rate-limits with 429 and it is not rare.** Four queries in quick succession
during one anchor ingest was enough to lose the cycleway layer. A missing layer is
recorded as an ingest error and, more importantly, as an un-queried anchor kind, so a
later evaluation says it could not check that rule instead of quietly clearing the space.

**httpx replaces a URL query string with `params`.** Passing an empty dict strips the
cursor off a paginated `next` link and silently re-fetches page 1 forever. The symptom is
a run that reports 4,000 fetched and 1,000 created.

**Turkish upper-casing is not English upper-casing.** `str.upper()` folds the dotless ı
and dotted İ by English rules, so comparing İSPARK's `parkType` without an explicit
translation table drops all 111 `AÇIK OTOPARK` sites into `UNKNOWN`. It raises no error
and loses 45 % of the network.

**Turkish decimals use a comma.** `float("110,00")` raises, which is the safe failure, and
`float("1.234,00")` silently reads as `1.234`, which is not. İSPARK tariffs are parsed
with the separators handled explicitly.

**WKT is longitude first.** İSPARK's `areaPolygon` is `POLYGON((lon lat, ...))` and
everything downstream here is latitude first. Getting it backwards puts Istanbul in
Somalia, and nothing raises: the coordinates are valid, just wrong.

**A non-empty index is not coverage.** An anchor index holding 4,477 Amsterdam anchors is
not empty, so an Istanbul point swept it, found nothing within a hundred metres because
everything in it was two thousand kilometres away, broke no rules, and came back **legal**,
indistinguishable from a space that had been checked. Anchor sets record their extent, and
the usable area is that extent eroded by the rulebook's own reach.

**A geocoder index leaks across borders.** The point-of-interest table held only Dutch
places and was searched whatever country was asked for, so "Tour Eiffel" scored the word
"Tour" against an Amsterdam boat-tour company and answered a French query with a canal in
the Netherlands at 0.42 confidence. Points of interest record their country and the search
filters on it.

**A junction is a road junction.** Deriving junction anchors from the union of the car and
foot graphs turned every footpath connection into a five-metre no-parking zone: 46,987
junctions in central Amsterdam instead of 1,089. Every statute here says roads, `kruispunt`
and `Kreuzungen` and `kavşaklar`, so the car graph alone is right.

## Licence obligations

**OpenStreetMap and Nominatim (ODbL)** require attribution and carry share-alike
obligations on derived databases. Attribution travels with the data through
`SourceLicence` and appears in the API index response and on the map. The legal anchors
are an OSM-derived database, so if this product ever redistributes them the share-alike
terms apply and need review before release.

**RDW, NDW and PDOK** are CC0 or equivalent: no attribution obligation. The adapters
record it anyway, because knowing where a number came from is worth more than the licence
requires.

**Gemeente Amsterdam** is CC-BY-4.0: attribution required.

**İBB / İSPARK** is a municipal licence with no clear commercial grant. Treated as
unknown, which means an attribution is shown and a commercial deployment needs a reading
of the licence first.

**Camera feeds are none of the above.** Each needs its own permission, recorded in the
camera registry with a reference to the agreement. Being visible on a web page is not a
licence for automated processing, and the registry is where that distinction is written
down rather than assumed. No feed is processed at permission status `UNVERIFIED`.

## What this product does not do with any of it

Constant across every source and every country, and not conditional on the licence:

* no face recognition and no licence-plate recognition, anywhere in the pipeline;
* camera frames are processed in memory and discarded, never written to disk outside a
  deliberately inspectable training set;
* no attempt to identify, track or follow an individual person or vehicle;
* only occupancy, geometry, confidence and timestamps are persisted.
