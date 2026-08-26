# Data sources

Every source, what it gives, and what its licence requires. The licence column is not
paperwork: OpenStreetMap is share-alike and the government sources are not, so a product
that mixes them has to know which is which before it publishes anything.

The machine-readable version lives in the `source_licences` table, written by each
adapter on its first run.

## Verified reachable, 2026-08-26

| Source | Dataset | Rows / coverage | Licence | Refresh |
|---|---|---|---|---|
| Gemeente Amsterdam | `parkeervakken` | **210,247 bay polygons** in EPSG:28992 | CC-BY-4.0 | weekly |
| RDW | `8u4d-s4q7` GEBIED | 14,748 parking areas | CC0 | daily |
| RDW | `b3us-f26s` SPECIFICATIES | 3,137 rows; **only 909 publish a height** | CC0 | daily |
| RDW | `t5pc-eb34` GEO garages | 237 geolocated | CC0 | daily |
| RDW | `6wzd-evwu` GEO P+R | 134 geolocated | CC0 | daily |
| RDW | `mz4f-59fw` GEBRUIKSDOEL | 14,691 usage codes | CC0 | daily |
| RDW | `edv8-qiyg` TOEGANG | 4,826 access windows | CC0 | daily |
| RDW | `2uc2-nnv3` GEBIEDSBEHEERDER | 461 operators | CC0 | daily |
| RDW | `f6v7-gjpa` INDEX | 164 organisations and their feed URLs | CC0 | daily |
| RDW | `m9d7-ebf2` vehicles | Lookup by licence plate | CC0 | daily |
| NDW | `Truckparking_Parking_Status.xml` | Live DATEX II v3 occupancy | CC0 | ~1 min |
| NDW | `Truckparking_Parking_Table.xml` | Static records, capacities, coordinates | CC0 | daily |
| NDW | `emissiezones.xml.gz` | Environmental zones | CC0 | daily |
| PDOK | Locatieserver v3.1 | Dutch address geocoding | CC0 | continuous |
| OpenStreetMap | Overpass | POIs, car parks, road network | **ODbL** | continuous |

## Things worth knowing before relying on them

Each of these cost real debugging time. They are recorded so the next person does not
have to rediscover them.

**RDW publishes `0` for an unknown height limit**, and 2,225 of 3,137 specification rows
do. Zero maps to `NULL`, never to unlimited — treating it as unlimited would route a van
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

**httpx replaces a URL query string with `params`.** Passing an empty dict strips the
cursor off a paginated `next` link and silently re-fetches page 1 forever. The symptom is
a run that reports 4,000 fetched and 1,000 created.

## Licence obligations

**OpenStreetMap (ODbL)** requires attribution and carries share-alike obligations on
derived databases. Attribution travels with the data through `SourceLicence`, and appears
in the API index response and on the map. If this product ever redistributes a database
derived from OSM, the share-alike terms apply to it and need review before release.

**RDW, NDW and PDOK** are CC0 or equivalent: no attribution obligation. The adapters
record it anyway, because knowing where a number came from is worth more than the licence
requires.

**Gemeente Amsterdam** is CC-BY-4.0: attribution required.

**Camera feeds are none of the above.** Each needs its own permission, recorded in the
camera registry with a reference to the agreement. Being visible on a web page is not a
licence for automated processing, and the registry is where that distinction is written
down rather than assumed.
