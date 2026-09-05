"""The ``pf`` command line.

One entry point for ingest, search, camera management and evaluation, so the whole
system can be driven without the API.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Annotated

from parkfit.numeric import limit_numeric_threads

# Before anything that pulls in numpy. BLAS and OpenMP size their scratch pools from the
# core count when the native library loads, and nothing here benefits from them; see
# parkfit.numeric for why this has to happen at the top of the file rather than in the
# command that trains a model.
limit_numeric_threads()

import typer  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from parkfit import __version__  # noqa: E402
from parkfit.config import get_settings  # noqa: E402

app = typer.Typer(
    name="pf",
    help="CamToParkingSlot: vehicle-aware parking search for the Netherlands.",
    no_args_is_help=True,
    add_completion=False,
)
ingest_app = typer.Typer(help="Pull open data into the local database.", no_args_is_help=True)
cameras_app = typer.Typer(help="Manage the camera registry.", no_args_is_help=True)
predict_app = typer.Typer(
    help="Occupancy prediction: history, decay rates, learned model.", no_args_is_help=True
)
detect_app = typer.Typer(
    help="The vehicle detector: dataset, training, ONNX export.", no_args_is_help=True
)
app.add_typer(ingest_app, name="ingest")
app.add_typer(cameras_app, name="cameras")
app.add_typer(predict_app, name="predict")
app.add_typer(detect_app, name="detect")

occupancy_app = typer.Typer(
    help="Bay occupancy: is this known parking space occupied right now?",
    no_args_is_help=True,
)
app.add_typer(occupancy_app, name="occupancy")

console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "httpcore", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------
@ingest_app.command("all")
def ingest_all(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    skip_bays: bool = typer.Option(False, help="Skip the Amsterdam bay ingest, which is large."),
) -> None:
    """Run every ingest adapter in dependency order."""
    _setup_logging(verbose)
    from parkfit.storage.session import checkpoint, create_all

    create_all()
    results = []
    results.append(_run_rdw())
    results.append(_run_ndw())
    results.append(_run_osm())
    if not skip_bays:
        results.append(_run_amsterdam())

    stats = checkpoint(analyze=True)
    console.print(f"\n[dim]maintenance: {stats}[/dim]")
    _print_ingest_table(results)


@ingest_app.command("rdw")
def ingest_rdw(
    geocoded_only: bool = typer.Option(True, help="Only areas that publish coordinates."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """National parking register: garages, park-and-ride, capacities, height limits."""
    _setup_logging(verbose)
    from parkfit.storage.session import create_all

    create_all()
    _print_ingest_table([_run_rdw(geocoded_only=geocoded_only)])


@ingest_app.command("ndw")
def ingest_ndw(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Live DATEX II parking occupancy."""
    _setup_logging(verbose)
    from parkfit.storage.session import create_all

    create_all()
    _print_ingest_table([_run_ndw()])


@ingest_app.command("amsterdam")
def ingest_amsterdam(
    limit: int = typer.Option(0, help="Stop after this many bays. 0 means the whole city."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Amsterdam parking bays: exact polygons, layout, sign codes, time regimes."""
    _setup_logging(verbose)
    from parkfit.storage.session import create_all

    create_all()
    _print_ingest_table([_run_amsterdam(limit=limit or None)])


@ingest_app.command("osm")
def ingest_osm(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """OpenStreetMap car parks and points of interest."""
    _setup_logging(verbose)
    from parkfit.storage.session import create_all

    create_all()
    _print_ingest_table([_run_osm()])


@ingest_app.command("roads")
def ingest_roads(
    south: float = 52.33,
    west: float = 4.82,
    north: float = 52.41,
    east: float = 4.97,
    country: str = typer.Option("NL", help="ISO 3166-1 alpha-2 code for this region."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Build and cache the routable road graph for a bounding box.

    Graphs are kept per region, so ingesting a second city adds to the set rather than
    replacing the first.
    """
    _setup_logging(verbose)
    from parkfit.ingest.osm import OsmAdapter
    from parkfit.ingest.osm import ingest_roads as build

    with OsmAdapter() as adapter:
        result = build(
            adapter, south=south, west=west, north=north, east=east, country=country
        )
    _print_ingest_table([result])


@ingest_app.command("ispark")
def ingest_ispark(
    details: bool = typer.Option(
        False, "--details", help="Also fetch tariffs, addresses and polygons (one call per site)."
    ),
    limit: int | None = typer.Option(None, help="Cap the detail pass, for a quick check."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Istanbul municipal parking: live free-space counts, tariffs and polygons."""
    _setup_logging(verbose)
    from parkfit.ingest.ispark import IsparkAdapter

    results = []
    with IsparkAdapter() as adapter:
        results.append(adapter.run())
        if details:
            # One request per site, so it is opt-in and never on the live status path.
            results.append(adapter.run_details(limit=limit))
    _print_ingest_table(results)


@ingest_app.command("france")
def ingest_france(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """French national off-street parking base: capacity, height limits, tariffs."""
    _setup_logging(verbose)
    from parkfit.ingest.france import FranceAdapter
    from parkfit.storage.session import create_all

    create_all()
    with FranceAdapter() as adapter:
        result = adapter.run()
    _print_ingest_table([result])


@ingest_app.command("autobahn")
def ingest_autobahn(
    roads: str = typer.Option("", help="Comma-separated road list, e.g. A1,A8. Empty means all."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """German motorway parking: capacity per rest area, never occupancy.

    One request per road, 108 of them, so this is a daily job rather than something a
    search triggers.
    """
    _setup_logging(verbose)
    from parkfit.ingest.autobahn import AutobahnAdapter
    from parkfit.storage.session import create_all

    create_all()
    wanted = [r.strip() for r in roads.split(",") if r.strip()] or None
    with AutobahnAdapter() as adapter:
        result = adapter.run(roads=wanted)
    _print_ingest_table([result])


@ingest_app.command("anchors")
def ingest_anchors(
    south: float = 52.33,
    west: float = 4.82,
    north: float = 52.41,
    east: float = 4.97,
    country: str = typer.Option("NL", help="ISO 3166-1 alpha-2 code for the rulebook."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Collect the map features road law measures distances from.

    Run ``pf ingest roads`` for the same box first: junctions are derived from the road
    graph, so without one this collects everything except junctions and says so.
    """
    _setup_logging(verbose)
    from parkfit.ingest.anchors import ingest_anchors as collect
    from parkfit.ingest.osm import OsmAdapter
    from parkfit.services.legality import reset_legality_service

    with OsmAdapter() as adapter:
        result = collect(
            adapter, south=south, west=west, north=north, east=east, country=country
        )
    # The service caches the built index for the process, so a fresh ingest in the same
    # process would otherwise keep answering from the old anchors.
    reset_legality_service()
    _print_ingest_table([result])


def _run_rdw(geocoded_only: bool = True):
    from parkfit.ingest.rdw import RdwAdapter

    with RdwAdapter() as adapter:
        return adapter.run(geocoded_only=geocoded_only)


def _run_ndw():
    from parkfit.ingest.ndw import NdwAdapter

    with NdwAdapter() as adapter:
        return adapter.run()


def _run_osm():
    from parkfit.ingest.osm import OsmAdapter, ingest_pois

    with OsmAdapter() as adapter:
        ingest_pois(adapter)
        return adapter.run()


def _run_amsterdam(limit: int | None = None):
    from parkfit.ingest.amsterdam import AmsterdamAdapter

    with AmsterdamAdapter(use_cache=False) as adapter:
        return adapter.run(limit=limit) if limit else adapter.run_all()


def _print_ingest_table(results) -> None:
    table = Table(title="Ingest", header_style="bold")
    for column in ("source", "fetched", "created", "updated", "skipped", "errors", "seconds"):
        table.add_column(column, justify="right" if column != "source" else "left")
    for r in results:
        table.add_row(
            r.source,
            str(r.fetched),
            str(r.created),
            str(r.updated),
            str(r.skipped),
            str(len(r.errors)),
            f"{r.duration_s:.1f}",
        )
    console.print(table)


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------
@app.command()
def search(
    destination: Annotated[str, typer.Argument(help="Where you want to go.")],
    car: str = typer.Option(
        "", "--car", help="A preset from `pf cars`, for example polo, s60, x5, transit."
    ),
    length: float = typer.Option(405.0, help="Vehicle length in cm."),
    width: float = typer.Option(175.0, help="Bodywork width in cm."),
    mirrors: float = typer.Option(0.0, help="Width across mirrors in cm. 0 infers it."),
    height: float = typer.Option(145.0, help="Height including anything on the roof, in cm."),
    weight: float = typer.Option(1100.0, help="Kerb weight in kg."),
    origin: str = typer.Option("52.3789,4.9002", help="Where you are starting from."),
    duration: int = typer.Option(120, help="How long you intend to stay, in minutes."),
    walk: float = typer.Option(12.0, help="Maximum walking time in minutes."),
    on_street: bool = typer.Option(True, help="Include individual on-street bays."),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Search for parking that fits a specific vehicle."""
    _setup_logging(verbose)
    from parkfit.domain import presets

    preset = presets.get(car) if car else None
    if car and preset is None:
        console.print(f"[red]unknown car {car!r}[/red]. Run `pf cars` for the list.")
        raise typer.Exit(code=2)
    if preset is not None:
        # Real registered dimensions replace the flag defaults entirely, rather than
        # merging with them, so a preset search is reproducible.
        length, width, height, weight = (
            preset.length_cm,
            preset.body_width_cm,
            preset.height_cm,
            preset.weight_kg,
        )
        mirrors = preset.width_with_mirrors_cm
        # To stderr when the caller asked for JSON, or this banner lands in the middle of
        # the document and every downstream parser chokes on it.
        Console(stderr=as_json).print(
            f"[dim]{preset.label}: {length:.0f} x {width:.0f} x {height:.0f} cm, "
            f"{weight:.0f} kg  ({preset.segment}, RDW {preset.rdw_body_type})[/dim]"
        )
    from parkfit.domain.vehicle import VehicleProfile
    from parkfit.services.search import SearchEngine, SearchPreferences, SearchRequest
    from parkfit.storage.session import session_scope

    try:
        lat_s, lon_s = origin.split(",")
        origin_lat, origin_lon = float(lat_s), float(lon_s)
    except ValueError:
        console.print("[red]--origin must look like 52.3789,4.9002[/red]")
        raise typer.Exit(2) from None

    vehicle = VehicleProfile(
        id="cli",
        nickname="cli vehicle",
        length_cm=length,
        body_width_cm=width,
        width_with_mirrors_cm=mirrors or (width + 36.0),
        height_cm=height,
        height_with_accessories_cm=height,
        weight_kg=weight,
        length_confirmed=True,
        width_confirmed=bool(mirrors),
        height_confirmed=True,
    )

    with session_scope() as session:
        engine = SearchEngine(session)
        try:
            response = engine.search(
                SearchRequest(
                    destination=destination,
                    vehicle=vehicle,
                    origin_lat=origin_lat,
                    origin_lon=origin_lon,
                    arrival_time=datetime.now(UTC),
                    duration_minutes=duration,
                    preferences=SearchPreferences(
                        max_walk_minutes=walk, include_on_street=on_street
                    ),
                )
            )
        finally:
            engine.close()

        if as_json:
            console.print_json(json.dumps(_search_to_dict(response)))
            return
        _print_search(response)


def _search_to_dict(response) -> dict:
    return {
        "search_id": response.search_id,
        "destination": (
            {
                "label": response.destination.label,
                "lat": response.destination.lat,
                "lon": response.destination.lon,
                "source": response.destination.source,
            }
            if response.destination
            else None
        ),
        "elapsed_ms": round(response.elapsed_ms, 1),
        "considered": response.considered,
        "warnings": response.warnings,
        "results": [
            {
                "rank": i,
                "name": c.name,
                "kind": c.kind,
                "drive_min": round(c.drive.duration_min, 1) if c.drive else None,
                "walk_min": round(c.walk.duration_min, 1) if c.walk else None,
                "price_eur": c.price_eur,
                "price_note": c.price_note,
                "probability": round(c.probability_at_eta, 3),
                "cost": round(c.generalised_cost, 2),
                "fit": c.fit_verdict,
                "confidence": c.confidence_label,
                "lat": c.lat,
                "lon": c.lon,
            }
            for i, c in enumerate(response.results)
        ],
    }


def _print_search(response) -> None:
    if response.destination is None:
        console.print("[red]Could not locate that destination.[/red]")
        for warning in response.warnings:
            console.print(f"  [yellow]{warning}[/yellow]")
        return

    d = response.destination
    console.print(
        f"\n[bold]{d.label}[/bold]  [dim]{d.lat:.5f}, {d.lon:.5f} via {d.source} "
        f"(confidence {d.confidence:.2f})[/dim]"
    )
    console.print(
        f"[dim]{response.considered} candidates in {response.radius_m:.0f} m | "
        f"ruled out: {response.rejected_illegal} illegal, {response.rejected_fit} too large, "
        f"{response.rejected_walk} too far to walk | routing: {response.routing_provider} | "
        f"{response.elapsed_ms:.0f} ms[/dim]"
    )
    for warning in response.warnings:
        console.print(f"  [yellow]! {warning}[/yellow]")

    table = Table(header_style="bold", show_lines=False)
    table.add_column("#", justify="right")
    table.add_column("option", max_width=30)
    table.add_column("drive", justify="right")
    table.add_column("walk", justify="right")
    table.add_column("price", justify="right")
    table.add_column("P(free)", justify="right")
    table.add_column("fit")
    table.add_column("evidence")

    for i, c in enumerate(response.results):
        fit_colour = {"FITS": "green", "TIGHT_FIT": "yellow", "UNVERIFIED": "dim"}.get(
            c.fit_verdict, "red"
        )
        table.add_row(
            str(i),
            c.name,
            f"{c.drive.duration_min:.0f}m" if c.drive else "-",
            f"{c.walk.duration_min:.0f}m" if c.walk else "-",
            f"EUR {c.price_eur:.2f}" if c.price_eur else "free*",
            f"{c.probability_at_eta:.2f}",
            f"[{fit_colour}]{c.fit_verdict}[/{fit_colour}]",
            c.confidence_label.replace("_", " ").lower(),
        )
    console.print(table)
    if any(c.price_eur == 0 for c in response.results):
        console.print("[dim]* not metered; check the signs on arrival[/dim]")


# ---------------------------------------------------------------------------
# cameras
# ---------------------------------------------------------------------------
@cameras_app.command("add")
def camera_add(
    camera_id: Annotated[str, typer.Option("--id", help="Identifier for this camera.")],
    url: Annotated[str, typer.Option("--url", help="Stream URL.")],
    stream_type: str = typer.Option("hls", "--type", help="hls, mjpeg, rtsp, snapshot or file."),
    owner: str = typer.Option("", help="Who owns the camera."),
    lat: float = typer.Option(0.0),
    lon: float = typer.Option(0.0),
    attest: str = typer.Option(
        "", help="Reference to the permission you hold. Sets owner_attested."
    ),
) -> None:
    """Register a camera feed.

    A camera is registered disabled and unverified. Pass ``--attest`` with a reference to
    the permission you hold to mark it owner-attested, which is the status a production
    deployment accepts. That is an assertion by you, not something the software can check.
    """
    from parkfit.cameras.registry import CameraRegistry
    from parkfit.storage.session import create_all, session_scope

    create_all()
    with session_scope() as session:
        registry = CameraRegistry(session)
        registry.register(
            camera_id,
            stream_url=url,
            stream_type=stream_type,
            owner=owner or None,
            lat=lat or None,
            lon=lon or None,
        )
        if attest:
            registry.attest_ownership(camera_id, agreement_reference=attest)
            console.print(f"[green]{camera_id} registered and attested[/green] ({attest})")
        else:
            console.print(
                f"[yellow]{camera_id} registered as unverified[/yellow]\n"
                "It will not run until its permission status is resolved. Use --attest "
                "with a reference to the permission you hold."
            )


@cameras_app.command("list")
def camera_list() -> None:
    """List registered cameras and whether each may run here."""
    from parkfit.cameras.registry import CameraRegistry
    from parkfit.storage.session import create_all, session_scope

    create_all()
    with session_scope() as session:
        rows = CameraRegistry(session).processable()

    if not rows:
        console.print("[dim]No cameras registered. Add one with: pf cameras add[/dim]")
        return

    table = Table(
        title=f"Camera registry ({get_settings().environment.value})", header_style="bold"
    )
    for column in ("camera", "status", "may run", "enabled", "type", "health"):
        table.add_column(column)
    for camera, decision in rows:
        table.add_row(
            camera.camera_id,
            camera.permission_status,
            "[green]yes[/green]" if decision.allowed else "[red]no[/red]",
            "yes" if camera.enabled else "no",
            camera.stream_type or "-",
            camera.technical_status,
        )
    console.print(table)
    for camera, decision in rows:
        if not decision.allowed:
            console.print(f"[dim]{camera.camera_id}: {decision.reason}[/dim]")


@cameras_app.command("enable")
def camera_enable(camera_id: str, off: bool = typer.Option(False, "--off")) -> None:
    """Enable or disable a camera. Enabling refuses if its permission does not allow it."""
    from parkfit.cameras.registry import CameraRegistry
    from parkfit.storage.session import session_scope

    with session_scope() as session:
        try:
            CameraRegistry(session).set_enabled(camera_id, not off)
        except PermissionError as exc:
            console.print(f"[red]refused:[/red] {exc}")
            raise typer.Exit(2) from None
        except KeyError:
            console.print(f"[red]unknown camera: {camera_id}[/red]")
            raise typer.Exit(2) from None
    console.print(f"[green]{camera_id} {'disabled' if off else 'enabled'}[/green]")


@cameras_app.command("audit")
def camera_audit(
    max_per_site: int = typer.Option(10, help="Candidates per listing page."),
    browser: bool = typer.Option(True, help="Render client-side pages with a headless browser."),
    report: str = typer.Option("docs/camera_registry/audit_report.md"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Crawl candidate webcam sources and record what may be processed.

    Honours robots.txt and performs no anti-bot evasion. It can rule a source out; it
    cannot rule one in, because that needs a person and a written permission.
    """
    _setup_logging(verbose)
    import pathlib

    from parkfit.cameras.auditor import SourceAuditor, render_audit_report

    with SourceAuditor(use_browser=browser) as auditor:
        candidates = auditor.audit(max_per_site=max_per_site)

    path = pathlib.Path(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_audit_report(candidates), encoding="utf-8")

    table = Table(title="Audit", header_style="bold")
    for column in ("site", "status", "type", "url"):
        table.add_column(column, overflow="fold")
    for candidate in candidates[:30]:
        table.add_row(
            candidate.source_site,
            candidate.permission_status,
            candidate.stream_type or "-",
            (candidate.stream_url or candidate.page_url)[:70],
        )
    console.print(table)
    console.print(f"\n[dim]report written to {path}[/dim]")


# ---------------------------------------------------------------------------
# meta
# ---------------------------------------------------------------------------
@app.command()
def evaluate(
    scenes: int = typer.Option(60, help="Synthetic scenes for gap measurement."),
    quick: bool = typer.Option(False, help="Fewer trials, for a fast check."),
    report: str = typer.Option("docs/architecture/evaluation.json"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Measure accuracy against the specification targets.

    The headline number is the false-free rate: how often a space is called free when it
    is not. Overall accuracy hides it, because a detector that calls everything occupied
    and one that calls everything free score identically on accuracy and differ entirely
    in whether they are safe to ship.
    """
    _setup_logging(verbose)
    import pathlib as _pathlib

    from parkfit.ml.evaluate.harness import format_report, run_all, write_report

    console.print("[dim]running evaluation, this takes a moment...[/dim]")
    result = run_all(scenes=scenes, quick=quick)
    console.print()
    console.print(format_report(result))
    path = write_report(result, _pathlib.Path(report))
    console.print()
    console.print(f"[dim]machine-readable report written to {path}[/dim]")


@app.command("synth")
def synth(
    out: str = typer.Option("data/synthetic", help="Where to write the dataset."),
    count: int = typer.Option(40, help="Number of scenes."),
    seed: int = typer.Option(0),
) -> None:
    """Render a synthetic dataset with exact ground-truth gap lengths."""
    import pathlib as _pathlib

    from parkfit.ml.synthetic.scene import write_dataset

    manifest = write_dataset(_pathlib.Path(out), count=count, seed=seed)
    total_gaps = sum(len(s["gaps_m"]) for s in manifest["scenes"])
    console.print(
        f"[green]{len(manifest['scenes'])} scenes[/green] with {total_gaps} ground-truth gaps "
        f"-> {out}"
    )
    console.print(f"[dim]control points: {len(manifest['control_points'])}[/dim]")


@predict_app.command("history")
def predict_history(
    days: int = typer.Option(21, help="How many days of history to simulate."),
    bays: int = typer.Option(150, help="How many kerb bays to give a history to."),
    facilities: int = typer.Option(40, help="How many car parks to give a history to."),
    interval: int = typer.Option(30, help="Minutes between persisted observations."),
    seed: int = typer.Option(20260826),
) -> None:
    """Simulate occupancy history for a sample of real targets.

    Replaces any history this command wrote before, and never touches an observation from
    a real source.
    """
    from parkfit.prediction.history import generate_history
    from parkfit.storage.session import session_scope

    with session_scope() as session:
        report, _ = generate_history(
            session,
            days=days,
            bays=bays,
            facilities=facilities,
            sample_interval_min=interval,
            seed=seed,
        )
    console.print(f"[green]{report.describe()}[/green]")


@predict_app.command("lambda")
def predict_lambda(
    days: int = typer.Option(21, help="Simulation window, matched to the stored history."),
    bays: int = typer.Option(150),
    facilities: int = typer.Option(40),
    interval: int = typer.Option(30),
    seed: int = typer.Option(20260826),
    from_observations: bool = typer.Option(
        False,
        "--from-observations",
        help="Estimate from stored samples only, as production must, instead of from the "
        "simulation's own one-minute transition counts.",
    ),
) -> None:
    """Estimate per-segment vacancy decay rates and store them.

    Re-runs the simulation to recover its transition counts, which are not persisted; the
    seed makes that reproduce the history exactly rather than generate a different one.
    """
    from parkfit.prediction import lambda_est
    from parkfit.prediction.history import generate_history
    from parkfit.storage.session import session_scope

    with session_scope() as session:
        _, simulated = generate_history(
            session,
            days=days,
            bays=bays,
            facilities=facilities,
            sample_interval_min=interval,
            seed=seed,
        )
        if from_observations:
            counts = lambda_est.counts_from_observations(session, list(simulated.keys()))
        else:
            counts = {k: v.counts for k, v in simulated.items()}

        report = lambda_est.estimate_and_store(session, counts, truth=simulated)
        console.print(f"[green]{report.describe()}[/green]")

        cost = lambda_est.measure_sampling_cost(session, simulated)

    if cost:
        table = Table(title=f"what {interval}-minute polling costs the decay estimate")
        table.add_column("measurement")
        table.add_column("value", justify="right")
        table.add_row("rate from 1-minute transitions", f"{cost['fine_lambda_mean']:.4f} /min")
        table.add_row(
            f"rate from {interval}-minute samples", f"{cost['coarse_lambda_mean']:.4f} /min"
        )
        table.add_row("recovered fraction", f"{cost['coarse_over_fine'] * 100:.0f} %")
        console.print(table)
        console.print(
            "[dim]Sparse polling misses turnovers that begin and end inside one sample, so "
            "it under-reports the rate. This is a property of the feed, not of the "
            "estimator.[/dim]"
        )


@predict_app.command("train")
def predict_train(
    source: str = typer.Option(
        "synthetic-history",
        help="Only train on observations from this source. Pass an empty string for all.",
    ),
    trees: int = typer.Option(400),
    threads: int = typer.Option(4),
    out: str = typer.Option("data/models/occupancy.lgb"),
) -> None:
    """Fit the occupancy model and score it against its baselines."""
    import pathlib as _pathlib

    from parkfit.prediction import model as occupancy_model
    from parkfit.storage.session import session_scope

    with session_scope() as session:
        report = occupancy_model.train(
            session,
            source_name=source or None,
            model_path=_pathlib.Path(out),
            num_trees=trees,
            num_threads=threads,
        )

    if not report.trained:
        console.print(f"[yellow]{report.describe()}[/yellow]")
        raise typer.Exit(code=1)

    console.print(f"[green]{report.rows:,} rows[/green] over {report.targets:,} targets")
    table = Table(title="occupancy model against its baselines (Brier score, lower is better)")
    table.add_column("held out")
    table.add_column("rows", justify="right")
    table.add_column("model", justify="right")
    table.add_column("flat prior", justify="right")
    table.add_column("per kind", justify="right")
    table.add_column("per target", justify="right")
    table.add_column("AUC", justify="right")
    for split in report.splits:
        per_target = "n/a" if split.per_target_brier is None else f"{split.per_target_brier:.4f}"
        table.add_row(
            split.name,
            f"{split.rows:,}",
            f"{split.model_brier:.4f}",
            f"{split.flat_prior_brier:.4f}",
            f"{split.per_kind_brier:.4f}",
            per_target,
            f"{split.model_auc:.3f}",
        )
    console.print(table)

    top = sorted(report.feature_importance.items(), key=lambda kv: -kv[1])[:6]
    console.print("[dim]top features: " + ", ".join(name for name, _ in top) + "[/dim]")
    console.print(f"[dim]model written to {report.model_path}[/dim]")


@predict_app.command("all")
def predict_all(
    days: int = typer.Option(21),
    bays: int = typer.Option(150),
    facilities: int = typer.Option(40),
    interval: int = typer.Option(30),
) -> None:
    """Run the whole prediction pipeline: history, decay rates, then the model."""
    predict_history(days=days, bays=bays, facilities=facilities, interval=interval, seed=20260826)
    predict_lambda(
        days=days,
        bays=bays,
        facilities=facilities,
        interval=interval,
        seed=20260826,
        from_observations=False,
    )
    predict_train(source="synthetic-history", trees=400, threads=4, out="data/models/occupancy.lgb")


@occupancy_app.command("stats")
def occupancy_stats(
    root: str = typer.Option("data/parking_ds"),
) -> None:
    """Describe the CNRPark-EXT splits actually present on disk."""
    import pathlib as _pathlib

    from parkfit.ml.datasets import occupancy as occ

    _setup_logging(verbose=False)
    base = _pathlib.Path(root)
    splits, patches = base / "splits" / "CNRPark-EXT", base / "PATCHES"
    if not splits.exists():
        console.print(f"[yellow]no CNRPark-EXT under {root}[/yellow]")
        raise typer.Exit(code=1)

    table = Table(title="CNRPark-EXT")
    for column in ("split", "patches", "occupied", "free", "cameras", "days", "weather"):
        table.add_column(column, justify="right" if column != "split" else "left")
    for name in ("train", "val", "test"):
        described = occ.describe(occ.read_split(splits / f"{name}.txt", patches))
        table.add_row(
            name,
            f"{described['patches']:,}",
            f"{described['occupied']:,}",
            f"{described['free']:,}",
            str(len(described["cameras"])),
            str(described["days"]),
            ", ".join(described["weather"]),
        )
    console.print(table)


@occupancy_app.command("train")
def occupancy_train(
    root: str = typer.Option("data/parking_ds"),
    protocol: str = typer.Option("official", help="official, camera or weather."),
    holdout: str = typer.Option("", help="Comma-separated cameras or weather to hold out."),
    epochs: int = typer.Option(6),
    batch: int = typer.Option(256),
    workers: int = typer.Option(4),
    device: str = typer.Option("cuda"),
    out: str = typer.Option("data/models/occupancy_cnn.pt"),
    report_path: str = typer.Option("docs/architecture/occupancy_cnn.json"),
) -> None:
    """Train the occupancy classifier on real parking-space crops."""
    import pathlib as _pathlib

    from parkfit.ml.train import occupancy_cnn

    _setup_logging(verbose=True)
    report = occupancy_cnn.train(
        _pathlib.Path(root),
        protocol=protocol,
        holdout={h.strip() for h in holdout.split(",") if h.strip()} or None,
        epochs=epochs,
        batch_size=batch,
        workers=workers,
        device=device,
        weights_path=_pathlib.Path(out),
    )
    if not report.trained:
        console.print(f"[yellow]{report.describe()}[/yellow]")
        raise typer.Exit(code=1)

    console.print(f"[green]{report.describe()}[/green]")
    if report.per_weather:
        table = Table(title="accuracy by weather, held-out set")
        table.add_column("weather")
        table.add_column("accuracy", justify="right")
        for name, value in report.per_weather.items():
            table.add_row(name, f"{value:.4f}")
        console.print(table)
    occupancy_cnn.write_report(report, _pathlib.Path(report_path))


@detect_app.command("dataset")
def detect_dataset(
    out: str = typer.Option("data/detector", help="Where to write the dataset."),
    train: int = typer.Option(600, help="Training scenes."),
    val: int = typer.Option(150, help="Validation scenes."),
    seed: int = typer.Option(7),
) -> None:
    """Render scenes into a detection dataset with exact ground-truth boxes."""
    import pathlib as _pathlib

    from parkfit.ml.datasets import scenes as scene_dataset

    report = scene_dataset.build(_pathlib.Path(out), train_scenes=train, val_scenes=val, seed=seed)
    console.print(f"[green]{report.describe()}[/green]")
    console.print(f"[dim]conditions: {report.per_condition}[/dim]")


@detect_app.command("train")
def detect_train(
    dataset: str = typer.Option("data/detector", help="Dataset directory."),
    epochs: int = typer.Option(26),
    batch: int = typer.Option(8),
    threads: int = typer.Option(4),
    out: str = typer.Option("data/models/detector.pt"),
) -> None:
    """Train the vehicle detector and score it on held-out scenes."""
    import pathlib as _pathlib

    from parkfit.ml.train import detector as detector_train

    _setup_logging(verbose=True)
    report = detector_train.train(
        _pathlib.Path(dataset),
        epochs=epochs,
        batch_size=batch,
        threads=threads,
        weights_path=_pathlib.Path(out),
    )
    if not report.trained:
        console.print(f"[yellow]{report.describe()}[/yellow]")
        raise typer.Exit(code=1)

    console.print(f"[green]{report.describe()}[/green]")
    table = Table(title="detection F1 by lighting condition")
    table.add_column("condition")
    table.add_column("F1", justify="right")
    for condition, score in sorted(report.per_condition.items()):
        table.add_row(condition, f"{score:.3f}")
    console.print(table)


@detect_app.command("harvest")
def detect_harvest(
    frames: int = typer.Option(22, help="Frames per camera."),
    spacing: float = typer.Option(5.0, help="Seconds between frames."),
    out: str = typer.Option("data/real/frames", help="Where the frames land."),
) -> None:
    """Pull real frames from every live public camera.

    A feed that is offline is skipped rather than fatal, because these are other
    people's cameras and they go down without telling us.
    """
    import pathlib as _pathlib

    from parkfit.cameras import harvest as harvester

    _setup_logging(verbose=True)
    got = harvester.harvest_all(
        _pathlib.Path(out), frames_per_camera=frames, spacing_seconds=spacing
    )
    table = Table(title="harvested real frames")
    table.add_column("camera")
    table.add_column("frames", justify="right")
    for camera_id, paths in sorted(got.items()):
        table.add_row(camera_id, str(len(paths)))
    console.print(table)
    total = sum(len(v) for v in got.values())
    console.print(f"[green]{total} real frames in {out}[/green]")


@detect_app.command("label")
def detect_label(
    frames: str = typer.Option("data/real/frames", help="Directory of harvested frames."),
    out: str = typer.Option("data/real/labels.json"),
    device: str = typer.Option("cuda", help="cuda or cpu."),
) -> None:
    """Label real frames with the COCO-pretrained teacher.

    These are pseudo-labels, not ground truth. The teacher has seen real photographs,
    which is the whole point, but it is wrong sometimes and the student inherits that.
    """
    import pathlib as _pathlib

    from parkfit.ml.datasets import real as real_ds

    _setup_logging(verbose=True)
    paths = sorted(_pathlib.Path(frames).glob("*.jpg"))
    if not paths:
        console.print(f"[yellow]no frames in {frames}; run `pf detect harvest` first[/yellow]")
        raise typer.Exit(code=1)

    labelled = real_ds.label_frames(paths, device=device)
    written = real_ds.write_labels(labelled, _pathlib.Path(out))
    boxes = sum(len(item.boxes) for item in labelled)
    from collections import Counter

    counts: Counter[str] = Counter()
    for item in labelled:
        for box in item.boxes:
            counts[real_ds.scenes.CLASS_NAMES[box["class"]]] += 1

    table = Table(title="teacher labels on real frames")
    table.add_column("class")
    table.add_column("instances", justify="right")
    for name, count in counts.most_common():
        table.add_row(name, str(count))
    console.print(table)
    console.print(f"[green]{boxes} boxes over {len(labelled)} frames -> {written}[/green]")


@detect_app.command("train-real")
def detect_train_real(
    labels: str = typer.Option("data/real/labels.json"),
    epochs: int = typer.Option(40),
    batch: int = typer.Option(8),
    holdout: str = typer.Option("", help="Comma-separated camera ids to hold out."),
    device: str = typer.Option("cuda"),
    backbone: str = typer.Option("pretrained", help="pretrained or scratch."),
    width: int = typer.Option(960, help="Model input width."),
    height: int = typer.Option(544, help="Model input height."),
    out: str = typer.Option("data/models/detector_real.pt"),
    report_path: str = typer.Option("docs/architecture/detector_real.json"),
) -> None:
    """Train the detector on real frames and score it on cameras it has never seen."""
    import pathlib as _pathlib

    from parkfit.ml.train import real_detector

    _setup_logging(verbose=True)
    holdout_set = {c.strip() for c in holdout.split(",") if c.strip()} or None
    report = real_detector.train_real(
        _pathlib.Path(labels),
        holdout_cameras=holdout_set,
        epochs=epochs,
        batch_size=batch,
        device=device,
        backbone=backbone,
        width=width,
        height=height,
        weights_path=_pathlib.Path(out),
    )
    if not report.trained:
        console.print(f"[yellow]{report.describe()}[/yellow]")
        raise typer.Exit(code=1)

    console.print(f"[green]{report.describe()}[/green]")
    if report.per_class:
        table = Table(title="recall by class, unseen cameras")
        table.add_column("class")
        table.add_column("recall", justify="right")
        for name, value in report.per_class.items():
            table.add_row(name, f"{value:.3f}")
        console.print(table)
    real_detector.write_report(report, _pathlib.Path(report_path))


@detect_app.command("export-real")
def detect_export_real(
    weights: str = typer.Option("data/models/detector_real.pt"),
    out: str = typer.Option("data/models/detector_real.onnx"),
    width: int = typer.Option(960),
    height: int = typer.Option(544),
    report_path: str = typer.Option("docs/architecture/detector_real.json"),
) -> None:
    """Export the real-frame detector to ONNX and write the C++ worker's spec."""
    import pathlib as _pathlib

    from parkfit.ml.export import onnx as onnx_export

    _setup_logging(verbose=True)
    result = onnx_export.export_real(
        _pathlib.Path(weights),
        _pathlib.Path(out),
        width=width,
        height=height,
        report_path=_pathlib.Path(report_path),
    )
    table = Table(title="ONNX parity against PyTorch")
    table.add_column("output")
    table.add_column("max relative diff", justify="right")
    for name, diff in result["diffs"].items():
        table.add_row(name, f"{diff:.2e}")
    console.print(table)
    spec = result["spec"]
    console.print(
        f"input {spec['input_width']}x{spec['input_height']}, "
        f"stride {spec['output_stride']}, {len(spec['class_names'])} classes"
    )
    if not result["ok"]:
        console.print("[yellow]parity outside tolerance[/yellow]")
        raise typer.Exit(code=1)
    console.print(f"[green]exported {out} with matching spec[/green]")


@detect_app.command("export")
def detect_export(
    weights: str = typer.Option("data/models/detector.pt"),
    out: str = typer.Option("data/models/detector.onnx"),
    dataset: str = typer.Option("data/detector"),
    frames: int = typer.Option(12, help="Frames to check PyTorch against ONNX Runtime."),
) -> None:
    """Export the detector to ONNX and verify the export against PyTorch."""
    import pathlib as _pathlib

    from parkfit.ml.export import onnx as onnx_export

    report = onnx_export.export(
        _pathlib.Path(weights),
        _pathlib.Path(out),
        dataset_root=_pathlib.Path(dataset),
        verify_frames=frames,
    )
    if not report.exported:
        console.print(f"[yellow]{report.describe()}[/yellow]")
        raise typer.Exit(code=1)

    colour = "green" if report.agrees else "red"
    console.print(f"[{colour}]{report.describe()}[/{colour}]")
    if not report.agrees:
        # An export that does not reproduce the model is worse than no export: it looks
        # like it works and shifts every box by an amount nobody measured.
        raise typer.Exit(code=1)


@detect_app.command("all")
def detect_all(
    epochs: int = typer.Option(26),
    train: int = typer.Option(600),
    val: int = typer.Option(150),
) -> None:
    """Build the dataset, train the detector, then export and verify it."""
    detect_dataset(out="data/detector", train=train, val=val, seed=7)
    detect_train(
        dataset="data/detector",
        epochs=epochs,
        batch=8,
        threads=4,
        out="data/models/detector.pt",
    )
    detect_export(
        weights="data/models/detector.pt",
        out="data/models/detector.onnx",
        dataset="data/detector",
        frames=12,
    )


@app.command("cars")
def cars() -> None:
    """The test fleet, with real dimensions from the Dutch vehicle register."""
    from parkfit.domain import presets

    table = Table(title="Test fleet (RDW registered dimensions, centimetres)")
    table.add_column("key")
    table.add_column("vehicle")
    table.add_column("segment")
    table.add_column("RDW type", style="dim")
    for column in ("L", "W", "H", "kg"):
        table.add_column(column, justify="right")

    for preset in presets.PRESETS:
        table.add_row(
            preset.key,
            preset.label,
            preset.segment,
            preset.rdw_body_type,
            f"{preset.length_cm:.0f}",
            f"{preset.body_width_cm:.0f}",
            f"{preset.height_cm:.0f}",
            f"{preset.weight_kg:.0f}",
        )
    console.print(table)
    console.print(
        "[dim]Width is bodywork; mirrors add 36 cm. Height excludes anything on the roof.[/dim]"
    )
    console.print('[dim]Use with: pf search "Dam" --car x5[/dim]')


@app.command()
def status() -> None:
    """Show what is loaded and how much data is present."""
    from sqlalchemy import func, select

    from parkfit.native import HAS_NATIVE, native_version
    from parkfit.routing.provider import get_routing_service
    from parkfit.storage.models import (
        AvailabilityObservation,
        CameraSource,
        ParkingBay,
        ParkingFacility,
        PointOfInterest,
    )
    from parkfit.storage.session import create_all, session_scope

    create_all()
    settings = get_settings()
    with session_scope() as session:
        counts = {
            "parking facilities": session.execute(
                select(func.count()).select_from(ParkingFacility)
            ).scalar(),
            "parking bays": session.execute(select(func.count()).select_from(ParkingBay)).scalar(),
            "points of interest": session.execute(
                select(func.count()).select_from(PointOfInterest)
            ).scalar(),
            "availability observations": session.execute(
                select(func.count()).select_from(AvailabilityObservation)
            ).scalar(),
            "registered cameras": session.execute(
                select(func.count()).select_from(CameraSource)
            ).scalar(),
        }

    table = Table(title=f"CamToParkingSlot {__version__}", header_style="bold")
    table.add_column("component")
    table.add_column("value", justify="right")
    table.add_row("environment", settings.environment.value)
    table.add_row("database", "postgres" if settings.is_postgres else "sqlite")
    table.add_row(
        "native module",
        f"[green]{native_version()}[/green]" if HAS_NATIVE else "[yellow]not built[/yellow]",
    )
    table.add_row("routing provider", get_routing_service().active_provider)
    for name, value in counts.items():
        table.add_row(name, f"{value:,}")
    console.print(table)
    if not HAS_NATIVE:
        console.print(
            "[yellow]Build the native module for the compiled path: .\\tasks.ps1 build[/yellow]"
        )


@app.command()
def version() -> None:
    """Print the version."""
    console.print(__version__)


def main() -> int:
    try:
        app()
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
