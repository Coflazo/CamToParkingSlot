/**
 * Map rendering.
 *
 * Uses OpenStreetMap raster tiles, which need no API key and no account, the product
 * runs on a laptop with nothing configured. The OSM tile usage policy asks for a
 * identifying User-Agent and modest volume, which a single-user dev app satisfies;
 * a real deployment should serve its own tiles from the same extract the router uses.
 *
 * Routes are drawn from the geometry the backend already computed. Re-deriving them in
 * the browser would risk showing a line that does not match the time quoted beside it.
 */

import maplibregl, { type LngLatLike, type Map as MapLibreMap, Marker, Popup } from "maplibre-gl";

import type { PublicCamera } from "./api";
import "maplibre-gl/dist/maplibre-gl.css";
import type { Recommendation } from "./api";

const AMSTERDAM: LngLatLike = [4.9041, 52.3676];

/** A deliberately plain dark basemap: the data on top of it is the point. */
const STYLE: maplibregl.StyleSpecification = {
  version: 8,
  sources: {
    osm: {
      type: "raster",
      tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
      tileSize: 256,
      maxzoom: 19,
      attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    },
  },
  layers: [
    { id: "background", type: "background", paint: { "background-color": "#0a0e14" } },
    {
      id: "osm",
      type: "raster",
      source: "osm",
      // Pulled down and desaturated so the markers and routes carry the contrast.
      // raster-opacity alone only blends toward the layer beneath and still reads as a
      // bright map; brightness-max is what actually darkens the highlights, which
      // matters when this is held up in a car at dusk.
      paint: {
        "raster-opacity": 0.85,
        "raster-saturation": -0.6,
        "raster-contrast": -0.15,
        "raster-brightness-min": 0.02,
        "raster-brightness-max": 0.42,
      },
    },
  ],
};

export class ParkingMap {
  private map: MapLibreMap;
  private markers: Marker[] = [];
  // Cameras are not results and must not be cleared with them. They stay put across
  // searches, because the reason to look at one is often "what is that street like
  // right now", which has nothing to do with the query that is currently on screen.
  private cameraMarkers: Marker[] = [];
  private destinationMarker: Marker | null = null;
  private ready = false;
  private pending: (() => void)[] = [];

  constructor(container: string, private onSelect: (id: string) => void) {
    this.map = new maplibregl.Map({
      container,
      style: STYLE,
      center: AMSTERDAM,
      zoom: 12.5,
      attributionControl: { compact: true },
    });
    this.map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    this.map.addControl(
      new maplibregl.GeolocateControl({ trackUserLocation: false }),
      "top-right",
    );
    this.map.on("load", () => {
      this.ready = true;
      for (const task of this.pending) task();
      this.pending = [];
    });
  }

  private whenReady(task: () => void): void {
    if (this.ready) task();
    else this.pending.push(task);
  }

  setDestination(lat: number, lon: number, label: string): void {
    this.destinationMarker?.remove();
    const element = document.createElement("div");
    element.className = "map-pin map-pin-destination";
    element.setAttribute("aria-label", label);
    this.destinationMarker = new Marker({ element })
      .setLngLat([lon, lat])
      .setPopup(new Popup({ offset: 14 }).setText(label))
      .addTo(this.map);
  }

  showResults(results: Recommendation[], destination: { lat: number; lon: number }): void {
    this.clearMarkers();

    results.forEach((result, index) => {
      const element = document.createElement("div");
      element.className = index === 0 ? "map-pin map-pin-best" : "map-pin map-pin-other";
      element.textContent = String(index + 1);
      element.addEventListener("click", () => this.onSelect(result.id));

      const marker = new Marker({ element })
        .setLngLat([result.lon, result.lat])
        .setPopup(
          new Popup({ offset: 14 }).setHTML(
            `<strong>${escapeHtml(result.name)}</strong><br>` +
              `${result.drive ? Math.round(result.drive.duration_min) + " min drive" : ""}` +
              `${result.walk ? " · " + Math.round(result.walk.duration_min) + " min walk" : ""}`,
          ),
        )
        .addTo(this.map);
      this.markers.push(marker);
    });

    this.whenReady(() => this.fit(results, destination));
  }

  /** Draw the drive and walk legs for one option, using the backend's own geometry. */
  showRoutes(result: Recommendation): void {
    this.whenReady(() => {
      this.setLine("route-drive", result.drive?.geometry ?? [], "#4c9aff", 4);
      this.setLine("route-walk", result.walk?.geometry ?? [], "#3fb950", 3, [1, 2]);
    });
  }

  clearRoutes(): void {
    this.whenReady(() => {
      this.setLine("route-drive", [], "#4c9aff", 4);
      this.setLine("route-walk", [], "#3fb950", 3, [1, 2]);
    });
  }

  private setLine(
    id: string,
    coordinates: [number, number][],
    colour: string,
    width: number,
    dash?: number[],
  ): void {
    const data: GeoJSON.Feature<GeoJSON.LineString> = {
      type: "Feature",
      properties: {},
      geometry: { type: "LineString", coordinates },
    };

    const existing = this.map.getSource(id) as maplibregl.GeoJSONSource | undefined;
    if (existing) {
      existing.setData(data);
      return;
    }
    if (coordinates.length === 0) return;

    this.map.addSource(id, { type: "geojson", data });
    this.map.addLayer({
      id,
      type: "line",
      source: id,
      layout: { "line-cap": "round", "line-join": "round" },
      paint: {
        "line-color": colour,
        "line-width": width,
        "line-opacity": 0.85,
        ...(dash ? { "line-dasharray": dash } : {}),
      },
    });
  }

  private fit(results: Recommendation[], destination: { lat: number; lon: number }): void {
    const bounds = new maplibregl.LngLatBounds(
      [destination.lon, destination.lat],
      [destination.lon, destination.lat],
    );
    for (const result of results) bounds.extend([result.lon, result.lat]);
    this.map.fitBounds(bounds, { padding: 70, maxZoom: 16, duration: 600 });
  }

  focus(result: Recommendation): void {
    this.map.easeTo({ center: [result.lon, result.lat], zoom: 16, duration: 500 });
    this.showRoutes(result);
  }

  /** Plot every watchable camera. The glyph is the brand mark: a viewfinder. */
  showCameras(cameras: PublicCamera[], onOpen: (camera: PublicCamera) => void): void {
    for (const marker of this.cameraMarkers) marker.remove();
    this.cameraMarkers = [];

    for (const camera of cameras) {
      const element = document.createElement("button");
      element.type = "button";
      element.className = "map-cam";
      element.title = `${camera.name} (${camera.operator})`;
      element.setAttribute("aria-label", `Open live camera: ${camera.name}`);
      element.innerHTML =
        `<svg viewBox="0 0 24 24" aria-hidden="true">` +
        `<path d="M4 9V6.5A2.5 2.5 0 0 1 6.5 4H9"/><path d="M15 4h2.5A2.5 2.5 0 0 1 20 6.5V9"/>` +
        `<path d="M20 15v2.5a2.5 2.5 0 0 1-2.5 2.5H15"/><path d="M9 20H6.5A2.5 2.5 0 0 1 4 17.5V15"/>` +
        `</svg>`;
      element.addEventListener("click", (event) => {
        event.stopPropagation();
        onOpen(camera);
      });

      this.cameraMarkers.push(
        new Marker({ element }).setLngLat([camera.lon, camera.lat]).addTo(this.map),
      );
    }
  }

  private clearMarkers(): void {
    for (const marker of this.markers) marker.remove();
    this.markers = [];
    this.clearRoutes();
  }
}

export function escapeHtml(value: string): string {
  const div = document.createElement("div");
  div.textContent = value;
  return div.innerHTML;
}
