/**
 * ParkFit NL, application entry point.
 *
 * The interface has one job beyond finding parking: never let a claim appear without
 * the evidence behind it. Every result shows what is known, where it came from and how
 * old it is, and a fit verdict the driver can act on, including "we could not check",
 * which is a real answer and is shown as one rather than being quietly rounded to yes.
 */

import "./style.css";
import { ApiError, api, getToken, setToken, streamAvailability } from "./api";
import type { GeocodeResult, Recommendation, SearchResponse, Vehicle } from "./api";
import { ParkingMap, escapeHtml } from "./map";
import * as navigate from "./navigate";

const el = <T extends HTMLElement>(id: string): T => {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing element: ${id}`);
  return node as T;
};

const form = el<HTMLFormElement>("search-form");
const destinationInput = el<HTMLInputElement>("destination");
const suggestionList = el<HTMLUListElement>("suggestions");
const durationSelect = el<HTMLSelectElement>("duration");
const walkSelect = el<HTMLSelectElement>("walk");
const vehicleSelect = el<HTMLSelectElement>("vehicle");
const onStreetToggle = el<HTMLInputElement>("on-street");
const evToggle = el<HTMLInputElement>("ev");
const disabledToggle = el<HTMLInputElement>("disabled-bay");
const searchButton = el<HTMLButtonElement>("search-button");
const statusBox = el<HTMLDivElement>("status");
const resultsBox = el<HTMLDivElement>("results");
const healthPill = el<HTMLSpanElement>("health-pill");
const vehicleButton = el<HTMLButtonElement>("vehicle-button");
const vehicleDialog = el<HTMLDialogElement>("vehicle-dialog");
const vehicleList = el<HTMLDivElement>("vehicle-list");
const vehicleAuth = el<HTMLDivElement>("vehicle-auth");
const legend = el<HTMLDivElement>("map-legend");

let currentResults: Recommendation[] = [];
let selectedId: string | null = null;
let originCoords: { lat: number; lon: number } | null = null;
let closeStream: (() => void) | null = null;

const map = new ParkingMap("map", (id) => selectResult(id));

// ---------------------------------------------------------------- health
async function refreshHealth(): Promise<void> {
  try {
    const health = await api.health();
    healthPill.className = "pill pill-live";
    healthPill.textContent = `${health.facilities.toLocaleString()} car parks · ${health.bays.toLocaleString()} bays`;
    healthPill.title =
      `ParkFit ${health.version} · ${health.database} · routing via ${health.routing_provider}` +
      (health.native_module ? " · native core loaded" : " · native core NOT built");
  } catch {
    healthPill.className = "pill pill-stale";
    healthPill.textContent = "backend offline";
    healthPill.title = "Start the API with: .\\tasks.ps1 serve";
  }
}

// ------------------------------------------------------------ suggestions
let suggestTimer: number | undefined;
let activeSuggestion = -1;

destinationInput.addEventListener("input", () => {
  window.clearTimeout(suggestTimer);
  const query = destinationInput.value.trim();
  if (query.length < 2) {
    hideSuggestions();
    return;
  }
  // Debounced: typing "Rembrandt House Museum" should not be twenty-one requests.
  suggestTimer = window.setTimeout(() => void loadSuggestions(query), 220);
});

destinationInput.addEventListener("keydown", (event) => {
  const items = Array.from(suggestionList.querySelectorAll("li"));
  if (items.length === 0) return;
  if (event.key === "ArrowDown" || event.key === "ArrowUp") {
    event.preventDefault();
    activeSuggestion =
      (activeSuggestion + (event.key === "ArrowDown" ? 1 : items.length - 1)) % items.length;
    items.forEach((item, index) =>
      item.setAttribute("aria-selected", String(index === activeSuggestion)),
    );
  } else if (event.key === "Enter" && activeSuggestion >= 0) {
    event.preventDefault();
    items[activeSuggestion]?.click();
  } else if (event.key === "Escape") {
    hideSuggestions();
  }
});

document.addEventListener("click", (event) => {
  if (!suggestionList.contains(event.target as Node) && event.target !== destinationInput) {
    hideSuggestions();
  }
});

// Close on blur so the dropdown cannot sit on top of the search button. The delay lets
// a click on a suggestion land first: blur fires before click, and hiding the list
// immediately would remove the element out from under the pointer.
destinationInput.addEventListener("blur", () => {
  window.setTimeout(hideSuggestions, 150);
});

async function loadSuggestions(query: string): Promise<void> {
  try {
    const { results } = await api.geocode(query, 6);
    if (results.length === 0) {
      hideSuggestions();
      return;
    }
    suggestionList.innerHTML = "";
    activeSuggestion = -1;
    for (const result of results) {
      suggestionList.appendChild(suggestionItem(result));
    }
    suggestionList.hidden = false;
  } catch {
    hideSuggestions();
  }
}

function suggestionItem(result: GeocodeResult): HTMLLIElement {
  const item = document.createElement("li");
  item.setAttribute("role", "option");
  item.innerHTML =
    `${escapeHtml(result.label)}<span class="suggestion-meta">` +
    `${escapeHtml(result.kind)} · ${result.source === "OpenStreetMap" ? "place" : "address"}` +
    `</span>`;
  item.addEventListener("click", () => {
    destinationInput.value = result.label;
    hideSuggestions();
    void runSearch();
  });
  return item;
}

function hideSuggestions(): void {
  suggestionList.hidden = true;
  suggestionList.innerHTML = "";
  activeSuggestion = -1;
}

// ---------------------------------------------------------------- search
form.addEventListener("submit", (event) => {
  event.preventDefault();
  hideSuggestions();
  void runSearch();
});

async function runSearch(): Promise<void> {
  const destination = destinationInput.value.trim();
  if (destination.length < 2) return;

  searchButton.disabled = true;
  searchButton.textContent = "Searching…";
  setStatus("Looking for parking…");
  closeStream?.();
  closeStream = null;

  try {
    const vehicleId = vehicleSelect.value ? Number(vehicleSelect.value) : null;
    const response = await api.search({
      destination,
      vehicleId,
      originLat: originCoords?.lat ?? null,
      originLon: originCoords?.lon ?? null,
      durationMinutes: Number(durationSelect.value),
      maxWalkMinutes: Number(walkSelect.value),
      includeOnStreet: onStreetToggle.checked,
      needsEvCharging: evToggle.checked,
      needsDisabledBay: disabledToggle.checked,
    });
    renderSearch(response);
  } catch (error) {
    const message =
      error instanceof ApiError
        ? error.message
        : "Could not reach the backend. Start it with: .\\tasks.ps1 serve";
    setStatus(message, "error");
    resultsBox.innerHTML = "";
  } finally {
    searchButton.disabled = false;
    searchButton.textContent = "Find parking";
  }
}

function renderSearch(response: SearchResponse): void {
  currentResults = response.results;
  selectedId = null;

  if (!response.destination) {
    setStatus(response.warnings[0] ?? "Could not locate that destination.", "error");
    resultsBox.innerHTML = `<div class="empty">Try a place name, or a full address.</div>`;
    return;
  }

  map.setDestination(response.destination.lat, response.destination.lon, response.destination.label);

  if (response.results.length === 0) {
    setStatus(
      response.warnings[0] ?? "Nothing found near there for this vehicle.",
      "warning",
    );
    resultsBox.innerHTML = `<div class="empty">
      No options survived the filters. Try a longer walk, or check the vehicle dimensions.
    </div>`;
    map.showResults([], response.destination);
    return;
  }

  // The counts are shown deliberately. "Why is this garage not listed?" is a fair
  // question, and the honest answer is that it was excluded and for which reason.
  const excluded: string[] = [];
  if (response.rejected_fit) excluded.push(`${response.rejected_fit} too large for your vehicle`);
  if (response.rejected_illegal) excluded.push(`${response.rejected_illegal} not permitted`);
  if (response.rejected_walk) excluded.push(`${response.rejected_walk} beyond your walking limit`);

  setStatus(
    `${response.results.length} options near ${response.destination.label} · ` +
      `${response.considered} considered within ${Math.round(response.radius_m)} m` +
      (excluded.length ? ` · ruled out: ${excluded.join(", ")}` : "") +
      ` · ${Math.round(response.elapsed_ms)} ms`,
    response.warnings.length ? "warning" : undefined,
  );
  if (response.warnings.length) {
    const note = document.createElement("div");
    note.className = "status-warning";
    note.textContent = response.warnings[0] ?? "";
    statusBox.appendChild(note);
  }

  resultsBox.innerHTML = "";
  response.results.forEach((result) => resultsBox.appendChild(resultCard(result)));

  map.showResults(response.results, response.destination);
  legend.hidden = false;

  const first = response.results[0];
  if (first) selectResult(first.id);

  // Live updates only for the options actually on screen, and only while this search
  // is the current one.
  closeStream = streamAvailability(
    response.results.map((r) => r.id),
    () => void 0,
  );
}

function resultCard(result: Recommendation): HTMLElement {
  const card = document.createElement("article");
  card.className = "result";
  card.dataset["fit"] = result.fit.verdict;
  card.dataset["id"] = result.id;
  card.tabIndex = 0;
  // Drives the entrance stagger, so results arrive in rank order rather than all at once.
  card.style.setProperty("--rank", String(result.rank));

  const drive = result.drive ? `${Math.round(result.drive.duration_min)} min drive` : "—";
  const walk = result.walk ? `${Math.round(result.walk.duration_min)} min walk` : "—";
  const price = result.price_eur > 0 ? `€${result.price_eur.toFixed(2)}` : "not metered";
  const probability = Math.round(result.probability_at_arrival * 100);

  const size = result.is_exact_space
    ? `${(result.bay_length_cm / 100).toFixed(1)} × ${(result.bay_width_cm / 100).toFixed(1)} m ${escapeHtml(result.orientation)} bay`
    : result.capacity
      ? `${result.capacity} spaces`
      : "";

  card.innerHTML = `
    <div class="result-head">
      <span class="result-name">${escapeHtml(result.name)}</span>
      <span class="result-rank">#${result.rank + 1}</span>
    </div>
    <div class="result-metrics">
      <span><strong>${drive}</strong></span>
      <span><strong>${walk}</strong></span>
      <span><strong>${price}</strong></span>
      ${size ? `<span>${size}</span>` : ""}
    </div>
    <div class="probability-bar" role="img"
         aria-label="${probability}% chance of being free when you arrive">
      <div class="probability-fill" style="width:${probability}%"></div>
    </div>
    <div class="result-detail">
      <span>${escapeHtml(result.fit.explanation)}</span>
      <span>
        ${evidencePill(result)}
        <span>${probability}% likely free on arrival · ${escapeHtml(result.price_note)}</span>
      </span>
      ${
        result.restriction_warnings.length
          ? `<span class="status-warning">${escapeHtml(result.restriction_warnings.join("; "))}</span>`
          : ""
      }
    </div>
    <button type="button" class="take-me" data-take="${escapeHtml(result.id)}"
            ${result.navigation ? "" : "disabled"}>
      <span>Take me there</span>
      <span class="take-me-arrow" aria-hidden="true">&rarr;</span>
    </button>
  `;

  // Stops the card's own click handler from firing as well, which would re-select the
  // card underneath the sheet that just opened.
  card.querySelector<HTMLButtonElement>(".take-me")?.addEventListener("click", (event) => {
    event.stopPropagation();
    selectResult(result.id);
    navigate.open(result);
  });

  card.addEventListener("click", () => selectResult(result.id));
  card.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      selectResult(result.id);
    }
  });
  return card;
}

/** The evidence badge. Wording comes from the server so every surface agrees. */
function evidencePill(result: Recommendation): string {
  const label = result.evidence.confidence_label;
  const cls = result.evidence.stale
    ? "pill-stale"
    : label === "CAMERA_CONFIRMED" || label === "AVAILABILITY_REPORTED_BY_OPERATOR"
      ? "pill-live"
      : "pill-static";
  const text = label.replaceAll("_", " ").toLowerCase();
  return `<span class="pill ${cls}" title="${escapeHtml(result.evidence.freshness)}">${escapeHtml(text)}</span>`;
}

function selectResult(id: string): void {
  selectedId = id;
  const result = currentResults.find((r) => r.id === id);
  if (!result) return;

  for (const card of Array.from(resultsBox.querySelectorAll<HTMLElement>(".result"))) {
    card.classList.toggle("is-active", card.dataset["id"] === id);
  }
  map.focus(result);
}

function setStatus(message: string, tone?: "warning" | "error"): void {
  statusBox.className = tone ? `status status-${tone}` : "status";
  statusBox.textContent = message;
}

// --------------------------------------------------------------- vehicles
vehicleButton.addEventListener("click", () => {
  void renderVehicleDialog();
  vehicleDialog.showModal();
});

async function loadVehicles(): Promise<void> {
  vehicleSelect.innerHTML = `<option value="">Not specified</option>`;
  if (!getToken()) return;
  try {
    const vehicles = await api.vehicles();
    for (const vehicle of vehicles) {
      const option = document.createElement("option");
      option.value = String(vehicle.id);
      option.textContent = vehicle.nickname;
      vehicleSelect.appendChild(option);
    }
    if (vehicles.length > 0) vehicleSelect.value = String(vehicles[0]?.id ?? "");
  } catch {
    setToken(null);
  }
}

async function renderVehicleDialog(): Promise<void> {
  vehicleList.innerHTML = "";
  vehicleAuth.innerHTML = "";

  if (!getToken()) {
    vehicleAuth.appendChild(authForm());
    return;
  }

  try {
    const vehicles = await api.vehicles();
    if (vehicles.length === 0) {
      vehicleList.innerHTML = `<p class="dialog-lead">No vehicles yet.</p>`;
    }
    for (const vehicle of vehicles) {
      vehicleList.appendChild(vehicleCard(vehicle));
    }
    vehicleAuth.appendChild(addVehicleForm());
  } catch {
    setToken(null);
    vehicleAuth.appendChild(authForm());
  }
}

function vehicleCard(vehicle: Vehicle): HTMLElement {
  const card = document.createElement("div");
  card.className = "vehicle-card";
  card.innerHTML = `
    <div class="vehicle-card-name">${escapeHtml(vehicle.nickname)}</div>
    <div class="vehicle-card-dims">
      ${(vehicle.length_cm / 100).toFixed(2)} × ${(vehicle.body_width_cm / 100).toFixed(2)} ×
      ${(vehicle.height_with_accessories_cm / 100).toFixed(2)} m
      ${vehicle.height_confirmed ? "" : " · height not confirmed"}
    </div>
  `;
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "button button-ghost";
  remove.textContent = "Remove";
  remove.addEventListener("click", async () => {
    await api.deleteVehicle(vehicle.id);
    await renderVehicleDialog();
    await loadVehicles();
  });
  card.appendChild(remove);
  return card;
}

function authForm(): HTMLElement {
  const wrapper = document.createElement("div");
  wrapper.className = "vehicle-auth";
  wrapper.innerHTML = `
    <p class="dialog-lead">Sign in to save vehicles. Searching works without an account.</p>
    <label class="field"><span class="field-label">Email</span>
      <input type="email" id="auth-email" autocomplete="email" /></label>
    <label class="field"><span class="field-label">Password</span>
      <input type="password" id="auth-password" autocomplete="current-password" />
      <span class="field-hint">At least 10 characters.</span></label>
    <div class="dialog-actions">
      <button type="button" class="button button-ghost" id="auth-register">Create account</button>
      <button type="button" class="button button-primary" id="auth-login">Sign in</button>
    </div>
    <div id="auth-error" class="status-error"></div>
  `;

  const run = async (action: "login" | "register") => {
    const email = (wrapper.querySelector("#auth-email") as HTMLInputElement).value.trim();
    const password = (wrapper.querySelector("#auth-password") as HTMLInputElement).value;
    const errorBox = wrapper.querySelector("#auth-error") as HTMLElement;
    errorBox.textContent = "";
    try {
      const result = action === "login" ? await api.login(email, password) : await api.register(email, password);
      setToken(result.access_token);
      await renderVehicleDialog();
      await loadVehicles();
    } catch (error) {
      errorBox.textContent = error instanceof ApiError ? error.message : "Something went wrong.";
    }
  };

  wrapper.querySelector("#auth-login")?.addEventListener("click", () => void run("login"));
  wrapper.querySelector("#auth-register")?.addEventListener("click", () => void run("register"));
  return wrapper;
}

function addVehicleForm(): HTMLElement {
  const wrapper = document.createElement("div");
  wrapper.className = "vehicle-auth";
  wrapper.innerHTML = `
    <h3 style="margin:0;font-size:14px">Add a vehicle</h3>
    <label class="field"><span class="field-label">Dutch licence plate</span>
      <input type="text" id="plate" placeholder="XT-994-N" />
      <span class="field-hint">Looked up at RDW, then discarded. Only the dimensions are kept.</span>
    </label>
    <button type="button" class="button button-ghost" id="lookup">Look up</button>
    <div class="field-row">
      <label class="field field-small"><span class="field-label">Length cm</span>
        <input type="number" id="v-length" min="100" max="2500" /></label>
      <label class="field field-small"><span class="field-label">Width cm</span>
        <input type="number" id="v-width" min="100" max="400" /></label>
      <label class="field field-small"><span class="field-label">Height cm</span>
        <input type="number" id="v-height" min="100" max="500" /></label>
    </div>
    <span class="field-hint" id="plate-note"></span>
    <button type="button" class="button button-primary" id="save-vehicle">Save vehicle</button>
    <div id="vehicle-error" class="status-error"></div>
  `;

  const field = (id: string) => wrapper.querySelector(id) as HTMLInputElement;

  wrapper.querySelector("#lookup")?.addEventListener("click", async () => {
    const note = wrapper.querySelector("#plate-note") as HTMLElement;
    try {
      const lookup = await api.lookupPlate(field("#plate").value);
      if (!lookup.found) {
        note.textContent = "No vehicle found for that plate. Enter the dimensions by hand.";
        return;
      }
      field("#v-length").value = String(Math.round(lookup.length_cm));
      field("#v-width").value = String(Math.round(lookup.body_width_cm));
      // Height is deliberately left blank: RDW does not publish it, and it is the
      // dimension a barrier physically stops.
      note.textContent = lookup.note;
    } catch (error) {
      note.textContent = error instanceof ApiError ? error.message : "Lookup failed.";
    }
  });

  wrapper.querySelector("#save-vehicle")?.addEventListener("click", async () => {
    const errorBox = wrapper.querySelector("#vehicle-error") as HTMLElement;
    errorBox.textContent = "";
    try {
      await api.createVehicle({
        nickname: field("#plate").value || "My car",
        length_cm: Number(field("#v-length").value),
        body_width_cm: Number(field("#v-width").value),
        width_with_mirrors_cm: 0,
        height_cm: Number(field("#v-height").value),
        weight_kg: 0,
      });
      await renderVehicleDialog();
      await loadVehicles();
    } catch (error) {
      errorBox.textContent =
        error instanceof ApiError ? error.message : "Could not save that vehicle.";
    }
  });

  return wrapper;
}

// ----------------------------------------------------------------- start
navigator.geolocation?.getCurrentPosition(
  (position) => {
    originCoords = { lat: position.coords.latitude, lon: position.coords.longitude };
  },
  () => {
    // Declining location is fine; the backend falls back to the destination as origin.
  },
  { timeout: 5000, maximumAge: 300000 },
);

void refreshHealth();
void loadVehicles();
window.setInterval(() => void refreshHealth(), 60000);

if ("serviceWorker" in navigator && import.meta.env.PROD) {
  window.addEventListener("load", () => {
    void navigator.serviceWorker.register("/sw.js").catch(() => {
      /* offline support is a bonus, not a requirement */
    });
  });
}

export { selectedId };
