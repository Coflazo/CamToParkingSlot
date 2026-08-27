/**
 * Typed client for the ParkFit API.
 *
 * The response types mirror the server schemas deliberately closely, including the
 * evidence block. Every availability claim arrives with its source, observation time
 * and confidence label, and the UI is built to show them rather than to flatten them
 * into a number, a parking app that displays "47 spaces" without saying when it last
 * knew that is teaching its users not to trust it.
 */

export interface GeocodeResult {
  label: string;
  lat: number;
  lon: number;
  kind: string;
  source: string;
  confidence: number;
  city: string | null;
}

export interface LegDetail {
  distance_m: number;
  duration_min: number;
  provider: string;
  is_estimate: boolean;
  geometry: [number, number][];
}

export interface FitDetail {
  verdict: "FITS" | "TIGHT_FIT" | "DOES_NOT_FIT" | "UNVERIFIED";
  slack_cm: number;
  binding_constraint: string | null;
  unverified: string[];
  explanation: string;
}

export interface EvidenceDetail {
  source: string;
  observed_at: string | null;
  age_seconds: number | null;
  freshness: string;
  confidence_label: string;
  stale: boolean;
  conflicting_sources: number;
  vacant_spaces: number | null;
  total_spaces: number | null;
}

/** One "open in ..." handoff. */
export interface NavigationLink {
  provider: string;
  display_name: string;
  url: string;
}

/**
 * Where to send the driver.
 *
 * `lat`/`lon` is the exact destination and is never an address: a street string gets
 * re-geocoded by the receiving app against its own database and lands somewhere else.
 * `point_description` says what the point actually is, because a driver routed to a car
 * park centroid is still looking for the way in.
 */
export interface Navigation {
  lat: number;
  lon: number;
  label: string;
  is_entrance: boolean;
  point_description: string;
  links: NavigationLink[];
}

export interface Recommendation {
  id: string;
  kind: string;
  name: string;
  lat: number;
  lon: number;
  rank: number;
  generalised_cost_eur: number;
  probability_at_arrival: number;
  price_eur: number;
  price_note: string;
  drive: LegDetail | null;
  walk: LegDetail | null;
  fit: FitDetail;
  evidence: EvidenceDetail;
  capacity: number | null;
  max_height_cm: number | null;
  bay_length_cm: number;
  bay_width_cm: number;
  orientation: string;
  restriction_warnings: string[];
  is_exact_space: boolean;
  expires_at: string | null;
  navigation: Navigation | null;
}

export interface SearchResponse {
  search_id: string;
  destination: GeocodeResult | null;
  results: Recommendation[];
  considered: number;
  merged_duplicates: number;
  rejected_illegal: number;
  rejected_fit: number;
  rejected_walk: number;
  radius_m: number;
  routing_provider: string;
  warnings: string[];
  elapsed_ms: number;
}

export interface Vehicle {
  id: number;
  nickname: string;
  make: string | null;
  model: string | null;
  length_cm: number;
  body_width_cm: number;
  width_with_mirrors_cm: number;
  height_cm: number;
  height_with_accessories_cm: number;
  weight_kg: number;
  is_ev: boolean;
  has_trailer: boolean;
  has_roof_box: boolean;
  extra_parallel_clearance_cm: number;
  height_confirmed: boolean;
  width_confirmed: boolean;
}

export interface PlateLookup {
  found: boolean;
  make: string | null;
  model: string | null;
  length_cm: number;
  body_width_cm: number;
  width_with_mirrors_cm: number;
  weight_kg: number;
  fuel_type: string | null;
  is_ev: boolean;
  unconfirmed_fields: string[];
  note: string;
}

export interface HealthResponse {
  status: string;
  version: string;
  native_module: boolean;
  native_version: string | null;
  database: string;
  routing_provider: string;
  facilities: number;
  bays: number;
  points_of_interest: number;
  live_observations_last_hour: number;
}

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

const TOKEN_KEY = "parkfit.token";

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    // Private browsing and blocked site data both throw here. An anonymous search
    // works perfectly well, so this is not worth failing over.
    return null;
  }
}

export function setToken(token: string | null): void {
  try {
    if (token === null) localStorage.removeItem(TOKEN_KEY);
    else localStorage.setItem(TOKEN_KEY, token);
  } catch {
    /* storage unavailable; the session simply does not persist */
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("Accept", "application/json");
  if (init.body) headers.set("Content-Type", "application/json");
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);

  const response = await fetch(path, { ...init, headers });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") detail = body.detail;
      else if (Array.isArray(body?.detail) && body.detail[0]?.msg) detail = body.detail[0].msg;
    } catch {
      /* the body was not JSON; the status line is all we have */
    }
    throw new ApiError(response.status, detail);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export interface SearchParams {
  destination: string;
  vehicleId?: number | null;
  originLat?: number | null;
  originLon?: number | null;
  durationMinutes: number;
  maxWalkMinutes: number;
  includeOnStreet: boolean;
  needsEvCharging?: boolean;
  needsDisabledBay?: boolean;
}


/** A camera whose operator publishes it, and which the user may therefore watch. */
export interface PublicCamera {
  camera_id: string;
  name: string;
  operator: string;
  lat: number;
  lon: number;
  embed_url: string;
  watch_url: string;
  note: string;
  free_spaces_seen: number;
}

export interface CameraList {
  cameras: PublicCamera[];
  count: number;
  disclaimer: string;
}


/** One vehicle the detector found in a camera frame, in frame pixels. */
export interface DetectedBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  label: string;
  score: number;
}

/** A stretch of kerb with nothing on it. Metres are estimates, never measurements. */
export interface FreeSpaceBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  length_m: number;
  depth_m: number;
  fits: string[];
}

export interface CameraAnalysis {
  camera_id: string;
  ok: boolean;
  reason: string;
  age_seconds: number;
  frame_width: number;
  frame_height: number;
  frame_data_uri: string;
  vehicles: DetectedBox[];
  free_spaces: FreeSpaceBox[];
  pixels_per_metre: number;
  scale_confident: boolean;
  note: string;
}

export const api = {
  health: () => request<HealthResponse>("/health"),

  geocode: (q: string, limit = 6) =>
    request<{ query: string; results: GeocodeResult[] }>(
      `/v1/geocode?q=${encodeURIComponent(q)}&limit=${limit}`,
    ),

  search: (params: SearchParams) =>
    request<SearchResponse>("/v1/searches", {
      method: "POST",
      body: JSON.stringify({
        destination: params.destination,
        vehicle_id: params.vehicleId ?? null,
        origin_lat: params.originLat ?? null,
        origin_lon: params.originLon ?? null,
        expected_duration_minutes: params.durationMinutes,
        preferences: {
          max_walk_minutes: params.maxWalkMinutes,
          include_on_street: params.includeOnStreet,
          needs_ev_charging: params.needsEvCharging ?? false,
          needs_disabled_bay: params.needsDisabledBay ?? false,
        },
      }),
    }),

  register: (email: string, password: string) =>
    request<{ access_token: string }>("/v1/auth/register", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),

  login: (email: string, password: string) =>
    request<{ access_token: string }>("/v1/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),

  vehicles: () => request<Vehicle[]>("/v1/vehicles"),

  // Unauthenticated on purpose: these feeds are public already, and needing an
  // account to look at a public webcam would be theatre.
  cameras: () => request<CameraList>("/v1/cameras"),

  // Asking marks the camera as watched, so the server keeps a fresh reading ready and
  // the next call comes back in milliseconds rather than seconds.
  cameraAnalysis: (id: string) =>
    request<CameraAnalysis>(`/v1/cameras/${encodeURIComponent(id)}/analysis`),

  lookupPlate: (plate: string) =>
    request<PlateLookup>("/v1/vehicles/lookup-rdw", {
      method: "POST",
      body: JSON.stringify({ plate }),
    }),

  createVehicle: (vehicle: Record<string, unknown>) =>
    request<Vehicle>("/v1/vehicles", { method: "POST", body: JSON.stringify(vehicle) }),

  deleteVehicle: (id: number) => request<void>(`/v1/vehicles/${id}`, { method: "DELETE" }),

  confirm: (targetKind: string, targetId: number, outcome: string) =>
    request<{ accepted: boolean; message: string }>("/v1/observations/user-confirmation", {
      method: "POST",
      body: JSON.stringify({ target_kind: targetKind, target_id: targetId, outcome }),
    }),
};

/**
 * Subscribe to live availability for the targets a driver is currently acting on.
 *
 * Only for an active search. Holding a stream open for an idle map view costs battery
 * on a phone in someone's car and buys nothing.
 */
export function streamAvailability(
  targets: string[],
  onUpdate: (items: unknown[]) => void,
): () => void {
  if (targets.length === 0) return () => {};
  const source = new EventSource(
    `/v1/availability/stream?targets=${encodeURIComponent(targets.slice(0, 50).join(","))}`,
  );
  source.addEventListener("availability", (event) => {
    try {
      const payload = JSON.parse((event as MessageEvent).data);
      onUpdate(payload.items ?? []);
    } catch {
      /* a malformed frame is not worth tearing the stream down for */
    }
  });
  return () => source.close();
}
