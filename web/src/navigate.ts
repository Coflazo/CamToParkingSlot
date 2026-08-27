/**
 * The handoff sheet: "take me there".
 *
 * This is the moment the product either keeps its promise or throws it away. Everything
 * upstream, the surveyed polygons, the fit engine, the ranking, exists to name one exact
 * place to put a car. Handing that over as a street name would undo all of it, because
 * the receiving app re-geocodes the text against its own database and lands somewhere
 * near, not somewhere exact.
 *
 * So the sheet shows the coordinate. Not because a driver reads coordinates, but because
 * showing it is the honest way to say what is being handed over, and because when the
 * point is a car park centroid rather than a door the sheet says that in words.
 *
 * The links themselves are built server-side by the C++ core, so there is one
 * implementation of the provider templates rather than a copy here that slowly drifts.
 */

import type { Navigation, Recommendation } from "./api";
import { escapeHtml } from "./map";

/** Brand marks, drawn rather than imported, so nothing depends on an icon CDN. */
const GLYPHS: Record<string, string> = {
  google_maps: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2a7 7 0 0 0-7 7c0 5.25 7 13 7 13s7-7.75 7-13a7 7 0 0 0-7-7Zm0 9.5A2.5 2.5 0 1 1 12 6.5a2.5 2.5 0 0 1 0 5Z"/></svg>`,
  apple_maps: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 20.5 9.5 3.5l3.2 8.2 2.1-2.6L21 20.5l-8.4-4.2Z"/></svg>`,
  waze: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3a8 8 0 0 0-8 8v3.2c0 .9-.4 1.6-1 2.2 1.6.9 4.9 1.6 9 1.6s7.4-.7 9-1.6c-.6-.6-1-1.3-1-2.2V11a8 8 0 0 0-8-8Zm-3 8.2a1.2 1.2 0 1 1 0-2.4 1.2 1.2 0 0 1 0 2.4Zm6 0a1.2 1.2 0 1 1 0-2.4 1.2 1.2 0 0 1 0 2.4Z"/></svg>`,
  yandex: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M13.6 3h3.1v18h-2.9v-6.6h-.6L9.6 21H6.3l4.2-7.2A5.2 5.2 0 0 1 7 8.7C7 5.4 9.5 3 13.6 3Zm.2 2.4c-2.2 0-3.6 1.4-3.6 3.4s1.2 3.3 3.5 3.3h.1V5.4Z"/></svg>`,
  openstreetmap: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2 2 7l10 5 10-5-10-5Zm-10 10 10 5 10-5m-20 5 10 5 10-5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/></svg>`,
  geo: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2a7 7 0 0 1 7 7c0 5.2-7 13-7 13S5 14.2 5 9a7 7 0 0 1 7-7Z" fill="none" stroke="currentColor" stroke-width="1.8"/><circle cx="12" cy="9" r="2.4"/></svg>`,
};

let sheet: HTMLDivElement | null = null;
let lastFocused: HTMLElement | null = null;

function ensureSheet(): HTMLDivElement {
  if (sheet) return sheet;

  sheet = document.createElement("div");
  sheet.className = "nav-sheet";
  sheet.setAttribute("role", "dialog");
  sheet.setAttribute("aria-modal", "true");
  sheet.setAttribute("aria-labelledby", "nav-sheet-title");
  sheet.hidden = true;
  document.body.appendChild(sheet);

  sheet.addEventListener("click", (event) => {
    if (event.target === sheet) close();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && sheet && !sheet.hidden) close();
  });

  return sheet;
}

/**
 * Coordinates, grouped for reading.
 *
 * A driver never types these, but seeing them is what makes the precision claim
 * checkable rather than a slogan. Seven decimals is roughly a centimetre.
 */
function formatCoordinate(value: number): string {
  return value.toFixed(7);
}

export function open(result: Recommendation): void {
  const nav: Navigation | null = result.navigation;
  const node = ensureSheet();

  if (!nav || nav.links.length === 0) {
    // No usable coordinate is a real state and gets a real answer, not a dead button.
    node.innerHTML = `
      <div class="nav-panel" data-state="unavailable">
        <h2 id="nav-sheet-title">Cannot route here</h2>
        <p class="nav-note">This result has no usable coordinate, so handing it to a
        navigation app would send you somewhere we cannot vouch for.</p>
        <button type="button" class="nav-close" data-close>Close</button>
      </div>`;
  } else {
    const precision = nav.is_entrance
      ? "entrance"
      : result.is_exact_space
        ? "surveyed bay"
        : "car park";

    node.innerHTML = `
      <div class="nav-panel">
        <header class="nav-head">
          <p class="nav-eyebrow">Take me there</p>
          <h2 id="nav-sheet-title">${escapeHtml(result.name)}</h2>
        </header>

        <div class="nav-point">
          <div class="nav-coords">
            <span class="nav-coord"><em>lat</em>${formatCoordinate(nav.lat)}</span>
            <span class="nav-coord"><em>lon</em>${formatCoordinate(nav.lon)}</span>
          </div>
          <p class="nav-note">
            <span class="nav-tag" data-kind="${escapeHtml(precision)}">${escapeHtml(precision)}</span>
            ${escapeHtml(nav.point_description)}
          </p>
        </div>

        <ul class="nav-apps">
          ${nav.links
            .map(
              (link, index) => `
            <li style="--i:${index}">
              <a href="${escapeHtml(link.url)}" target="_blank" rel="noopener noreferrer"
                 class="nav-app" data-provider="${escapeHtml(link.provider)}">
                <span class="nav-glyph">${GLYPHS[link.provider] ?? GLYPHS["geo"]}</span>
                <span class="nav-app-name">${escapeHtml(link.display_name)}</span>
                <span class="nav-arrow" aria-hidden="true">&rarr;</span>
              </a>
            </li>`,
            )
            .join("")}
        </ul>

        <div class="nav-foot">
          <button type="button" class="nav-copy" data-copy="${formatCoordinate(nav.lat)}, ${formatCoordinate(nav.lon)}">
            Copy coordinates
          </button>
          <button type="button" class="nav-close" data-close>Close</button>
        </div>
      </div>`;
  }

  lastFocused = document.activeElement as HTMLElement | null;
  node.hidden = false;
  // Two frames: one to un-hide, one so the transition has a start state to run from.
  requestAnimationFrame(() => requestAnimationFrame(() => node.classList.add("is-open")));

  node.querySelector<HTMLElement>("[data-close]")?.addEventListener("click", close);

  const copyButton = node.querySelector<HTMLButtonElement>("[data-copy]");
  copyButton?.addEventListener("click", async () => {
    const text = copyButton.dataset["copy"] ?? "";
    try {
      await navigator.clipboard.writeText(text);
      copyButton.textContent = "Copied";
    } catch {
      // Clipboard access is refused in plenty of ordinary situations. Saying so beats a
      // button that silently does nothing.
      copyButton.textContent = "Copy blocked by the browser";
    }
    window.setTimeout(() => {
      copyButton.textContent = "Copy coordinates";
    }, 2000);
  });

  node.querySelector<HTMLElement>(".nav-app, .nav-close")?.focus();
}

export function close(): void {
  if (!sheet) return;
  sheet.classList.remove("is-open");
  // Matches the transition, so the sheet is not yanked out from under the animation.
  window.setTimeout(() => {
    if (sheet) sheet.hidden = true;
  }, 260);
  lastFocused?.focus();
}
