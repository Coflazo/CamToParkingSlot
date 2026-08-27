/* CamToParkingSlot service worker.
 *
 * Caches the shell so the app opens without a network, which matters in an
 * underground car park. Search responses are deliberately NOT cached: a parking
 * availability answer served from yesterday is worse than no answer at all, and
 * the whole product rests on never presenting stale data as current.
 */

const SHELL_CACHE = "parkfit-shell-v1";
const SHELL = ["/", "/index.html", "/manifest.webmanifest"];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(SHELL_CACHE).then((cache) => cache.addAll(SHELL)).then(() => self.skipWaiting()),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(keys.filter((key) => key !== SHELL_CACHE).map((key) => caches.delete(key))),
      )
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  // Never serve an API response from cache. Availability has a shelf life measured in
  // seconds, and the freshness labels the UI shows would become lies.
  if (url.pathname.startsWith("/v1/") || url.pathname === "/health") return;
  if (event.request.method !== "GET") return;

  event.respondWith(
    caches.match(event.request).then((cached) => cached ?? fetch(event.request)),
  );
});
