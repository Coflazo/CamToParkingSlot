import { defineConfig } from "vite";

export default defineConfig({
  server: {
    port: 5173,
    host: "127.0.0.1",
    proxy: {
      // Same-origin during development, so the browser never sees a cross-origin
      // request and no CORS preflight sits between a search and its results.
      "/v1": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/health": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
  build: {
    target: "es2022",
    sourcemap: true,
    rollupOptions: {
      output: {
        // MapLibre is most of the bundle. Splitting it lets the shell and the search
        // form paint before the map library has finished downloading, which is the
        // difference between a usable and an unusable first load on mobile data.
        manualChunks: { maplibre: ["maplibre-gl"] },
      },
    },
  },
});
