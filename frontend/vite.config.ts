import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const backend = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  // The lazily loaded three.js viewer chunk is ~630 kB; that's fine for a local app.
  build: { chunkSizeWarningLimit: 800 },
  server: {
    proxy: {
      "/api": { target: backend, ws: true },
      "/files": backend,
    },
  },
});
