import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/** Offline demo build: mocked backend (src/demo), no API keys needed. */
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    outDir: "dist-demo",
    rollupOptions: { input: "demo.html" },
    chunkSizeWarningLimit: 2000,
  },
});
