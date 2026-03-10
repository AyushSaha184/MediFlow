import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // VITE_API_URL is injected at build time by Vercel.
  // During local dev the proxy below handles it (no env var needed).
  server: {
    proxy: {
      "/session": "http://localhost:8000",
      "/intake": "http://localhost:8000",
      "/analyze-medical-session": "http://localhost:8000",
      "/health": "http://localhost:8000",
    },
  },
});
