import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: "0.0.0.0",
    allowedHosts: [
      "fatimahstudio.local", "fatimahstudio", ".local",
      "100.107.31.9",          // host's Meshnet IP
      ".nord",                  // any *.nord hostname (mesh peers)
    ],
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        ws: true,
      },
    },
  },
});
