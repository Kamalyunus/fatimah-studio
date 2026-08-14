#!/usr/bin/env bash
# Fatimah Studio launcher — resource-safe.
#   * The frontend is a static build served by the backend on :8000 (no vite/node
#     dev server), so there is nothing for the hermes gateway to reap.
#   * A single-instance lock stops double-clicks from racing.
#   * Each service is started ONLY if it isn't already active (systemctl start is a
#     no-op on a running unit anyway, so services can never double-launch).
#   * If the studio is already open in a Chrome app window, we do NOT open a second one.
set -u

URL="http://127.0.0.1:8000"

# --- 1) Single-instance guard: if another launch is mid-flight, bail out quietly. ---
LOCK="/tmp/fatimah-studio-launch.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
  command -v notify-send >/dev/null && notify-send "Fatimah Studio" "Already starting…"
  exit 0
fi

# --- 2) Start each service only if it isn't already running. ---
start_if_down() {  # $1 = scope flag ("" or "--user"), $2 = unit
  if [ "$(systemctl $1 is-active "$2" 2>/dev/null)" != "active" ]; then
    systemctl $1 start "$2" 2>/dev/null
  fi
}
start_if_down "--user" comfyui.service
# ollama may be a system unit or a user unit depending on the box; try system first.
if [ "$(systemctl is-active ollama 2>/dev/null)" != "active" ] \
   && [ "$(systemctl --user is-active ollama 2>/dev/null)" != "active" ]; then
  systemctl start ollama 2>/dev/null || systemctl --user start ollama 2>/dev/null
fi
# The backend also serves the built frontend, so this is the only web service to start.
start_if_down "--user" fatimah-backend.service

# --- 3) Wait (up to ~30s) for the app to actually answer. ---
for _ in $(seq 1 30); do
  [ "$(curl -s -m 3 -o /dev/null -w "%{http_code}" "$URL/api/health" 2>/dev/null)" = "200" ] && break
  sleep 1
done

# --- 4) Open the app — but not if a studio app window is already open. ---
if pgrep -f -- "--app=http://(127\.0\.0\.1|localhost):8000" >/dev/null 2>&1; then
  command -v notify-send >/dev/null && notify-send "Fatimah Studio" "Already open."
  exit 0
fi
if command -v google-chrome >/dev/null 2>&1; then
  exec google-chrome --app="$URL" >/dev/null 2>&1
else
  exec xdg-open "$URL" >/dev/null 2>&1
fi
