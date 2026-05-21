#!/usr/bin/env bash
# Run an EVA smoke against the IIFL bot, with the simulated CUSTOMER voice
# routed through our local supertonic_server (instead of cloud Cartesia).
#
# Effect: in the resulting EVA run, the user-side audio you hear is generated
# by supertonic; the bot side is still real Cartesia (Priya) — we don't touch
# outbound_voice_agent.
#
# Prereqs:
#   1. supertonic_server is running and warm:
#        cd ../ && PYTHONPATH=src /Users/harsh/Desktop/audio_test/.venv/bin/python \
#          -m supertonic_server.server
#      (default port 7799)
#   2. outbound_voice_agent (prod bot) is running and reachable via the
#      BOT_WS_URL already set in eva/.env. Nothing to change there.
#
# Usage:
#     bash scripts/run_eva_with_supertonic.sh                 # voice=F1, record iifl.1.1
#     SUPERTONIC_VOICE=M3 bash scripts/run_eva_with_supertonic.sh
#     EVA_RECORD_IDS='["iifl.1.2"]' bash scripts/run_eva_with_supertonic.sh

set -euo pipefail

# ---- config (env-overridable) ----------------------------------------------
SUPERTONIC_HOST="${SUPERTONIC_HOST:-127.0.0.1}"
SUPERTONIC_PORT="${SUPERTONIC_PORT:-7799}"
SUPERTONIC_VOICE="${SUPERTONIC_VOICE:-F1}"
EVA_DOMAIN="${EVA_DOMAIN:-iifl}"
EVA_RECORD_IDS="${EVA_RECORD_IDS:-[\"iifl.1.1\"]}"

EVA_DIR="${EVA_DIR:-/Users/harsh/Desktop/audio_test/eva}"
EVA_BIN="${EVA_DIR}/.venv/bin/eva"

SUPERTONIC_WS_URL="ws://${SUPERTONIC_HOST}:${SUPERTONIC_PORT}/tts/websocket"
SUPERTONIC_READY_URL="http://${SUPERTONIC_HOST}:${SUPERTONIC_PORT}/ready"

# ---- preflight -------------------------------------------------------------
echo "» Verifying supertonic_server at ${SUPERTONIC_READY_URL}"
if ! ready_json="$(curl --silent --show-error --max-time 2 "${SUPERTONIC_READY_URL}" 2>&1)"; then
    echo "ERROR: supertonic_server is not reachable at ${SUPERTONIC_HOST}:${SUPERTONIC_PORT}." >&2
    echo "       Start it in another terminal first:" >&2
    echo "         cd /Users/harsh/Desktop/audio_test/supertonic_server" >&2
    echo "         PYTHONPATH=src /Users/harsh/Desktop/audio_test/.venv/bin/python -m supertonic_server.server" >&2
    exit 1
fi
if ! echo "${ready_json}" | grep -q '"ready":true'; then
    echo "ERROR: supertonic_server replied ${ready_json} — not ready yet." >&2
    exit 1
fi
echo "  ready: ${ready_json}"

# Sanity-check the requested voice is in the catalog.
voices_json="$(curl --silent --max-time 2 "http://${SUPERTONIC_HOST}:${SUPERTONIC_PORT}/voices")"
if ! echo "${voices_json}" | grep -q "\"name\":\"${SUPERTONIC_VOICE}\""; then
    echo "ERROR: voice '${SUPERTONIC_VOICE}' is not in supertonic's catalog." >&2
    echo "       Available: ${voices_json}" >&2
    exit 1
fi
echo "  voice OK: ${SUPERTONIC_VOICE}"

# ---- run -------------------------------------------------------------------
echo "» Launching EVA"
echo "  EVA_USER_SIM_TTS    = supertonic"
echo "  EVA_SUPERTONIC_URL  = ${SUPERTONIC_WS_URL}"
echo "  EVA_SUPERTONIC_VOICE = ${SUPERTONIC_VOICE}"
echo "  domain               = ${EVA_DOMAIN}"
echo "  record-ids           = ${EVA_RECORD_IDS}"
echo

cd "${EVA_DIR}"

EVA_USER_SIM_TTS="supertonic" \
EVA_SUPERTONIC_URL="${SUPERTONIC_WS_URL}" \
EVA_SUPERTONIC_VOICE="${SUPERTONIC_VOICE}" \
"${EVA_BIN}" --domain "${EVA_DOMAIN}" --record-ids "${EVA_RECORD_IDS}" 2>&1 | tee /tmp/eva_supertonic_run.log

rc=${PIPESTATUS[0]}

# ---- post-run summary ------------------------------------------------------
echo
run_dir="$(grep -oE "output/[0-9-]+_[0-9-]+\.[0-9]+_[^ ]+" /tmp/eva_supertonic_run.log | tail -1 || true)"
if [[ -n "${run_dir}" ]]; then
    echo "» Run dir: ${EVA_DIR}/${run_dir}"
fi

echo "» supertonic_server hits during this run:"
# Pull ctx ... done lines from the most recent server log. Use lsof to find
# the listening pid; if its stdout is not redirected to a file we can read,
# fall back to a hint.
srv_pid="$(lsof -i ":${SUPERTONIC_PORT}" -t 2>/dev/null | head -n1 || true)"
if [[ -n "${srv_pid}" ]]; then
    echo "  (server pid=${srv_pid}; tail its stdout to see 'ctx ... done: TTFB=...' lines)"
fi

exit "${rc}"
