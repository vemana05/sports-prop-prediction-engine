#!/bin/bash
export PATH="/Users/vemana/.docker/bin:/usr/local/bin:/opt/homebrew/bin:$PATH"
# Daily forward-collection job (spec Section 5.7): log predictions for every
# upcoming game, THEN pull real odds — in that order, so a prediction is
# always on record before the matching line. Meant to run once a day,
# comfortably before the earliest possible NBA tip-off (see the launchd
# plist in this same directory for the schedule).
#
# What this needs to actually run, every time it fires:
#   1. Docker Desktop running (the daemon, not just installed) — `docker
#      compose run` fails immediately otherwise.
#   2. The Mac awake (not asleep) at the scheduled time — see this repo's
#      setup notes for why and how to arrange that with `pmset`.
#   3. ODDS_API_KEY set in .env (odds_theoddsapi.py no-ops cleanly with a
#      warning, not a crash, if it's missing — but no key means no odds).
#
# Runs under `caffeinate -i` (see below) so the Mac can't idle-sleep mid-job.
#
# Logs its own output to logs/daily_odds_job_<date>.log so a failure
# overnight is visible the next time someone looks, not just lost.

# Keep the Mac awake until the job finishes: re-run this script under
# `caffeinate -i` (idle sleep blocked while it runs), once. Absent outside
# macOS (e.g. the test container), where it just runs as is.
if [ -z "${PARLAY_CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
    export PARLAY_CAFFEINATED=1
    exec caffeinate -i /bin/bash "$0" "$@"
fi

set -uo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs
LOG_FILE="logs/daily_odds_job_$(date +%Y%m%d_%H%M%S).log"
# requests puts ?apiKey=... into connection-error URLs; keep keys out of logs.
redact() { sed -E 's/(apiKey|api_key)=[^&" )]+/\1=REDACTED/g'; }

echo "=== daily_odds_job starting $(date) ===" | tee -a "$LOG_FILE"

if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker daemon is not running — cannot proceed. Is Docker Desktop open?" | tee -a "$LOG_FILE"
    exit 1
fi

# DuckDB allows a single writer: if the api container has warehouse.duckdb
# open, stop it for the run and bring it back on exit (success or failure).
if [ -n "$(docker compose ps -q --status running api)" ]; then
    echo "--- stopping api container (holds warehouse.duckdb lock) ---" | tee -a "$LOG_FILE"
    docker compose stop api >>"$LOG_FILE" 2>&1
    trap 'docker compose start api >>"$LOG_FILE" 2>&1' EXIT
fi

# After a late wake the Mac can be up with no network for a while (2026-10-14:
# every call failed DNS for the whole run, and the job exited 1). Wait up to
# 30 min for the containers to resolve the Odds API host before spending
# anything; if it never does, fail clearly instead of logging tracebacks.
echo "--- waiting for network (DNS from inside Docker) ---" | tee -a "$LOG_FILE"
NET_OK=0
for i in $(seq 1 30); do
    if docker compose run --rm --no-deps backend python -c \
        "import socket; socket.getaddrinfo('api.the-odds-api.com', 443)" >/dev/null 2>&1; then
        NET_OK=1; break
    fi
    echo "no network yet (attempt $i/30); retrying in 60s" | tee -a "$LOG_FILE"
    sleep 60
done
if [ "$NET_OK" -ne 1 ]; then
    echo "ERROR: no network after 30 min — nothing collected today" | tee -a "$LOG_FILE"
    exit 1
fi

# Refresh availability before predicting: log_predictions reads the latest
# injury report for teammates-out. Failures here are logged, not fatal — a
# stale report shouldn't stop predictions or odds collection.
echo "--- refresh: NBA injury report (latest) ---" | tee -a "$LOG_FILE"
docker compose run --rm backend python -m backend.ingest.injuries_nba_official --latest 2>&1 | redact | tee -a "$LOG_FILE"

echo "--- refresh: Sleeper player status ---" | tee -a "$LOG_FILE"
docker compose run --rm backend python -m backend.ingest.sleeper 2>&1 | redact | tee -a "$LOG_FILE"

# Load finished games (player logs, schedule results, box status) and
# rebuild nba_features BEFORE anything is predicted, so current form and
# the early-season counter see last night's games. Resumable (skips games
# already loaded) and waits for stats.nba.com DNS itself. Models are not
# retrained here. A failure is logged and reported in the finish line, but
# predictions and odds still run on whatever is loaded, and the step alone is
# retried at 1 PM by nba_gamelog_retry_job.sh (it looks for RETRY_MARKER).
# The step caps itself at ~10 min (nba_daily_games.STEP_BUDGET_S).
RETRY_MARKER="logs/nba_gamelogs_retry_$(date +%Y%m%d)"
echo "--- refresh: NBA game logs (finished games) + nba_features ---" | tee -a "$LOG_FILE"
docker compose run --rm backend python -m backend.ingest.nba_daily_games 2>&1 | redact | tee -a "$LOG_FILE"
GAMELOG_STATUS=${PIPESTATUS[0]}
if [ "$GAMELOG_STATUS" -ne 0 ]; then
    echo "game logs failed; $RETRY_MARKER left for the 1 PM retry" | tee -a "$LOG_FILE"
    touch "$RETRY_MARKER"
else
    rm -f "$RETRY_MARKER"
fi

# Grade logged NBA points predictions against the lines and box scores
# (reports/nba_model_vs_lines.json, the "NBA model vs. lines" page). Read-only
# on the warehouse, and runs even if game logs failed (the page flags a stale
# box-score date). A failure is reported in the finish line but doesn't stop
# predictions or odds. Recommends a shrink/cap; nothing applies it.
echo "--- grade: NBA model vs. lines ---" | tee -a "$LOG_FILE"
docker compose run --rm backend python -m backend.backtest.nba_model_vs_lines 2>&1 | redact | tee -a "$LOG_FILE"
GRADE_STATUS=${PIPESTATUS[0]}

echo "--- step 0: warming schedule projections (next game day) ---" | tee -a "$LOG_FILE"
docker compose run --rm backend python -m backend.predict.schedule_projections 2>&1 | redact | tee -a "$LOG_FILE"

echo "--- step 1/2: logging predictions ---" | tee -a "$LOG_FILE"
docker compose run --rm backend python -m backend.predict.log_predictions 2>&1 | redact | tee -a "$LOG_FILE"
PREDICT_STATUS=${PIPESTATUS[0]}

echo "--- step 2/2: collecting odds ---" | tee -a "$LOG_FILE"
docker compose run --rm backend python -m backend.ingest.odds_theoddsapi 2>&1 | redact | tee -a "$LOG_FILE"
ODDS_STATUS=${PIPESTATUS[0]}

echo "=== daily_odds_job finished $(date) (gamelogs exit $GAMELOG_STATUS, grade exit $GRADE_STATUS, predict exit $PREDICT_STATUS, odds exit $ODDS_STATUS) ===" | tee -a "$LOG_FILE"

if [ "$GAMELOG_STATUS" -ne 0 ] || [ "$GRADE_STATUS" -ne 0 ] || [ "$PREDICT_STATUS" -ne 0 ] || [ "$ODDS_STATUS" -ne 0 ]; then
    exit 1
fi
