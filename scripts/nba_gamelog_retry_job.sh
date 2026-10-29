#!/bin/bash
export PATH="/Users/vemana/.docker/bin:/usr/local/bin:/opt/homebrew/bin:$PATH"
# 1 PM ET retry of daily_odds_job.sh's NBA game-log step, ONLY when this
# morning's run failed it (daily_odds_job.sh leaves logs/nba_gamelogs_retry_
# <YYYYMMDD> behind). Re-runs backend.ingest.nba_daily_games once — game
# logs, results, box status, nba_features — and nothing else: no odds, no
# predictions (the morning's predictions record the stale game date they
# were made on in model_predictions_log.features_through).
#
# Same requirements as the daily job: Docker Desktop running and the Mac
# awake (see com.parlaybuilder.nbagamelogretry.plist).

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
LOG_FILE="logs/nba_gamelog_retry_$(date +%Y%m%d_%H%M%S).log"
RETRY_MARKER="logs/nba_gamelogs_retry_$(date +%Y%m%d)"
redact() { sed -E 's/(apiKey|api_key)=[^&" )]+/\1=REDACTED/g'; }

echo "=== nba_gamelog_retry starting $(date) ===" | tee -a "$LOG_FILE"

if [ ! -e "$RETRY_MARKER" ]; then
    echo "morning game-log step did not fail today; nothing to retry" | tee -a "$LOG_FILE"
    echo "=== nba_gamelog_retry finished $(date) (gamelogs exit 0) ===" | tee -a "$LOG_FILE"
    exit 0
fi
# Once only, whatever the outcome: the next chance is tomorrow's daily job.
rm -f "$RETRY_MARKER"

if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker daemon is not running — cannot proceed. Is Docker Desktop open?" | tee -a "$LOG_FILE"
    exit 1
fi

# DuckDB allows a single writer: stop the api container for the run.
if [ -n "$(docker compose ps -q --status running api)" ]; then
    echo "--- stopping api container (holds warehouse.duckdb lock) ---" | tee -a "$LOG_FILE"
    docker compose stop api >>"$LOG_FILE" 2>&1
    trap 'docker compose start api >>"$LOG_FILE" 2>&1' EXIT
fi

echo "--- retry: NBA game logs (finished games) + nba_features ---" | tee -a "$LOG_FILE"
docker compose run --rm backend python -m backend.ingest.nba_daily_games 2>&1 | redact | tee -a "$LOG_FILE"
GAMELOG_STATUS=${PIPESTATUS[0]}

echo "=== nba_gamelog_retry finished $(date) (gamelogs exit $GAMELOG_STATUS) ===" | tee -a "$LOG_FILE"
exit "$GAMELOG_STATUS"
