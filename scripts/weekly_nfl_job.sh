#!/bin/bash
export PATH="/Users/vemana/.docker/bin:/usr/local/bin:/opt/homebrew/bin:$PATH"
# Weekly NFL refresh into data/nfl_warehouse.duckdb (backend/nfl_db.py), two modes:
#   prep     (default; Saturday)  schedule, injuries, depth charts before the games
#   results  (Tuesday)            scores, player stats, snap counts, injuries and
#                                 depth charts for finished games, then rebuilds
#                                 nfl_features and backs up. Never retrains models.
# Separate from daily_odds_job.sh on purpose: it touches only the NFL
# warehouse, never warehouse.duckdb, and spends no Odds API credits.
#
# Same requirements as the daily job: Docker Desktop running and the Mac
# awake at the scheduled time (see com.parlaybuilder.weeklynfljob.plist).

set -uo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs
MODE="${1:-prep}"
case "$MODE" in prep|results) ;; *) echo "usage: $0 [prep|results]"; exit 2 ;; esac
LOG_FILE="logs/weekly_nfl_job_${MODE}_$(date +%Y%m%d_%H%M%S).log"

echo "=== weekly_nfl_job ($MODE) starting $(date) ===" | tee -a "$LOG_FILE"

if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker daemon is not running — cannot proceed. Is Docker Desktop open?" | tee -a "$LOG_FILE"
    exit 1
fi

# The api container opens nfl_warehouse.duckdb (read-only) per NFL request,
# which blocks a writer; stop it for the run and bring it back on exit.
if [ -n "$(docker compose ps -q --status running api)" ]; then
    echo "--- stopping api container (may hold nfl_warehouse.duckdb) ---" | tee -a "$LOG_FILE"
    docker compose stop api >>"$LOG_FILE" 2>&1
    trap 'docker compose start api >>"$LOG_FILE" 2>&1' EXIT
fi

# Current season only (matches nfl_nflverse.current_season): completed
# seasons never change, so re-reading them would be wasted work.
YEAR=$(date +%Y)
if [ "$(date +%-m)" -ge 9 ]; then SEASON=$YEAR; else SEASON=$((YEAR - 1)); fi

if [ "$MODE" = prep ]; then
    echo "--- nflverse: schedule, injuries, depth charts (season $SEASON) ---" | tee -a "$LOG_FILE"
    docker compose run --rm backend python -m backend.ingest.nfl_nflverse \
        --stage schedule --stage injuries --stage depth_charts --seasons "$SEASON" 2>&1 | tee -a "$LOG_FILE"
    STATUS=${PIPESTATUS[0]}
else
    echo "--- nflverse: schedule, player stats, snap counts, injuries, depth charts (season $SEASON) ---" | tee -a "$LOG_FILE"
    docker compose run --rm backend python -m backend.ingest.nfl_nflverse --no-backup \
        --stage schedule --stage player_stats --stage snap_counts --stage injuries --stage depth_charts \
        --seasons "$SEASON" 2>&1 | tee -a "$LOG_FILE"
    STATUS=${PIPESTATUS[0]}
    if [ "$STATUS" -eq 0 ]; then
        # Rebuilds features from the warehouse and backs up nfl_warehouse.duckdb.
        echo "--- rebuilding nfl_features + backup ---" | tee -a "$LOG_FILE"
        docker compose run --rm backend python -m backend.features.nfl_features 2>&1 | tee -a "$LOG_FILE"
        STATUS=${PIPESTATUS[0]}
        # The Schedule page caches each week's projections for a day; drop this
        # season's caches so new actuals show now instead of tomorrow.
        rm -f data/projections/nfl_${SEASON}_w*.json
    fi
fi

echo "=== weekly_nfl_job ($MODE) finished $(date) (exit $STATUS) ===" | tee -a "$LOG_FILE"
exit "$STATUS"
