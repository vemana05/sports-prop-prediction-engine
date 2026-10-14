#!/bin/bash
export PATH="/Users/vemana/.docker/bin:/usr/local/bin:/opt/homebrew/bin:$PATH"
# Weekly NFL refresh into data/nfl_warehouse.duckdb (backend/nfl_db.py), three modes:
#   prep     (default; Saturday)  schedule, injuries, depth charts before the games,
#                                 then logs projections for players/games not yet
#                                 logged and retries OddsPapi /odds for games that
#                                 had no prop lines on Wednesday
#   lines    (Wednesday)          logs this week's NFL projections
#                                 (backend/predict/log_nfl_predictions.py), then
#                                 OddsPapi: the week's fixtures, player names and
#                                 opening prop lines (backend/ingest/odds_oddspapi_nfl.py)
#   results  (Tuesday)            scores, player stats, snap counts, injuries and
#                                 depth charts for finished games, then rebuilds
#                                 nfl_features and backs up, then OddsPapi closing
#                                 lines for last week's games, then (only with
#                                 NFL_WEEKLY_RETRAIN=1 in .env) trains last week's
#                                 walk-forward NFL models (nfl_<market>_thru<season>w<NN>,
#                                 older versions kept), rebuilds the Replay table and
#                                 warms this week's projection cache.
# Separate from daily_odds_job.sh on purpose: it touches only the NFL
# warehouse, never warehouse.duckdb, and spends no Odds API credits. The
# OddsPapi steps spend nothing unless ODDSPAPI_API_KEY is set and
# ODDSPAPI_COLLECT_ENABLED=1 in .env, and stop at 200 of 250 requests/month.
# Steps added since the image was last built run in the api service, whose
# source is mounted (the backend service runs the code baked into the image).
#
# Same requirements as the daily job: Docker Desktop running and the Mac
# awake at the scheduled time (see com.parlaybuilder.weeklynfljob.plist).

set -uo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs
MODE="${1:-prep}"
case "$MODE" in prep|lines|results) ;; *) echo "usage: $0 [prep|lines|results]"; exit 2 ;; esac
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

# Projections go on record before lines are collected (same order as the
# NBA daily job). Never overwritten, so a rerun only adds what's missing.
log_predictions() {
    echo "--- logging NFL projections for the next 7 days ---" | tee -a "$LOG_FILE"
    docker compose run --rm --no-deps api python -m backend.predict.log_nfl_predictions 2>&1 | tee -a "$LOG_FILE"
    [ "${PIPESTATUS[0]}" -eq 0 ] || echo "WARNING: NFL prediction logging failed" | tee -a "$LOG_FILE"
}

if [ "$MODE" = lines ]; then
    log_predictions
    echo "--- OddsPapi: this week's fixtures, names, opening prop lines ---" | tee -a "$LOG_FILE"
    docker compose run --rm --no-deps api python -m backend.ingest.odds_oddspapi_nfl wednesday 2>&1 | tee -a "$LOG_FILE"
    STATUS=${PIPESTATUS[0]}
elif [ "$MODE" = prep ]; then
    echo "--- nflverse: schedule, injuries, depth charts (season $SEASON) ---" | tee -a "$LOG_FILE"
    docker compose run --rm backend python -m backend.ingest.nfl_nflverse \
        --stage schedule --stage injuries --stage depth_charts --seasons "$SEASON" 2>&1 | tee -a "$LOG_FILE"
    STATUS=${PIPESTATUS[0]}
    log_predictions
    echo "--- OddsPapi: retry games with no prop lines on Wednesday ---" | tee -a "$LOG_FILE"
    docker compose run --rm --no-deps api python -m backend.ingest.odds_oddspapi_nfl saturday 2>&1 | tee -a "$LOG_FILE"
    [ "${PIPESTATUS[0]}" -eq 0 ] || echo "WARNING: OddsPapi Saturday retry failed (see above)" | tee -a "$LOG_FILE"
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
        # Warm the models + this week's projections so the first page load
        # doesn't have to. Runs in the api service for its live-mounted source;
        # `run` doesn't publish ports, so it won't clash with the stopped api.
        echo "--- weekly NFL retrain + projection cache ---" | tee -a "$LOG_FILE"
        docker compose run --rm --no-deps api python -m backend.predict.nfl_projections --retrain 2>&1 | tee -a "$LOG_FILE"
        [ "${PIPESTATUS[0]}" -eq 0 ] || echo "WARNING: NFL retrain/warm-up failed; the API will train in the background on first request" | tee -a "$LOG_FILE"
        # Replay rows for the new week, projected by the version that existed before it.
        echo "--- rebuilding NFL replay ---" | tee -a "$LOG_FILE"
        docker compose run --rm --no-deps api python -m backend.backtest.nfl_replay 2>&1 | tee -a "$LOG_FILE"
        [ "${PIPESTATUS[0]}" -eq 0 ] || echo "WARNING: NFL replay rebuild failed" | tee -a "$LOG_FILE"
    fi
    # Closing prop lines for last week's games. A failure here is logged but
    # doesn't fail the results job: stats and features are already in.
    echo "--- OddsPapi: closing prop lines for last week's games ---" | tee -a "$LOG_FILE"
    docker compose run --rm --no-deps api python -m backend.ingest.odds_oddspapi_nfl tuesday 2>&1 | tee -a "$LOG_FILE"
    [ "${PIPESTATUS[0]}" -eq 0 ] || echo "WARNING: OddsPapi closing-line step failed (see above)" | tee -a "$LOG_FILE"
fi

echo "=== weekly_nfl_job ($MODE) finished $(date) (exit $STATUS) ===" | tee -a "$LOG_FILE"
exit "$STATUS"
