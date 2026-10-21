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
#                                 older versions kept), rebuilds the Replay table,
#                                 warms this week's projection cache, and last
#                                 rebuilds the Model vs. lines report
#                                 (backend/backtest/nfl_model_vs_lines.py).
# Separate from daily_odds_job.sh on purpose: it touches only the NFL
# warehouse, never warehouse.duckdb, and spends no Odds API credits. The
# OddsPapi steps spend nothing unless ODDSPAPI_API_KEY is set and
# ODDSPAPI_COLLECT_ENABLED=1 in .env, and stop at 200 of 250 requests/month.
# Both the backend and api services mount ./backend, so every step runs the
# current source.
#
# Same requirements as the daily job: Docker Desktop running and the Mac
# awake at the scheduled time (see com.parlaybuilder.weeklynfljob.plist).

set -uo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs
MODE="${1:-prep}"
case "$MODE" in prep|lines|results) ;; *) echo "usage: $0 [prep|lines|results]"; exit 2 ;; esac
LOG_FILE="logs/weekly_nfl_job_${MODE}_$(date +%Y%m%d_%H%M%S).log"
# requests puts ?apiKey=... into connection-error URLs; keep keys out of logs.
redact() { sed -E 's/(apiKey|api_key)=[^&" )]+/\1=REDACTED/g'; }

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

# After a late wake the Mac can be up with no network for a while (2026-10-20:
# the closing-line step died on RemoteDisconnected). Same wait as
# daily_odds_job.sh: up to 30 min for the containers to resolve the OddsPapi
# host before any OddsPapi step spends a request.
wait_for_network() {
    echo "--- waiting for network (DNS from inside Docker) ---" | tee -a "$LOG_FILE"
    for i in $(seq 1 30); do
        if docker compose run --rm --no-deps api python -c \
            "import socket; socket.getaddrinfo('api.oddspapi.io', 443)" >/dev/null 2>&1; then
            return 0
        fi
        echo "no network yet (attempt $i/30); retrying in 60s" | tee -a "$LOG_FILE"
        sleep 60
    done
    echo "ERROR: no network after 30 min — skipping OddsPapi, nothing spent" | tee -a "$LOG_FILE"
    return 1
}

# Error text requests/urllib3 print when the network, not the API, failed.
NET_ERROR='ConnectionError|NameResolutionError|Failed to resolve|Temporary failure in name resolution|Connection aborted|RemoteDisconnected|Max retries exceeded|Network is unreachable|timed out'

# One OddsPapi step ($1: wednesday|saturday|tuesday); sets ODDS_STATUS. With
# $2=retry, a run that failed on a network error is retried once after the
# network is back (fixtures are cached and games already collected are
# skipped, so a rerun spends only what the first run didn't).
run_oddspapi() {
    ODDS_STATUS=1
    wait_for_network || return 0
    local out
    out=$(mktemp)
    docker compose run --rm --no-deps api python -m backend.ingest.odds_oddspapi_nfl "$1" 2>&1 | redact | tee "$out" | tee -a "$LOG_FILE"
    ODDS_STATUS=${PIPESTATUS[0]}
    if [ "$ODDS_STATUS" -ne 0 ] && [ "${2:-}" = retry ] && grep -qE "$NET_ERROR" "$out"; then
        echo "--- OddsPapi $1 failed on a network error; retrying once ---" | tee -a "$LOG_FILE"
        if wait_for_network; then
            docker compose run --rm --no-deps api python -m backend.ingest.odds_oddspapi_nfl "$1" 2>&1 | redact | tee -a "$LOG_FILE"
            ODDS_STATUS=${PIPESTATUS[0]}
        fi
    fi
    rm -f "$out"
}

# Projections go on record before lines are collected (same order as the
# NBA daily job). Never overwritten, so a rerun only adds what's missing.
log_predictions() {
    echo "--- logging NFL projections for the next 7 days ---" | tee -a "$LOG_FILE"
    docker compose run --rm --no-deps api python -m backend.predict.log_nfl_predictions 2>&1 | redact | tee -a "$LOG_FILE"
    [ "${PIPESTATUS[0]}" -eq 0 ] || echo "WARNING: NFL prediction logging failed" | tee -a "$LOG_FILE"
}

if [ "$MODE" = lines ]; then
    log_predictions
    echo "--- OddsPapi: this week's fixtures, names, opening prop lines ---" | tee -a "$LOG_FILE"
    run_oddspapi wednesday retry
    STATUS=$ODDS_STATUS
elif [ "$MODE" = prep ]; then
    echo "--- nflverse: schedule, injuries, depth charts (season $SEASON) ---" | tee -a "$LOG_FILE"
    docker compose run --rm backend python -m backend.ingest.nfl_nflverse \
        --stage schedule --stage injuries --stage depth_charts --seasons "$SEASON" 2>&1 | redact | tee -a "$LOG_FILE"
    STATUS=${PIPESTATUS[0]}
    log_predictions
    echo "--- OddsPapi: retry games with no prop lines on Wednesday ---" | tee -a "$LOG_FILE"
    run_oddspapi saturday
    [ "$ODDS_STATUS" -eq 0 ] || echo "WARNING: OddsPapi Saturday retry failed (see above)" | tee -a "$LOG_FILE"
else
    echo "--- nflverse: schedule, player stats, snap counts, injuries, depth charts (season $SEASON) ---" | tee -a "$LOG_FILE"
    docker compose run --rm backend python -m backend.ingest.nfl_nflverse --no-backup \
        --stage schedule --stage player_stats --stage snap_counts --stage injuries --stage depth_charts \
        --seasons "$SEASON" 2>&1 | redact | tee -a "$LOG_FILE"
    STATUS=${PIPESTATUS[0]}
    if [ "$STATUS" -eq 0 ]; then
        # Rebuilds features from the warehouse and backs up nfl_warehouse.duckdb.
        echo "--- rebuilding nfl_features + backup ---" | tee -a "$LOG_FILE"
        docker compose run --rm backend python -m backend.features.nfl_features 2>&1 | redact | tee -a "$LOG_FILE"
        STATUS=${PIPESTATUS[0]}
        # The Schedule page caches each week's projections for a day; drop this
        # season's caches so new actuals show now instead of tomorrow.
        rm -f data/projections/nfl_${SEASON}_w*.json
        # Warm the models + this week's projections so the first page load
        # doesn't have to. Runs in the api service for its live-mounted source;
        # `run` doesn't publish ports, so it won't clash with the stopped api.
        echo "--- weekly NFL retrain + projection cache ---" | tee -a "$LOG_FILE"
        docker compose run --rm --no-deps api python -m backend.predict.nfl_projections --retrain 2>&1 | redact | tee -a "$LOG_FILE"
        [ "${PIPESTATUS[0]}" -eq 0 ] || echo "WARNING: NFL retrain/warm-up failed; the API will train in the background on first request" | tee -a "$LOG_FILE"
        # Replay rows for the new week, projected by the version that existed before it.
        echo "--- rebuilding NFL replay ---" | tee -a "$LOG_FILE"
        docker compose run --rm --no-deps api python -m backend.backtest.nfl_replay 2>&1 | redact | tee -a "$LOG_FILE"
        [ "${PIPESTATUS[0]}" -eq 0 ] || echo "WARNING: NFL replay rebuild failed" | tee -a "$LOG_FILE"
    fi
    # Closing prop lines for last week's games. A failure here is logged but
    # doesn't fail the results job: stats and features are already in.
    echo "--- OddsPapi: closing prop lines for last week's games ---" | tee -a "$LOG_FILE"
    run_oddspapi tuesday
    [ "$ODDS_STATUS" -eq 0 ] || echo "WARNING: OddsPapi closing-line step failed (see above)" | tee -a "$LOG_FILE"
    # Model vs. lines: logged projections against those closing lines and the
    # actual stats (reports/nfl_model_vs_lines.json). Local data only; no API calls.
    echo "--- model vs. closing lines report ---" | tee -a "$LOG_FILE"
    docker compose run --rm --no-deps api python -m backend.backtest.nfl_model_vs_lines 2>&1 | redact | tee -a "$LOG_FILE"
    [ "${PIPESTATUS[0]}" -eq 0 ] || echo "WARNING: model vs. lines report failed" | tee -a "$LOG_FILE"
fi

echo "=== weekly_nfl_job ($MODE) finished $(date) (exit $STATUS) ===" | tee -a "$LOG_FILE"
exit "$STATUS"
