#!/bin/bash
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
# Logs its own output to logs/daily_odds_job_<date>.log so a failure
# overnight is visible the next time someone looks, not just lost.

set -uo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs
LOG_FILE="logs/daily_odds_job_$(date +%Y%m%d_%H%M%S).log"

echo "=== daily_odds_job starting $(date) ===" | tee -a "$LOG_FILE"

if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker daemon is not running — cannot proceed. Is Docker Desktop open?" | tee -a "$LOG_FILE"
    exit 1
fi

echo "--- step 1/2: logging predictions ---" | tee -a "$LOG_FILE"
docker compose run --rm backend python -m backend.predict.log_predictions 2>&1 | tee -a "$LOG_FILE"
PREDICT_STATUS=${PIPESTATUS[0]}

echo "--- step 2/2: collecting odds ---" | tee -a "$LOG_FILE"
docker compose run --rm backend python -m backend.ingest.odds_theoddsapi 2>&1 | tee -a "$LOG_FILE"
ODDS_STATUS=${PIPESTATUS[0]}

echo "=== daily_odds_job finished $(date) (predict exit $PREDICT_STATUS, odds exit $ODDS_STATUS) ===" | tee -a "$LOG_FILE"

if [ "$PREDICT_STATUS" -ne 0 ] || [ "$ODDS_STATUS" -ne 0 ]; then
    exit 1
fi
