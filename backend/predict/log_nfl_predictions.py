"""Logs the NFL projection for every projected player and market, for every
NFL game kicking off in the next `LOOKAHEAD`, into nfl_predictions_log in
data/nfl_warehouse.duckdb — the NFL counterpart of log_predictions.py (NBA).

Run BEFORE the OddsPapi lines step, so a projection is on record before the
matching line is collected; the Wednesday lines job and the Saturday prep
job both run it (weekly_nfl_job.sh). Rows are written once per
(game, player, market) and never overwritten (the primary key enforces
it): Wednesday's projection is the one on record, and Saturday only adds
players who weren't projected Wednesday (depth-chart changes) and games
that kick off before the next Wednesday.

Projections come from nfl_projections.compute — the same depth-chart
players, Out filter, features and model version the Schedule page shows —
computed fresh rather than read from the page's day cache, so a run right
after an injury refresh uses it. Already-started games are skipped.

    docker compose run --rm --no-deps api python -m backend.predict.log_nfl_predictions
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb

from backend.ingest.nfl_nflverse import current_season
from backend.ingest.util import logger
from backend.nfl_db import get_nfl_connection
from backend.predict import nfl_projections

LOOKAHEAD = timedelta(days=7)


def upcoming_weeks(conn: duckdb.DuckDBPyConnection, season: int, now: datetime) -> list[int]:
    return [w for (w,) in conn.execute(
        "SELECT DISTINCT week FROM nfl_schedule WHERE season = ? AND home_score IS NULL "
        "AND kickoff_utc > ? AND kickoff_utc <= ? ORDER BY week",
        [season, now, now + LOOKAHEAD],
    ).fetchall()]


def prediction_rows(data: dict, now: datetime) -> list[tuple]:
    """(logged_at, game_id, player_id, market, mean, lo, hi, model_version)
    for every game in a compute() payload that hasn't kicked off by now."""
    rows = []
    for g in data["games"]:
        kickoff = datetime.fromisoformat(g["kickoff_utc"]) if g["kickoff_utc"] else None
        if g["final"] or kickoff is None or kickoff <= now or kickoff > now + LOOKAHEAD:
            continue
        for side in ("away", "home"):
            for p in g["teams"][side]["players"]:
                for market, proj in p["projections"].items():
                    rows.append((now.replace(tzinfo=None), g["game_id"], p["player_id"], market,
                                 proj["mean"], proj["lo"], proj["hi"], data["model_version"]))
    return rows


def run(now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    season = current_season()
    conn = get_nfl_connection(read_only=True)
    try:
        weeks = upcoming_weeks(conn, season, now)
    finally:
        conn.close()  # compute() attaches the warehouse read-only on its own
    if not weeks:
        logger.info("log_nfl_predictions: no NFL games in the next %s", LOOKAHEAD)
        return {"weeks": [], "rows": 0, "new": 0}

    rows = []
    for week in weeks:
        rows += prediction_rows(nfl_projections.compute(season, week), now)

    conn = get_nfl_connection()
    try:
        before = conn.execute("SELECT count(*) FROM nfl_predictions_log").fetchone()[0]
        if rows:
            conn.executemany(
                "INSERT INTO nfl_predictions_log VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (game_id, player_id, market) DO NOTHING",
                rows,
            )
        new = conn.execute("SELECT count(*) FROM nfl_predictions_log").fetchone()[0] - before
    finally:
        conn.close()
    logger.info("log_nfl_predictions: %d projections for weeks %s, %d new", len(rows), weeks, new)
    return {"weeks": weeks, "rows": len(rows), "new": new}


if __name__ == "__main__":
    print(f"log_nfl_predictions complete: {run()}")
