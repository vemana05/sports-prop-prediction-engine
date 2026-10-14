"""NFL pre-game prediction log: only games not yet kicked off are logged,
and a logged projection is never overwritten by a later run."""

from datetime import datetime, timedelta, timezone

import duckdb

from backend.nfl_db import NFL_SCHEMA
from backend.predict import log_nfl_predictions as lp

NOW = datetime(2026, 10, 14, 13, 0, tzinfo=timezone.utc)


def _game(game_id, kickoff, final=False, mean=60.0):
    player = {"player_id": "00-1", "projections": {"receiving_yards": {"mean": mean, "lo": 40.0, "hi": 80.0, "actual": None}}}
    return {"game_id": game_id, "kickoff_utc": kickoff.isoformat(), "final": final,
            "teams": {"away": {"players": [player]}, "home": {"players": []}}}


def _payload(mean=60.0):
    return {"model_version": "thru2025", "games": [
        _game("upcoming", NOW + timedelta(days=1), mean=mean),
        _game("started", NOW - timedelta(minutes=5)),
        _game("final", NOW - timedelta(days=3), final=True),
        _game("next_week", NOW + timedelta(days=8)),
    ]}


def test_only_unstarted_games_in_window_are_logged():
    rows = lp.prediction_rows(_payload(), NOW)
    assert [(r[1], r[2], r[3], r[4], r[7]) for r in rows] == [("upcoming", "00-1", "receiving_yards", 60.0, "thru2025")]


def test_first_logged_projection_is_kept():
    conn = duckdb.connect(":memory:")
    conn.execute(NFL_SCHEMA)
    sql = "INSERT INTO nfl_predictions_log VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (game_id, player_id, market) DO NOTHING"
    conn.executemany(sql, lp.prediction_rows(_payload(60.0), NOW))
    conn.executemany(sql, lp.prediction_rows(_payload(75.0), NOW + timedelta(days=0.5)))
    assert conn.execute("SELECT mean FROM nfl_predictions_log").fetchall() == [(60.0,)]
