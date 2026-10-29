"""backend/predict/log_predictions.py records, on every prediction row, the
newest game date in the game logs its features were built from, so
predictions made on stale features (a failed game-log refresh) can be left
out of evaluation. Models and roster inputs are stubbed; no network."""

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from backend.db import get_connection
from backend.models.nba_minutes_model import FEATURE_COLUMNS
from backend.predict import log_predictions as lp

BOS, NYK = "1610612738", "1610612752"


@pytest.fixture
def conn(tmp_path):
    c = get_connection(str(tmp_path / "w.duckdb"))
    for gid, day in (("0022600001", "2026-10-20"), ("0022600002", "2026-10-22")):
        c.execute(
            """INSERT INTO nba_player_games (player_id_nba, player_name, team, opponent, game_id, game_date, season,
                   minutes, pts, reb, ast, fg3m, stl, blk, tov, fga, fgm, fta, ftm, oreb, dreb, ingested_at)
               VALUES ('201', 'Jaylen Brown', 'BOS', 'NYK', ?, ?, '2026-27',
                   30, 20, 5, 4, 2, 1, 0, 2, 15, 7, 4, 3, 1, 4, now())""",
            [gid, day],
        )
    yield c
    c.close()


def test_features_through_is_the_newest_game_before_as_of(conn):
    assert lp.features_through(conn, date(2026, 10, 23)) == date(2026, 10, 22)
    assert lp.features_through(conn, date(2026, 10, 22)) == date(2026, 10, 20)  # strictly before, like build_current_form
    assert lp.features_through(conn, date(2026, 10, 1)) is None


class _Model:
    def predict(self, X):
        return np.full(len(X), 10.0)


def test_every_logged_prediction_records_features_through(conn, monkeypatch):
    tip = datetime.now(timezone.utc) + timedelta(hours=6)
    monkeypatch.setattr(lp, "upcoming_games", lambda c: pd.DataFrame(
        [{"game_id": "0022600009", "season": "2026-27", "home_team_id": BOS, "away_team_id": NYK, "tip_utc": tip}]))
    monkeypatch.setattr(lp, "teammates_out_from_injury_report", lambda c, d: {})
    monkeypatch.setattr(lp, "current_team_by_player", lambda c: {})
    seen = {}

    def form(c, as_of=None, **kw):
        seen["as_of"] = as_of
        return pd.DataFrame([{"player_id_nba": "201", "team": "BOS", **{col: 1.0 for col in FEATURE_COLUMNS}}])

    monkeypatch.setattr(lp, "build_current_form", form)
    monkeypatch.setattr(lp, "get_or_train_booster", lambda name, train, max_age: (_Model(), {"sigma": 5.0, "r": 3.0}))

    result = lp.run(conn)

    expected = lp.features_through(conn, seen["as_of"])
    assert expected is not None
    rows = conn.execute("SELECT market, features_through FROM model_predictions_log").fetchall()
    assert len(rows) == 5 and {r[1] for r in rows} == {expected}
    assert result.notes["features_through"] == str(expected)


def test_existing_predictions_log_gains_the_column(tmp_path):
    path = str(tmp_path / "old.duckdb")
    c = get_connection(path)
    c.execute("ALTER TABLE model_predictions_log DROP COLUMN features_through")
    c.execute("INSERT INTO model_predictions_log VALUES (now(), 'g', 'p', 'points', 20.0, 5.0, 'old')")
    c.close()
    c = get_connection(path)
    assert c.execute("SELECT features_through FROM model_predictions_log").fetchall() == [(None,)]
    c.close()
