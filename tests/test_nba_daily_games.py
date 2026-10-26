"""backend/ingest/nba_daily_games.py against a stubbed nba_api: loads only
finished games not already loaded, is resumable, waits for DNS, rebuilds
nba_features, and keeps the early-season counter and the model's season
features on the same game logs. No network: every nba_api endpoint is a
fake and real HTTP is blocked."""

import json
from datetime import date, datetime, timezone

import pandas as pd
import pytest
import requests

from backend.db import get_connection
from backend.features.nba_current_form import build_current_form
from backend.ingest import nba_box_status, nba_daily_games, nba_nbaapi, nba_schedule
from backend.predict.early_season import games_missing_logs, team_games_played

NOW = datetime(2026, 10, 23, 15, 0, tzinfo=timezone.utc)  # Fri 11 AM ET
BOS, NYK, LAL = 1610612738, 1610612752, 1610612747
ABBR = {BOS: "BOS", NYK: "NYK", LAL: "LAL"}
# (game_id, ET date, home, away, tip_utc, gameStatus)
SCHEDULE = [
    ("0022600001", "2026-10-20", BOS, NYK, "2026-10-20T23:30:00Z", 3),  # already loaded before the run
    ("0022600002", "2026-10-21", LAL, BOS, "2026-10-22T02:30:00Z", 3),
    ("0022600003", "2026-10-22", NYK, LAL, "2026-10-22T23:30:00Z", 3),  # final, not in LeagueGameLog yet
    ("0022600004", "2026-10-23", BOS, LAL, "2026-10-23T23:30:00Z", 1),  # tonight
]
PLAYERS = {BOS: ("201", "Jaylen Brown"), NYK: ("202", "Jalen Brunson"), LAL: ("203", "Luka Doncic")}


class FakeNbaApi:
    """Stands in for the three nba_api endpoints; records every call."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.logged = {"0022600001", "0022600002"}  # games stats.nba.com has player logs for

    def schedule(self, season, timeout=30):
        self.calls.append(("schedule", season))
        rows = [{"gameId": g, "gameDate": d, "gameDateTimeUTC": t, "homeTeam_teamId": h, "awayTeam_teamId": a,
                 "gameStatus": st, "homeTeam_score": 110 if st == 3 else 0, "awayTeam_score": 100 if st == 3 else 0}
                for g, d, h, a, t, st in SCHEDULE]
        return _frames(pd.DataFrame(rows))

    def game_log(self, season, season_type_all_star, player_or_team_abbreviation, date_from_nullable="", timeout=60):
        self.calls.append(("game_log", season, date_from_nullable))
        start = datetime.strptime(date_from_nullable, "%m/%d/%Y").date() if date_from_nullable else date.min
        rows = []
        for g, d, h, a, _, _ in SCHEDULE:
            if g not in self.logged or date.fromisoformat(d) < start:
                continue
            for team, opp, sep in ((h, a, "vs."), (a, h, "@")):
                pid, name = PLAYERS[team]
                rows.append({"PLAYER_ID": int(pid), "PLAYER_NAME": name, "TEAM_ABBREVIATION": ABBR[team], "GAME_ID": g,
                             "GAME_DATE": d, "MATCHUP": f"{ABBR[team]} {sep} {ABBR[opp]}", "MIN": 34, "PTS": 25, "REB": 6,
                             "AST": 5, "FG3M": 3, "STL": 1, "BLK": 0, "TOV": 3, "FGA": 18, "FGM": 9, "FTA": 6, "FTM": 4,
                             "OREB": 1, "DREB": 5})
        return _frames(pd.DataFrame(rows))

    def box(self, game_id, timeout=30):
        self.calls.append(("box", game_id))
        g = next(s for s in SCHEDULE if s[0] == game_id)
        rows = [{"gameId": game_id, "teamId": t, "personId": int(PLAYERS[t][0]), "firstName": PLAYERS[t][1].split()[0],
                 "familyName": PLAYERS[t][1].split()[1], "minutes": "34:00", "comment": ""} for t in (g[2], g[3])]
        return _frames(pd.DataFrame(rows))

    def count(self, kind):
        return sum(c[0] == kind for c in self.calls)


def _frames(df):
    obj = type("Resp", (), {})()
    obj.get_data_frames = lambda: [df]
    obj.get_json = lambda: json.dumps({"stub": True})
    return obj


@pytest.fixture
def api(monkeypatch, tmp_path):
    fake = FakeNbaApi()
    monkeypatch.setattr(nba_schedule.scheduleleaguev2, "ScheduleLeagueV2", fake.schedule)
    monkeypatch.setattr(nba_nbaapi.leaguegamelog, "LeagueGameLog", fake.game_log)
    monkeypatch.setattr(nba_box_status.boxscoretraditionalv3, "BoxScoreTraditionalV3", fake.box)
    monkeypatch.setenv("NBA_BOX_STATUS_CACHE_DIR", str(tmp_path / "box_cache"))
    monkeypatch.setattr(requests.Session, "send", lambda *a, **k: pytest.fail("real HTTP call attempted"))
    return fake


@pytest.fixture
def conn(tmp_path):
    """Last season's logs plus the 2026-27 opener already loaded (with box status)."""
    c = get_connection(str(tmp_path / "w.duckdb"))
    for team, opp, gid, day, season in [(BOS, NYK, "0022501200", "2026-04-12", "2025-26"),
                                        (NYK, BOS, "0022501200", "2026-04-12", "2025-26"),
                                        (BOS, NYK, "0022600001", "2026-10-20", "2026-27"),
                                        (NYK, BOS, "0022600001", "2026-10-20", "2026-27")]:
        pid, name = PLAYERS[team]
        c.execute(
            """INSERT INTO nba_player_games (player_id_nba, player_name, team, opponent, game_id, game_date, season,
                   minutes, pts, reb, ast, fg3m, stl, blk, tov, fga, fgm, fta, ftm, oreb, dreb, ingested_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 30, 20, 5, 4, 2, 1, 0, 2, 15, 7, 4, 3, 1, 4, now())""",
            [pid, name, ABBR[team], ABBR[opp], gid, day, season],
        )
        c.execute("INSERT INTO nba_box_status VALUES (?, ?, ?, ?, '30:00', '', now()) ON CONFLICT DO NOTHING",
                  [gid, str(team), pid, name])
    for gid, d, h, a, t, _ in SCHEDULE[:1]:  # schedule as of the last refresh: opener only
        c.execute("INSERT INTO nba_schedule VALUES (?, '2026-27', ?, ?, ?, ?, now())", [gid, d, str(h), str(a), t])
    yield c
    c.close()


def _game_ids(conn, table):
    return {r[0] for r in conn.execute(f"SELECT DISTINCT game_id FROM {table}").fetchall()}


def test_loads_only_new_finished_games_then_rebuilds_features(conn, api):
    result = nba_daily_games.run(conn, now=NOW, wait_dns=False)

    assert result.notes["errors"] == []
    assert api.calls[0] == ("schedule", "2026-27")
    # One LeagueGameLog call, starting at the earliest missing finished game.
    assert [c for c in api.calls if c[0] == "game_log"] == [("game_log", "2026-27", "10/21/2026")]
    assert _game_ids(conn, "nba_player_games") == {"0022501200", "0022600001", "0022600002"}
    assert result.notes["log_rows"] == 2 and result.notes["log_games_still_pending"] == 1  # 0022600003
    # Box status fetched only for the newly logged game.
    assert [c for c in api.calls if c[0] == "box"] == [("box", "0022600002")]
    assert "0022600002" in _game_ids(conn, "nba_player_availability")
    # Schedule results stored; tonight's game isn't final.
    assert dict(conn.execute("SELECT game_id, game_status FROM nba_game_results").fetchall())["0022600004"] == 1
    # nba_features rebuilt with the new game.
    assert result.notes["features"] > 0
    assert "0022600002" in _game_ids(conn, "nba_features")


def test_rerun_is_resumable_and_skips_loaded_games(conn, api):
    nba_daily_games.run(conn, now=NOW, wait_dns=False)
    rows_before = conn.execute("SELECT count(*) FROM nba_player_games").fetchone()[0]

    api.logged.add("0022600003")  # stats.nba.com has caught up
    api.calls.clear()
    result = nba_daily_games.run(conn, now=NOW, wait_dns=False)
    assert [c for c in api.calls if c[0] == "game_log"] == [("game_log", "2026-27", "10/22/2026")]
    assert [c for c in api.calls if c[0] == "box"] == [("box", "0022600003")]
    assert conn.execute("SELECT count(*) FROM nba_player_games").fetchone()[0] == rows_before + 2

    api.calls.clear()
    result = nba_daily_games.run(conn, now=NOW, wait_dns=False)
    assert api.count("game_log") == 0 and api.count("box") == 0
    assert result.rows_written == 0 and result.notes["features"] is None  # nothing new, features current


def test_failed_box_game_stays_pending_for_next_run(conn, api, monkeypatch):
    def boom(game_id, timeout=30):
        api.calls.append(("box", game_id))
        raise requests.ConnectionError("stats.nba.com timed out")

    monkeypatch.setattr(nba_box_status.boxscoretraditionalv3, "BoxScoreTraditionalV3", boom)
    monkeypatch.setattr("backend.ingest.util.time.sleep", lambda s: None)
    result = nba_daily_games.run(conn, now=NOW, wait_dns=False)
    assert result.notes["errors"] == ["box_status"] and result.notes["box_failed"] == 1
    assert nba_daily_games.pending_box_games(conn, "2026-27") == ["0022600002"]


def test_schedule_failure_falls_back_to_stored_schedule(conn, api, monkeypatch):
    monkeypatch.setattr(nba_schedule, "fetch_season_schedule", lambda season: (_ for _ in ()).throw(RuntimeError("down")))
    monkeypatch.setattr(nba_daily_games, "features_behind", lambda conn: False)
    result = nba_daily_games.run(conn, now=NOW, wait_dns=False)
    assert result.notes["errors"] == ["schedule"]
    assert result.notes["log_games_pending"] == 0  # only the already-loaded opener is on file


def test_waits_for_dns_before_any_call(conn, api):
    attempts, sleeps = [], []

    def resolve(host, port):
        attempts.append(host)
        if len(attempts) < 3:
            raise OSError("Temporary failure in name resolution")

    assert nba_daily_games.wait_for_dns(resolve=resolve, sleep=sleeps.append, delay_s=5)
    assert attempts == ["stats.nba.com"] * 3 and sleeps == [5, 5]


def test_no_dns_loads_nothing(conn, api, monkeypatch):
    monkeypatch.setattr(nba_daily_games, "wait_for_dns", lambda: False)
    result = nba_daily_games.run(conn, now=NOW)
    assert result.notes["errors"] == ["dns"] and api.calls == []


def test_season_label():
    assert nba_daily_games.season_for(date(2026, 10, 23)) == "2026-27"
    assert nba_daily_games.season_for(date(2027, 3, 1)) == "2026-27"
    assert nba_daily_games.season_for(date(2026, 9, 30)) == "2025-26"


def test_early_season_counter_and_model_inputs_read_the_same_logs(conn, api):
    nba_daily_games.run(conn, now=NOW, wait_dns=False)
    as_of = NOW.date()
    season, played = team_games_played(conn, as_of)
    form = build_current_form(conn, as_of=as_of).set_index("player_id_nba")
    for team, (pid, _) in PLAYERS.items():
        # One player per team plays every game: the model's games-played
        # feature and the banner's team count agree.
        assert form.loc[pid, "games_played_prior_season"] == played[ABBR[team]]
    assert played == {"BOS": 2, "NYK": 1, "LAL": 1}  # 0022600003 tipped but isn't logged yet
    assert games_missing_logs(conn, season, as_of) == 1
