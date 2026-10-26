"""NBA legs: a player's team is their current one (Sleeper via player_xref,
else their last game log), it must be one of the two teams in their odds
event, and the opponent is always the event's other team. Players that
can't be placed get no legs and are listed instead."""

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from backend.api import main
from backend.api.slate import Game, Slate
from backend.db import get_connection
from backend.features.nba_current_form import build_current_form, current_team_by_player
from backend.parlay import leg_pool
from backend.parlay.leg_pool import EventTeams, place_players
from backend.parlay.monte_carlo import Leg

NOW = datetime(2026, 10, 26, 15, 0, tzinfo=timezone.utc)
TIP = datetime(2026, 10, 26, 23, 40, tzinfo=timezone.utc)
TEAM_ID = {"BOS": "1610612738", "CHI": "1610612741", "NYK": "1610612752", "PHI": "1610612755", "MIA": "1610612748"}


def _player_game(c, pid, name, team, opp, gid, day, season="2025-26"):
    c.execute(
        """INSERT INTO nba_player_games (player_id_nba, player_name, team, opponent, game_id, game_date, season,
               minutes, pts, reb, ast, fg3m, stl, blk, tov, fga, fgm, fta, ftm, oreb, dreb, ingested_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, 30, 20, 5, 4, 2, 1, 0, 2, 15, 7, 4, 3, 1, 4, now())""",
        [pid, name, team, opp, gid, day, season],
    )


def _sleeper(c, pid, name, team):
    c.execute("INSERT INTO sleeper_players (sleeper_id, full_name, team, position, ingested_at) VALUES (?, ?, ?, 'SF', now())",
              [f"s{pid}", name, team])
    c.execute("INSERT INTO player_xref VALUES (?, ?, ?, ?, 'exact', 100, now(), now())", [pid, pid, f"s{pid}", name])


@pytest.fixture
def conn(tmp_path):
    """CHI @ BOS tonight. Last season's logs put Mitchell Robinson on NYK and
    Paul George on PHI; Sleeper has Robinson on BOS now and George on no team
    (so his old PHI team stands). Jaylen Brown and Coby White are unchanged."""
    c = get_connection(str(tmp_path / "w.duckdb"))
    last = date(2026, 4, 12)
    _player_game(c, "robinson", "Mitchell Robinson", "NYK", "CHI", "0022501200", last)
    _player_game(c, "george", "Paul George", "PHI", "MIA", "0022501201", last)
    _player_game(c, "brown", "Jaylen Brown", "BOS", "NYK", "0022501202", last)
    _player_game(c, "white", "Coby White", "CHI", "NYK", "0022501200", last)
    _sleeper(c, "robinson", "Mitchell Robinson", "BOS")
    _sleeper(c, "george", "Paul George", None)
    for pid, name in [("robinson", "Mitchell Robinson"), ("george", "Paul George"), ("brown", "Jaylen Brown"), ("white", "Coby White")]:
        c.execute("INSERT INTO odds_snapshots VALUES (?, 'ev1', NULL, ?, ?, 'player_points', 'dk', 15.5, 1.9, 1.9)",
                  [NOW - timedelta(hours=1), TIP, name])
    c.execute("INSERT INTO nba_schedule VALUES ('0022600121', '2026-27', '2026-10-26', ?, ?, ?, now())",
              [TEAM_ID["BOS"], TEAM_ID["CHI"], TIP - timedelta(minutes=10)])
    yield c
    c.close()


@pytest.fixture
def stub_models(monkeypatch):
    """Predictions and calibration stubbed; records what current form was asked."""
    seen = {}

    def points(conn, ids, out, team_by_player, opponent_by_player):
        seen.update(ids=set(ids), team=team_by_player, opp=opponent_by_player)
        return {i: (20.0, 5.0) for i in ids}

    names = {"Mitchell Robinson": "robinson", "Paul George": "george", "Jaylen Brown": "brown", "Coby White": "white"}
    monkeypatch.setattr(leg_pool, "_resolve_player_names", lambda conn, raw: {n: names[n] for n in raw})
    monkeypatch.setattr(leg_pool, "teammates_out_from_injury_report", lambda conn, d: {})
    monkeypatch.setattr(leg_pool, "_points_predictions", points)
    monkeypatch.setattr(leg_pool, "get_or_fit_calibrator", lambda conn, market: None)
    return seen


def test_current_team_prefers_sleeper_then_last_game(conn):
    teams = current_team_by_player(conn)
    assert teams["robinson"] == "BOS"  # traded: Sleeper wins over last season's NYK
    assert teams["george"] == "PHI"  # Sleeper has no team: last game log stands
    assert teams["brown"] == "BOS" and teams["white"] == "CHI"


def test_place_players_uses_event_for_opponent_and_lists_misfits():
    odds = pd.DataFrame({"player_id_nba": ["a", "b", "c", "d"], "player_name_raw": ["A", "B", "C", "D"],
                         "event_id": ["ev1", "ev1", "ev1", "ev2"]})
    placed, unplaced = place_players(odds, {"ev1": EventTeams(home="BOS", away="CHI")},
                                     {"a": "BOS", "b": "CHI", "c": "PHI", "d": "MIA"})
    assert placed == {("a", "ev1"): ("BOS", "CHI", True), ("b", "ev1"): ("CHI", "BOS", False)}
    assert {(u.player_id_nba, u.team, u.event) for u in unplaced} == {("c", "PHI", "CHI @ BOS"), ("d", "MIA", None)}


def test_leg_pool_places_traded_player_and_drops_stale_one(conn, stub_models):
    unplaced = []
    legs = leg_pool.build_leg_pool(conn, now=NOW, unplaced=unplaced)
    by_player = {l.player_id_nba: l for l in legs}
    assert set(by_player) == {"robinson", "brown", "white"}
    assert (by_player["robinson"].team, by_player["robinson"].opponent, by_player["robinson"].is_home) == ("BOS", "CHI", True)
    assert (by_player["white"].team, by_player["white"].opponent, by_player["white"].is_home) == ("CHI", "BOS", False)
    assert [(u.player_name, u.team, u.event) for u in unplaced] == [("Paul George", "PHI", "CHI @ BOS")]
    # Model inputs get the same current team and event opponent as the leg.
    assert stub_models["ids"] == {"robinson", "brown", "white"}
    assert stub_models["team"]["robinson"] == "BOS" and stub_models["opp"]["robinson"] == "CHI"


def test_odds_event_teams_win_over_schedule_match(conn, stub_models):
    # The odds feed says this event is NYK @ BOS: Coby White (CHI) no longer fits.
    conn.execute("INSERT INTO odds_events VALUES ('ev1', ?, 'BOS', 'NYK', now())", [TIP])
    unplaced = []
    legs = leg_pool.build_leg_pool(conn, now=NOW, unplaced=unplaced)
    assert {l.player_id_nba for l in legs} == {"robinson", "brown"}
    assert all(l.opponent == "NYK" for l in legs)
    assert {u.player_name for u in unplaced} == {"Paul George", "Coby White"}


def test_unknown_event_teams_place_nobody(conn, stub_models):
    conn.execute("DELETE FROM nba_schedule")
    unplaced = []
    assert leg_pool.build_leg_pool(conn, now=NOW, unplaced=unplaced) == []
    assert len(unplaced) == 4 and all(u.event is None for u in unplaced)


def test_check_placement_needs_no_models(conn, monkeypatch):
    monkeypatch.setattr(leg_pool, "_resolve_player_names",
                        lambda conn, raw: {n: n.split()[-1].lower() for n in raw})
    monkeypatch.setattr(leg_pool, "_points_predictions", lambda *a: pytest.fail("no model work in --check"))
    n, unplaced = leg_pool.check_placement(conn, now=NOW)
    assert n == 4 and [u.player_name for u in unplaced] == ["Paul George"]


def test_current_form_takes_team_and_opponent_overrides(conn):
    form = build_current_form(conn, as_of=date(2026, 10, 26), team_by_player={"robinson": "BOS"},
                              opponent_by_player={"robinson": "CHI"})
    row = form.set_index("player_id_nba").loc["robinson"]
    assert row["team"] == "BOS"
    default = build_current_form(conn, as_of=date(2026, 10, 26)).set_index("player_id_nba").loc["robinson"]
    assert default["team"] == "NYK"  # point-in-time default unchanged


def test_leg_json_opponent_comes_from_the_leg_never_the_old_team(monkeypatch):
    s = Slate()
    s.games = {"ev1": Game("ev1", "BOS", "CHI", TIP)}
    monkeypatch.setattr(main, "SLATE", s)
    placed = Leg("robinson", "Mitchell Robinson", "ev1", "BOS", "points", "over", 8.5, 9, 3, 1.9, opponent="CHI", is_home=True)
    j = main._leg_json(placed)
    assert (j["team"], j["opponent"], j["is_home"]) == ("BOS", "CHI", True)
    # A leg built without placement whose team isn't in the game: no guessed opponent.
    stale = Leg("george", "Paul George", "ev1", "PHI", "points", "over", 15.5, 16, 5, 1.9)
    j = main._leg_json(stale)
    assert j["opponent"] is None and j["is_home"] is None


def test_odds_collection_records_event_teams(tmp_path):
    from backend.ingest.odds_theoddsapi import _write_snapshot

    c = get_connection(str(tmp_path / "w.duckdb"))
    event = {"id": "ev9", "commence_time": "2026-10-26T23:40:00Z", "home_team": "Boston Celtics",
             "away_team": "Chicago Bulls", "bookmakers": []}
    _write_snapshot(c, event)
    _write_snapshot(c, {**event, "home_team": "Nowhere Nobodies"})  # unrecognized: logged, row kept
    assert c.execute("SELECT event_id, home_team, away_team FROM odds_events").fetchall() == [("ev9", "BOS", "CHI")]
    c.close()
