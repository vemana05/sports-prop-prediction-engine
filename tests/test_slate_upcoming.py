"""The NBA slate holds only games that haven't tipped: stale past events in
odds_snapshots stay out of the leg pool, the "Next slate" date and the game
filter, and odds events are labelled "away @ home" from nba_schedule."""

import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from backend.api import main, slate as slate_mod
from backend.api.slate import Game, Slate, _load_games, _match_schedule
from backend.db import get_connection
from backend.parlay import leg_pool
from backend.parlay.monte_carlo import Leg

NOW = datetime(2026, 10, 21, 17, 16, tzinfo=timezone.utc)  # Wed Oct 21, 1:16 PM ET
TEAM_ID = {"BOS": "1610612738", "NYK": "1610612752", "LAL": "1610612747", "GSW": "1610612744",
           "DEN": "1610612743", "OKC": "1610612760", "PHI": "1610612755"}

# (event_id, commence_time, {player: team}) — team as the player's last game log has it
EVENTS = [
    ("ev_past", datetime(2026, 10, 20, 23, 0, tzinfo=timezone.utc), {"p_nyk": "NYK", "p_phi": "PHI"}),
    ("ev_lal", datetime(2026, 10, 22, 2, 10, tzinfo=timezone.utc), {"p_lal": "LAL", "p_gsw": "GSW", "p_moved": "BOS"}),
    ("ev_bos", datetime(2026, 10, 22, 2, 10, tzinfo=timezone.utc), {"p_bos": "BOS", "p_nyk2": "NYK"}),
    ("ev_okc", datetime(2026, 10, 23, 1, 40, tzinfo=timezone.utc), {"p_den": "DEN", "p_okc": "OKC"}),
]
# (game_id, ET game_date, away, home, tip_utc) — schedule tips are rounded vs the odds feed's
SCHEDULE = [
    ("0022600002", date(2026, 10, 20), "PHI", "NYK", datetime(2026, 10, 20, 23, 0, tzinfo=timezone.utc)),
    ("0022600005", date(2026, 10, 21), "GSW", "LAL", datetime(2026, 10, 22, 2, 0, tzinfo=timezone.utc)),
    ("0022600092", date(2026, 10, 21), "NYK", "BOS", datetime(2026, 10, 22, 2, 0, tzinfo=timezone.utc)),
    ("0022600007", date(2026, 10, 22), "DEN", "OKC", datetime(2026, 10, 23, 1, 30, tzinfo=timezone.utc)),
]
TEAM_BY_PLAYER = {p: t for _, _, players in EVENTS for p, t in players.items()}


@pytest.fixture
def conn(tmp_path):
    c = get_connection(str(tmp_path / "w.duckdb"))
    for event_id, commence, players in EVENTS:
        pulled = commence - timedelta(hours=8)
        for p in players:
            c.execute(
                "INSERT INTO odds_snapshots VALUES (?, ?, NULL, ?, ?, 'player_points', 'dk', 20.5, 1.9, 1.9)",
                [pulled, event_id, commence, p],
            )
    for gid, day, away, home, tip in SCHEDULE:
        c.execute("INSERT INTO nba_schedule VALUES (?, '2026-27', ?, ?, ?, ?, now())", [gid, day, TEAM_ID[home], TEAM_ID[away], tip])
    yield c
    c.close()


@pytest.fixture
def stub_models(monkeypatch):
    monkeypatch.setattr(leg_pool, "_resolve_player_names", lambda conn, names: {n: n for n in names})
    monkeypatch.setattr(leg_pool, "teammates_out_from_injury_report", lambda conn, d: {})
    monkeypatch.setattr(leg_pool, "_points_predictions", lambda conn, ids, *a: {i: (20.0, 5.0) for i in ids})
    monkeypatch.setattr(leg_pool, "get_or_fit_calibrator", lambda conn, market: None)
    monkeypatch.setattr(leg_pool, "current_team_by_player", lambda conn: dict(TEAM_BY_PLAYER))
    monkeypatch.setattr(
        leg_pool, "build_current_form",
        lambda conn, **kw: pd.DataFrame({"player_id_nba": list(TEAM_BY_PLAYER), "team": list(TEAM_BY_PLAYER.values())}),
    )


def test_leg_pool_drops_events_that_already_started(conn, stub_models):
    legs = leg_pool.build_leg_pool(conn, now=NOW)
    assert {l.game_id for l in legs} == {"ev_lal", "ev_bos", "ev_okc"}
    assert not {"p_nyk", "p_phi"} & {l.player_id_nba for l in legs}


def test_games_are_upcoming_only_and_matched_away_at_home(conn, stub_models):
    legs = leg_pool.build_leg_pool(conn, now=NOW)
    games, _ = _load_games(conn, legs, NOW)
    assert set(games) == {"ev_lal", "ev_bos", "ev_okc"}
    # same tip time, told apart by the players' teams (the stale one is dropped)
    assert games["ev_lal"].label == "GSW @ LAL" and games["ev_lal"].nba_game_id == "0022600005"
    assert games["ev_bos"].label == "NYK @ BOS" and games["ev_bos"].nba_game_id == "0022600092"
    assert games["ev_okc"].label == "DEN @ OKC"
    assert games["ev_lal"].tip_utc == EVENTS[1][1]  # odds feed's tip, not the schedule's rounded one


def test_match_never_reuses_a_scheduled_game_or_crosses_days():
    tip = datetime(2026, 10, 22, 2, 10, tzinfo=timezone.utc)
    legs = [Leg("a", "a", "e1", "LAL", "points", "over", 20, 20, 5, 1.9), Leg("b", "b", "e2", "LAL", "points", "over", 20, 20, 5, 1.9)]
    sched = [("g1", "LAL", "GSW", datetime(2026, 10, 22, 2, 0, tzinfo=timezone.utc)),
             ("g_next_day", "LAL", "DEN", datetime(2026, 10, 23, 2, 0, tzinfo=timezone.utc))]
    assert _match_schedule([("e1", None, tip), ("e2", None, tip)], sched, legs) == {"e1": "g1"}


def _ready_slate(games: dict[str, Game], legs: list[Leg]) -> Slate:
    s = Slate()
    s.games, s.legs = games, legs
    s.status, s.built_at, s.built_day = "ready", time.time(), date(2026, 10, 21)
    return s


def test_slate_date_is_earliest_upcoming_tip_and_tipped_games_leave_the_pool(monkeypatch):
    past, soon, later = (NOW - timedelta(hours=1), NOW + timedelta(hours=6), NOW + timedelta(days=1, hours=8))
    legs = [Leg(p, p, gid, "LAL", "points", "over", 20, 20, 5, 1.9) for p, gid in [("a", "g_past"), ("b", "g_soon"), ("c", "g_later")]]
    s = _ready_slate({"g_past": Game("g_past", "LAL", "GSW", past), "g_soon": Game("g_soon", "BOS", "NYK", soon),
                      "g_later": Game("g_later", "OKC", "DEN", later)}, legs)
    monkeypatch.setattr(slate_mod, "datetime", type("dt", (datetime,), {"now": staticmethod(lambda tz=None: NOW)}))
    assert s.slate_date == date(2026, 10, 21)  # 6 PM ET tonight, not yesterday's g_past
    assert {l.game_id for l in s.pool()} == {"g_soon", "g_later"}

    monkeypatch.setattr(main, "SLATE", s)
    out = main._slate_json()
    assert out["slate_date"] == "2026-10-21" and out["n_games"] == 2 and out["n_legs"] == 2
    assert [g["label"] for g in out["games"]] == ["NYK @ BOS", "DEN @ OKC"]
    assert out["first_tip_utc"] == soon.isoformat()


def test_ensure_fresh_rebuilds_on_new_day_or_finished_daily_job(monkeypatch, tmp_path):
    rebuilt = []
    s = _ready_slate({}, [])
    monkeypatch.setattr(s, "start_build", lambda: rebuilt.append(True))
    monkeypatch.setattr(slate_mod, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(slate_mod, "today_et", lambda: date(2026, 10, 21))

    s.ensure_fresh()
    assert not rebuilt  # same day, no job log

    log = tmp_path / "daily_odds_job_20261021_100414.log"
    log.write_text("=== daily_odds_job starting ===\n")
    s._checked_at = 0
    s.ensure_fresh(now=time.time() + 120)
    assert not rebuilt  # job still running

    log.write_text(log.read_text() + "=== daily_odds_job finished Wed Oct 21 13:16:44 EDT 2026 (predict exit 0, odds exit 0) ===\n")
    s.built_at = log.stat().st_mtime - 10
    s.ensure_fresh(now=time.time() + 240)
    assert rebuilt == [True]

    rebuilt.clear()
    s.built_at = time.time() + 1000  # built after the job finished
    monkeypatch.setattr(slate_mod, "today_et", lambda: date(2026, 10, 22))
    s.ensure_fresh()
    assert rebuilt == [True]
