"""Model vs. lines (NFL): name matching, no-vig and push handling, one book
per prop, and the report's metrics. In-memory NFL warehouse; no network."""

from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd
import pytest

from backend.backtest import nfl_model_vs_lines as mvl
from backend.models.nfl_market_model import Spread
from backend.nfl_db import NFL_SCHEMA

YARDS = Spread(a=10.0, b=0.0, z_sorted=np.sort(np.random.default_rng(0).standard_normal(20001)))
COUNTS = Spread(r=8.0)


def _spread_for(version, market):
    return COUNTS if market == "receptions" else YARDS


# ---- names --------------------------------------------------------------------

ROSTER = pd.DataFrame([
    (2026, 5, "KC", "00-1", "Patrick Mahomes", "QB"),
    (2026, 5, "KC", "00-2", "Kenny Gainwell", "RB"),
    (2026, 5, "BUF", "00-3", "Cam Ward", "QB"),
    (2026, 5, "BUF", "00-4", "Charvarius Ward", "CB"),
    (2026, 5, "BUF", "00-5", "José Ramírez Jr.", "WR"),
    (2026, 5, "BUF", "00-6", "Jacory Croskey-Merritt", "RB"),
    (2026, 5, "KC", "00-7", "Mike Williams", "WR"),
    (2026, 5, "KC", "00-8", "Mike Williams", "TE"),
    (2026, 9, "KC", "00-9", "Future Signing", "WR"),  # rostered after the game: not a candidate
], columns=["season", "week", "team", "player_id", "player_name", "position"])


def _players(*names):
    return pd.DataFrame([("fx1", str(i), n, 2026, 6, "KC", "BUF") for i, n in enumerate(names)],
                        columns=["fixture_id", "op_player_id", "player_name", "season", "week", "home_team", "away_team"])


def test_name_matching_rules():
    m = mvl.match_names(_players("Mahomes, Patrick", "Jose Ramirez", "Gainwell, Kenneth", "Ward, Cameron",
                                 "Jacory Merritt", "Mike Williams", "Future Signing", "Nobody Here"), ROSTER)
    got = dict(zip(m["player_name"], zip(m["player_id"], m["match_method"])))
    assert got["Mahomes, Patrick"] == ("00-1", "exact")          # "Last, First"
    assert got["Jose Ramirez"] == ("00-5", "exact")              # accents + suffix
    assert got["Gainwell, Kenneth"] == ("00-2", "initial")       # nickname via last name + initial
    assert got["Ward, Cameron"] == ("00-3", "initial")           # skill player wins over the CB
    assert got["Jacory Merritt"] == ("00-6", "initial")          # half of a hyphenated last name
    unmatched = m[m["player_id"].isna()].set_index("player_name")["unmatched_reason"]
    assert set(unmatched.index) == {"Mike Williams", "Future Signing", "Nobody Here"}
    assert "2 roster players" in unmatched["Mike Williams"]      # two skill players: never guessed


# ---- probabilities ------------------------------------------------------------

def test_no_vig():
    assert mvl.no_vig_over(1.91, 1.91) == pytest.approx(0.5)
    assert mvl.no_vig_over(1.5, 2.5) == pytest.approx((1 / 1.5) / (1 / 1.5 + 1 / 2.5))


def test_whole_number_line_takes_the_push_out():
    from scipy import stats
    mu, r = 5.0, COUNTS.r
    p = r / (r + mu)
    over, push = 1 - stats.nbinom.cdf(5, r, p), stats.nbinom.pmf(5, r, p)
    cal, raw = mvl.model_p_over(mu, 5.0, COUNTS)
    assert raw == pytest.approx(over / (1 - push))
    assert cal == raw  # no calibrator
    _, half = mvl.model_p_over(mu, 4.5, COUNTS)
    assert half == pytest.approx(1 - stats.nbinom.cdf(4, r, p))
    # Yards, symmetric residuals: a line at the mean is ~50/50 with or without the push band.
    assert mvl.model_p_over(60.0, 60.0, YARDS)[1] == pytest.approx(0.5, abs=0.01)


def test_calibrator_is_applied():
    class Half:
        def predict(self, x):
            return np.asarray(x) / 2
    cal, raw = mvl.model_p_over(60.0, 50.5, YARDS, Half())
    assert cal == pytest.approx(raw / 2)


# ---- end to end on a tiny warehouse --------------------------------------------

@pytest.fixture
def conn():
    c = duckdb.connect(":memory:")
    c.execute(NFL_SCHEMA)
    now = datetime(2026, 10, 13, tzinfo=timezone.utc)
    c.execute("INSERT INTO nfl_schedule (game_id, season, week, game_type, home_team, away_team, home_score, away_score, ingested_at) "
              "VALUES ('2026_06_BUF_KC', 2026, 6, 'REG', 'KC', 'BUF', 27, 24, ?)", [now])
    c.execute("INSERT INTO nfl_oddspapi_fixtures (fixture_id, game_id, season, week, kickoff_utc, home_team, away_team) "
              "VALUES ('fx1', '2026_06_BUF_KC', 2026, 6, '2026-10-11 17:00:00+00', 'KC', 'BUF')")
    c.executemany("INSERT INTO nfl_rosters_weekly (season, week, team, player_id, player_name, position, ingested_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                  [(*r, now) for r in ROSTER.itertuples(index=False)])
    c.executemany("INSERT INTO nfl_oddspapi_players VALUES ('fx1', ?, ?, NULL)",
                  [("p1", "Mahomes, Patrick"), ("p2", "Gainwell, Kenneth"), ("p3", "Ward, Cameron"), ("p4", "Nobody Here"),
                   ("p5", "Jose Ramirez")])

    def line(book, market, pid, kind, ln, o, u):
        c.execute("INSERT INTO nfl_prop_lines VALUES ('fx1', '2026_06_BUF_KC', ?, ?, ?, ?, ?, ?, ?, NULL, ?)",
                  [book, market, pid, kind, ln, o, u, now])
    # Mahomes: FanDuel and DraftKings close; FanDuel must be the one used, with FanDuel's open.
    line("fanduel", "passing_yards", "p1", "close", 260.5, 1.91, 1.91)
    line("draftkings", "passing_yards", "p1", "close", 250.5, 1.80, 2.00)
    line("fanduel", "passing_yards", "p1", "open", 255.5, 1.91, 1.91)
    line("draftkings", "passing_yards", "p1", "open", 240.5, 1.91, 1.91)
    # Gainwell: DraftKings only (FanDuel had no line); actual 4 receptions on a 4.0 line -> push.
    line("draftkings", "receptions", "p2", "close", 4.0, 1.95, 1.87)
    # Ward: no projection logged. Nobody Here: unmatched name. Ramirez: projected but didn't play.
    line("fanduel", "passing_yards", "p3", "close", 210.5, 1.91, 1.91)
    line("fanduel", "receiving_yards", "p4", "close", 30.5, 1.91, 1.91)
    line("fanduel", "receiving_yards", "p5", "close", 40.5, 1.91, 1.91)
    c.executemany("INSERT INTO nfl_predictions_log VALUES (?, '2026_06_BUF_KC', ?, ?, ?, NULL, NULL, 'thru2025')",
                  [(datetime(2026, 10, 7), "00-1", "passing_yards", 280.0),
                   (datetime(2026, 10, 7), "00-2", "receptions", 3.5),
                   (datetime(2026, 10, 7), "00-5", "receiving_yards", 45.0)])
    c.execute("INSERT INTO nfl_player_weekly (player_id, game_id, season, week, passing_yards, receptions, ingested_at) "
              "VALUES ('00-1', '2026_06_BUF_KC', 2026, 6, 301, 0, ?), ('00-2', '2026_06_BUF_KC', 2026, 6, 0, 4, ?)", [now, now])
    yield c
    c.close()


def test_end_to_end(conn):
    props, names = mvl.load_rows(conn)
    assert len(props) == 5  # one row per (game, player, market): DraftKings' Mahomes close is not a second row
    scored = mvl.score_rows(props, {}, _spread_for)
    by = scored.set_index("player_name")
    mahomes = by.loc["Mahomes, Patrick"]
    assert (mahomes["bookmaker"], mahomes["line"], mahomes["open_line"]) == ("fanduel", 260.5, 255.5)
    assert mahomes["status"] == "graded" and mahomes["outcome"] == 1.0
    assert by.loc["Gainwell, Kenneth", "bookmaker"] == "draftkings" and bool(by.loc["Gainwell, Kenneth", "push"])
    assert by.loc["Ward, Cameron", "status"] == "no_projection"
    assert by.loc["Nobody Here", "status"] == "unmatched_name"
    assert by.loc["Jose Ramirez", "status"] == "did_not_play"

    report = mvl.build_report(scored, names, now=datetime(2026, 10, 14, tzinfo=timezone.utc))
    o = report["overall"]
    assert (o["props"], o["graded"], o["pushes"]) == (5, 1, 1)
    assert not o["enough_sample"]
    assert report["names"]["matched"] == 4 and report["names"]["match_rate"] == pytest.approx(0.8)
    assert [u["name"] for u in report["names"]["unmatched"]] == ["Nobody Here"]
    assert {r["status"] for r in report["not_graded"]} == {"no_projection", "did_not_play"}
    assert report["coverage"] == {"graded": 2, "no_projection": 1, "unmatched_name": 1, "did_not_play": 1}
    # Model 280 vs a 260.5 line: over (p > 0.5 = book). Over hit at 1.91 -> +0.91; Gainwell's push returns the stake.
    roi0 = report["overall"]["roi"][0]
    assert roi0["margin"] == 0 and roi0["bets"] == 2 and roi0["pushes"] == 1 and roi0["profit"] == pytest.approx(0.91)
    assert o["brier_book"] == pytest.approx(0.25)
    assert o["side_hit_rate"] == 1.0
    # Opening FanDuel line 255.5, model said over, closed 260.5: beat the close by 5 yards.
    assert o["clv"] == {"n": 1, "beat": 1, "same": 0, "worse": 0, "beat_rate": 1.0, "mean_line_move": 5.0}
    assert "too small to conclude" in report["summary"][0]
    assert report["markets"]["passing_yards"]["graded"] == 1


def test_roi_margins_and_clv_same_line():
    df = pd.DataFrame({
        "status": "graded", "push": False, "market": "receptions", "bookmaker": "fanduel",
        "p_model": [0.60, 0.53, 0.40], "p_model_raw": [0.6, 0.53, 0.4], "p_book": [0.50, 0.50, 0.50],
        "over_price": [1.91] * 3, "under_price": [1.91] * 3, "outcome": [1.0, 0.0, 1.0], "line": [4.5] * 3,
        "open_line": [4.5, np.nan, np.nan], "open_over": [2.0, np.nan, np.nan], "open_under": [1.8, np.nan, np.nan],
        "p_model_open": [0.6, np.nan, np.nan],
    })
    roi = {r["margin"]: r for r in mvl.roi_table(df)}
    assert roi[0.0]["bets"] == 3 and roi[0.0]["profit"] == pytest.approx(0.91 - 1 - 1)
    assert roi[0.05]["bets"] == 2 and roi[0.10]["bets"] == 2 and roi[0.08]["wins"] == 1
    # Same line, over's no-vig went from ~47% (2.00 / 1.80) to 50%: the opening over beat the close.
    assert mvl.clv(df)["beat"] == 1


def test_empty_report_says_nothing_to_conclude():
    report = mvl.build_report(mvl._empty(), mvl.match_names(_players(), ROSTER))
    assert report["overall"]["graded"] == 0 and report["overall"]["roi"] == []
    assert "nothing to conclude" in report["summary"][0]
