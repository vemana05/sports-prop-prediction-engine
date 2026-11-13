"""NBA model vs. lines: one book per prop (DraftKings, else FanDuel, last
snapshot before tip), the features-date rule, push and did-not-play
handling, the report's metrics, the Builder replay and its parlay grading,
the build-time suggestion log, and the shrink/cap recommendation. Temp
warehouse; no network."""

import json
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from backend.api import main
from backend.backtest import nba_model_vs_lines as mvl
from backend.db import get_connection
from backend.parlay import suggestion_log
from backend.parlay.monte_carlo import Leg
from backend.parlay.search import score_parlay

UTC = timezone.utc
HOME, AWAY = "1610612747", "1610612744"  # LAL, GSW
ABBR = {HOME: "LAL", AWAY: "GSW"}
DAY = date(2026, 11, 5)
TIP = datetime(2026, 11, 6, 2, 0, tzinfo=UTC)  # 9 PM ET Nov 5
NOW = datetime(2026, 11, 13, 12, 0, tzinfo=UTC)
CORR = {rel: {("points", "points"): (0.0, 0)} for rel in ("same_player", "teammate", "opponent")}
TIER = {"legs_min": 2, "legs_max": 2, "min_leg_prob": 0.0, "target_parlay_prob_min": 0.0,
        "target_parlay_prob_max": 1.0, "max_stake_pct_bankroll": 0.01}


def _no_resolve(conn, names):
    return {n: n.lower().replace(" ", "_") for n in names if n != "Unknown Guy"}


# ---- fixture warehouse ------------------------------------------------------------

# player: (predicted mean, features_through, actual pts or None = not in box score)
PLAYERS = {
    "a_star": (30.0, date(2026, 11, 4), 34.0),   # over hits
    "b_role": (10.0, date(2026, 11, 4), 8.0),    # under hits
    "c_push": (20.0, date(2026, 11, 4), 20.0),   # whole line, push
    "d_dnp": (12.0, date(2026, 11, 4), None),    # box loaded, didn't play
    "e_old": (15.0, None, 18.0),                 # logged before features_through existed
    "f_same": (15.0, DAY, 18.0),                 # features dated the game day
    "g_noline": (15.0, date(2026, 11, 4), 18.0),
}
LINES = {"a_star": 29.5, "b_role": 9.5, "c_push": 20.0, "d_dnp": 11.5, "e_old": 14.5, "f_same": 14.5}


@pytest.fixture
def conn(tmp_path):
    c = get_connection(str(tmp_path / "w.duckdb"))
    c.execute("INSERT INTO nba_schedule VALUES ('G1', '2026-27', ?, ?, ?, ?, now())", [DAY, HOME, AWAY, TIP])
    c.execute("INSERT INTO nba_game_results VALUES ('G1', '2026-27', 3, 110, 100, now())")
    # a later game that tipped but has no box score yet
    c.execute("INSERT INTO nba_schedule VALUES ('G2', '2026-27', ?, ?, ?, ?, now())",
              [date(2026, 11, 7), HOME, AWAY, datetime(2026, 11, 8, 2, 0, tzinfo=UTC)])
    for pid, (mean, ft, actual) in PLAYERS.items():
        c.execute("INSERT INTO model_predictions_log VALUES (?, 'G1', ?, 'points', ?, 6.0, 'v', ?)",
                  [TIP - timedelta(hours=20), pid, mean, ft])
        if actual is not None:
            c.execute("""INSERT INTO nba_player_games (player_id_nba, player_name, team, opponent, game_id, game_date, season, pts, ingested_at)
                         VALUES (?, ?, 'LAL', 'GSW', 'G1', ?, '2026-27', ?, now())""", [pid, pid.replace("_", " ").title(), DAY, actual])
    c.execute("INSERT INTO model_predictions_log VALUES (?, 'G2', 'a_star', 'points', 30.0, 6.0, 'v', ?)",
              [datetime(2026, 11, 7, 12), date(2026, 11, 6)])
    for pid, line in LINES.items():
        name = pid.replace("_", " ")
        # DraftKings: an early snapshot, the last pre-tip one, and one after tip (ignored)
        for pulled, ln in [(TIP - timedelta(hours=30), line + 3), (TIP - timedelta(hours=6), line), (TIP + timedelta(hours=1), line + 5)]:
            c.execute("INSERT INTO odds_snapshots VALUES (?, 'ev1', NULL, ?, ?, 'player_points', 'draftkings', ?, 1.91, 1.91)",
                      [pulled.replace(tzinfo=None), TIP, name, ln])
        c.execute("INSERT INTO odds_snapshots VALUES (?, 'ev1', NULL, ?, ?, 'player_points', 'fanduel', ?, 1.8, 2.0)",
                  [(TIP - timedelta(hours=5)).replace(tzinfo=None), TIP, name, line + 1])
    c.execute("INSERT INTO odds_snapshots VALUES (?, 'ev2', NULL, ?, 'a star', 'player_points', 'draftkings', 29.5, 1.91, 1.91)",
              [datetime(2026, 11, 7, 15), datetime(2026, 11, 8, 2, 0, tzinfo=UTC)])
    yield c
    c.close()


def _graded(conn, calibrator=None):
    preds = mvl.load_predictions(conn)
    odds, unresolved = mvl.load_odds(conn, resolve=_no_resolve)
    g = mvl.grade(preds, mvl.pick_lines(odds, preds), NOW, calibrator)
    g["star"] = g["player_id_nba"] == "a_star"
    return g.set_index(["game_id", "player_id_nba"]), odds


# ---- lines and statuses -------------------------------------------------------------

def test_one_book_last_snapshot_before_tip_draftkings_first():
    odds = pd.DataFrame({
        "player_id_nba": ["p"] * 4 + ["q"], "game_date": [DAY] * 5,
        "bookmaker": ["draftkings", "draftkings", "fanduel", "draftkings", "fanduel"],
        "pulled_at": pd.to_datetime(["2026-11-05 10:00", "2026-11-05 15:00", "2026-11-05 20:00", "2026-11-06 03:00", "2026-11-05 15:00"]),
        "commence": pd.to_datetime(["2026-11-06 02:00"] * 5),
        "line": [20.5, 21.5, 22.5, 25.5, 9.5], "over_price": 1.9, "under_price": 1.9, "event_id": "e",
    })
    games = pd.DataFrame({"game_id": ["G1", "G1"], "player_id_nba": ["p", "q"], "game_date": [DAY, DAY],
                          "tip": pd.to_datetime(["2026-11-06 02:00"] * 2)})
    got = mvl.pick_lines(odds, games).set_index("player_id_nba")
    assert (got.loc["p", "bookmaker"], got.loc["p", "line"]) == ("draftkings", 21.5)  # FanDuel is later, but DK wins; post-tip DK ignored
    assert (got.loc["q", "bookmaker"], got.loc["q", "line"]) == ("fanduel", 9.5)


def test_statuses_features_rule_push_and_void(conn):
    g, _ = _graded(conn)
    s = g["status"]
    assert s["G1", "a_star"] == "graded" and s["G1", "b_role"] == "graded"
    assert s["G1", "c_push"] == "push"
    assert s["G1", "d_dnp"] == "did_not_play"
    assert s["G1", "e_old"] == "no_features_date"
    assert s["G1", "f_same"] == "features_not_before_game"
    assert s["G1", "g_noline"] == "no_line"
    assert s["G2", "a_star"] == "no_box_score"  # tipped, finished by the clock, box score not loaded
    assert g.loc[("G1", "a_star"), "bookmaker"] == "draftkings" and g.loc[("G1", "a_star"), "line"] == 29.5
    assert g.loc[("G1", "a_star"), "outcome"] == 1.0 and g.loc[("G1", "b_role"), "outcome"] == 0.0
    assert np.isnan(g.loc[("G1", "c_push"), "outcome"])


def test_game_not_finished_before_tip_plus_buffer(conn):
    preds = mvl.load_predictions(conn)
    odds, _ = mvl.load_odds(conn, resolve=_no_resolve)
    g = mvl.grade(preds, mvl.pick_lines(odds, preds), datetime(2026, 11, 8, 3, 0, tzinfo=UTC))
    assert g.set_index(["game_id", "player_id_nba"]).loc[("G2", "a_star"), "status"] == "not_finished"


def test_model_probability_matches_builder_and_removes_push():
    cal, raw = mvl.model_p_over([25.0], [6.0], [22.5])
    leg = Leg("p", "p", "g", "LAL", "points", "over", 22.5, 25.0, 6.0, 1.9)
    assert cal[0] == pytest.approx(leg.p_over())
    cal, _ = mvl.model_p_over([20.0], [6.0], [20.0])  # whole line at the mean: 50/50 once the push is out
    assert cal[0] == pytest.approx(0.5)
    over = 1 - stats.norm.cdf(23.5, 20, 6)
    push = stats.norm.cdf(23.5, 20, 6) - stats.norm.cdf(22.5, 20, 6)
    assert mvl.model_p_over([20.0], [6.0], [23.0])[0][0] == pytest.approx(over / (1 - push))
    assert mvl.no_vig_over(1.8, 2.0) == pytest.approx((1 / 1.8) / (1 / 1.8 + 1 / 2.0))


# ---- metrics ------------------------------------------------------------------------

def _synthetic(n=400, seed=0, model_noise=0.15):
    rng = np.random.default_rng(seed)
    pb = rng.uniform(0.35, 0.65, n)
    y = (rng.uniform(size=n) < pb).astype(float)
    pm = np.clip(pb + rng.normal(0, model_noise, n), 0.01, 0.99)
    return pd.DataFrame({"status": "graded", "p_model": pm, "p_model_raw": pm, "p_book": pb, "outcome": y,
                         "player_id_nba": [f"p{i % 50}" for i in range(n)], "star": np.arange(n) % 5 == 0,
                         "predicted_mean": 20.0 + rng.normal(0, 3, n), "actual": 20.0 + rng.normal(0, 5, n),
                         "line": 20.5, "bookmaker": "draftkings"})


def test_summary_sides_gap_buckets_and_stars():
    df = _synthetic()
    s = mvl.summarize(df)
    assert s["graded"] == 400 and s["enough_sample"]
    assert s["over"]["n"] == s["under"]["n"] == 400
    assert s["over"]["hit_rate"] + s["under"]["hit_rate"] == pytest.approx(1)
    assert sum(b["n"] for b in s["over"]["calibration_model"]) == 400
    assert sum(g["n"] for g in s["gap"]) == 400
    assert [(g["lo"], g["hi"]) for g in s["gap"]] == [(0, 5), (5, 10), (10, 15), (15, None)]
    stars = df[df["star"]]
    assert s["stars"]["n"] == len(stars)
    assert s["stars"]["mean_error"] == pytest.approx((stars["predicted_mean"] - stars["actual"]).mean(), abs=0.01)
    assert s["brier_book"] < s["brier_model"]  # the noisy model loses to the line it was built from


def test_gap_hit_rate_uses_model_side():
    df = pd.DataFrame({"status": "graded", "p_model": [0.70, 0.30, 0.52], "p_book": [0.5, 0.5, 0.5],
                       "outcome": [1.0, 1.0, 0.0]})
    gap = mvl.gap_table(df)
    assert gap[3]["n"] == 2 and gap[3]["hits"] == 1  # over picked and hit; under picked and missed
    assert gap[0]["n"] == 1 and gap[0]["hit_rate"] == 0.0


def test_recommendation_shrinks_a_noisy_model_and_needs_a_sample():
    rec = mvl.recommend(_synthetic(model_noise=0.15))
    assert rec["available"]
    assert rec["shrink"] >= 0.5 or (rec["cap"] is not None and rec["cap"] <= 0.06)
    assert rec["brier_after"] <= rec["brier_before"]
    assert not mvl.recommend(_synthetic(n=20))["available"]


def test_adjust_shrinks_then_caps():
    assert mvl.adjust([0.8], [0.5], 0.5, None)[0] == pytest.approx(0.65)
    assert mvl.adjust([0.8], [0.5], 0.5, 0.05)[0] == pytest.approx(0.55)
    assert mvl.adjust([0.2], [0.5], 0.0, 0.1)[0] == pytest.approx(0.4)


def test_report_on_an_empty_sample_says_so(conn):
    g, _ = _graded(conn)
    g = g.reset_index()
    g["status"] = g["status"].replace({"graded": "no_box_score", "push": "no_box_score"})
    r = mvl.build_report(g, [], 6, [], [], None, [], ["safe"], date(2026, 10, 27), NOW)
    assert r["overall"]["graded"] == 0 and not r["recommendation"]["available"]
    assert "No points props can be graded yet" in r["summary"][0] and "2026-10-27" in r["summary"][0]


# ---- parlays ------------------------------------------------------------------------

def test_leg_and_parlay_grading():
    assert mvl.grade_leg(30.0, 29.5, "over", "final") == "hit"
    assert mvl.grade_leg(30.0, 29.5, "under", "final") == "miss"
    assert mvl.grade_leg(20.0, 20.0, "over", "final") == "push"
    assert mvl.grade_leg(None, 20.5, "over", "did_not_play") == "void"
    assert mvl.grade_leg(None, 20.5, "over", "pending") == "pending"
    assert mvl.grade_parlay(["hit", "hit"]) == "won"
    assert mvl.grade_parlay(["hit", "miss", "pending"]) == "lost"
    assert mvl.grade_parlay(["hit", "pending"]) == "pending"
    assert mvl.grade_parlay(["hit", "push"]) == "reduced"
    assert mvl.grade_parlay(["void"]) == "void"


def test_replay_builds_from_snapshot_at_build_and_grades(conn):
    g, odds = _graded(conn)
    preds = mvl.load_predictions(conn)
    days = mvl.replay_days(conn, preds, odds, mvl.last_teams(conn), ABBR, lambda d: False)
    assert [d.day for d in days] == [DAY, date(2026, 11, 7)]
    d = days[0]
    # last pull before the first tip is the FanDuel one at tip-5h, but DraftKings' tip-6h snapshot wins the book rule
    assert d.build_at == (TIP - timedelta(hours=5)).replace(tzinfo=None)
    # features rule applies here too; d_dnp has no game log to place him on a team, so he gets no leg
    assert set(d.rows["player_id_nba"]) == {"a_star", "b_role", "c_push"}
    assert set(d.rows["bookmaker"]) == {"draftkings"}
    states = mvl.leg_state(g.reset_index())
    parlays = mvl.run_replay(days[:1], None, CORR, {"risky": TIER}, states, n_simulations=500)
    assert parlays and all(p["tier"] == "risky" and len(p["legs"]) == 2 for p in parlays)
    results = {p["result"] for p in parlays}
    assert results <= {"won", "lost", "reduced"}
    s = mvl.parlay_summary(parlays, ["risky"])["risky"]
    assert s["suggested"] == len(parlays) and s["settled"] == sum(p["result"] in ("won", "lost") for p in parlays)


def test_adjusted_calibrator_reaches_the_leg():
    leg = Leg("p", "p", "g", "LAL", "points", "over", 20.5, 30.0, 6.0, 1.9, over_price=1.91, under_price=1.91,
              calibrator=mvl.AdjustedCalibrator(None, 0.5, 0.0, 0.05))
    assert leg.p_hit() == pytest.approx(0.55)


def test_suggestions_logged_at_build_and_graded(conn):
    legs = [Leg("a_star", "A Star", "ev1", "LAL", "points", "over", 29.5, 30.0, 6.0, 1.91, 1.91, 1.91),
            Leg("b_role", "B Role", "ev1", "LAL", "points", "under", 9.5, 10.0, 6.0, 1.91, 1.91, 1.91)]
    c = score_parlay(legs, TIER, CORR, 100.0, n_simulations=500, seed=1)
    game = type("G", (), {"nba_game_id": "G1", "tip_utc": TIP})()
    built = datetime(2026, 11, 5, 15, 0, tzinfo=UTC)
    assert suggestion_log.log_suggestions(conn, built, {"risky": [c]}, {"ev1": game}, True) == 1
    suggestion_log.log_suggestions(conn, built + timedelta(hours=1), {"risky": [c]}, {"ev1": game}, True)  # rebuild
    row = conn.execute(f"SELECT tier, rank, n_legs, legs, early_season FROM {suggestion_log.TABLE} ORDER BY built_at").fetchone()
    assert row[:3] == ("risky", 1, 2) and row[4] is True
    assert json.loads(row[3])[0]["nba_game_id"] == "G1"
    g, _ = _graded(conn)
    logged = mvl.grade_logged(conn, mvl.leg_state(g.reset_index()))
    assert len(logged) == 1 and logged[0]["result"] == "won" and logged[0]["date"] == "2026-11-05"


def test_api_reports_unavailable_until_built(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "REPORTS_DIR", tmp_path)
    assert main.nba_model_vs_lines() == {"available": False}
    (tmp_path / "nba_model_vs_lines.json").write_text(json.dumps({"summary": ["x"]}))
    assert main.nba_model_vs_lines() == {"available": True, "summary": ["x"]}
