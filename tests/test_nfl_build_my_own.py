"""NFL "Build my own": custom legs at any line and side, scored by the same
engine as suggestions (Monte Carlo with correlation, game-script note,
30%-shrink Kelly, sanity guard). Synthetic week; never opens a warehouse."""

import time

import numpy as np
import pytest
from fastapi import HTTPException

from backend.api import main, nfl_builder
from backend.models.nfl_market_model import Spread
from backend.parlay import nfl_legs, nfl_search
from backend.parlay.search import leg_id

TIER = {"legs_min": 2, "legs_max": 5, "min_leg_prob": 0.5, "target_parlay_prob_min": 0.1,
        "target_parlay_prob_max": 0.4, "max_stake_pct_bankroll": 0.02}
YARDS = Spread(a=10.0, b=0.25, z_sorted=np.linspace(-2.5, 2.5, 201))
SPREADS = {"passing_yards": YARDS, "rushing_yards": YARDS, "receiving_yards": YARDS, "receptions": Spread(r=8.0)}
CALS = {m: None for m in SPREADS}
GAME = {"game_id": "2026_08_KC_BUF", "home": "BUF", "away": "KC", "kickoff_utc": "2099-10-25T20:25:00+00:00"}


def _player(pid, name, pos, team, opp, means):
    return {"player_id": pid, "player_name": name, "position": pos, "team": team, "opponent": opp,
            "is_home": team == "BUF", "game_id": GAME["game_id"], "kickoff_utc": GAME["kickoff_utc"],
            "questionable": False, "means": means}


PLAYERS = {
    "00-0033873": _player("00-0033873", "Pat QB", "QB", "KC", "BUF", {"passing_yards": 255.0}),
    "00-0039001": _player("00-0039001", "Kay RB", "RB", "KC", "BUF", {"rushing_yards": 62.0, "receptions": 2.5}),
    "00-0039002": _player("00-0039002", "Bee WR", "WR", "BUF", "KC", {"receiving_yards": 68.0, "receptions": 5.6}),
}
OUT = [{"player_id": "00-0039999", "player_name": "Hurt WR", "team": "BUF", "position": "WR"}]


def _corr(teammate_pass_rush=0.0):
    pairs = {(a, b): (0.0, 0) for a in SPREADS for b in SPREADS}
    team = dict(pairs)
    team[("passing_yards", "rushing_yards")] = team[("rushing_yards", "passing_yards")] = (teammate_pass_rush, 100)
    return {"same_player": dict(pairs), "teammate": team, "opponent": dict(pairs)}


@pytest.fixture
def wp():
    meta = {"season": 2026, "week": 8, "weeks": [], "injuries_posted": True, "calibrated": False,
            "players": PLAYERS, "out": OUT, "spreads": SPREADS, "calibrators": CALS, "narrow": {}}
    legs = [nfl_legs.make_leg(p, m, "over", nfl_legs.rung_line(r), SPREADS, CALS, {}, rung=r)
            for p in PLAYERS.values() for m in p["means"] for r in nfl_legs.LADDERS[m]]
    pool = nfl_builder.WeekPool(week=8, legs=legs, games={GAME["game_id"]: GAME}, meta=meta, built_at=time.time())
    pool.legs_by_id = {leg_id(l): l for l in legs}
    pool.players = PLAYERS
    return pool


@pytest.fixture
def api(wp, monkeypatch):
    monkeypatch.setattr(main, "_nfl_pool", lambda week: wp)
    monkeypatch.setattr(nfl_builder.NFL_SLATE, "_corr", _corr())
    monkeypatch.setattr(main, "load_tiers", lambda: {"risky": TIER, "safe": TIER, "dangerous": TIER})
    return wp


def _score(leg_ids, prices=None):
    req = main.ScoreRequest(league="nfl", week=8, leg_ids=leg_ids, prices=prices or {}, tier="risky", bankroll=1000)
    return main.score(req)["parlay"]


def test_custom_id_parses_hyphenated_nfl_player_ids():
    assert nfl_builder.parse_leg_id("00-0033873-passing_yards-under-249.5") == ("00-0033873", "passing_yards", "under", 249.5)


def test_any_ladder_value_and_custom_line_over_or_under(wp):
    over = nfl_builder.find_leg(wp, "00-0039001-rushing_yards-over-89.5")  # 90+, off the ladder
    under = nfl_builder.find_leg(wp, "00-0039001-rushing_yards-under-89.5")
    assert over.p_hit() + under.p_hit() == pytest.approx(1.0)
    assert nfl_builder.leg_json(under)["label"] == "Under 89.5 rush yds"
    assert nfl_builder.leg_json(under)["side"] == "under"
    rung = nfl_builder.find_leg(wp, "00-0039001-rushing_yards-over-59.5")
    assert nfl_builder.leg_json(rung)["label"] == "60+ rush yds"


def test_out_and_unknown_players_and_bad_lines_are_refused(wp):
    with pytest.raises(HTTPException) as e:
        nfl_builder.find_leg(wp, "00-0039999-receiving_yards-over-49.5")
    assert e.value.status_code == 422 and "Out" in e.value.detail
    with pytest.raises(HTTPException):
        nfl_builder.find_leg(wp, "00-0033873-rushing_yards-over-9.5")  # no rushing projection
    with pytest.raises(HTTPException):
        nfl_builder.find_leg(wp, "00-0033873-passing_yards-sideways-249.5")
    with pytest.raises(HTTPException):
        nfl_builder.find_leg(wp, "00-0033873-passing_yards-over-9999")


def test_players_json_lists_every_ladder_rung_and_out_players_disabled(wp):
    data = nfl_builder.players_json(wp)
    qb = next(p for p in data["players"] if p["player_id"] == "00-0033873")
    assert [r["rung"] for r in qb["markets"]["passing_yards"]["ladder"]] == nfl_legs.LADDERS["passing_yards"]
    ps = [r["p_over"] for r in qb["markets"]["passing_yards"]["ladder"]]
    assert ps == sorted(ps, reverse=True)
    assert [o["player_name"] for o in data["out"]] == ["Hurt WR"] and data["out"][0]["out"] is True
    assert "never against real NFL sportsbook lines" in data["untested_note"]


def test_unpriced_slip_scores_hit_chance_with_correlation_only(api, monkeypatch):
    monkeypatch.setattr(nfl_builder.NFL_SLATE, "_corr", _corr(-0.3))
    p = _score(["00-0033873-passing_yards-over-249.5", "00-0039001-rushing_yards-over-59.5"])
    assert 0 < p["p_hit"] < 1 and p["priced"] is False and p["stake"] == 0
    assert p["american_odds"] is None and p["fair_american_odds"] is not None
    assert p["p_hit"] < p["independent_p_hit"] - 0.01  # negative teammate correlation is applied
    assert any(n.startswith("Game script: KC @ BUF") for n in p["notes"])


def test_under_on_passing_with_rb_over_is_not_a_script_conflict(api):
    p = _score(["00-0033873-passing_yards-under-249.5", "00-0039001-rushing_yards-over-59.5"])
    assert not p["notes"]


def test_typed_odds_give_combined_odds_ev_and_shrunk_kelly(api):
    ids = ["00-0033873-passing_yards-under-259.5", "00-0039002-receptions-over-5.5"]  # near 50/50, like the book
    p = _score(ids, {ids[0]: {"under": -110, "over": -110}, ids[1]: {"over": -110, "under": -110}})
    assert p["priced"] is True
    assert p["decimal_odds"] == pytest.approx((1 + 100 / 110) ** 2)
    assert p["no_vig_p"] == pytest.approx(0.25)
    assert p["p_sized"] == pytest.approx(0.7 * p["p_hit"] + 0.3 * 0.25)  # 30% shrink toward the book
    assert p["ev_per_dollar"] == pytest.approx(p["p_sized"] * p["decimal_odds"] - 1)
    assert not p["flagged"]
    leg = p["legs"][0]
    assert leg["side"] == "under" and leg["american_odds"] == -110


def test_under_leg_is_priced_by_its_own_side(api):
    i = "00-0033873-passing_yards-under-249.5"
    assert _score([i, "00-0039002-receptions-over-3.5"], {i: {"over": -110}})["legs"][0]["priced"] is False
    leg = nfl_search.apply_price(nfl_builder.find_leg(api, i), None, None, 150)
    assert leg.priced and leg.decimal_odds == pytest.approx(2.5) and leg.no_vig_p_hit() == pytest.approx(0.4)


def test_sanity_guard_zeroes_the_stake_on_a_wild_model_vs_book_gap(api):
    ids = ["00-0039001-rushing_yards-over-24.5", "00-0039002-receptions-over-2.5"]  # both very likely by the model
    p = _score(ids, {ids[0]: {"over": 400, "under": -600}, ids[1]: {"over": -110, "under": -110}})
    assert p["flagged"] is True and p["stake"] == 0
    assert p["reason"].startswith("CHECK DATA")


def test_same_player_market_twice_is_refused(api):
    with pytest.raises(HTTPException) as e:
        _score(["00-0039001-rushing_yards-over-59.5", "00-0039001-rushing_yards-under-40.5"])
    assert e.value.status_code == 422


def test_typed_line_keeps_the_picked_leg_id_as_base(api):
    i = "00-0039002-receiving_yards-over-69.5"
    p = _score([i, "00-0033873-passing_yards-over-249.5"], {i: {"line": "64.5", "over": -120}})
    assert p["legs"][0]["base_id"] == i and p["legs"][0]["line"] == 64.5
