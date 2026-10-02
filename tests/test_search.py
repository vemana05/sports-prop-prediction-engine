"""Unit tests for the beam search parlay builder — synthetic legs and a
fixed correlation table, no DB needed."""

import pytest

from backend.parlay.monte_carlo import Leg
from backend.parlay.search import (
    _beam_search_by_naive_ev,
    _is_duplicate_or_contradiction,
    search_tier,
)


def _leg(player, market, line, mean, sigma, side="over", odds=1.9, game_id="g1", team="A"):
    return Leg(
        player_id_nba=player, player_name=player, game_id=game_id, team=team, market=market,
        side=side, line=line, predicted_mean=mean, dispersion=sigma,
        decimal_odds=odds, over_price=odds, under_price=odds,
    )


_ZERO_CORR = {"same_player": {}, "teammate": {}, "opponent": {}}


def _zero_corr_matrices():
    markets = ["points", "rebounds", "assists", "fg3m"]
    table = {(a, b): (1.0 if a == b else 0.0, 10_000) for a in markets for b in markets}
    return {"same_player": table, "teammate": table, "opponent": table}


def test_duplicate_same_player_same_market_same_side_is_flagged():
    a = _leg("p1", "points", 20.5, 22, 5)
    b = _leg("p1", "points", 20.5, 22, 5)
    assert _is_duplicate_or_contradiction(b, [a]) is True


def test_contradiction_same_player_same_market_opposite_side_is_flagged():
    over = _leg("p1", "points", 20.5, 22, 5, side="over")
    under = _leg("p1", "points", 20.5, 22, 5, side="under")
    assert _is_duplicate_or_contradiction(under, [over]) is True


def test_same_player_different_market_is_not_flagged():
    pts = _leg("p1", "points", 20.5, 22, 5)
    reb = _leg("p1", "rebounds", 8.5, 9, 15)
    assert _is_duplicate_or_contradiction(reb, [pts]) is False


def test_different_players_never_flagged():
    a = _leg("p1", "points", 20.5, 22, 5)
    b = _leg("p2", "points", 18.5, 20, 5)
    assert _is_duplicate_or_contradiction(b, [a]) is False


def test_beam_search_never_includes_a_contradiction_within_one_parlay():
    over = _leg("p1", "points", 20.5, 25, 5, odds=1.9)  # strong favorite, high naive EV both sides look tempting
    under = _leg("p1", "points", 20.5, 25, 5, side="under", odds=1.9)
    other = _leg("p2", "points", 10.5, 15, 4, odds=1.9, game_id="g2")
    finalists = _beam_search_by_naive_ev([over, under, other], legs_min=1, legs_max=2, beam_width=50, finalists_per_size=10)
    for legs in finalists:
        ids = [(l.player_id_nba, l.market, l.side) for l in legs]
        assert not ("p1", "points", "over") in ids or not ("p1", "points", "under") in ids


def test_beam_search_respects_legs_min_and_legs_max():
    legs = [_leg(f"p{i}", "points", 10 + i, 12 + i, 4, game_id=f"g{i}") for i in range(6)]
    finalists = _beam_search_by_naive_ev(legs, legs_min=3, legs_max=4, beam_width=200, finalists_per_size=20)
    assert finalists, "expected at least one finalist"
    for f in finalists:
        assert 3 <= len(f) <= 4


def test_search_tier_filters_by_min_leg_prob():
    # a leg with a very low hit probability should never appear even though
    # it offers attractive-looking odds
    strong = _leg("p1", "points", 10.5, 20, 3, odds=1.5)  # very likely to hit
    weak = _leg("p2", "points", 30.5, 10, 3, odds=5.0, game_id="g2")  # very unlikely to hit, but big payout
    tier_config = {"legs_min": 1, "legs_max": 1, "min_leg_prob": 0.7, "target_parlay_prob_min": 0.0, "target_parlay_prob_max": 1.0, "max_stake_pct_bankroll": 0.02}
    results = search_tier([strong, weak], "test", tier_config, _zero_corr_matrices(), bankroll=500, n_simulations=2000, seed=1)
    for c in results:
        assert len(c.legs) == 1
        assert c.legs[0].player_id_nba == "p1"


def test_search_tier_respects_target_parlay_probability_range():
    legs = [_leg(f"p{i}", "points", 15, 20, 4, odds=1.9, game_id=f"g{i}") for i in range(5)]
    tight_tier = {"legs_min": 2, "legs_max": 2, "min_leg_prob": 0.0, "target_parlay_prob_min": 0.99, "target_parlay_prob_max": 1.0, "max_stake_pct_bankroll": 0.02}
    results = search_tier(legs, "test", tight_tier, _zero_corr_matrices(), bankroll=500, n_simulations=2000, seed=1)
    for c in results:
        assert 0.99 <= c.p_hit <= 1.0


def test_search_tier_returns_nothing_when_no_legs_meet_min_prob():
    weak = _leg("p1", "points", 30.5, 10, 3, odds=5.0)
    tier_config = {"legs_min": 1, "legs_max": 1, "min_leg_prob": 0.99, "target_parlay_prob_min": 0.0, "target_parlay_prob_max": 1.0, "max_stake_pct_bankroll": 0.02}
    results = search_tier([weak], "test", tier_config, _zero_corr_matrices(), bankroll=500, n_simulations=1000, seed=1)
    assert results == []


def test_search_tier_flags_a_leg_that_wildly_disagrees_with_the_no_vig_price():
    # _leg's over_price == under_price == decimal_odds => no_vig_p_hit() is
    # always exactly 0.5; a line deep in the tail of the model's own
    # distribution (mean 20, sigma 2, line 10.5) makes the model's own
    # probability close to 1.0 — a >15-point divergence that, in the real
    # Oct 20 slate, turned out to be a feature/data bug, not a real edge.
    suspicious = _leg("p1", "points", 10.5, 20, 2, odds=1.9)
    tier_config = {"legs_min": 1, "legs_max": 1, "min_leg_prob": 0.0, "target_parlay_prob_min": 0.0, "target_parlay_prob_max": 1.0, "max_stake_pct_bankroll": 0.02}
    results = search_tier([suspicious], "test", tier_config, _zero_corr_matrices(), bankroll=500, n_simulations=1000, seed=1)
    assert len(results) == 1
    assert results[0].flagged is True
    assert results[0].stake.stake == 0.0
    assert "CHECK DATA" in results[0].stake.reason


def test_search_tier_does_not_flag_a_leg_close_to_the_no_vig_price():
    reasonable = _leg("p1", "points", 19.5, 20, 5, odds=1.9)
    tier_config = {"legs_min": 1, "legs_max": 1, "min_leg_prob": 0.0, "target_parlay_prob_min": 0.0, "target_parlay_prob_max": 1.0, "max_stake_pct_bankroll": 0.02}
    results = search_tier([reasonable], "test", tier_config, _zero_corr_matrices(), bankroll=500, n_simulations=1000, seed=1)
    assert len(results) == 1
    assert results[0].flagged is False
    assert results[0].flag_reasons == []
