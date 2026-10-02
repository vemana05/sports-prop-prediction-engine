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


# --- Section 8 step 3: size on the shrunk probability, not the raw one ---

_WIDE_TIER = {"legs_min": 1, "legs_max": 1, "min_leg_prob": 0.0, "target_parlay_prob_min": 0.0, "target_parlay_prob_max": 1.0, "max_stake_pct_bankroll": 0.05}


def test_stake_is_sized_on_probability_shrunk_toward_no_vig():
    from backend.bankroll.kelly import recommend_stake, shrink_to_market

    # model P(over 19.5 | mean 21, sigma 5) ~= 0.618; no-vig is 0.5 (equal prices)
    leg = _leg("p1", "points", 19.5, 21, 5, odds=1.9)
    [c] = search_tier([leg], "test", _WIDE_TIER, _zero_corr_matrices(), bankroll=1000, n_simulations=1000, seed=1)
    assert c.flagged is False
    assert c.no_vig_p == pytest.approx(0.5)
    assert c.p_sized == pytest.approx(shrink_to_market(c.p_hit, 0.5))
    assert c.p_sized < c.p_hit

    sized = recommend_stake(c.p_sized, 1.9, 1000, 0.05)
    unshrunk = recommend_stake(c.p_hit, 1.9, 1000, 0.05)
    assert c.stake.stake == sized.stake == 29.5  # 1000 * f*(0.5825) 0.1187 * 0.25, floored to $0.50
    assert c.stake.ev_per_dollar == pytest.approx(sized.ev_per_dollar)
    assert unshrunk.stake > c.stake.stake  # the bug: sizing on raw p overstakes


def test_parlay_no_vig_is_the_product_of_leg_no_vig_prices():
    from backend.parlay.search import parlay_no_vig_p

    a = _leg("p1", "points", 19.5, 21, 5)
    b = _leg("p2", "points", 19.5, 21, 5, game_id="g2")
    b.over_price, b.under_price = 1.5, 3.0  # no-vig over = (1/1.5) / (1/1.5 + 1/3) = 2/3
    assert parlay_no_vig_p([a, b]) == pytest.approx(0.5 * 2 / 3)


# --- side preference, veto filtering, empty-tier reasons ---

def test_filter_pool_keeps_only_the_chosen_side_and_drops_vetoes_and_games():
    from backend.parlay.search import filter_pool, leg_id

    over = _leg("p1", "points", 20.5, 22, 5, side="over")
    under = _leg("p1", "points", 20.5, 22, 5, side="under")
    other = _leg("p2", "points", 18.5, 20, 5, side="under", game_id="g2")
    assert filter_pool([over, under, other], "over") == [over]
    assert filter_pool([over, under, other], "under") == [under, other]
    assert filter_pool([over, under, other], "both", vetoed={leg_id(under)}, excluded_games={"g2"}) == [over]


def test_report_explains_when_no_leg_clears_the_tier_minimum():
    from backend.parlay.search import run_search

    weak = _leg("p1", "points", 30.5, 25, 5, odds=1.9)
    tier = {**_WIDE_TIER, "min_leg_prob": 0.70}
    results, report = run_search([weak], "safe", tier, _zero_corr_matrices(), bankroll=500, n_simulations=500, seed=1)
    assert results == []
    assert report.reason.startswith("No legs clear 70% in the current pool")
    assert report.best_leg_p < 0.70


def test_report_explains_when_no_combination_lands_in_the_hit_band():
    from backend.parlay.search import run_search

    legs = [_leg(f"p{i}", "points", 15.5, 20, 4, odds=1.9, game_id=f"g{i}") for i in range(6)]
    tier = {"legs_min": 5, "legs_max": 6, "min_leg_prob": 0.0, "target_parlay_prob_min": 0.0, "target_parlay_prob_max": 0.001, "max_stake_pct_bankroll": 0.005}
    results, report = run_search(legs, "dangerous", tier, _zero_corr_matrices(), bankroll=500, n_simulations=2000, seed=1)
    assert results == []
    assert report.reason.startswith("No 5-6 leg combinations fall under 0.1% hit chance")
    assert report.n_combinations > 0 and report.combo_p_min > 0.001


def test_report_names_the_side_when_a_side_filter_leaves_nothing():
    from backend.parlay.search import filter_pool, run_search

    over = _leg("p1", "points", 20.5, 22, 5, side="over")
    results, report = run_search(filter_pool([over], "under"), "risky", _WIDE_TIER, _zero_corr_matrices(), 500, sides="under")
    assert results == [] and report.reason == "No under legs left in the current pool."


# --- veto + replacement ---

def _two_leg_tier():
    return {"legs_min": 2, "legs_max": 2, "min_leg_prob": 0.5, "target_parlay_prob_min": 0.0, "target_parlay_prob_max": 1.0, "max_stake_pct_bankroll": 0.02}


def test_find_replacement_picks_an_eligible_leg_from_a_new_player():
    from backend.parlay.search import find_replacement

    a = _leg("p1", "points", 19.5, 21, 5, game_id="g1")
    b = _leg("p2", "points", 19.5, 21, 5, game_id="g2")
    same_player_as_a = _leg("p1", "points", 19.5, 21, 5, side="under", game_id="g1")
    too_weak = _leg("p3", "points", 30.5, 20, 5, game_id="g3")
    good = _leg("p4", "points", 18.5, 21, 5, game_id="g4")
    pool = [a, b, same_player_as_a, too_weak, good]
    c, why = find_replacement([a, b], b, pool, "test", _two_leg_tier(), _zero_corr_matrices(), 500, n_simulations=1000, seed=1)
    assert why is None
    assert [l.player_id_nba for l in c.legs] == ["p1", "p4"]


def test_find_replacement_says_why_when_nothing_fits():
    from backend.parlay.search import find_replacement

    a = _leg("p1", "points", 19.5, 21, 5, game_id="g1")
    b = _leg("p2", "points", 19.5, 21, 5, game_id="g2")
    too_weak = _leg("p3", "points", 30.5, 20, 5, game_id="g3")
    c, why = find_replacement([a, b], b, [a, b, too_weak], "test", _two_leg_tier(), _zero_corr_matrices(), 500, n_simulations=500, seed=1)
    assert c is None
    assert "clears 50%" in why
