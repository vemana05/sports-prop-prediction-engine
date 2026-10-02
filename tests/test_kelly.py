"""Unit tests for the Kelly sizing math (spec Section 8) — pure functions,
no DB needed."""

import os

import pytest

from backend.bankroll.kelly import ev_per_dollar, kelly_f_star, recommend_stake


def test_ev_per_dollar_breakeven_at_fair_coinflip_even_money():
    # p=0.5, d=2.0 (even money) -> EV = 0.5*(2-1) - 0.5 = 0
    assert ev_per_dollar(0.5, 2.0) == pytest.approx(0.0)


def test_ev_per_dollar_positive_when_model_beats_the_price():
    # p=0.6 at even money (d=2.0): EV = 0.6*1 - 0.4 = 0.2
    assert ev_per_dollar(0.6, 2.0) == pytest.approx(0.2)


def test_ev_per_dollar_negative_when_model_is_worse_than_the_price():
    assert ev_per_dollar(0.4, 2.0) == pytest.approx(-0.2)


def test_kelly_f_star_matches_known_formula():
    # p=0.6, d=2.0 -> f* = (0.6*2 - 1) / (2 - 1) = 0.2
    assert kelly_f_star(0.6, 2.0) == pytest.approx(0.2)


def test_kelly_f_star_zero_at_breakeven():
    assert kelly_f_star(0.5, 2.0) == pytest.approx(0.0)


def test_kelly_f_star_handles_d_equal_to_one_without_dividing_by_zero():
    assert kelly_f_star(0.9, 1.0) == 0.0


def test_negative_ev_gives_zero_stake_with_a_reason():
    rec = recommend_stake(p=0.3, decimal_odds=2.0, bankroll=500, tier_cap_pct=0.02)
    assert rec.stake == 0.0
    assert rec.reason is not None
    assert "edge" in rec.reason.lower()


def test_positive_ev_gives_a_positive_quarter_kelly_stake():
    os.environ["KELLY_FRACTION"] = "0.25"
    # p=0.6, d=2.0 -> f*=0.2, quarter Kelly = 0.05 of bankroll
    rec = recommend_stake(p=0.6, decimal_odds=2.0, bankroll=500, tier_cap_pct=0.02)
    # tier cap (2% = $10) binds before quarter-Kelly ($25) does
    assert rec.stake > 0
    assert rec.capped_by_tier is True
    assert rec.stake == pytest.approx(10.0)


def test_stake_capped_by_tier_max_not_by_kelly_when_kelly_is_smaller():
    # small edge -> quarter-Kelly stake well under the tier cap
    rec = recommend_stake(p=0.52, decimal_odds=2.0, bankroll=1000, tier_cap_pct=0.02)
    quarter_kelly_stake = 1000 * kelly_f_star(0.52, 2.0) * 0.25
    assert rec.capped_by_tier is False
    assert rec.stake == pytest.approx(round(quarter_kelly_stake * 2) / 2, abs=0.5)


def test_stake_is_rounded_to_50_cents():
    rec = recommend_stake(p=0.65, decimal_odds=2.2, bankroll=777, tier_cap_pct=0.02)
    assert rec.stake >= 0
    assert (rec.stake * 2) == pytest.approx(round(rec.stake * 2)), "stake should be a multiple of $0.50"


def test_stake_below_book_minimum_becomes_zero():
    # a tiny bankroll + tight tier cap produces a stake under a $5 minimum
    rec = recommend_stake(p=0.6, decimal_odds=2.0, bankroll=10, tier_cap_pct=0.02, book_minimum=5.0)
    assert rec.stake == 0.0


def test_zero_bankroll_gives_zero_stake_not_an_error():
    rec = recommend_stake(p=0.7, decimal_odds=2.0, bankroll=0, tier_cap_pct=0.02)
    assert rec.stake == 0.0


def test_shrink_to_market_moves_thirty_percent_toward_the_no_vig_price():
    # 0.70 model, 0.50 market, default weight 0.30 -> 0.70*0.7 + 0.50*0.3 = 0.64
    from backend.bankroll.kelly import shrink_to_market

    assert shrink_to_market(0.70, 0.50) == pytest.approx(0.64)


def test_shrink_to_market_without_a_market_price_leaves_the_model_alone():
    from backend.bankroll.kelly import shrink_to_market

    assert shrink_to_market(0.70, None) == 0.70
