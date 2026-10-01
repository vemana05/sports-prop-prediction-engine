"""Bankroll & stake recommendation (spec Section 8). Pure math, no I/O —
takes a probability and decimal odds, returns a stake. Works the same for
a single leg or a whole parlay; the caller decides what `p` and `d` mean
(for a parlay, `p` is the Monte Carlo-simulated, market-shrunk hit
probability and `d` is the product of each leg's own decimal odds).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass


def kelly_fraction() -> float:
    return float(os.environ.get("KELLY_FRACTION", "0.25"))


@dataclass
class StakeRecommendation:
    model_p: float
    decimal_odds: float
    ev_per_dollar: float
    kelly_f_star: float
    stake: float
    capped_by_tier: bool
    reason: str | None = None  # set when stake is $0 and the user should know why


def ev_per_dollar(p: float, d: float) -> float:
    """Section 8 step 1: EV per $1 = p*(d-1) - (1-p)."""
    return p * (d - 1) - (1 - p)


def kelly_f_star(p: float, d: float) -> float:
    """Section 8 step 2: Kelly f* = (p*d - 1) / (d - 1)."""
    if d <= 1:
        return 0.0
    return (p * d - 1) / (d - 1)


def recommend_stake(
    p: float,
    decimal_odds: float,
    bankroll: float,
    tier_cap_pct: float,
    book_minimum: float = 0.0,
    round_to: float = 0.50,
) -> StakeRecommendation:
    """`p` should already be market-shrunk (Section 8 step 3) — this
    function doesn't shrink, it just sizes. Steps 4-7 of Section 8:
    quarter Kelly, capped by the tier's max stake, EV<=0 -> $0, rounded,
    never below the book's minimum (if given)."""
    ev = ev_per_dollar(p, decimal_odds)
    f_star = kelly_f_star(p, decimal_odds)

    if ev <= 0 or f_star <= 0:
        return StakeRecommendation(
            model_p=p, decimal_odds=decimal_odds, ev_per_dollar=ev, kelly_f_star=f_star,
            stake=0.0, capped_by_tier=False,
            reason="The model doesn't see an edge on this parlay. If you play it anyway, treat it as entertainment money.",
        )

    raw_stake = bankroll * f_star * kelly_fraction()
    tier_cap_stake = bankroll * tier_cap_pct
    capped = raw_stake > tier_cap_stake
    stake = min(raw_stake, tier_cap_stake)

    stake = math.floor(stake / round_to) * round_to
    if book_minimum and 0 < stake < book_minimum:
        stake = 0.0  # can't actually place a bet this small

    return StakeRecommendation(
        model_p=p, decimal_odds=decimal_odds, ev_per_dollar=ev, kelly_f_star=f_star,
        stake=stake, capped_by_tier=capped,
    )
