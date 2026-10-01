"""Beam search parlay builder (spec Section 7).

1. Filter the leg pool to the tier's `min_leg_prob` (using each leg's
   market-shrunk probability, not the raw model one — Section 8).
2. Beam search (width `beam_width`, default ~200) ranking candidates by a
   FAST, independence-assumed EV estimate — running the real Monte Carlo
   simulator on every candidate at every step would be far too slow, and
   the spec's own phrasing ("beam search maximizing EV ... simulate
   finalists") implies exactly this two-speed approach.
3. Only the surviving finalists (a handful per leg-count) get the real,
   correlation-aware Monte Carlo treatment — that's what decides the
   reported probability, EV, and stake.
4. Keep finalists whose simulated parlay probability falls in the tier's
   [target_parlay_prob_min, target_parlay_prob_max], ranked by true EV.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.bankroll.kelly import StakeRecommendation, ev_per_dollar, recommend_stake
from backend.parlay.monte_carlo import DEFAULT_N_SIMULATIONS, Leg, simulate_parlay_hit_probability

DEFAULT_BEAM_WIDTH = 200
DEFAULT_FINALISTS_PER_SIZE = 15
DEFAULT_RESULTS_PER_TIER = 5


@dataclass
class ParlayCandidate:
    legs: list[Leg]
    p_hit: float  # Monte Carlo-simulated, correlation-aware
    independent_p_hit: float  # what it'd be if legs were independent — shown so correlation's effect is visible
    combined_decimal_odds: float
    stake: StakeRecommendation


def _combined_decimal_odds(legs: list[Leg]) -> float:
    d = 1.0
    for leg in legs:
        d *= leg.decimal_odds
    return d


def _naive_independent_p(legs: list[Leg]) -> float:
    p = 1.0
    for leg in legs:
        p *= leg.shrunk_p_hit()
    return p


def _is_duplicate_or_contradiction(leg: Leg, chosen: list[Leg]) -> bool:
    """Section 7 step 2: never two legs on the same (player, market) in one
    parlay — same side is a pointless duplicate, opposite side is a
    guaranteed-loss contradiction (can't win both an over and an under on
    the same prop)."""
    return any(leg.player_id_nba == c.player_id_nba and leg.market == c.market for c in chosen)


def _beam_search_by_naive_ev(
    eligible_legs: list[Leg], legs_min: int, legs_max: int, beam_width: int, finalists_per_size: int
) -> list[list[Leg]]:
    beam: list[tuple[list[Leg], float]] = [
        ([leg], ev_per_dollar(leg.shrunk_p_hit(), leg.decimal_odds)) for leg in eligible_legs
    ]
    beam.sort(key=lambda x: x[1], reverse=True)
    beam = beam[:beam_width]

    finalists: list[list[Leg]] = []
    for size in range(1, legs_max + 1):
        if size >= legs_min:
            top_this_size = sorted(beam, key=lambda x: x[1], reverse=True)[:finalists_per_size]
            finalists.extend(legs for legs, _ in top_this_size)
        if size == legs_max:
            break

        new_beam: list[tuple[list[Leg], float]] = []
        for chosen, _ in beam:
            for candidate in eligible_legs:
                if _is_duplicate_or_contradiction(candidate, chosen):
                    continue
                new_legs = chosen + [candidate]
                new_p = _naive_independent_p(new_legs)
                new_d = _combined_decimal_odds(new_legs)
                new_beam.append((new_legs, ev_per_dollar(new_p, new_d)))
        new_beam.sort(key=lambda x: x[1], reverse=True)
        beam = new_beam[:beam_width]

    seen = set()
    unique: list[list[Leg]] = []
    for legs in finalists:
        key = frozenset((l.player_id_nba, l.market, l.side, l.line) for l in legs)
        if key in seen:
            continue
        seen.add(key)
        unique.append(legs)
    return unique


def search_tier(
    all_legs: list[Leg],
    tier_name: str,
    tier_config: dict,
    corr_matrices: dict,
    bankroll: float,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    finalists_per_size: int = DEFAULT_FINALISTS_PER_SIZE,
    n_results: int = DEFAULT_RESULTS_PER_TIER,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    seed: int | None = None,
) -> list[ParlayCandidate]:
    eligible = [leg for leg in all_legs if leg.shrunk_p_hit() >= tier_config["min_leg_prob"]]
    if not eligible:
        return []

    finalist_leg_sets = _beam_search_by_naive_ev(
        eligible, tier_config["legs_min"], tier_config["legs_max"], beam_width, finalists_per_size
    )

    candidates: list[ParlayCandidate] = []
    for legs in finalist_leg_sets:
        sim = simulate_parlay_hit_probability(legs, corr_matrices, n_simulations=n_simulations, seed=seed)
        p_hit = sim["p_hit"]
        if not (tier_config["target_parlay_prob_min"] <= p_hit <= tier_config["target_parlay_prob_max"]):
            continue
        d = _combined_decimal_odds(legs)
        stake = recommend_stake(p_hit, d, bankroll, tier_config["max_stake_pct_bankroll"])
        candidates.append(
            ParlayCandidate(legs=legs, p_hit=p_hit, independent_p_hit=sim["independent_p_hit"], combined_decimal_odds=d, stake=stake)
        )

    candidates.sort(key=lambda c: c.stake.ev_per_dollar, reverse=True)
    return candidates[:n_results]
