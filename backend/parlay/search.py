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
5. Size the stake (Section 8 step 3) on the simulated probability shrunk
   toward the book's no-vig parlay probability (the product of each leg's
   no-vig price) by MODEL_SHRINK_TO_MARKET — never on the raw simulated
   probability. EV and ranking use that same shrunk probability.

Sanity guard: a model probability this far from the book's own no-vig
price, or a parlay EV this large, is far more likely to be a feature/data
bug than a real edge (the Oct 20 opening-night slate's investigation found
exactly this — e.g. a training-serving skew in build_current_form that
made every leg look like a lopsided "under") — `_sanity_flags` below
catches both patterns and zeroes the stake with a "check data" reason
instead of silently recommending it.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.bankroll.kelly import StakeRecommendation, ev_per_dollar, model_shrink_to_market, recommend_stake, shrink_to_market
from backend.parlay.monte_carlo import DEFAULT_N_SIMULATIONS, Leg, simulate_parlay_hit_probability

DEFAULT_BEAM_WIDTH = 200
DEFAULT_FINALISTS_PER_SIZE = 15
DEFAULT_RESULTS_PER_TIER = 5

LEG_MODEL_VS_MARKET_DIVERGENCE_LIMIT = 0.15  # probability points
PARLAY_EV_SANITY_LIMIT = 0.30  # per $1
# Stricter limits until every team has played EARLY_SEASON_GAMES
# (backend/predict/early_season.py): features still lean on last season and
# star projections run low (spec 5.9), so smaller gaps are already suspect.
EARLY_LEG_MODEL_VS_MARKET_DIVERGENCE_LIMIT = 0.10
EARLY_PARLAY_EV_SANITY_LIMIT = 0.20


SIDES = ("both", "over", "under")
REPLACEMENT_SHORTLIST = 25  # replacement candidates that get the full Monte Carlo, by naive EV


@dataclass
class ParlayCandidate:
    legs: list[Leg]
    p_hit: float  # Monte Carlo-simulated, correlation-aware, calibrated model probability
    independent_p_hit: float  # what it'd be if legs were independent — shown so correlation's effect is visible
    combined_decimal_odds: float
    stake: StakeRecommendation  # sized on `p_sized`, not `p_hit`
    flagged: bool = False
    flag_reasons: list[str] = None
    no_vig_p: float | None = None  # the book's own no-vig parlay probability
    p_sized: float | None = None  # p_hit shrunk toward no_vig_p (Section 8 step 3) — what EV and stake use
    model_ev_per_dollar: float | None = None  # EV on the unshrunk p_hit, used only by the sanity guard

    def __post_init__(self):
        if self.flag_reasons is None:
            self.flag_reasons = []
        if self.p_sized is None:
            self.p_sized = self.p_hit


@dataclass
class SearchReport:
    """Why a tier came back the way it did, with the numbers behind it —
    shown in place of a blank result list."""
    n_pool: int = 0  # legs left after side/veto/game filters
    n_eligible: int = 0  # ... that clear the tier's min_leg_prob
    n_eligible_players: int = 0
    best_leg_p: float | None = None  # best shrunk leg probability in the filtered pool
    n_combinations: int = 0  # finalists that got the full Monte Carlo
    combo_p_min: float | None = None
    combo_p_max: float | None = None
    n_in_range: int = 0
    reason: str | None = None


def leg_id(leg: Leg) -> str:
    return f"{leg.player_id_nba}-{leg.market}-{leg.side}-{leg.line}"


def filter_pool(
    legs: list[Leg], sides: str = "both", vetoed: set[str] | frozenset = frozenset(),
    excluded_games: set[str] | frozenset = frozenset(),
) -> list[Leg]:
    """The leg pool a search may draw from: only the chosen side(s), no
    vetoed legs, no legs from games the user switched off."""
    if sides not in SIDES:
        raise ValueError(f"sides must be one of {SIDES}, got {sides!r}")
    return [
        leg for leg in legs
        if (sides == "both" or leg.side == sides) and leg_id(leg) not in vetoed and leg.game_id not in excluded_games
    ]


def parlay_no_vig_p(legs: list[Leg]) -> float | None:
    """The book's no-vig probability for the whole parlay: the product of
    each leg's no-vig probability (the book prices a parlay as if its legs
    were independent). A leg without both sides quoted has no market price
    to shrink toward, so it contributes its own model probability — i.e.
    that leg isn't shrunk. None if no leg has a market price at all."""
    priced = [leg.no_vig_p_hit() for leg in legs]
    if all(p is None for p in priced):
        return None
    out = 1.0
    for leg, p in zip(legs, priced):
        out *= leg.p_hit() if p is None else p
    return out


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


def _sanity_flags(legs: list[Leg], model_ev: float, early_season: bool = False) -> list[str]:
    """Reasons to distrust this parlay's numbers rather than recommend it:
    any leg whose (calibrated) model probability and the book's own
    no-vig probability disagree by more than
    `LEG_MODEL_VS_MARKET_DIVERGENCE_LIMIT`, or a parlay EV (on the
    unshrunk model probability) above `PARLAY_EV_SANITY_LIMIT` per $1. A
    real, exploitable edge this big against a liquid market is possible
    but rare; a feature/data bug upstream of the model is far more common —
    see this module's docstring. Checked on the unshrunk numbers so the
    shrink can't hide a bug the guard exists to catch. `early_season`
    switches to the stricter EARLY_* limits."""
    leg_limit = EARLY_LEG_MODEL_VS_MARKET_DIVERGENCE_LIMIT if early_season else LEG_MODEL_VS_MARKET_DIVERGENCE_LIMIT
    ev_limit = EARLY_PARLAY_EV_SANITY_LIMIT if early_season else PARLAY_EV_SANITY_LIMIT
    note = " (early-season limit)" if early_season else ""
    reasons = []
    for leg in legs:
        no_vig = leg.no_vig_p_hit()
        if no_vig is None:
            continue
        divergence = abs(leg.p_hit() - no_vig)
        if divergence > leg_limit:
            reasons.append(
                f"{leg.player_name} {leg.market} {leg.side} {leg.line}: model {leg.p_hit():.2f} vs. "
                f"no-vig {no_vig:.2f} ({divergence * 100:.0f} points apart){note}"
            )
    if model_ev > ev_limit:
        reasons.append(f"parlay EV {model_ev:+.2f} per $1 is implausibly large{note}")
    return reasons


def score_parlay(
    legs: list[Leg],
    tier_config: dict,
    corr_matrices: dict,
    bankroll: float,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    seed: int | None = None,
    early_season: bool = False,
) -> ParlayCandidate:
    """The one scoring path for any leg set — suggested, swapped-in after a
    veto, or hand-picked in manual mode: correlation-aware Monte Carlo hit
    probability, shrink toward the book's no-vig parlay probability, then
    quarter-Kelly capped at the tier's max stake. The sanity guard zeroes
    the stake of anything that trips it."""
    sim = simulate_parlay_hit_probability(legs, corr_matrices, n_simulations=n_simulations, seed=seed)
    p_hit = sim["p_hit"]
    d = _combined_decimal_odds(legs)
    no_vig = parlay_no_vig_p(legs)
    p_sized = shrink_to_market(p_hit, no_vig)
    stake = recommend_stake(p_sized, d, bankroll, tier_config["max_stake_pct_bankroll"])
    model_ev = ev_per_dollar(p_hit, d)

    flag_reasons = _sanity_flags(legs, model_ev, early_season)
    if flag_reasons:
        stake = StakeRecommendation(
            model_p=stake.model_p, decimal_odds=stake.decimal_odds, ev_per_dollar=stake.ev_per_dollar,
            kelly_f_star=stake.kelly_f_star, stake=0.0, capped_by_tier=False,
            reason="CHECK DATA (not recommended) — " + "; ".join(flag_reasons),
        )
    return ParlayCandidate(
        legs=legs, p_hit=p_hit, independent_p_hit=sim["independent_p_hit"], combined_decimal_odds=d,
        stake=stake, flagged=bool(flag_reasons), flag_reasons=flag_reasons,
        no_vig_p=no_vig, p_sized=p_sized, model_ev_per_dollar=model_ev,
    )


def in_tier_range(p_hit: float, tier_config: dict) -> bool:
    return tier_config["target_parlay_prob_min"] <= p_hit <= tier_config["target_parlay_prob_max"]


def _rank_key(c: ParlayCandidate):
    # Flagged candidates sort after every clean one, regardless of their
    # (likely bogus) EV — they're only kept at all so a caller can see
    # *why* something was suppressed, not to let a suspiciously huge EV
    # win the ranking it's being flagged for.
    return (c.flagged, -c.stake.ev_per_dollar)


def _pct(p: float) -> str:
    return f"{p * 100:.0f}%" if p >= 0.1 or p == 0 else f"{p * 100:.1f}%"


def _band(tier_config: dict) -> str:
    lo, hi = tier_config["target_parlay_prob_min"], tier_config["target_parlay_prob_max"]
    if hi >= 1:
        return f"at or above {_pct(lo)}"
    if lo <= 0:
        return f"under {_pct(hi)}"
    return f"between {_pct(lo)} and {_pct(hi)}"


def _side_phrase(sides: str) -> str:
    return {"both": "", "over": " over", "under": " under"}[sides]


def run_search(
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
    sides: str = "both",
    early_season: bool = False,
) -> tuple[list[ParlayCandidate], SearchReport]:
    """`all_legs` should already be filtered (`filter_pool`); `sides` is
    only used to word the report."""
    report = SearchReport(n_pool=len(all_legs))
    legs_min, legs_max, min_p = tier_config["legs_min"], tier_config["legs_max"], tier_config["min_leg_prob"]
    size = f"{legs_min}-{legs_max} leg" if legs_min != legs_max else f"{legs_min}-leg"
    side = _side_phrase(sides)

    if not all_legs:
        report.reason = f"No{side} legs left in the current pool."
        return [], report

    report.best_leg_p = max(leg.shrunk_p_hit() for leg in all_legs)
    eligible = [leg for leg in all_legs if leg.shrunk_p_hit() >= min_p]
    report.n_eligible = len(eligible)
    report.n_eligible_players = len({(leg.player_id_nba, leg.market) for leg in eligible})
    if not eligible:
        report.reason = (
            f"No{side} legs clear {_pct(min_p)} in the current pool — the best of {len(all_legs)} is "
            f"{_pct(report.best_leg_p)} (after the {model_shrink_to_market():.0%} shrink toward the book)."
        )
        return [], report
    if report.n_eligible_players < legs_min:
        report.reason = (
            f"Only {report.n_eligible_players} player prop{'s' if report.n_eligible_players != 1 else ''} "
            f"clear{'s' if report.n_eligible_players == 1 else ''} {_pct(min_p)} ({report.n_eligible} of "
            f"{len(all_legs)}{side} legs); {tier_name.capitalize()} needs at least {legs_min} different ones."
        )
        return [], report

    finalist_leg_sets = _beam_search_by_naive_ev(eligible, legs_min, legs_max, beam_width, finalists_per_size)
    scored = [score_parlay(legs, tier_config, corr_matrices, bankroll, n_simulations, seed, early_season) for legs in finalist_leg_sets]
    report.n_combinations = len(scored)
    if scored:
        report.combo_p_min = min(c.p_hit for c in scored)
        report.combo_p_max = max(c.p_hit for c in scored)
    candidates = [c for c in scored if in_tier_range(c.p_hit, tier_config)]
    report.n_in_range = len(candidates)
    if not candidates:
        if not scored:
            spread = "none could be formed"
        elif len(scored) == 1:
            spread = f"the only one that can be formed simulates to {_pct(report.combo_p_min)}"
        else:
            spread = f"the {len(scored)} best by EV simulate to {_pct(report.combo_p_min)}–{_pct(report.combo_p_max)}"
        report.reason = f"No {size} combinations fall {_band(tier_config)} hit chance — {spread}."
        return [], report

    candidates.sort(key=_rank_key)
    return candidates[:n_results], report


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
    return run_search(
        all_legs, tier_name, tier_config, corr_matrices, bankroll,
        beam_width, finalists_per_size, n_results, n_simulations, seed,
    )[0]


def find_replacement(
    parlay_legs: list[Leg],
    vetoed_leg: Leg,
    pool: list[Leg],
    tier_name: str,
    tier_config: dict,
    corr_matrices: dict,
    bankroll: float,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    seed: int | None = None,
    sides: str = "both",
    early_season: bool = False,
) -> tuple[ParlayCandidate | None, str | None]:
    """Swap `vetoed_leg` out of `parlay_legs` for the best leg in `pool`
    (already filtered for side, vetoes and games) that clears the tier's
    min_leg_prob, isn't another leg on a player+market already in the
    parlay, and keeps the parlay's simulated hit chance inside the tier's
    band. Best = highest EV on the shrunk probability, clean before
    flagged. Returns (parlay, None) or (None, why-not)."""
    vetoed_key = leg_id(vetoed_leg)
    remaining = [leg for leg in parlay_legs if leg_id(leg) != vetoed_key]
    in_parlay = {leg_id(leg) for leg in remaining}
    min_p = tier_config["min_leg_prob"]
    side = _side_phrase(sides)

    fits = [
        leg for leg in pool
        if leg_id(leg) not in in_parlay and leg_id(leg) != vetoed_key
        and leg.shrunk_p_hit() >= min_p and not _is_duplicate_or_contradiction(leg, remaining)
    ]
    if not fits:
        return None, (
            f"No other{side} leg clears {_pct(min_p)} without repeating a player already in this parlay."
        )

    # Rank by the cheap independence estimate first; only the shortlist gets
    # the real Monte Carlo.
    fits.sort(key=lambda leg: ev_per_dollar(_naive_independent_p(remaining + [leg]), _combined_decimal_odds(remaining + [leg])), reverse=True)
    scored = [score_parlay(remaining + [leg], tier_config, corr_matrices, bankroll, n_simulations, seed, early_season) for leg in fits[:REPLACEMENT_SHORTLIST]]
    in_band = [c for c in scored if in_tier_range(c.p_hit, tier_config)]
    if not in_band:
        lo, hi = min(c.p_hit for c in scored), max(c.p_hit for c in scored)
        return None, (
            f"{len(fits)}{side} leg{'s' if len(fits) != 1 else ''} could fill the spot, but every swap puts the "
            f"hit chance at {_pct(lo)}–{_pct(hi)}, outside {tier_name.capitalize()}'s {_band(tier_config)}."
        )
    in_band.sort(key=_rank_key)
    return in_band[0], None
