"""NFL parlay search and scoring without odds (spec Section 7, adapted).

NBA's beam search ranks by EV, which needs a price on every leg. NFL legs
have none until the user types one, so the search instead aims each tier
at a target hit chance inside its band (config/tiers.yaml):

    target = geometric middle of [target_parlay_prob_min, target_parlay_prob_max]
             (Safe: sqrt(0.35 * 1.0) ~ 0.59; Dangerous: half its 12% cap)

and, for each parlay size L in [legs_min, legs_max], runs a beam whose
partial parlays track the target path target**(s/L) under the
independence assumption. Finalists get the real correlation-aware Monte
Carlo (monte_carlo.py, same as NBA); those inside the tier's band are
kept, closest to target first, skipping near-duplicates.

Scoring (`score`) is the NBA path with prices made optional: Monte Carlo
hit chance always; combined odds, EV and a quarter-Kelly stake (tier cap,
30% shrink toward the book's no-vig price, sanity guard) only once every
leg has a typed price.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np

from backend.bankroll.kelly import StakeRecommendation, ev_per_dollar, recommend_stake, shrink_to_market
from backend.parlay.monte_carlo import DEFAULT_N_SIMULATIONS, simulate_parlay_hit_probability
from backend.parlay.nfl_legs import NflLeg, rung_line
from backend.parlay.search import SearchReport, _band, _pct, _sanity_flags, in_tier_range, leg_id, parlay_no_vig_p

BEAM_WIDTH = 150
FINALISTS_PER_SIZE = 12
RESULTS_PER_TIER = 5
MAX_SHARED_LEGS = 0.5  # a result may share at most half its legs with a better one
REPLACEMENT_SHORTLIST = 25
SCREEN_SIMULATIONS = 6_000  # finalists are screened at this count; results are re-scored at the full count

PASS_HEAVY = {"passing_yards"}
CATCH_MARKETS = {"receiving_yards", "receptions"}


@dataclass
class NflParlay:
    legs: list[NflLeg]
    p_hit: float
    independent_p_hit: float
    priced: bool = False
    decimal_odds: float | None = None
    no_vig_p: float | None = None
    p_sized: float | None = None
    stake: StakeRecommendation | None = None
    model_ev_per_dollar: float | None = None
    flagged: bool = False
    flag_reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def target_p(tier_cfg: dict) -> float:
    lo, hi = tier_cfg["target_parlay_prob_min"], min(tier_cfg["target_parlay_prob_max"], 1.0)
    return math.sqrt(lo * hi) if lo > 0 else hi / 2


# ---- notes -----------------------------------------------------------------------

def _script_lean(l: NflLeg) -> str | None:
    """Which game script a leg needs: "pass" (passing yards or WR/TE catches
    over, RB rushing under) or "run" (the reverse). None if neither."""
    passing = l.market in PASS_HEAVY or (l.market in CATCH_MARKETS and l.position in ("WR", "TE"))
    running = l.market == "rushing_yards" and l.position == "RB"
    if not (passing or running):
        return None
    pass_lean = passing == (l.side == "over")
    return "pass" if pass_lean else "run"


def game_script_notes(legs: list[NflLeg], games: dict) -> list[str]:
    """A same-game mix of legs that need a pass-heavy script (passing yards,
    WR/TE catches over; RB rushing under) and legs that need a run-heavy one
    (the reverse) depends on how the game unfolds."""
    notes = []
    for gid in dict.fromkeys(l.game_id for l in legs):
        leans = {_script_lean(l) for l in legs if l.game_id == gid}
        if {"pass", "run"} <= leans:
            g = games.get(gid, {})
            matchup = f"{g.get('away', '?')} @ {g.get('home', '?')}"
            notes.append(
                f"Game script: {matchup} mixes pass-heavy and run-heavy legs. A team that leads early tends to run more "
                f"and pass less (the trailing team the reverse), so these legs can pull against each other. The "
                f"correlation model only captures this on average."
            )
    return notes


def _warnings(legs: list[NflLeg]) -> list[str]:
    return list(dict.fromkeys(w for l in legs for w in l.warnings))


# ---- scoring ----------------------------------------------------------------------

def score(legs: list[NflLeg], tier_cfg: dict, corr: dict, bankroll: float, games: dict,
          n_simulations: int = DEFAULT_N_SIMULATIONS, seed: int | None = None) -> NflParlay:
    sim = simulate_parlay_hit_probability(legs, corr, n_simulations=n_simulations, seed=seed)
    out = NflParlay(legs=legs, p_hit=sim["p_hit"], independent_p_hit=sim["independent_p_hit"],
                    notes=game_script_notes(legs, games), warnings=_warnings(legs))
    if not all(l.priced for l in legs):
        return out

    d = math.prod(l.decimal_odds for l in legs)
    no_vig = parlay_no_vig_p(legs)
    p_sized = shrink_to_market(out.p_hit, no_vig)
    stake = recommend_stake(p_sized, d, bankroll, tier_cfg["max_stake_pct_bankroll"])
    model_ev = ev_per_dollar(out.p_hit, d)
    flags = _sanity_flags(legs, model_ev)
    if flags:
        stake = StakeRecommendation(
            model_p=stake.model_p, decimal_odds=d, ev_per_dollar=stake.ev_per_dollar, kelly_f_star=stake.kelly_f_star,
            stake=0.0, capped_by_tier=False, reason="CHECK DATA (not recommended) — " + "; ".join(flags),
        )
    return dataclasses.replace(out, priced=True, decimal_odds=d, no_vig_p=no_vig, p_sized=p_sized, stake=stake,
                               model_ev_per_dollar=model_ev, flagged=bool(flags), flag_reasons=flags)


def american_to_decimal(a: float) -> float:
    if -100 < a < 100:
        raise ValueError("American odds must be at least +100 or at most -100")
    return 1 + (a / 100 if a > 0 else 100 / -a)


def parse_line(text: str | float) -> float:
    """A typed book line: "54.5" stays 54.5; "55" or "55+" means 55 or
    more, i.e. over 54.5."""
    s = str(text).strip().rstrip("+")
    v = float(s)
    return v - 0.5 if v == int(v) else v


def apply_price(leg: NflLeg, line: str | float | None, over: float | None, under: float | None) -> NflLeg:
    """The pool leg with a user-typed book line and/or American prices.
    The leg's own side's price (over, or under for an under leg) prices it;
    the other side's is optional and only removes the vig."""
    new_line = parse_line(line) if line not in (None, "") else leg.line
    over_d = american_to_decimal(over) if over is not None else None
    under_d = american_to_decimal(under) if under is not None else None
    own = over_d if leg.side == "over" else under_d
    return dataclasses.replace(
        leg, line=new_line, rung=leg.rung if new_line == leg.line else None,
        over_price=over_d, under_price=under_d, decimal_odds=own or 0.0,
        base_id=leg.base_id or leg_id(leg),
    )


# ---- search -----------------------------------------------------------------------

def _beam(p: np.ndarray, key: np.ndarray, size: int, target: float, width: int) -> list[tuple[int, ...]]:
    """Index tuples of length `size` whose independent hit chance tracks
    target**(s/size). Legs are taken in index order (p descending), which
    makes every combination unique, and never two legs on one player+market."""
    log_p = np.log(p)
    log_t = math.log(target)
    order = np.argsort(np.abs(log_p - log_t / size))[:width]
    beam = [((int(i),), log_p[i]) for i in order]
    for s in range(2, size + 1):
        cand_idx, cand_lp, parents = [], [], []
        for b, (chosen, lp) in enumerate(beam):
            start = chosen[-1] + 1
            if start >= len(p):
                continue
            idx = np.arange(start, len(p))
            idx = idx[~np.isin(key[idx], key[list(chosen)])]
            cand_idx.append(idx)
            cand_lp.append(lp + log_p[idx])
            parents.append(np.full(len(idx), b))
        if not cand_idx:
            return []
        cand_idx, cand_lp, parents = map(np.concatenate, (cand_idx, cand_lp, parents))
        dist = np.abs(cand_lp - log_t * s / size)
        top = np.argsort(dist)[:width]
        beam = [(beam[parents[t]][0] + (int(cand_idx[t]),), cand_lp[t]) for t in top]
    return [chosen for chosen, _ in beam]


def search(pool: list[NflLeg], tier_name: str, tier_cfg: dict, corr: dict, bankroll: float, games: dict,
           seed: int | None = None, n_results: int = RESULTS_PER_TIER) -> tuple[list[NflParlay], SearchReport]:
    report = SearchReport(n_pool=len(pool))
    legs_min, legs_max, min_p = tier_cfg["legs_min"], tier_cfg["legs_max"], tier_cfg["min_leg_prob"]
    if not pool:
        report.reason = "No legs left in this week's pool."
        return [], report
    report.best_leg_p = max(l.p_hit() for l in pool)
    eligible = sorted((l for l in pool if l.p_hit() >= min_p), key=lambda l: -l.p_hit())
    report.n_eligible = len(eligible)
    report.n_eligible_players = len({(l.player_id_nba, l.market) for l in eligible})
    if report.n_eligible_players < legs_min:
        report.reason = (f"Only {report.n_eligible_players} player props clear {_pct(min_p)}; "
                         f"{tier_name.capitalize()} needs at least {legs_min} different ones.")
        return [], report

    p = np.array([l.p_hit() for l in eligible])
    keys = {k: i for i, k in enumerate(dict.fromkeys((l.player_id_nba, l.market) for l in eligible))}
    key = np.array([keys[(l.player_id_nba, l.market)] for l in eligible])
    target = target_p(tier_cfg)

    finalists: list[tuple[int, ...]] = []
    for size in range(legs_min, legs_max + 1):
        finalists += _beam(p, key, size, target, BEAM_WIDTH)[:FINALISTS_PER_SIZE]
    scored = [score([eligible[i] for i in combo], tier_cfg, corr, bankroll, games, SCREEN_SIMULATIONS, seed) for combo in finalists]
    report.n_combinations = len(scored)
    if scored:
        report.combo_p_min, report.combo_p_max = min(c.p_hit for c in scored), max(c.p_hit for c in scored)
    in_band = [c for c in scored if in_tier_range(c.p_hit, tier_cfg)]
    report.n_in_range = len(in_band)
    if not in_band:
        report.reason = (f"No {legs_min}-{legs_max} leg combinations fall {_band(tier_cfg)} hit chance — "
                         f"the {len(scored)} closest simulate to {_pct(report.combo_p_min or 0)}–{_pct(report.combo_p_max or 0)}.")
        return [], report

    in_band.sort(key=lambda c: abs(math.log(max(c.p_hit, 1e-9)) - math.log(target)))
    results: list[NflParlay] = []
    for c in in_band:
        ids = {leg_id(l) for l in c.legs}
        if any(len(ids & {leg_id(l) for l in r.legs}) > MAX_SHARED_LEGS * len(ids) for r in results):
            continue
        results.append(c)
        if len(results) == n_results:
            break
    return [score(c.legs, tier_cfg, corr, bankroll, games, seed=seed) for c in results], report


def find_replacement(current: list[NflLeg], vetoed: NflLeg, pool: list[NflLeg], tier_name: str, tier_cfg: dict,
                     corr: dict, bankroll: float, games: dict, seed: int | None = None) -> tuple[NflParlay | None, str | None]:
    """Swap the vetoed leg for the pool leg that keeps the parlay closest to
    the tier's target hit chance (inside its band), never repeating a
    player+market already in the parlay."""
    vid = leg_id(vetoed)
    remaining = [l for l in current if leg_id(l) != vid and (l.base_id or leg_id(l)) != vid]
    taken = {(l.player_id_nba, l.market) for l in remaining}
    fits = [l for l in pool if l.p_hit() >= tier_cfg["min_leg_prob"] and (l.player_id_nba, l.market) not in taken
            and (l.player_id_nba, l.market) != (vetoed.player_id_nba, vetoed.market)]
    if not fits:
        return None, f"No other leg clears {_pct(tier_cfg['min_leg_prob'])} without repeating a player prop already in this parlay."
    target = target_p(tier_cfg)
    base = math.prod(l.p_hit() for l in remaining)
    fits.sort(key=lambda l: abs(math.log(base * l.p_hit()) - math.log(target)))
    scored = [score(remaining + [l], tier_cfg, corr, bankroll, games, SCREEN_SIMULATIONS, seed) for l in fits[:REPLACEMENT_SHORTLIST]]
    in_band = [c for c in scored if in_tier_range(c.p_hit, tier_cfg)]
    if not in_band:
        lo, hi = min(c.p_hit for c in scored), max(c.p_hit for c in scored)
        return None, (f"{len(fits)} legs could fill the spot, but every swap puts the hit chance at {_pct(lo)}–{_pct(hi)}, "
                      f"outside {tier_name.capitalize()}'s {_band(tier_cfg)}.")
    in_band.sort(key=lambda c: abs(math.log(max(c.p_hit, 1e-9)) - math.log(target)))
    return score(in_band[0].legs, tier_cfg, corr, bankroll, games, seed=seed), None


__all__ = ["NflParlay", "score", "search", "find_replacement", "apply_price", "rung_line", "target_p"]
