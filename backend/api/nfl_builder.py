"""NFL side of the Builder API: per-week leg pool + correlations cached in
memory, searches cached per (week, tier, vetoes), and JSON shaped like the
NBA parlay/leg payloads with prices optional. Never calls The Odds API.

Builds synchronously on first request for a week (about a second once the
ladder calibrators are cached) and refreshes after POOL_TTL_S, so a new
injury report or projection recompute shows up within the hour.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field

from fastapi import HTTPException

from backend.parlay import nfl_legs, nfl_search
from backend.parlay.nfl_legs import MARKET_ABBR, NflLeg, UNTESTED_NOTE
from backend.parlay.search import SearchReport, in_tier_range, leg_id

POOL_TTL_S = 30 * 60
SEARCH_SEED = 7


@dataclass
class WeekPool:
    week: int
    legs: list[NflLeg]
    games: dict
    meta: dict
    built_at: float
    legs_by_id: dict[str, NflLeg] = field(default_factory=dict)
    searches: dict[tuple, tuple[list[nfl_search.NflParlay], SearchReport]] = field(default_factory=dict)


class NflSlate:
    def __init__(self) -> None:
        self._pools: dict[int | None, WeekPool] = {}
        self._corr: dict | None = None
        self._lock = threading.Lock()

    def corr(self) -> dict:
        if self._corr is None:
            self._corr = nfl_legs.correlation_matrices()
        return self._corr

    def pool(self, week: int | None) -> WeekPool:
        with self._lock:
            cached = self._pools.get(week)
            if cached and time.time() - cached.built_at < POOL_TTL_S:
                return cached
            built = nfl_legs.build_pool(week)
            wp = WeekPool(week=built["week"], legs=built["legs"], games=built["games"], meta=built, built_at=time.time())
            wp.legs_by_id = {leg_id(l): l for l in wp.legs}
            self._pools[week] = self._pools[wp.week] = wp
            return wp

    def search(self, wp: WeekPool, tier: str, tier_cfg: dict, vetoed: frozenset, excluded: frozenset):
        key = (tier, vetoed, excluded)
        if key not in wp.searches:
            pool = [l for l in wp.legs if leg_id(l) not in vetoed and l.game_id not in excluded]
            wp.searches[key] = nfl_search.search(pool, tier, tier_cfg, self.corr(), 100.0, wp.games, seed=SEARCH_SEED)
        return wp.searches[key]


NFL_SLATE = NflSlate()


def american(d: float) -> int:
    return round((d - 1) * 100) if d >= 2 else round(-100 / (d - 1))


def leg_label(l: NflLeg) -> str:
    unit = MARKET_ABBR[l.market]
    return f"{l.rung}+ {unit}" if l.rung is not None else f"Over {l.line:g} {unit}"


def leg_json(l: NflLeg) -> dict:
    unit = MARKET_ABBR[l.market]
    need = f"{l.rung}+" if l.rung is not None else f"over {l.line:g}"
    return {
        "id": leg_id(l),
        "base_id": l.base_id or leg_id(l),
        "player_id": l.player_id_nba,
        "player_name": l.player_name,
        "position": l.position,
        "team": l.team,
        "opponent": l.opponent,
        "is_home": l.is_home,
        "game_id": l.game_id,
        "tip_utc": l.kickoff_utc,
        "market": l.market,
        "side": "over",
        "line": l.line,
        "rung": l.rung,
        "label": leg_label(l),
        "model_p": l.p_hit(),
        "book_p": l.no_vig_p_hit(),
        "shrunk_p": l.shrunk_p_hit(),
        "priced": l.priced,
        "vig_removed": bool(l.over_price and l.under_price),
        "decimal_odds": l.decimal_odds if l.priced else None,
        "american_odds": american(l.decimal_odds) if l.priced else None,
        "under_american": american(l.under_price) if l.under_price else None,
        "predicted_mean": l.predicted_mean,
        "reason": f"Model projects {l.predicted_mean:.1f} {unit}; needs {need}",
        "headshot_url": None,
        "injury": "Questionable" if l.questionable else None,
        "warnings": list(l.warnings),
    }


def parlay_json(c: nfl_search.NflParlay, tier_cfg: dict, name: str) -> dict:
    s = c.stake
    priced_amount = 0.0 if (c.flagged or s is None) else s.stake
    legs = [leg_json(l) for l in c.legs]
    return {
        "id": "|".join(sorted(l["id"] for l in legs)),
        "name": name,
        "legs": legs,
        "p_hit": c.p_hit,
        "independent_p_hit": c.independent_p_hit,
        "priced": c.priced,
        "n_priced": sum(l.priced for l in c.legs),
        "p_sized": c.p_sized if c.p_sized is not None else c.p_hit,
        "no_vig_p": c.no_vig_p,
        "book_p": 1 / c.decimal_odds if c.decimal_odds else None,
        "decimal_odds": c.decimal_odds,
        "american_odds": american(c.decimal_odds) if c.decimal_odds else None,
        "fair_american_odds": american(1 / c.p_hit) if 0 < c.p_hit < 1 else None,
        "ev_per_dollar": s.ev_per_dollar if s else None,
        "model_ev_per_dollar": c.model_ev_per_dollar,
        "stake": priced_amount,
        "payout": round(priced_amount * c.decimal_odds, 2) if c.decimal_odds else 0.0,
        "capped_by_tier": bool(s and s.capped_by_tier and not c.flagged),
        "reason": s.reason if s else None,
        "flagged": c.flagged,
        "flag_reasons": c.flag_reasons,
        "in_tier_band": in_tier_range(c.p_hit, tier_cfg),
        "notes": c.notes,
        "warnings": c.warnings,
    }


def slate_json(wp: WeekPool) -> dict:
    games = sorted(wp.games.values(), key=lambda g: (g["kickoff_utc"] or "", g["game_id"]))
    m = wp.meta
    return {
        "status": "ready",
        "league": "nfl",
        "season": m["season"],
        "week": wp.week,
        "weeks": m["weeks"],
        "n_games": len(games),
        "n_legs": len(wp.legs),
        "games": [{"game_id": g["game_id"], "teams": [g["away"], g["home"]], "tip_utc": g["kickoff_utc"]} for g in games],
        "first_tip_utc": games[0]["kickoff_utc"] if games else None,
        "odds_pulled_at": None,
        "injuries_posted": m["injuries_posted"],
        "calibrated": m["calibrated"],
        "untested_note": UNTESTED_NOTE,
    }


def resolve(wp: WeekPool, ids: list[str], prices: dict[str, dict]) -> list[NflLeg]:
    """Pool legs by base id, with any typed line/price applied."""
    missing = [i for i in ids if i not in wp.legs_by_id]
    if missing:
        raise HTTPException(409, f"These legs aren't in this week's pool any more (projections refreshed?): {', '.join(missing)}")
    out = []
    for i in ids:
        leg, p = wp.legs_by_id[i], prices.get(i) or {}
        if any(p.get(k) not in (None, "") for k in ("line", "over", "under")):
            try:
                leg = nfl_search.apply_price(leg, p.get("line"), p.get("over"), p.get("under"))
            except ValueError as e:
                raise HTTPException(422, f"{leg.player_name}: {e}")
        out.append(leg)
    seen = set()
    for l in out:
        if (l.player_id_nba, l.market) in seen:
            raise HTTPException(422, f"{l.player_name} already has a {MARKET_ABBR[l.market]} leg in this parlay.")
        seen.add((l.player_id_nba, l.market))
    return out


def report_json(r: SearchReport) -> dict:
    out = {k: getattr(r, k) for k in r.__dataclass_fields__}
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in out.items()}
