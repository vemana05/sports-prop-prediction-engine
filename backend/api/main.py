"""FastAPI routes for the Builder UI (spec Section 10.4 A) over the existing
Phase 4 parlay search and Kelly sizing code.

    GET  /api/health
    GET  /api/tiers
    GET  /api/slate?league=nba
    POST /api/slate/refresh
    GET  /api/parlays?league=nba&tier=risky&bankroll=500&sides=both&veto=<leg id>&exclude=<game id>
    GET  /api/legs?sides=both&veto=...&exclude=...
    POST /api/parlay/replace     veto one leg, swap in the best leg that still fits
    POST /api/parlay/score       score a hand-picked leg set ("Build my own")
    GET  /api/schedule?date=YYYY-MM-DD   games + tentative projections (default: next game day)
    GET  /api/schedule?league=nfl&week=N NFL week + tentative projections (default: current week)
    GET  /api/replay?league=nba&market=points&period=YYYY-MM-DD&weeks=8   out-of-sample replay of a past day
    GET  /api/replay?league=nfl&market=receiving_yards&period=2026-4    ... or NFL week (season-week)
    GET  /api/performance        walk-forward + calibration summary from reports/
    GET  /api/reports/{file}     calibration charts
    GET/POST /api/bets, PATCH/DELETE /api/bets/{id}   bet tracker

NFL odds aren't collected (they'd share the NBA's 500-credit Odds API
budget, spec 3.5). With `league=nfl` the slate/parlay/replace/score routes
work from model-only ladder legs (api/nfl_builder.py): hit chance always,
odds/EV/stake only for prices the user types into the slip.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import duckdb
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.api import data_sources, nfl_builder, replay as replay_mod, tracker
from backend.predict import nfl_projections, schedule_projections
from backend.bankroll.kelly import recommend_stake
from backend.parlay import nfl_search
from backend.parlay.monte_carlo import Leg
from backend.parlay.search import ParlayCandidate, SearchReport, find_replacement, in_tier_range, leg_id, score_parlay
from backend.api.slate import REFERENCE_BANKROLL, SEARCH_SEED, SLATE, Game, load_tiers

app = FastAPI(title="Parlay Builder API")

League = Literal["nba", "nfl"]
Tier = Literal["safe", "risky", "dangerous"]
Sides = Literal["both", "over", "under"]
REPORTS_DIR = Path("reports")

MARKET_ABBR = {"points": "pts", "rebounds": "reb", "assists": "ast", "fg3m": "3PM"}
NFL_MANUAL_UNAVAILABLE = "Build my own isn't available for NFL yet; veto legs in a suggested parlay instead."


def _nfl_pool(week: int | None) -> nfl_builder.WeekPool:
    # GET routes return the "warming" payload themselves; this covers POSTs.
    if warming := nfl_projections.ensure_models(week=week):
        raise HTTPException(503, warming["message"])
    try:
        return nfl_builder.NFL_SLATE.pool(week)
    except duckdb.IOException:
        raise HTTPException(503, "The NFL warehouse is busy (a data job is running). Try again in a few seconds.")


@app.on_event("startup")
def _warm() -> None:
    SLATE.start_build()


def american(d: float) -> int:
    return round((d - 1) * 100) if d >= 2 else round(-100 / (d - 1))


def _iso(dt) -> str | None:
    return dt.isoformat() if dt else None


def _leg_json(leg: Leg) -> dict:
    game = SLATE.games.get(leg.game_id) or Game(leg.game_id, None, None, None)
    is_home = game.home == leg.team if game.home else None
    opponent = (game.away if is_home else game.home) if is_home is not None else None
    unit = MARKET_ABBR.get(leg.market, leg.market)
    return {
        "id": leg_id(leg),
        "player_id": leg.player_id_nba,
        "player_name": leg.player_name,
        "team": leg.team or None,
        "opponent": opponent,
        "is_home": is_home,
        "game_id": leg.game_id,
        "tip_utc": _iso(game.tip_utc),
        "market": leg.market,
        "side": leg.side,
        "line": leg.line,
        "label": f"{leg.side.capitalize()} {leg.line:g} {unit}",
        "model_p": leg.p_hit(),
        "book_p": leg.no_vig_p_hit(),
        "shrunk_p": leg.shrunk_p_hit(),
        "decimal_odds": leg.decimal_odds,
        "american_odds": american(leg.decimal_odds),
        "predicted_mean": leg.predicted_mean,
        "reason": f"Model projects {leg.predicted_mean:.1f} {unit} against a {leg.line:g} line",
        "headshot_url": f"https://cdn.nba.com/headshots/nba/latest/260x190/{leg.player_id_nba}.png",
        "injury": None,  # injury chips land with the player drawer
    }


def _parlay_json(c: ParlayCandidate, bankroll: float, tier_cfg: dict, name: str) -> dict:
    """Stake is re-sized for this bankroll on the same shrunk probability
    the search ranked by (Section 8 step 3) — never on raw `p_hit`."""
    d = c.combined_decimal_odds
    stake = recommend_stake(c.p_sized, d, bankroll, tier_cfg["max_stake_pct_bankroll"])
    amount, reason = (0.0, c.stake.reason) if c.flagged else (stake.stake, stake.reason)
    legs = [_leg_json(leg) for leg in c.legs]
    return {
        "id": "|".join(sorted(leg["id"] for leg in legs)),
        "name": name,
        "legs": legs,
        "p_hit": c.p_hit,
        "independent_p_hit": c.independent_p_hit,
        "p_sized": c.p_sized,
        "no_vig_p": c.no_vig_p,
        "book_p": 1 / d,
        "decimal_odds": d,
        "american_odds": american(d),
        "ev_per_dollar": stake.ev_per_dollar,
        "model_ev_per_dollar": c.model_ev_per_dollar,
        "stake": amount,
        "payout": round(amount * d, 2),
        "capped_by_tier": stake.capped_by_tier and not c.flagged,
        "reason": reason,
        "flagged": c.flagged,
        "flag_reasons": c.flag_reasons,
        "in_tier_band": in_tier_range(c.p_hit, tier_cfg),
    }


def _report_json(r: SearchReport) -> dict:
    return {k: getattr(r, k) for k in r.__dataclass_fields__}


def _games_json() -> list[dict]:
    """Games in the pool with the teams that have legs in them — odds events
    aren't always resolved to an nba game_id yet, so home/away can be
    unknown even when the teams aren't."""
    teams: dict[str, set[str]] = {}
    for leg in SLATE.legs:
        teams.setdefault(leg.game_id, set()).update([leg.team] if leg.team else [])
    for g in SLATE.games.values():
        teams.setdefault(g.game_id, set()).update(t for t in (g.away, g.home) if t)
    return sorted(
        (
            {"game_id": gid, "teams": sorted(t), "tip_utc": _iso(SLATE.games[gid].tip_utc) if gid in SLATE.games else None}
            for gid, t in teams.items()
        ),
        key=lambda g: (g["tip_utc"] or "", g["teams"]),
    )


def _slate_json() -> dict:
    tips = sorted(g.tip_utc for g in SLATE.games.values() if g.tip_utc)
    return {
        "status": SLATE.status,
        "error": SLATE.error,
        "n_games": len(SLATE.games),
        "n_legs": len(SLATE.legs),
        "games": _games_json(),
        "first_tip_utc": _iso(tips[0]) if tips else None,
        "odds_pulled_at": _iso(SLATE.odds_pulled_at),
        "build_seconds": SLATE.build_seconds,
    }


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "slate": SLATE.status}


@app.get("/api/tiers")
def tiers() -> dict:
    return load_tiers()


@app.get("/api/slate")
def slate(league: League = "nba", week: int | None = Query(None, ge=1, le=23)) -> dict:
    if league == "nfl":
        return nfl_projections.ensure_models(week=week) or nfl_builder.slate_json(_nfl_pool(week))
    return _slate_json()


@app.post("/api/slate/refresh")
def refresh() -> dict:
    SLATE.start_build()
    return _slate_json()


def _filters(veto: list[str], exclude: list[str]) -> tuple[frozenset, frozenset]:
    return frozenset(veto), frozenset(exclude)


@app.get("/api/parlays")
def parlays(
    league: League = "nba",
    tier: Tier = "risky",
    bankroll: float = Query(500.0, gt=0, le=10_000_000),
    sides: Sides = "both",
    veto: list[str] = Query(default=[]),
    exclude: list[str] = Query(default=[]),
    week: int | None = Query(None, ge=1, le=23),
) -> dict:
    if league == "nfl":
        if warming := nfl_projections.ensure_models(week=week):
            return {**warming, "parlays": []}
        tier_cfg = load_tiers()[tier]
        wp = _nfl_pool(week)
        results, report = nfl_builder.NFL_SLATE.search(wp, tier, tier_cfg, *_filters(veto, exclude))
        return {
            **nfl_builder.slate_json(wp),
            "tier": tier,
            "sides": "over",
            "report": nfl_builder.report_json(report),
            "parlays": [nfl_builder.parlay_json(c, tier_cfg, f"Parlay {chr(ord('A') + i)}") for i, c in enumerate(results)],
        }
    if SLATE.status != "ready":
        return {**_slate_json(), "parlays": []}
    tier_cfg = load_tiers()[tier]
    results, report = SLATE.search(tier, sides, *_filters(veto, exclude))
    return {
        **_slate_json(),
        "tier": tier,
        "sides": sides,
        "report": _report_json(report),
        "parlays": [_parlay_json(c, bankroll, tier_cfg, f"Parlay {chr(ord('A') + i)}") for i, c in enumerate(results)],
    }


@app.get("/api/legs")
def legs(
    league: League = "nba",
    sides: Sides = "both",
    veto: list[str] = Query(default=[]),
    exclude: list[str] = Query(default=[]),
) -> dict:
    if league == "nfl":
        return {"status": "unavailable", "message": NFL_MANUAL_UNAVAILABLE, "legs": []}
    if SLATE.status != "ready":
        return {"status": SLATE.status, "legs": []}
    pool = SLATE.pool(sides, *_filters(veto, exclude))
    return {"status": "ready", "legs": [_leg_json(leg) for leg in sorted(pool, key=lambda l: -l.shrunk_p_hit())]}


def _lookup(ids: list[str]) -> list[Leg]:
    missing = [i for i in ids if i not in SLATE.legs_by_id]
    if missing:
        raise HTTPException(409, f"These legs aren't in the current pool any more (odds refreshed?): {', '.join(missing)}")
    return [SLATE.legs_by_id[i] for i in ids]


def _require_ready() -> None:
    if SLATE.status != "ready":
        raise HTTPException(503, "The slate is still building. Try again in a few seconds.")


class PriceIn(BaseModel):
    """A sportsbook price typed into the slip for one NFL leg."""
    line: str | None = None  # "54.5", or "55" / "55+" meaning 55 or more
    over: float | None = None  # American
    under: float | None = None  # American, optional; enables the no-vig price


class ReplaceRequest(BaseModel):
    league: League = "nba"
    week: int | None = None
    prices: dict[str, PriceIn] = {}  # NFL: keyed by base leg id
    leg_ids: list[str] = Field(min_length=1)
    veto_id: str
    tier: Tier = "risky"
    sides: Sides = "both"
    vetoed: list[str] = []  # every leg vetoed this session, including veto_id
    excluded_games: list[str] = []
    bankroll: float = Field(500.0, gt=0, le=10_000_000)
    name: str = "Your parlay"


@app.post("/api/parlay/replace")
def replace_leg(req: ReplaceRequest) -> dict:
    """Veto one leg: drop it, swap in the best leg that fits the tier, the
    side preference, and the one-leg-per-player-prop rule, and re-score.
    If nothing fits, the leg still goes (a veto is a veto) and the parlay
    is re-scored on what's left, with the reason no swap was possible."""
    if req.veto_id not in req.leg_ids:
        raise HTTPException(422, "veto_id must be one of leg_ids")
    if req.league == "nfl":
        return _nfl_replace(req)
    _require_ready()
    tier_cfg = load_tiers()[req.tier]
    current = _lookup(req.leg_ids)
    vetoed_leg = SLATE.legs_by_id.get(req.veto_id)
    pool = SLATE.pool(req.sides, *_filters(req.vetoed + [req.veto_id], req.excluded_games))

    replaced, why_not = find_replacement(
        current, vetoed_leg, pool, req.tier, tier_cfg, SLATE.corr, REFERENCE_BANKROLL, seed=SEARCH_SEED, sides=req.sides
    )
    if replaced is not None:
        new_leg = next(l for l in replaced.legs if leg_id(l) not in req.leg_ids)
        return {
            "parlay": _parlay_json(replaced, req.bankroll, tier_cfg, req.name),
            "added": _leg_json(new_leg),
            "message": None,
        }

    remaining = [l for l in current if leg_id(l) != req.veto_id]
    message = f"No replacement fits: {why_not}"
    if not remaining:
        return {"parlay": None, "added": None, "message": message}
    c = score_parlay(remaining, tier_cfg, SLATE.corr, REFERENCE_BANKROLL, seed=SEARCH_SEED)
    if len(remaining) < tier_cfg["legs_min"]:
        message += f" The parlay now has {len(remaining)} leg{'s' if len(remaining) != 1 else ''}, below {req.tier.capitalize()}'s {tier_cfg['legs_min']}-leg minimum."
    return {"parlay": _parlay_json(c, req.bankroll, tier_cfg, req.name), "added": None, "message": message}


def _nfl_prices(prices: dict[str, PriceIn]) -> dict[str, dict]:
    return {k: v.model_dump() for k, v in prices.items()}


def _nfl_replace(req: ReplaceRequest) -> dict:
    tier_cfg = load_tiers()[req.tier]
    wp = _nfl_pool(req.week)
    prices = _nfl_prices(req.prices)
    current = nfl_builder.resolve(wp, req.leg_ids, prices)
    vetoed_leg = wp.legs_by_id[req.veto_id]
    vetoed = set(req.vetoed) | {req.veto_id}
    pool = [l for l in wp.legs if leg_id(l) not in vetoed and l.game_id not in set(req.excluded_games)]
    corr = nfl_builder.NFL_SLATE.corr()
    replaced, why_not = nfl_search.find_replacement(current, vetoed_leg, pool, req.tier, tier_cfg, corr, req.bankroll, wp.games, seed=SEARCH_SEED)
    if replaced is not None:
        new_leg = next(l for l in replaced.legs if (l.base_id or leg_id(l)) not in req.leg_ids)
        return {"parlay": nfl_builder.parlay_json(replaced, tier_cfg, req.name), "added": nfl_builder.leg_json(new_leg), "message": None}
    remaining = [l for l in current if (l.base_id or leg_id(l)) != req.veto_id]
    message = f"No replacement fits: {why_not}"
    if not remaining:
        return {"parlay": None, "added": None, "message": message}
    c = nfl_search.score(remaining, tier_cfg, corr, req.bankroll, wp.games, seed=SEARCH_SEED)
    if len(remaining) < tier_cfg["legs_min"]:
        message += f" The parlay now has {len(remaining)} leg{'s' if len(remaining) != 1 else ''}, below {req.tier.capitalize()}'s {tier_cfg['legs_min']}-leg minimum."
    return {"parlay": nfl_builder.parlay_json(c, tier_cfg, req.name), "added": None, "message": message}


class ScoreRequest(BaseModel):
    league: League = "nba"
    week: int | None = None
    prices: dict[str, PriceIn] = {}  # NFL: keyed by base leg id
    name: str = "Your parlay"
    leg_ids: list[str] = Field(min_length=1, max_length=12)
    tier: Tier = "risky"
    bankroll: float = Field(500.0, gt=0, le=10_000_000)


@app.post("/api/parlay/score")
def score(req: ScoreRequest) -> dict:
    """Manual mode: the same engine (Monte Carlo, shrink, Kelly, sanity
    guard) on legs the user picked. Stake is capped at the selected tier.
    NFL: re-scores a slip with the user's typed lines/prices applied."""
    if req.league == "nfl":
        tier_cfg = load_tiers()[req.tier]
        wp = _nfl_pool(req.week)
        legs = nfl_builder.resolve(wp, req.leg_ids, _nfl_prices(req.prices))
        c = nfl_search.score(legs, tier_cfg, nfl_builder.NFL_SLATE.corr(), req.bankroll, wp.games, seed=SEARCH_SEED)
        return {"parlay": nfl_builder.parlay_json(c, tier_cfg, req.name)}
    _require_ready()
    chosen = _lookup(req.leg_ids)
    seen = set()
    for leg in chosen:
        if (leg.player_id_nba, leg.market) in seen:
            raise HTTPException(422, f"{leg.player_name} already has a {leg.market} leg in this parlay.")
        seen.add((leg.player_id_nba, leg.market))
    tier_cfg = load_tiers()[req.tier]
    c = score_parlay(chosen, tier_cfg, SLATE.corr, REFERENCE_BANKROLL, seed=SEARCH_SEED)
    return {"parlay": _parlay_json(c, req.bankroll, tier_cfg, "Your parlay")}


def _slate_game_for(home: str | None, away: str | None, tip_utc: str) -> str | None:
    """The Builder's game key for a scheduled game, if its lines are in the
    current slate. Odds events aren't always resolved to an nba game_id, so
    match on the teams of the slate's legs plus the same tip date."""
    if SLATE.status != "ready":
        return None
    day = tip_utc[:10]
    teams = {t for t in (home, away) if t}
    for gid, game in SLATE.games.items():
        leg_teams = {l.team for l in SLATE.legs if l.game_id == gid and l.team}
        same_day = game.tip_utc is None or game.tip_utc.isoformat()[:10] == day
        if leg_teams and leg_teams <= teams and same_day:
            return gid
    return None


@app.get("/api/schedule")
def schedule(
    league: League = "nba",
    date_: str | None = Query(None, alias="date"),
    week: int | None = Query(None, ge=1, le=23),
) -> dict:
    """Never calls The Odds API: "lines posted" only reflects odds already
    collected into the slate (NBA). NFL is projections only."""
    if league == "nfl":
        if warming := nfl_projections.ensure_models(week=week):
            return warming
        try:
            return {"status": "ready", **nfl_projections.get(week)}
        except duckdb.IOException:
            raise HTTPException(503, "The NFL warehouse is busy (a data job is running). Try again in a few seconds.")
    try:
        if date_:
            try:
                day = date.fromisoformat(date_)
            except ValueError:
                raise HTTPException(422, "date must be YYYY-MM-DD")
        else:
            conn = schedule_projections._connect()
            try:
                day = schedule_projections.next_game_day(conn) or schedule_projections.today_et()
            finally:
                conn.close()
        data = schedule_projections.get(day)
        with_lines = schedule_projections.games_with_lines([g["game_id"] for g in data["games"]])
    except duckdb.IOException:
        raise HTTPException(503, "The warehouse is busy (a data job or slate build is running). Try again in a few seconds.")
    for g in data["games"]:
        g["builder_game_id"] = _slate_game_for(g["home"], g["away"], g["tip_utc"])
        g["lines_posted"] = g["game_id"] in with_lines or g["builder_game_id"] is not None
    return {"status": "ready", "today": schedule_projections.today_et().isoformat(), **data}


@app.get("/api/replay")
def replay(
    league: League = "nba",
    market: str | None = None,
    period: str | None = None,
    weeks: int = Query(8, ge=1, le=52),
) -> dict:
    """Out-of-sample projections vs. actual results. Never calls The Odds API."""
    try:
        return replay_mod.replay(league, market, period, weeks)
    except replay_mod.NotBuilt as exc:
        return {"status": "unavailable", "league": league, "detail": str(exc)}
    except ValueError:
        raise HTTPException(422, "period must be YYYY-MM-DD (NBA) or SEASON-WEEK (NFL)")
    except duckdb.IOException:
        raise HTTPException(503, "The warehouse is busy (a data job is running). Try again in a few seconds.")


@app.get("/api/performance")
def performance() -> dict:
    path = REPORTS_DIR / "model_performance.json"
    if not path.exists():
        raise HTTPException(404, "reports/model_performance.json is missing")
    data = json.loads(path.read_text())
    for row in data["calibration"]["rows"]:
        row["chart_url"] = f"/api/reports/{row['chart']}" if (REPORTS_DIR / row["chart"]).exists() else None
    data["nfl"] = _nfl_performance()
    return data


def _nfl_performance() -> dict | None:
    path = REPORTS_DIR / "nfl_model_performance.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    rows = []
    for name, m in raw["markets"].items():
        b, c = m["baselines"], m["calibration"]
        chart = Path(c["chart"]).name
        rows.append({
            "market": name,
            "label": m["label"],
            "n": b["n"],
            "model_mae": b["model_mae"],
            "last5_mae": b["last5_avg_mae"],
            "season_mae": b["season_avg_mae"],
            "shrunk_mae": b["shrunk_last5_mae"],
            "no_history": b.get("no_history_rows"),
            "error_before": c["raw_mean_abs_calibration_error"],
            "error_after": c["calibrated_mean_abs_calibration_error"],
            "chart_url": f"/api/reports/{chart}" if (REPORTS_DIR / chart).exists() else None,
        })
    return {"test_seasons": raw["test_seasons"], "rows": rows}


@app.get("/api/data-sources")
def data_sources_status() -> dict:
    """Data Sources page: jobs, API usage, table freshness, models, review queue.
    Read-only; no API calls; key values are redacted."""
    return data_sources.build_status()


@app.get("/api/nfl/model-vs-lines")
def nfl_model_vs_lines() -> dict:
    """reports/nfl_model_vs_lines.json, rebuilt by the Tuesday results job
    (backend/backtest/nfl_model_vs_lines.py). available=False until the first run."""
    path = REPORTS_DIR / "nfl_model_vs_lines.json"
    if not path.exists():
        return {"available": False}
    return {"available": True, **json.loads(path.read_text())}


@app.get("/api/reports/{name}")
def report_file(name: str) -> FileResponse:
    path = REPORTS_DIR / name
    if "/" in name or not name.endswith(".png") or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/png")


class BetIn(BaseModel):
    league: League = "nba"
    tier: Tier
    name: str
    legs: list[dict]
    decimal_odds: float = Field(gt=1)
    american_odds: int
    model_p: float = Field(ge=0, le=1)
    stake: float = Field(gt=0)
    paper: bool = False


class Settle(BaseModel):
    result: Literal["pending", "won", "lost", "push"]


@app.get("/api/bets")
def bets() -> dict:
    return {"bets": tracker.list_bets()}


@app.post("/api/bets")
def log_bet(bet: BetIn) -> dict:
    return tracker.add_bet(bet.model_dump())


@app.patch("/api/bets/{bet_id}")
def settle(bet_id: int, body: Settle) -> dict:
    bet = tracker.settle_bet(bet_id, body.result)
    if bet is None:
        raise HTTPException(404)
    return bet


@app.delete("/api/bets/{bet_id}")
def delete(bet_id: int) -> dict:
    if not tracker.delete_bet(bet_id):
        raise HTTPException(404)
    return {"ok": True}
