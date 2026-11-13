"""Logs the parlays the NBA Builder suggests, at slate build time, to
nba_suggested_parlays_log (schema in backend/db.py) so they can be graded
against results later (backend/backtest/nba_model_vs_lines.py).

One row per suggested parlay per build: each tier's default search (both
sides, nothing vetoed or switched off), in the order the Builder shows them.
A rebuild later the same day logs again; the grader keeps each parlay's
first appearance.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import duckdb

from backend.parlay.monte_carlo import Leg
from backend.parlay.search import ParlayCandidate, leg_id
from backend.predict.schedule_projections import ET

TABLE = "nba_suggested_parlays_log"


def leg_record(leg: Leg, nba_game_id: str | None, tip_utc: datetime | None) -> dict:
    return {
        "id": leg_id(leg),
        "player_id_nba": leg.player_id_nba,
        "player_name": leg.player_name,
        "game_id": leg.game_id,
        "nba_game_id": nba_game_id,
        "game_date": tip_utc.astimezone(ET).date().isoformat() if tip_utc else None,
        "team": leg.team,
        "market": leg.market,
        "side": leg.side,
        "line": leg.line,
        "decimal_odds": leg.decimal_odds,
        "predicted_mean": leg.predicted_mean,
        "p_hit": leg.p_hit(),
        "no_vig_p": leg.no_vig_p_hit(),
    }


def rows(built_at: datetime, results: dict[str, list[ParlayCandidate]], games: dict, early_season: bool) -> list[tuple]:
    """`games` is the slate's {leg.game_id: Game}; it supplies each leg's
    nba game id and tip time (odds events are rarely resolved on their own)."""
    out = []
    built_utc = built_at if built_at.tzinfo is None else built_at.astimezone(timezone.utc).replace(tzinfo=None)
    for tier, candidates in results.items():
        for rank, c in enumerate(candidates, start=1):
            legs = []
            for leg in c.legs:
                g = games.get(leg.game_id)
                legs.append(leg_record(leg, getattr(g, "nba_game_id", None), getattr(g, "tip_utc", None)))
            out.append((
                built_utc, tier, rank, "|".join(sorted(l["id"] for l in legs)), len(legs), json.dumps(legs),
                c.p_hit, c.independent_p_hit, c.no_vig_p, c.p_sized, c.combined_decimal_odds,
                c.stake.ev_per_dollar, c.flagged, "; ".join(c.flag_reasons) or None, early_season,
            ))
    return out


def log_suggestions(conn: duckdb.DuckDBPyConnection, built_at: datetime, results: dict[str, list[ParlayCandidate]],
                    games: dict, early_season: bool) -> int:
    out = rows(built_at, results, games, early_season)
    if out:
        conn.executemany(f"INSERT INTO {TABLE} VALUES ({', '.join('?' * 15)}) ON CONFLICT DO NOTHING", out)
    return len(out)
