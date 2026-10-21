"""Builds today's leg pool: for every real odds line in `odds_snapshots`,
resolve the player, compute a fresh model prediction, attach this market's
isotonic calibrator (spec Section 2's pipeline — calibration happens here,
before the `Leg` ever reaches shrinkage or the Monte Carlo simulator), and
produce both the "over" and "under" `Leg`.

Deliberately does NOT read or write `model_predictions_log` — that table's
whole point is "log a prediction once, near game day, and never let it
drift" (see log_predictions.py). Computing a fresh prediction here instead
means the leg pool is always honest about "what would the model say right
now," including when (as during this build) that's weeks before tip-off,
without permanently logging a stale early snapshot that would block the
real, close-to-game-time prediction later.

Only `points` has real odds right now (odds_theoddsapi.py only collects
`player_points` — see its own docstring for the credit-budget reasoning);
rebounds/assists/fg3m legs would need those markets added to collection
first. The pool and everything downstream (correlation, Monte Carlo, beam
search) already supports all 4 markets — it's purely an odds-collection
scope decision, not a modeling limitation.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import duckdb
import pandas as pd
from nba_api.stats.static import teams as nba_static_teams
from rapidfuzz import fuzz, process

from backend.db import get_connection
from backend.features.nba_current_form import build_current_form, teammates_out_from_injury_report
from backend.ingest.util import logger
from backend.models.calibration import get_or_fit_calibrator
from backend.models.model_cache import get_or_train_booster
from backend.models.nba_minutes_model import FEATURE_COLUMNS
from backend.models.nba_points_model import train_final_model as train_points_final
from backend.parlay.monte_carlo import Leg
from backend.parlay.predictor_choice import use_model_for
from backend.xref.build_xref import normalize_name

FUZZY_MATCH_THRESHOLD = 88.0


def _opponent_by_team(conn: duckdb.DuckDBPyConnection, game_date: date) -> dict[str, str]:
    """{team_abbr: opponent_abbr} for every game on `game_date`, from the
    real nba_schedule table — used to fill in opp_pace_roll10 /
    opp_def_vs_position_pts_roll10, which are about the player's NEXT game,
    not a historical one. nba_schedule stores nba.com numeric team ids;
    nba_api's static team list is the standard way to resolve those back to
    the same 3-letter abbreviations nba_player_games uses."""
    id_to_abbr = {str(t["id"]): t["abbreviation"] for t in nba_static_teams.get_teams()}
    games = conn.execute(
        "SELECT home_team_id, away_team_id FROM nba_schedule WHERE game_date = ?", [game_date]
    ).fetchall()
    out: dict[str, str] = {}
    for home_id, away_id in games:
        home, away = id_to_abbr.get(home_id), id_to_abbr.get(away_id)
        if home and away:
            out[home] = away
            out[away] = home
    return out


def _resolve_player_names(conn: duckdb.DuckDBPyConnection, raw_names: list[str]) -> dict[str, str]:
    """{raw_name: player_id_nba}, fuzzy-matched against player_xref — same
    approach as build_xref.py / match_injury_names.py."""
    xref = conn.execute("SELECT player_id, full_name FROM player_xref WHERE full_name IS NOT NULL").fetchall()
    index = {normalize_name(name): pid for pid, name in xref}
    choices = list(index.keys())

    out = {}
    for raw in raw_names:
        norm = normalize_name(raw)
        if norm in index:
            out[raw] = index[norm]
            continue
        result = process.extractOne(norm, choices, scorer=fuzz.WRatio)
        if result and result[1] >= FUZZY_MATCH_THRESHOLD:
            out[raw] = index[result[0]]
        else:
            logger.warning("leg_pool: could not resolve odds player name %r to a player_id", raw)
    return out


def _points_predictions(
    conn: duckdb.DuckDBPyConnection,
    player_ids: list[str],
    opponent_by_team: dict[str, str],
    out_players_by_team: dict[str, list[str]],
) -> dict[str, tuple[float, float]]:
    """{player_id_nba: (predicted_mean, sigma)} for the points market,
    using whichever predictor won the walk-forward comparison."""
    current_form = build_current_form(conn, opponent_by_team=opponent_by_team, out_players_by_team=out_players_by_team)
    current_form = current_form[current_form["player_id_nba"].isin(player_ids)]
    if current_form.empty:
        return {}

    if use_model_for("points"):
        model, extra = get_or_train_booster("points", lambda: _train_points(conn))
        sigma = extra["sigma"]
        X = current_form[[c for c in FEATURE_COLUMNS if c != "home_flag"]].copy()
        X["home_flag"] = 0  # home/away unknown without a specific game joined in; a small, acknowledged simplification
        X["is_back_to_back"] = X["is_back_to_back"].astype(float)
        preds = model.predict(X[FEATURE_COLUMNS])
        return {pid: (float(p), sigma) for pid, p in zip(current_form["player_id_nba"], preds)}
    else:
        sigma = current_form["pts_std10"].median()  # fallback dispersion for the naive path
        return {
            pid: (float(roll), float(sigma) if pd.notna(sigma) else 5.0)
            for pid, roll in zip(current_form["player_id_nba"], current_form["pts_roll10"])
        }


def _train_points(conn):
    model, sigma = train_points_final(conn)
    return model, {"sigma": sigma}


def build_leg_pool(conn: duckdb.DuckDBPyConnection | None = None, now: datetime | None = None) -> list[Leg]:
    """Legs for events that haven't started yet: odds_snapshots keeps every
    pull, so yesterday's games are still in it."""
    own_conn = conn is None
    now = now or datetime.now(timezone.utc)
    conn = conn or get_connection()
    try:
        odds = conn.execute(
            """
            SELECT player_name_raw, event_id, game_id_nba, commence_time, bookmaker, line, over_price, under_price
            FROM odds_snapshots
            WHERE market = 'player_points' AND commence_time > ?
            QUALIFY ROW_NUMBER() OVER (PARTITION BY player_name_raw, event_id ORDER BY pulled_at DESC) = 1
            """,
            [now],
        ).fetchdf()
        if odds.empty:
            logger.warning("leg_pool: no upcoming player_points odds in odds_snapshots")
            return []

        name_map = _resolve_player_names(conn, odds["player_name_raw"].unique().tolist())
        odds["player_id_nba"] = odds["player_name_raw"].map(name_map)
        odds = odds.dropna(subset=["player_id_nba"])

        opponent_by_team: dict[str, str] = {}
        out_players_by_team: dict[str, list[str]] = {}
        for game_date in odds["commence_time"].dt.date.unique():
            opponent_by_team.update(_opponent_by_team(conn, game_date))
            out_players_by_team.update(teammates_out_from_injury_report(conn, game_date))

        preds = _points_predictions(
            conn, odds["player_id_nba"].unique().tolist(), opponent_by_team, out_players_by_team
        )
        points_calibrator = get_or_fit_calibrator(conn, "points")

        current_form = build_current_form(conn, opponent_by_team=opponent_by_team)
        team_by_player = dict(zip(current_form["player_id_nba"], current_form["team"]))

        legs: list[Leg] = []
        for row in odds.itertuples():
            if row.player_id_nba not in preds:
                continue
            predicted_mean, sigma = preds[row.player_id_nba]
            team = team_by_player.get(row.player_id_nba, "")
            base = dict(
                player_id_nba=row.player_id_nba,
                player_name=row.player_name_raw,
                game_id=row.game_id_nba or row.event_id,  # falls back to event_id if not yet resolved to an nba game_id
                team=team,
                market="points",
                line=row.line,
                predicted_mean=predicted_mean,
                dispersion=sigma,
                over_price=row.over_price,
                under_price=row.under_price,
                calibrator=points_calibrator,
            )
            if row.over_price:
                legs.append(Leg(side="over", decimal_odds=row.over_price, **base))
            if row.under_price:
                legs.append(Leg(side="under", decimal_odds=row.under_price, **base))

        logger.info("leg_pool: built %d legs from %d odds rows (%d players resolved)", len(legs), len(odds), len(preds))
        return legs
    finally:
        if own_conn:
            conn.close()
