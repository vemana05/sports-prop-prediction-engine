"""Logs a prediction for every market, for every likely-active player, for
every NBA game starting in the next `LOOKAHEAD_HOURS` — the free-tier
substitute for comparing against real historical odds (Section 3.5 / 5.7).
Run this once a day, BEFORE `backend/ingest/odds_theoddsapi.py`, so a
prediction is always on record before the matching line shows up; once
both exist for a game, they can be joined for paper-trading.

Predictions are logged once per (game, player, market) and never
overwritten (model_predictions_log's primary key enforces this) — so a
prediction can't quietly drift if this is re-run closer to tip-off.

Roster source: `nba_current_form` — whoever last played for a team, from
actual game logs, not a real depth chart. A player traded or signed since
their last logged game won't show up under their new team yet; see that
module's docstring for the full caveat.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb
import pandas as pd
from nba_api.stats.static import teams as nba_static_teams

from backend.db import get_connection
from backend.features.nba_current_form import build_current_form
from backend.ingest.util import IngestResult, logger, utcnow
from backend.models import nba_market_model as mm
from backend.models import nba_minutes_model, nba_points_model
from backend.models.nba_minutes_model import FEATURE_COLUMNS

LOOKAHEAD_HOURS = 48
MODEL_VERSION = "phase2-v1"

_MARKET_MODELS = [
    ("rebounds", "reb", "reb_roll10"),
    ("assists", "ast", "ast_roll10"),
    ("fg3m", "fg3m", "fg3m_roll10"),
]


def _team_id_to_abbrev() -> dict[str, str]:
    return {str(t["id"]): t["abbreviation"] for t in nba_static_teams.get_teams()}


def upcoming_games(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=LOOKAHEAD_HOURS)
    return conn.execute(
        "SELECT game_id, season, home_team_id, away_team_id, tip_utc FROM nba_schedule WHERE tip_utc BETWEEN ? AND ?",
        [now, cutoff],
    ).fetchdf()


def _build_feature_rows(games: pd.DataFrame, current_form: pd.DataFrame) -> pd.DataFrame:
    team_abbrev = _team_id_to_abbrev()
    rows = []
    for g in games.itertuples():
        home_abbrev = team_abbrev.get(str(g.home_team_id))
        away_abbrev = team_abbrev.get(str(g.away_team_id))
        for abbrev, home_flag in [(home_abbrev, 1), (away_abbrev, 0)]:
            roster = current_form[current_form["team"] == abbrev]
            for p in roster.itertuples():
                row = {"game_id": g.game_id, "player_id_nba": p.player_id_nba, "home_flag": home_flag}
                for col in FEATURE_COLUMNS:
                    if col == "home_flag":
                        continue
                    row[col] = getattr(p, col, None)
                rows.append(row)
    return pd.DataFrame(rows)


def run(conn: duckdb.DuckDBPyConnection | None = None) -> IngestResult:
    own_conn = conn is None
    conn = conn or get_connection()
    try:
        games = upcoming_games(conn)
        if games.empty:
            logger.info("log_predictions: no games in the next %dh", LOOKAHEAD_HOURS)
            return IngestResult(rows_written=0, notes={"games": 0})

        current_form = build_current_form(conn)
        if current_form.empty:
            logger.warning("log_predictions: no current-form data available")
            return IngestResult(rows_written=0, notes={"error": "no current form"})

        feature_rows = _build_feature_rows(games, current_form)
        if feature_rows.empty:
            logger.warning("log_predictions: no rostered players matched to the %d upcoming games", len(games))
            return IngestResult(rows_written=0, notes={"games": len(games), "players": 0})

        X = feature_rows[FEATURE_COLUMNS].copy()
        X["is_back_to_back"] = X["is_back_to_back"].astype(float)

        logger.info("log_predictions: training models on full history (minutes, points, rebounds, assists, fg3m)")
        minutes_model = nba_minutes_model.train_final_model(conn)
        points_model, points_sigma = nba_points_model.train_final_model(conn)
        market_models = {
            name: mm.train_final_model(conn, target, line_col) for name, target, line_col in _MARKET_MODELS
        }

        now = utcnow()
        log_rows = []
        minutes_preds = minutes_model.predict(X)
        points_preds = points_model.predict(X)
        for i, frow in enumerate(feature_rows.itertuples()):
            log_rows.append((now, frow.game_id, frow.player_id_nba, "minutes", float(minutes_preds[i]), None, MODEL_VERSION))
            log_rows.append((now, frow.game_id, frow.player_id_nba, "points", float(points_preds[i]), points_sigma, MODEL_VERSION))
        for name, (model, r) in market_models.items():
            preds = model.predict(X)
            for i, frow in enumerate(feature_rows.itertuples()):
                log_rows.append((now, frow.game_id, frow.player_id_nba, name, float(max(preds[i], 0)), r, MODEL_VERSION))

        conn.executemany(
            """
            INSERT INTO model_predictions_log VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (game_id, player_id_nba, market) DO NOTHING
            """,
            log_rows,
        )
        logger.info(
            "log_predictions: logged %d predictions (%d players x 5 markets) for %d games",
            len(log_rows), len(feature_rows), len(games),
        )
        return IngestResult(rows_written=len(log_rows), notes={"games": len(games), "players": len(feature_rows)})
    finally:
        if own_conn:
            conn.close()


if __name__ == "__main__":
    result = run()
    print(f"log_predictions complete: {result.rows_written} rows ({result.notes})")
