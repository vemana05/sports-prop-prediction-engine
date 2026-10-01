"""Builds and caches the out-of-sample walk-forward residual for every
(player, game, market) — the raw material correlation estimation
(`correlations.py`) needs. Uses whichever predictor actually won the
walk-forward comparison per market (`predictor_choice.py`) — for fg3m
that's the naive last-10 average, not the trained model.

Cached to a DuckDB table (`nba_walk_forward_residuals`) because computing
it means running all 4 markets' full walk-forward backtests — several
minutes of LightGBM training — which correlation estimation would
otherwise redo on every call. Rebuild with `run(force=True)` after
retraining any market model.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from backend.db import get_connection
from backend.ingest.util import IngestResult, logger, utcnow
from backend.models import nba_market_model as mm
from backend.models import nba_points_model
from backend.parlay.predictor_choice import use_model_for

RESIDUALS_TABLE = "nba_walk_forward_residuals"

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {RESIDUALS_TABLE} (
    player_id_nba VARCHAR NOT NULL,
    game_id       VARCHAR NOT NULL,
    team          VARCHAR,
    market        VARCHAR NOT NULL,
    actual        DOUBLE,
    predicted     DOUBLE,
    residual      DOUBLE,
    predictor     VARCHAR NOT NULL,  -- "model" | "naive"
    ingested_at   TIMESTAMP NOT NULL,
    PRIMARY KEY (player_id_nba, game_id, market)
);
"""

_MARKET_SPECS = [
    # (market, target_column, naive_column)
    ("points", "pts", "pts_roll10"),
    ("rebounds", "reb", "reb_roll10"),
    ("assists", "ast", "ast_roll10"),
    ("fg3m", "fg3m", "fg3m_roll10"),
]


def _points_residuals() -> pd.DataFrame:
    conn = get_connection()
    try:
        df = nba_points_model.walk_forward_backtest(conn)
    finally:
        conn.close()
    predictor = "model" if use_model_for("points") else "naive"
    predicted = df["predicted_mean"] if predictor == "model" else df["pts_roll10"]
    return pd.DataFrame(
        {
            "player_id_nba": df["player_id_nba"],
            "game_id": df["game_id"],
            "market": "points",
            "actual": df["pts"],
            "predicted": predicted,
            "predictor": predictor,
        }
    )


def _nb_market_residuals(conn: duckdb.DuckDBPyConnection, market: str, target_col: str, naive_col: str) -> pd.DataFrame:
    df = mm.walk_forward_backtest(conn, target_col, naive_col)
    predictor = "model" if use_model_for(market) else "naive"
    predicted = df["predicted_mean"] if predictor == "model" else df[naive_col]
    return pd.DataFrame(
        {
            "player_id_nba": df["player_id_nba"],
            "game_id": df["game_id"],
            "market": market,
            "actual": df[target_col],
            "predicted": predicted,
            "predictor": predictor,
        }
    )


def build(conn: duckdb.DuckDBPyConnection) -> IngestResult:
    conn.execute(_SCHEMA)
    conn.execute(f"DELETE FROM {RESIDUALS_TABLE}")

    frames = [_points_residuals()]
    for market, target_col, naive_col in _MARKET_SPECS:
        if market == "points":
            continue
        frames.append(_nb_market_residuals(conn, market, target_col, naive_col))

    combined = pd.concat(frames, ignore_index=True)
    combined["residual"] = combined["actual"] - combined["predicted"]
    combined = combined.dropna(subset=["predicted", "residual"])

    team_lookup = conn.execute("SELECT DISTINCT player_id_nba, game_id, team FROM nba_player_games").fetchdf()
    combined = combined.merge(team_lookup, on=["player_id_nba", "game_id"], how="left")
    combined["ingested_at"] = utcnow()

    conn.register("residuals_df", combined)
    conn.execute(
        f"""
        INSERT INTO {RESIDUALS_TABLE}
            (player_id_nba, game_id, team, market, actual, predicted, residual, predictor, ingested_at)
        SELECT player_id_nba, game_id, team, market, actual, predicted, residual, predictor, ingested_at
        FROM residuals_df
        """
    )
    conn.unregister("residuals_df")

    n = conn.execute(f"SELECT COUNT(*) FROM {RESIDUALS_TABLE}").fetchone()[0]
    logger.info("nba_walk_forward_residuals: built %d rows", n)
    return IngestResult(rows_written=n)


def run(conn: duckdb.DuckDBPyConnection | None = None, force: bool = False) -> IngestResult:
    own_conn = conn is None
    conn = conn or get_connection()
    try:
        conn.execute(_SCHEMA)
        existing = conn.execute(f"SELECT COUNT(*) FROM {RESIDUALS_TABLE}").fetchone()[0]
        if existing and not force:
            logger.info("nba_walk_forward_residuals: using cached %d rows (force=True to rebuild)", existing)
            return IngestResult(rows_written=existing, notes={"cached": True})
        return build(conn)
    finally:
        if own_conn:
            conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Rebuild even if a cached table exists")
    args = parser.parse_args()
    result = run(force=args.force)
    print(f"residuals build complete: {result.rows_written} rows ({result.notes})")
