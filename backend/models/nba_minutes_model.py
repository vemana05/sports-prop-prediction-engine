"""Minutes model (spec Section 5.3 step 1): predicts a player's minutes for
a game from `nba_features`. This is the most important input per the spec —
everything downstream (points, rebounds, etc.) is driven largely by how
long a player is on the floor.

Walk-forward by season (Section 5.4: "never random splits"): train on every
season strictly before the test season, evaluate on the test season, repeat
for each season after the first. A model trained on 2021-22 only predicting
2022-23 is one fold; by the last fold it's trained on 2021-22..2024-25
predicting 2025-26. Out-of-sample predictions from every fold are pooled for
the reported metrics — nothing is ever evaluated on data the model that
made the prediction was trained on.

Rows with `games_played_prior_season == 0` and no prior-season history are
still predictable (the 3/5/10-game rolling windows blend in the player's
most recent games from ANY prior season, including the one right before —
see nba_features.py's window definitions), but a genuinely brand-new player
(rookie, first game of their career) has all-NULL rolling features. LightGBM
handles NaN natively by learning a split direction for missing values, so
these rows are kept, not dropped — but a rookie's very first game or two
should be expected to have very wide/unreliable uncertainty; that's simply
a feature-quality gap, not a bug to route around at the modeling layer.
"""

from __future__ import annotations

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from backend.db import get_connection

FEATURE_COLUMNS = [
    "rest_days", "is_back_to_back", "games_played_prior_season",
    "min_roll3", "min_roll5", "min_roll10", "min_roll_season", "min_std10",
    "pts_roll3", "pts_roll5", "pts_roll10", "pts_roll_season", "pts_std10",
    "reb_roll10", "ast_roll10", "fg3m_roll10", "stl_roll10", "blk_roll10", "tov_roll10",
    "home_flag",
]
TARGET_COLUMN = "minutes"

SEASON_ORDER = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]

LGB_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "verbose": -1,
}
NUM_BOOST_ROUND = 300


def load_training_frame(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = conn.execute(
        f"""
        SELECT player_id_nba, game_id, game_date, season,
               {', '.join(c for c in FEATURE_COLUMNS if c != 'home_flag')},
               home_away, {TARGET_COLUMN}
        FROM nba_features
        """
    ).fetchdf()
    df["home_flag"] = (df["home_away"] == "home").astype(int)
    df["is_back_to_back"] = df["is_back_to_back"].astype(float)
    return df


def walk_forward_splits(df: pd.DataFrame):
    """Yields (train_df, test_df, test_season) for each season after the first."""
    for i in range(1, len(SEASON_ORDER)):
        train_seasons = SEASON_ORDER[:i]
        test_season = SEASON_ORDER[i]
        train_df = df[df["season"].isin(train_seasons)]
        test_df = df[df["season"] == test_season]
        yield train_df, test_df, test_season


def train_fold(train_df: pd.DataFrame) -> lgb.Booster:
    X = train_df[FEATURE_COLUMNS]
    y = train_df[TARGET_COLUMN]
    dataset = lgb.Dataset(X, label=y)
    return lgb.train(LGB_PARAMS, dataset, num_boost_round=NUM_BOOST_ROUND)


def walk_forward_backtest(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = load_training_frame(conn)
    results = []
    for train_df, test_df, test_season in walk_forward_splits(df):
        if train_df.empty or test_df.empty:
            continue
        model = train_fold(train_df)
        preds = model.predict(test_df[FEATURE_COLUMNS])
        fold = test_df[["player_id_nba", "game_id", "game_date", "season", TARGET_COLUMN]].copy()
        fold["predicted_minutes"] = preds
        fold["abs_error"] = (fold["predicted_minutes"] - fold[TARGET_COLUMN]).abs()
        results.append(fold)
    return pd.concat(results, ignore_index=True)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, g in results.groupby("season"):
        mae = g["abs_error"].mean()
        rmse = np.sqrt((g["abs_error"] ** 2).mean())
        # naive baseline: "predict last known rolling-3 average" — the model should beat this
        rows.append({"test_season": season, "n": len(g), "mae": mae, "rmse": rmse})
    overall_mae = results["abs_error"].mean()
    overall_rmse = np.sqrt((results["abs_error"] ** 2).mean())
    rows.append({"test_season": "ALL", "n": len(results), "mae": overall_mae, "rmse": overall_rmse})
    return pd.DataFrame(rows)


def train_final_model(conn: duckdb.DuckDBPyConnection) -> lgb.Booster:
    """Trains on every ingested season — for scoring upcoming games, not for backtest metrics."""
    df = load_training_frame(conn)
    return train_fold(df)


if __name__ == "__main__":
    conn = get_connection()
    try:
        results = walk_forward_backtest(conn)
        summary = summarize(results)
        print(summary.to_string(index=False))

        # naive baseline for context: predicting each game's min_roll3 directly (no model at all)
        df = load_training_frame(conn)
        baseline = df.dropna(subset=["min_roll3", TARGET_COLUMN])
        baseline_mae = (baseline["min_roll3"] - baseline[TARGET_COLUMN]).abs().mean()
        print(f"\nnaive baseline (predict min_roll3 directly, no model): MAE = {baseline_mae:.3f}")
    finally:
        conn.close()
