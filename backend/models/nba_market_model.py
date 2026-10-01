"""Shared machinery for NBA per-market count-stat models — rebounds,
assists, 3PM (spec Section 5.3: "Counts (reb, ast, 3PM, receptions, TDs):
Negative Binomial"). Points uses a different distribution (Normal) and
lives in `nba_points_model.py`, since the two families need different
math; everything else (feature set, walk-forward split, LightGBM training)
is identical and lives here once, so `nba_rebounds_model.py` /
`nba_assists_model.py` / `nba_fg3m_model.py` are thin wrappers.

Dispersion is estimated globally per walk-forward fold, not truly
per-player (the spec says "player-specific variance"): one shared r from
that fold's training-set mean/variance via method of moments. True
per-player dispersion would need enough games per player to estimate a
second moment reliably, which most players in a single season don't have
— global-per-fold is the defensible simplification until that changes.
"""

from __future__ import annotations

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

from backend.models.nba_minutes_model import FEATURE_COLUMNS, walk_forward_splits

LGB_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "verbose": -1,
}
NUM_BOOST_ROUND = 300


def load_training_frame(conn: duckdb.DuckDBPyConnection, target_column: str, line_source_column: str) -> pd.DataFrame:
    extra_cols = {target_column, line_source_column} - set(FEATURE_COLUMNS) - {"home_away"}
    base_cols = [c for c in FEATURE_COLUMNS if c != "home_flag"]
    select_cols = base_cols + sorted(extra_cols)
    df = conn.execute(
        f"""
        SELECT player_id_nba, game_id, game_date, season, home_away,
               {', '.join(select_cols)}
        FROM nba_features
        """
    ).fetchdf()
    df["home_flag"] = (df["home_away"] == "home").astype(int)
    df["is_back_to_back"] = df["is_back_to_back"].astype(float)
    return df


def estimate_nb_dispersion(y_train: np.ndarray, predicted_train: np.ndarray) -> float:
    """Method-of-moments global dispersion r for NegBinom(mean=mu, var=mu + mu^2/r),
    using the fold's own training actuals/predictions — never test data."""
    mu_bar = float(np.mean(predicted_train))
    var_actual = float(np.var(y_train))
    if mu_bar <= 0 or var_actual <= mu_bar:
        return 1e6  # no evidence of overdispersion vs. Poisson -> effectively Poisson
    r = mu_bar ** 2 / (var_actual - mu_bar)
    return max(r, 1.0)  # floor so a noisy fold can't produce a degenerate near-zero r


def p_over_nb(predicted_mean: np.ndarray, r: np.ndarray, line: np.ndarray) -> np.ndarray:
    """P(actual > line) under NegativeBinomial(mean=predicted_mean, dispersion=r).
    `line` is always a .5 value (see synthetic_line_calibration.make_synthetic_line),
    so there's never a push to handle; actual is a non-negative integer count."""
    mu = np.clip(predicted_mean, 1e-6, None)
    p = r / (r + mu)
    k = np.floor(line).astype(int)
    return 1 - stats.nbinom.cdf(k, r, p)


def train_final_model(conn: duckdb.DuckDBPyConnection, target_column: str, line_source_column: str) -> tuple[lgb.Booster, float]:
    """Trains on every ingested season — for scoring upcoming games, not
    backtest metrics. Returns (model, dispersion_r) from this full
    training set (in-sample, same simplification each walk-forward fold
    uses)."""
    df = load_training_frame(conn, target_column, line_source_column)
    X, y = df[FEATURE_COLUMNS], df[target_column]
    model = lgb.train(LGB_PARAMS, lgb.Dataset(X, label=y), num_boost_round=NUM_BOOST_ROUND)
    r = estimate_nb_dispersion(y.values, model.predict(X))
    return model, r


def walk_forward_backtest(conn: duckdb.DuckDBPyConnection, target_column: str, line_source_column: str) -> pd.DataFrame:
    df = load_training_frame(conn, target_column, line_source_column)
    results = []
    for train_df, test_df, test_season in walk_forward_splits(df):
        if train_df.empty or test_df.empty:
            continue
        X_train, y_train = train_df[FEATURE_COLUMNS], train_df[target_column]
        model = lgb.train(LGB_PARAMS, lgb.Dataset(X_train, label=y_train), num_boost_round=NUM_BOOST_ROUND)

        train_preds = model.predict(X_train)
        r = estimate_nb_dispersion(y_train.values, train_preds)

        preds = model.predict(test_df[FEATURE_COLUMNS])
        fold = test_df[["player_id_nba", "game_id", "game_date", "season", target_column, line_source_column]].copy()
        fold["predicted_mean"] = np.clip(preds, 0, None)
        fold["dispersion_r"] = r
        results.append(fold)
    return pd.concat(results, ignore_index=True)


def naive_baselines(df: pd.DataFrame, target_column: str, roll10_col: str, roll_season_col: str | None) -> dict:
    """MAE of 'just predict the rolling average, no model at all' — what
    the model needs to beat to be worth anything."""
    out = {}
    valid10 = df.dropna(subset=[roll10_col, target_column])
    out["last_10_avg"] = {"mae": (valid10[roll10_col] - valid10[target_column]).abs().mean(), "n": len(valid10)}
    if roll_season_col and roll_season_col in df.columns:
        valid_season = df.dropna(subset=[roll_season_col, target_column])
        out["season_avg"] = {"mae": (valid_season[roll_season_col] - valid_season[target_column]).abs().mean(), "n": len(valid_season)}
    return out
