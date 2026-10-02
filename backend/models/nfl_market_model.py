"""NFL per-market models (spec Section 5.3, Phase 6): passing yards,
rushing yards, receiving yards, receptions.

Walk-forward by season, never random splits (Section 5.4): each test
season is predicted by a model trained on every earlier season, 2020 on.
Test seasons are 2022-2026; 2026 is partial (the weeks played so far).

Each market trains only on its own pre-game universe (`in_*_market` in
nfl_features: starting QBs for passing, RB1/RB2 and starting QBs for
rushing, WR1-3 / TE1 / RB1 or anyone already seeing targets for receiving).
These are the players books actually post lines for.

Shrinkage toward position averages comes in through the features: each
target has a `*_shrunk` average pulled toward last season's mean for the
same position and depth rank (see nfl_features.py), so a rookie or a
newly promoted WR2 starts near what that role usually produces.

Distributions (P(over line)):
- Yards: the spec suggests quantile regression for skewed yardage. This
  uses a lighter equivalent: a spread that grows with the predicted mean,
  sigma(mu) = a + b*mu, and the empirical distribution of standardized
  residuals z = (y - mu) / sigma(mu). That keeps the real right skew (long
  touchdowns) without assuming a Normal shape.
- Receptions: Negative Binomial (Section 5.3 "counts").
Both are fit on OUT-OF-FOLD residuals within the training seasons (inner
leave-one-season-out), not in-sample residuals. In-sample residuals from a
boosted model are too small and make every probability overconfident.

    python -m backend.models.nfl_market_model
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

from backend.features.nfl_features import ROLL_STATS, TARGETS
from backend.nfl_db import get_nfl_connection

TEST_SEASONS = [2022, 2023, 2024, 2025, 2026]
REPORT_PATH = Path("reports/nfl_model_performance.json")

FEATURE_COLUMNS = (
    [f"{s}_roll{w}" for s in ROLL_STATS for w in ("3", "5", "_season")]
    + [f"{t}_std8" for t in TARGETS]
    + [f"{t}_shrunk" for t in TARGETS]
    + [f"prior_season_{t}" for t in TARGETS + ["targets", "carries", "attempts"]]
    + [
        "prior_season_games", "career_games_prior", "games_last5", "depth_rank",
        "opp_allowed_pass_yds_roll6", "opp_allowed_rush_yds_roll6", "opp_allowed_rec_yds_roll6", "opp_allowed_rec_roll6",
        "team_attempts_roll5", "team_carries_roll5",
        "team_out_target_share", "team_out_carry_share", "team_starting_qb_out",
        "questionable_flag", "no_history_flag",
        "home_flag", "rest_days", "div_game", "playoff_flag", "week",
        "indoor_flag", "grass_flag", "temperature_c", "wind_kph", "precip_mm",
        "pos_code",
    ]
)
POS_CODES = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}

LGB_PARAMS = {
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "verbose": -1,
    "seed": 7,
}
NUM_BOOST_ROUND = 400


@dataclass(frozen=True)
class Market:
    name: str
    label: str
    target: str
    universe: str
    family: str  # "yards" | "count"
    objective: str


MARKETS = [
    Market("passing_yards", "Passing yards", "passing_yards", "in_passing_market", "yards", "regression"),
    Market("rushing_yards", "Rushing yards", "rushing_yards", "in_rushing_market", "yards", "regression"),
    Market("receiving_yards", "Receiving yards", "receiving_yards", "in_receiving_market", "yards", "regression"),
    Market("receptions", "Receptions", "receptions", "in_receiving_market", "count", "poisson"),
]


def load_frame(conn: duckdb.DuckDBPyConnection, market: Market) -> pd.DataFrame:
    # Fixed row order: LightGBM bagging samples by row index, so an unordered
    # table rebuild would otherwise change results from run to run.
    df = conn.execute(f"SELECT * FROM nfl_features WHERE {market.universe} ORDER BY kickoff_utc, game_id, player_id").fetchdf()
    df["pos_code"] = df["position"].map(POS_CODES)
    return df


def _train(train: pd.DataFrame, market: Market) -> lgb.Booster:
    params = {**LGB_PARAMS, "objective": market.objective}
    return lgb.train(params, lgb.Dataset(train[FEATURE_COLUMNS].astype(float), label=train[market.target]),
                     num_boost_round=NUM_BOOST_ROUND)


def _predict(model: lgb.Booster, df: pd.DataFrame) -> np.ndarray:
    return np.clip(model.predict(df[FEATURE_COLUMNS].astype(float)), 0, None)


def _oof_predictions(train: pd.DataFrame, market: Market) -> np.ndarray:
    """Leave-one-season-out predictions inside the training seasons — the
    honest residuals the spread/dispersion is fit on."""
    oof = np.full(len(train), np.nan)
    for season in train["season"].unique():
        inner_test = (train["season"] == season).to_numpy()
        oof[inner_test] = _predict(_train(train[~inner_test], market), train[inner_test])
    return oof


@dataclass
class Spread:
    """Yards: sigma(mu) = a + b*mu plus sorted standardized residuals.
    Counts: Negative Binomial dispersion r."""
    a: float = 0.0
    b: float = 0.0
    z_sorted: np.ndarray | None = None
    r: float | None = None


def fit_spread(y: np.ndarray, mu: np.ndarray, family: str) -> Spread:
    if family == "count":
        mean_mu, mse = float(np.mean(mu)), float(np.mean((y - mu) ** 2))
        excess = mse - mean_mu
        r = float(np.mean(mu ** 2) / excess) if excess > 0 else 1e6
        return Spread(r=max(r, 1.0))
    # E|resid| ~ (a + b*mu) * sqrt(2/pi) for a Normal-ish spread; fit by least squares.
    abs_resid = np.abs(y - mu) * np.sqrt(np.pi / 2)
    b, a = np.polyfit(mu, abs_resid, 1)
    a, b = max(a, 5.0), max(b, 0.0)
    z = (y - mu) / (a + b * mu)
    return Spread(a=a, b=b, z_sorted=np.sort(z))


def p_over(mu: np.ndarray, line: np.ndarray, spread: Spread) -> np.ndarray:
    if spread.r is not None:
        r = spread.r
        p = r / (r + np.clip(mu, 1e-6, None))
        return 1 - stats.nbinom.cdf(np.floor(line).astype(int), r, p)
    z_line = (line - mu) / (spread.a + spread.b * mu)
    below = np.searchsorted(spread.z_sorted, z_line, side="right") / len(spread.z_sorted)
    return 1 - below


def walk_forward(conn: duckdb.DuckDBPyConnection, market: Market) -> tuple[pd.DataFrame, dict[int, Spread]]:
    df = load_frame(conn, market)
    folds, spreads = [], {}
    for test_season in TEST_SEASONS:
        train, test = df[df["season"] < test_season], df[df["season"] == test_season]
        if test.empty:
            continue
        oof = _oof_predictions(train, market)
        spreads[test_season] = fit_spread(train[market.target].to_numpy(float), oof, market.family)
        fold = test[[
            "player_id", "player_name", "game_id", "season", "week", "position", "depth_rank", market.target,
            f"{market.target}_roll5", f"{market.target}_roll_season", f"{market.target}_shrunk",
        ]].copy()
        fold["predicted_mean"] = _predict(_train(train, market), test)
        folds.append(fold)
    return pd.concat(folds, ignore_index=True), spreads


def baseline_report(results: pd.DataFrame, market: Market) -> dict:
    """Model vs. naive baselines on the SAME rows: those where the
    player's own last-5 average exists (i.e. not his first game), so no
    baseline gets to skip the hard cases the model has to predict."""
    t = market.target
    rows = results.dropna(subset=[f"{t}_roll5"])
    y = rows[t]

    def mae(pred):
        return round(float((pred - y).abs().mean()), 3)

    out = {
        "n": len(rows),
        "model_mae": mae(rows["predicted_mean"]),
        "last5_avg_mae": mae(rows[f"{t}_roll5"]),
        "season_avg_mae": mae(rows[f"{t}_roll_season"].fillna(rows[f"{t}_roll5"])),
        "shrunk_last5_mae": mae(rows[f"{t}_shrunk"]),
        "by_season": {},
        "by_position": {},
    }
    for season, g in rows.groupby("season"):
        out["by_season"][int(season)] = {
            "n": len(g),
            "model_mae": round(float((g["predicted_mean"] - g[t]).abs().mean()), 3),
            "last5_avg_mae": round(float((g[f"{t}_roll5"] - g[t]).abs().mean()), 3),
        }
    for pos, g in rows.groupby("position"):
        if len(g) < 200:
            continue
        out["by_position"][pos] = {
            "n": len(g),
            "model_mae": round(float((g["predicted_mean"] - g[t]).abs().mean()), 3),
            "last5_avg_mae": round(float((g[f"{t}_roll5"] - g[t]).abs().mean()), 3),
        }
    # First-game rows: no own history, so only the position prior and the model can say anything.
    first = results[results[f"{t}_roll5"].isna()]
    if len(first):
        out["no_history_rows"] = {
            "n": len(first),
            "model_mae": round(float((first["predicted_mean"] - first[t]).abs().mean()), 3),
            "position_prior_mae": round(float((first[f"{t}_shrunk"] - first[t]).abs().mean()), 3),
        }
    return out


def calibration_report(results: pd.DataFrame, spreads: dict[int, Spread], market: Market) -> dict:
    from backend.backtest.synthetic_line_calibration import run_synthetic_line_calibration

    def p_over_frame(rows: pd.DataFrame, line: np.ndarray) -> np.ndarray:
        out = np.empty(len(rows))
        for season, spread in spreads.items():
            m = (rows["season"] == season).to_numpy()
            out[m] = p_over(rows["predicted_mean"].to_numpy()[m], line[m], spread)
        return out

    calib = run_synthetic_line_calibration(
        results.assign(_extra=1.0), target_column=market.target, line_source_column=f"{market.target}_roll5",
        p_over_fn=None, market_name=f"NFL {market.label.lower()}", chart_path=f"reports/calibration_nfl_{market.name}.png",
        extra_col="_extra", p_over_frame_fn=p_over_frame,
    )
    return {
        "n": int(calib["n"]),
        "raw_mean_abs_calibration_error": round(float(calib["raw_mean_abs_calibration_error"]), 4),
        "calibrated_mean_abs_calibration_error": round(float(calib["calibrated_mean_abs_calibration_error"]), 4),
        "raw_buckets": calib["raw"].round(3).to_dict(orient="records"),
        "chart": f"reports/calibration_nfl_{market.name}.png",
    }


def run_all() -> dict:
    conn = get_nfl_connection(read_only=True)
    try:
        report = {"test_seasons": TEST_SEASONS, "markets": {}}
        for market in MARKETS:
            results, spreads = walk_forward(conn, market)
            report["markets"][market.name] = {
                "label": market.label,
                "baselines": baseline_report(results, market),
                "calibration": calibration_report(results, spreads, market),
            }
            b = report["markets"][market.name]["baselines"]
            print(f"{market.label:16s} n={b['n']:5d}  model {b['model_mae']:7.3f}  last5 {b['last5_avg_mae']:7.3f}  "
                  f"season {b['season_avg_mae']:7.3f}  shrunk {b['shrunk_last5_mae']:7.3f}")
    finally:
        conn.close()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=1, default=str))
    return report


if __name__ == "__main__":
    run_all()
