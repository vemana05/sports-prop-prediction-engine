"""Free calibration backtest — no odds API needed.

For each held-out (walk-forward) game, generate a synthetic prop line from
the player's own pre-game rolling average (10-game, the same `*_roll10`
feature already in `nba_features`), rounded to the nearest .5 so there's
never a push. Check whether the model's predicted P(over that line)
matches the actual hit rate, out of sample.

This is deliberately NOT the same thing as comparing to a real sportsbook
line (spec Section 5.4) — a synthetic line built from recent form is a
plausible stand-in, not a market price, so this doesn't tell us whether the
model has real edge. What it does tell us, for free, before any odds
account exists: are the model's own probabilities honest? If predicted 70%
isn't actually hitting ~70% of the time, no amount of real-line comparison
later will fix that — the probability layer itself is broken.

Isotonic calibration (spec Section 5.4): fit on out-of-sample predictions
and applied to get a second, calibrated probability — but fit genuinely
out-of-sample, not just on the same held-out set it's then evaluated
against. `results` carries each row's walk-forward test `season`; the
isotonic mapping for season N's rows is fit ONLY on season-N-1-and-earlier
rows (themselves already out-of-sample relative to the model that scored
them). The very first tested season has no earlier calibration data, so
it's left uncalibrated (reported as-is) rather than faked. Fitting and
evaluating calibration on the exact same pooled set — which would make
"calibration improves accuracy" a bit circular — is exactly what this
avoids.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

MIN_CALIBRATION_FIT_ROWS = 500  # below this, a season's isotonic fit is too noisy to trust


def make_synthetic_line(rolling_average: pd.Series) -> pd.Series:
    """Round a rolling average to the nearest .5 — a simple, defensible
    stand-in for what a sportsbook might post as a line. Rounding to .5
    (never a whole number) means a player's actual integer count can never
    exactly equal the line, so there's no push to handle."""
    return (rolling_average * 2).round() / 2


def _fit_walk_forward_isotonic(df: pd.DataFrame, p_col: str, actual_col: str) -> np.ndarray:
    """Returns calibrated probabilities, one per row, each produced by an
    isotonic model fit only on seasons strictly before that row's own
    season. A season with no (or too little) prior data falls back to its
    raw probability rather than a calibration fit on too few points."""
    seasons = sorted(df["season"].unique())
    calibrated = df[p_col].to_numpy(copy=True)
    for i, season in enumerate(seasons):
        train_mask = df["season"].isin(seasons[:i])
        test_mask = (df["season"] == season).to_numpy()
        if train_mask.sum() < MIN_CALIBRATION_FIT_ROWS:
            continue  # leave this season's rows as raw (already copied above)
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(df.loc[train_mask, p_col], df.loc[train_mask, actual_col])
        calibrated[test_mask] = iso.predict(df.loc[test_mask, p_col])
    return calibrated


def _bucket(df: pd.DataFrame, p_col: str, actual_col: str, n_buckets: int) -> pd.DataFrame:
    bucket = pd.qcut(df[p_col], n_buckets, duplicates="drop")
    return (
        df.groupby(bucket, observed=True)
        .agg(predicted=(p_col, "mean"), actual=(actual_col, "mean"), n=(actual_col, "size"))
        .reset_index(drop=True)
        .sort_values("predicted")
        .reset_index(drop=True)
    )


def run_synthetic_line_calibration(
    results: pd.DataFrame,
    target_column: str,
    line_source_column: str,
    p_over_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    market_name: str,
    chart_path: str,
    predicted_mean_col: str = "predicted_mean",
    extra_col: str = "sigma",
    n_buckets: int = 10,
    p_over_frame_fn: Callable[[pd.DataFrame, np.ndarray], np.ndarray] | None = None,
) -> dict:
    """`p_over_fn(predicted_mean, extra, line) -> P(actual > line)` —
    `extra` is whatever the distribution needs beyond the mean (sigma for
    Normal, the dispersion parameter r for Negative Binomial). Returns a
    dict with `raw` and `calibrated` bucketed tables (predicted vs. actual
    hit rate per bucket) and summary calibration-error numbers; saves a
    before/after chart to `chart_path`.

    `p_over_frame_fn(rows, line)`, if given, replaces `p_over_fn` for
    distributions that need more than one column per row (NFL yards use a
    per-fold empirical residual distribution — see nfl_market_model.py).
    """
    df = results.dropna(subset=[line_source_column, predicted_mean_col, extra_col]).copy()
    df["synthetic_line"] = make_synthetic_line(df[line_source_column])
    if p_over_frame_fn is not None:
        df["predicted_p_over"] = p_over_frame_fn(df, df["synthetic_line"].values)
    else:
        df["predicted_p_over"] = p_over_fn(
            df[predicted_mean_col].values, df[extra_col].values, df["synthetic_line"].values
        )
    df["actual_over"] = (df[target_column] > df["synthetic_line"]).astype(int)
    df["calibrated_p_over"] = _fit_walk_forward_isotonic(df, "predicted_p_over", "actual_over")

    raw_bucketed = _bucket(df, "predicted_p_over", "actual_over", n_buckets)
    calibrated_bucketed = _bucket(df, "calibrated_p_over", "actual_over", n_buckets)

    raw_error = (raw_bucketed["predicted"] - raw_bucketed["actual"]).abs().mean()
    calibrated_error = (calibrated_bucketed["predicted"] - calibrated_bucketed["actual"]).abs().mean()

    _save_chart(raw_bucketed, calibrated_bucketed, market_name, chart_path)

    return {
        "raw": raw_bucketed,
        "calibrated": calibrated_bucketed,
        "raw_mean_abs_calibration_error": raw_error,
        "calibrated_mean_abs_calibration_error": calibrated_error,
        "n": len(df),
    }


def _save_chart(raw: pd.DataFrame, calibrated: pd.DataFrame, market_name: str, chart_path: str) -> None:
    Path(chart_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")

    raw_sizes = raw["n"] / raw["n"].max() * 160 + 20
    ax.scatter(raw["predicted"], raw["actual"], s=raw_sizes, alpha=0.55, color="tab:red", label="before (raw)")

    cal_sizes = calibrated["n"] / calibrated["n"].max() * 160 + 20
    ax.scatter(calibrated["predicted"], calibrated["actual"], s=cal_sizes, alpha=0.85, color="tab:blue", label="after (isotonic)")

    ax.set_xlabel("Predicted P(over synthetic line)")
    ax.set_ylabel("Actual hit rate")
    ax.set_title(f"{market_name[:1].upper() + market_name[1:]} — synthetic-line calibration, before vs. after")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
