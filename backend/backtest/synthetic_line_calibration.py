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
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_synthetic_line(rolling_average: pd.Series) -> pd.Series:
    """Round a rolling average to the nearest .5 — a simple, defensible
    stand-in for what a sportsbook might post as a line. Rounding to .5
    (never a whole number) means a player's actual integer count can never
    exactly equal the line, so there's no push to handle."""
    return (rolling_average * 2).round() / 2


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
) -> pd.DataFrame:
    """`p_over_fn(predicted_mean, extra, line) -> P(actual > line)` —
    `extra` is whatever the distribution needs beyond the mean (sigma for
    Normal, the dispersion parameter r for Negative Binomial). Buckets
    predictions into `n_buckets` groups by predicted probability and
    reports predicted vs. actual hit rate per bucket, plus saves a
    calibration chart to `chart_path`.
    """
    df = results.dropna(subset=[line_source_column, predicted_mean_col, extra_col]).copy()
    df["synthetic_line"] = make_synthetic_line(df[line_source_column])
    df["predicted_p_over"] = p_over_fn(
        df[predicted_mean_col].values, df[extra_col].values, df["synthetic_line"].values
    )
    df["actual_over"] = (df[target_column] > df["synthetic_line"]).astype(int)

    df["bucket"] = pd.qcut(df["predicted_p_over"], n_buckets, duplicates="drop")
    grouped = (
        df.groupby("bucket", observed=True)
        .agg(predicted=("predicted_p_over", "mean"), actual=("actual_over", "mean"), n=("actual_over", "size"))
        .reset_index(drop=True)
        .sort_values("predicted")
        .reset_index(drop=True)
    )

    _save_chart(grouped, market_name, chart_path)
    return grouped


def _save_chart(grouped: pd.DataFrame, market_name: str, chart_path: str) -> None:
    Path(chart_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    sizes = grouped["n"] / grouped["n"].max() * 200 + 20
    ax.scatter(grouped["predicted"], grouped["actual"], s=sizes, alpha=0.8)
    for _, row in grouped.iterrows():
        ax.annotate(f"n={int(row['n'])}", (row["predicted"], row["actual"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Predicted P(over synthetic line)")
    ax.set_ylabel("Actual hit rate")
    ax.set_title(f"{market_name.title()} — synthetic-line calibration (out-of-sample)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
