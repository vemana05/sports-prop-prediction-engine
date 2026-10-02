"""Per-market isotonic calibrator mapping the model's raw P(over line) to
a calibrated probability — spec Section 2's pipeline: "Probability
distribution -> P(over line) -> Calibration (isotonic) -> Correlation-aware
Monte Carlo". `leg_pool.py` / `monte_carlo.py` were skipping this step
entirely: every leg's raw model probability went straight into shrinkage
and the simulator, uncalibrated. `synthetic_line_calibration.py` already
measured how miscalibrated the raw probabilities are (points: 0.0729 mean
abs calibration error before isotonic, 0.0201 after — spec Section 5.8),
but that measurement was never turned into an artifact the live pipeline
actually applies.

Fit genuinely out-of-sample, same discipline as
`synthetic_line_calibration._fit_walk_forward_isotonic`: pool every
walk-forward fold's held-out predictions (each already out-of-sample
relative to the model that scored it) into ONE isotonic regression per
market. There's no data truly out-of-sample for the FINAL model (it's
trained on every ingested season) — pooling walk-forward OOS folds is the
same proxy the backtest already relies on to report "before vs. after".
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Callable

import duckdb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from backend.backtest.synthetic_line_calibration import make_synthetic_line

CACHE_DIR = Path("data/models")
MAX_AGE_DAYS = 7  # matches model_cache's weekly-retrain cadence

_MARKET_SPECS = {
    "points": dict(target_column="pts", line_source_column="pts_roll10", extra_col="sigma"),
    "rebounds": dict(target_column="reb", line_source_column="reb_roll10", extra_col="dispersion_r"),
    "assists": dict(target_column="ast", line_source_column="ast_roll10", extra_col="dispersion_r"),
    "fg3m": dict(target_column="fg3m", line_source_column="fg3m_roll10", extra_col="dispersion_r"),
}


def _walk_forward_results(conn: duckdb.DuckDBPyConnection, market: str) -> tuple[pd.DataFrame, Callable]:
    if market == "points":
        from backend.models.nba_points_model import p_over, walk_forward_backtest

        return walk_forward_backtest(conn), p_over
    from backend.models import nba_market_model as mm

    spec = _MARKET_SPECS[market]
    return mm.walk_forward_backtest(conn, spec["target_column"], spec["line_source_column"]), mm.p_over_nb


def fit_calibrator(conn: duckdb.DuckDBPyConnection, market: str) -> IsotonicRegression:
    """One isotonic regression per market, fit on every walk-forward fold's
    pooled out-of-sample predictions against a synthetic line (same
    construction `synthetic_line_calibration.py` already uses to measure
    calibration error) — the calibrator applied to the FINAL model's live
    predictions going forward."""
    if market not in _MARKET_SPECS:
        raise ValueError(f"unknown market {market!r}, expected one of {list(_MARKET_SPECS)}")
    spec = _MARKET_SPECS[market]
    results, p_over_fn = _walk_forward_results(conn, market)

    df = results.dropna(subset=[spec["line_source_column"], "predicted_mean", spec["extra_col"]]).copy()
    line = make_synthetic_line(df[spec["line_source_column"]])
    raw_p_over = p_over_fn(df["predicted_mean"].values, df[spec["extra_col"]].values, line.values)
    actual_over = (df[spec["target_column"]] > line).astype(int).values

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_p_over, actual_over)
    return iso


def get_or_fit_calibrator(
    conn: duckdb.DuckDBPyConnection, market: str, max_age_days: int = MAX_AGE_DAYS
) -> IsotonicRegression:
    """Loads a cached calibrator fit within the last `max_age_days` (same
    weekly-retrain cadence as `model_cache.get_or_train_booster`), or fits
    and caches a fresh one."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{market}_calibration.pkl"
    if path.exists():
        age_days = (time.time() - path.stat().st_mtime) / 86400
        if age_days <= max_age_days:
            return pickle.loads(path.read_bytes())

    iso = fit_calibrator(conn, market)
    path.write_bytes(pickle.dumps(iso))
    return iso


def apply_calibration(calibrator: IsotonicRegression | None, raw_p_over: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return raw_p_over
    return calibrator.predict(raw_p_over)
