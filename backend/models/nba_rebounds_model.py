"""Rebounds model — Negative Binomial distribution (spec Section 5.3).
Thin wrapper around backend.models.nba_market_model; see that module's
docstring for the shared walk-forward/LightGBM/dispersion methodology.
"""

from __future__ import annotations

from backend.backtest.synthetic_line_calibration import run_synthetic_line_calibration
from backend.db import get_connection
from backend.models import nba_market_model as mm

TARGET_COLUMN = "reb"
LINE_SOURCE_COLUMN = "reb_roll10"

if __name__ == "__main__":
    conn = get_connection()
    try:
        results = mm.walk_forward_backtest(conn, TARGET_COLUMN, LINE_SOURCE_COLUMN)

        print("Naive baseline vs. model:")
        baselines = mm.naive_baselines(results, TARGET_COLUMN, LINE_SOURCE_COLUMN, None)
        print(f"  last_10_avg: MAE {baselines['last_10_avg']['mae']:.3f} (n={baselines['last_10_avg']['n']})")
        model_mae = (results["predicted_mean"] - results[TARGET_COLUMN]).abs().mean()
        print(f"  model:       MAE {model_mae:.3f} (n={len(results)})")

        print("\nFree synthetic-line calibration (line = player's own 10-game rolling average, rounded to .5):")
        calib = run_synthetic_line_calibration(
            results, target_column=TARGET_COLUMN, line_source_column=LINE_SOURCE_COLUMN,
            predicted_mean_col="predicted_mean", extra_col="dispersion_r",
            p_over_fn=mm.p_over_nb, market_name="rebounds", chart_path="reports/calibration_rebounds.png",
        )
        print(calib.to_string(index=False))
        print("Chart saved to reports/calibration_rebounds.png")
    finally:
        conn.close()
