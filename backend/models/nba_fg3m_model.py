"""3-pointers-made model — Negative Binomial distribution (spec Section
5.3). Thin wrapper around backend.models.nba_market_model; see that
module's docstring for the shared walk-forward/LightGBM/dispersion
methodology.
"""

from __future__ import annotations

from backend.backtest.synthetic_line_calibration import run_synthetic_line_calibration
from backend.db import get_connection
from backend.models import nba_market_model as mm

TARGET_COLUMN = "fg3m"
LINE_SOURCE_COLUMN = "fg3m_roll10"

if __name__ == "__main__":
    conn = get_connection()
    try:
        results = mm.walk_forward_backtest(conn, TARGET_COLUMN, LINE_SOURCE_COLUMN)

        print("Naive baseline vs. model:")
        baselines = mm.naive_baselines(results, TARGET_COLUMN, LINE_SOURCE_COLUMN, None)
        print(f"  last_10_avg: MAE {baselines['last_10_avg']['mae']:.3f} (n={baselines['last_10_avg']['n']})")
        model_mae = (results["predicted_mean"] - results[TARGET_COLUMN]).abs().mean()
        print(f"  model:       MAE {model_mae:.3f} (n={len(results)})")

        print("\nFree synthetic-line calibration, before vs. after isotonic (line = player's own 10-game rolling average, rounded to .5):")
        calib = run_synthetic_line_calibration(
            results, target_column=TARGET_COLUMN, line_source_column=LINE_SOURCE_COLUMN,
            predicted_mean_col="predicted_mean", extra_col="dispersion_r",
            p_over_fn=mm.p_over_nb, market_name="3-pointers made", chart_path="reports/calibration_fg3m.png",
        )
        print("Before (raw):")
        print(calib["raw"].to_string(index=False))
        print("After (isotonic):")
        print(calib["calibrated"].to_string(index=False))
        print(f"Mean abs calibration error: {calib['raw_mean_abs_calibration_error']:.4f} -> {calib['calibrated_mean_abs_calibration_error']:.4f} (n={calib['n']})")
        print("Chart saved to reports/calibration_fg3m.png")
    finally:
        conn.close()
