"""Does weekly walk-forward retraining help? Compares, on the weeks played
so far this season, the base model (thru last season) against the weekly
model the live page would use for each week (thru week N-1). Same rows,
same features; week 1 is identical by construction (no weekly model yet).

Metrics per market and week: MAE of the projected mean, and coverage of
the 25th-75th range (should be near 50%) with its mean width.

Trains any missing weekly versions (saved under data/models and reused by
serving), so the first run takes a few minutes per week.

    python -m backend.backtest.nfl_weekly_eval --season 2026 --weeks 1-4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from backend.models.nfl_market_model import MARKETS, POS_CODES, _predict
from backend.predict import nfl_projections as proj
from backend.predict.nfl_projections import Q_HI, Q_LO, Through, quantiles

REPORT_PATH = Path("reports/nfl_weekly_retrain_eval.json")


def _metrics(df: pd.DataFrame, target: str, mu: np.ndarray, spread) -> dict:
    y = df[target].to_numpy(float)
    lo, hi = quantiles(mu, spread, Q_LO), quantiles(mu, spread, Q_HI)
    return {
        "mae": round(float(np.mean(np.abs(mu - y))), 3),
        "coverage": round(float(np.mean((y >= lo) & (y <= hi))), 3),
        "width": round(float(np.mean(hi - lo)), 2),
    }


def evaluate(season: int, weeks: list[int]) -> dict:
    conn = duckdb.connect(proj._db_path(), read_only=True)
    try:
        feats = conn.execute("SELECT * FROM nfl_features WHERE season = ? AND week IN (SELECT UNNEST(?::INT[]))", [season, weeks]).fetchdf()
    finally:
        conn.close()
    feats["pos_code"] = feats["position"].map(POS_CODES)

    report: dict = {"season": season, "weeks": weeks, "markets": {}}
    for m in MARKETS:
        rows, pooled = [], {"base": [], "weekly": []}
        for w in weeks:
            df = feats[(feats["week"] == w) & feats[m.universe].astype("boolean").fillna(False).to_numpy(bool)].dropna(subset=[m.target])
            if df.empty:
                continue
            base_v, weekly_v = Through(season - 1), proj.target_version(season, w)
            out = {"week": w, "n": len(df), "weekly_version": weekly_v.tag}
            for name, v in (("base", base_v), ("weekly", weekly_v)):
                model, spread = proj._get_or_train(m, v)
                mu = _predict(model, df)
                out[name] = _metrics(df, m.target, mu, spread)
                pooled[name].append((df, mu, spread))
            rows.append(out)
        total = {}
        for name, parts in pooled.items():
            y = np.concatenate([d[m.target].to_numpy(float) for d, _, _ in parts])
            mu = np.concatenate([p for _, p, _ in parts])
            lo = np.concatenate([quantiles(p, s, Q_LO) for _, p, s in parts])
            hi = np.concatenate([quantiles(p, s, Q_HI) for _, p, s in parts])
            total[name] = {"mae": round(float(np.mean(np.abs(mu - y))), 3),
                           "coverage": round(float(np.mean((y >= lo) & (y <= hi))), 3),
                           "width": round(float(np.mean(hi - lo)), 2)}
        report["markets"][m.name] = {"by_week": rows, "all": {"n": sum(r["n"] for r in rows), **total}}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--season", type=int, default=proj.current_season())
    parser.add_argument("--weeks", default="1-4")
    args = parser.parse_args()
    a, _, b = args.weeks.partition("-")
    weeks = list(range(int(a), int(b or a) + 1))
    report = evaluate(args.season, weeks)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=1))
    print(f"{'market':16s} {'n':>5s}  {'MAE base':>9s} {'weekly':>7s}   {'cov base':>8s} {'weekly':>7s}   {'width base':>10s} {'weekly':>7s}")
    for name, r in report["markets"].items():
        t = r["all"]
        print(f"{name:16s} {t['n']:5d}  {t['base']['mae']:9.2f} {t['weekly']['mae']:7.2f}   "
              f"{t['base']['coverage']:8.3f} {t['weekly']['coverage']:7.3f}   {t['base']['width']:10.1f} {t['weekly']['width']:7.1f}")
        for row in r["by_week"]:
            print(f"   w{row['week']:<2d} {row['weekly_version']:13s} n={row['n']:4d}  MAE {row['base']['mae']:7.2f} -> {row['weekly']['mae']:7.2f}"
                  f"   cov {row['base']['coverage']:.3f} -> {row['weekly']['coverage']:.3f}")


if __name__ == "__main__":
    main()
