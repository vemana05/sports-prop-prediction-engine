"""Replay page (GET /api/replay): how the model did on a past NBA date or
NFL week, from out-of-sample predictions only.

    NBA  nba_walk_forward_residuals (parlay/residuals.py): each season is
         predicted by a model trained on earlier seasons; range from that
         fold's Normal sigma (points) or NB r (other markets).
    NFL  nfl_walk_forward_replay (backtest/nfl_replay.py): each season is
         predicted by models trained on earlier seasons only.

Everything is compared against actual results, never sportsbook lines,
and nothing here calls The Odds API.
"""

from __future__ import annotations

from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd

from backend.models.nfl_market_model import MARKETS as NFL_MARKETS
from backend.parlay.residuals import RESIDUALS_TABLE
from backend.predict import nfl_projections, schedule_projections
from backend.backtest.nfl_replay import REPLAY_TABLE

NBA_MARKETS = {"points": "Points", "rebounds": "Rebounds", "assists": "Assists", "fg3m": "3-pointers made"}
NFL_LABELS = {m.name: m.label for m in NFL_MARKETS}
NOTE = (
    "This compares projections against actual results, not against sportsbook lines. "
    "Beating a recent-average baseline doesn't show the model beats the market, so it does not prove betting edge."
)


class NotBuilt(Exception):
    pass


def summarize(df: pd.DataFrame) -> dict | None:
    """Model vs. naive MAE on the same rows (where the baseline exists),
    range coverage, and bias (projection minus actual: positive means the
    model projected too high)."""
    if df.empty:
        return None
    miss = df["predicted"] - df["actual"]
    with_naive = df.dropna(subset=["naive"])
    return {
        "n": len(df),
        "model_mae": float(miss.abs().mean()),
        "naive_mae": float((with_naive["naive"] - with_naive["actual"]).abs().mean()) if len(with_naive) else None,
        "model_mae_same_rows": float((with_naive["predicted"] - with_naive["actual"]).abs().mean()) if len(with_naive) else None,
        "n_naive": len(with_naive),
        "coverage": float(((df["actual"] >= df["lo"]) & (df["actual"] <= df["hi"])).mean()),
        "bias": float(miss.mean()),
    }


def _rows(df: pd.DataFrame) -> list[dict]:
    out = df.assign(miss=df["actual"] - df["predicted"], inside=(df["actual"] >= df["lo"]) & (df["actual"] <= df["hi"]))
    cols = ["player_name", "team", "opponent", "predicted", "lo", "hi", "actual", "miss", "inside", "naive"]
    return [
        {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()}
        for r in out[cols].astype(object).to_dict("records")
    ]


# ---- NBA --------------------------------------------------------------------

def _nba_frame(conn: duckdb.DuckDBPyConnection, market: str, start: date, end: date) -> pd.DataFrame:
    cols = {c[0] for c in conn.execute(f"DESCRIBE {RESIDUALS_TABLE}").fetchall()}
    if "dist" not in cols:
        raise NotBuilt("NBA replay needs the residuals table rebuilt: python -m backend.parlay.residuals --force")
    df = conn.execute(
        f"""SELECT g.game_date, g.player_name, r.team, g.opponent, r.actual, r.predicted, r.naive, r.dist, r.dispersion
            FROM {RESIDUALS_TABLE} r JOIN nba_player_games g USING (player_id_nba, game_id)
            WHERE r.market = ? AND g.game_date BETWEEN ? AND ? AND r.dist IS NOT NULL""",
        [market, start, end],
    ).fetchdf()
    mean = df["predicted"].to_numpy(float)
    lo, hi = np.empty(len(df)), np.empty(len(df))
    normal = (df["dist"] == "normal").to_numpy()
    if normal.any():
        lo[normal], hi[normal] = schedule_projections._normal_range(mean[normal], df["dispersion"].to_numpy(float)[normal])
    if (~normal).any():
        lo[~normal], hi[~normal] = schedule_projections._nb_range(mean[~normal], df["dispersion"].to_numpy(float)[~normal])
    return df.assign(lo=lo, hi=hi)


def nba(market: str, day: date | None, weeks: int) -> dict:
    conn = schedule_projections._connect()
    try:
        try:
            dates = [d for (d,) in conn.execute(
                f"""SELECT DISTINCT g.game_date FROM {RESIDUALS_TABLE} r JOIN nba_player_games g USING (player_id_nba, game_id)
                    WHERE r.market = ? ORDER BY 1 DESC""", [market]).fetchall()]
        except duckdb.CatalogException:
            raise NotBuilt("NBA replay needs the residuals table: python -m backend.parlay.residuals")
        if not dates:
            raise NotBuilt("No NBA walk-forward rows for this market.")
        # Snap a non-game day to the closest earlier game day.
        day = dates[0] if day is None else next((d for d in dates if d <= day), dates[-1])

        # Rolling: the last N calendar weeks (Mon-Sun) that had games, ending with the chosen day's week.
        monday = lambda d: d - timedelta(days=d.weekday())  # noqa: E731
        week_starts = sorted({monday(d) for d in dates if d <= day}, reverse=True)[:weeks]
        df = _nba_frame(conn, market, week_starts[-1], day)
    finally:
        conn.close()

    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    day_df = df[df["game_date"] == day].sort_values("predicted", ascending=False)
    rolling = []
    for ws in sorted(week_starts):
        wk = df[(df["game_date"] >= ws) & (df["game_date"] < ws + timedelta(days=7))]
        if s := summarize(wk):
            rolling.append({"label": f"Week of {ws:%b %-d, %Y}", **s})
    return {
        "period": {"key": day.isoformat(), "label": f"{day:%a %b %-d, %Y}"},
        "periods": [{"key": d.isoformat(), "label": f"{d:%a %b %-d, %Y}"} for d in dates],
        "markets": [{"key": k, "label": v} for k, v in NBA_MARKETS.items()],
        "market": {"key": market, "label": NBA_MARKETS[market]},
        "day": day_df,
        "rolling": rolling,
        "source": "NBA walk-forward residuals: each season predicted by a model trained only on earlier seasons.",
        "baseline_name": "Last-10 game average",
        "naive_is_model": market == "fg3m",
    }


# ---- NFL --------------------------------------------------------------------

def nfl(market: str, season: int | None, week: int | None, weeks: int) -> dict:
    conn = duckdb.connect(nfl_projections._db_path(), read_only=True)
    try:
        try:
            periods = conn.execute(f"SELECT DISTINCT season, week FROM {REPLAY_TABLE} WHERE market = ? ORDER BY 1 DESC, 2 DESC", [market]).fetchall()
        except duckdb.CatalogException:
            raise NotBuilt("NFL replay table missing: python -m backend.backtest.nfl_replay")
        if not periods:
            raise NotBuilt("No NFL replay rows for this market.")
        chosen = (season, week) if (season, week) in periods else periods[0]
        window = [p for p in periods if p <= chosen][:weeks]
        try:
            df = conn.execute(
                f"""SELECT season, week, player_name, team, opponent, position, actual, predicted, lo, hi, naive, model_version
                    FROM {REPLAY_TABLE} WHERE market = ? AND season * 100 + week IN (SELECT UNNEST(?::INT[]))""",
                [market, [s * 100 + w for s, w in window]],
            ).fetchdf()
        except duckdb.BinderException:
            raise NotBuilt("NFL replay table predates model versions: python -m backend.backtest.nfl_replay")
    finally:
        conn.close()

    label = lambda s, w: f"{s} · Week {w}"  # noqa: E731
    versions = lambda d: ", ".join(sorted(d["model_version"].unique())) or None  # noqa: E731
    day_df = df[(df["season"] == chosen[0]) & (df["week"] == chosen[1])].sort_values("predicted", ascending=False)
    rolling = []
    for s, w in reversed(window):
        wk = df[(df["season"] == s) & (df["week"] == w)]
        if summ := summarize(wk):
            rolling.append({"label": label(s, w), "model_version": versions(wk), **summ})
    return {
        "period": {"key": f"{chosen[0]}-{chosen[1]}", "label": label(*chosen)},
        "periods": [{"key": f"{s}-{w}", "label": label(s, w)} for s, w in periods],
        "markets": [{"key": k, "label": v} for k, v in NFL_LABELS.items()],
        "market": {"key": market, "label": NFL_LABELS[market]},
        "day": day_df,
        "rolling": rolling,
        "model_version": versions(day_df),
        "source": "NFL walk-forward: each week projected by a model trained only on games before it "
                  "(earlier seasons, plus this season through the previous week once weekly models exist). "
                  "Model versions read thruYYYY (seasons through YYYY) or thruYYYYwNN (plus that season through week NN).",
        "baseline_name": "Last-5 game average",
        "naive_is_model": False,
    }


def replay(league: str, market: str | None, period: str | None, weeks: int) -> dict:
    if league == "nfl":
        market = market if market in NFL_LABELS else "receiving_yards"
        season = week = None
        if period and "-" in period:
            season, week = (int(x) for x in period.split("-", 1))
        data = nfl(market, season, week, weeks)
    else:
        market = market if market in NBA_MARKETS else "points"
        day = date.fromisoformat(period) if period else None
        data = nba(market, day, weeks)

    day_df = data.pop("day")
    data["summary"] = summarize(day_df)
    biggest = day_df.assign(abs_miss=(day_df["actual"] - day_df["predicted"]).abs()).nlargest(5, "abs_miss")
    data["biggest_misses"] = _rows(biggest)
    data["rows"] = _rows(day_df)
    data["weeks"] = weeks
    data["note"] = NOTE
    return {"status": "ready", "league": league, **data}
