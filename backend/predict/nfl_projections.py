"""Tentative NFL projections for any week of the current season (the
Schedule page's NFL week picker, spec Section 10.4 H / Phase 6).

Per game, every relevant player gets a projected passing/rushing/receiving
yards and receptions line (only the markets whose pre-game universe he's in;
see nfl_features.py) with a 25th-75th percentile range from that market's
own distribution (nfl_market_model.Spread).

Projection-only: NFL odds collection is not enabled (shared 500-credit Odds
API budget, spec 3.5), and this module never calls The Odds API.

Models train on COMPLETED seasons only (2020 through last season), so
every projection in the current season, past weeks included, is genuinely
out-of-sample. Played games show the projection that was knowable before
kickoff next to the actual result.

Unplayed games: each team's players come from its latest depth chart
(QB1, RB1-3, WR1-5, TE1-2), minus anyone listed Out on that week's injury
report if one has posted. Their features are built by the same SQL as
training, on a scratch in-memory database that attaches the NFL warehouse
read-only, so serving never writes to (or locks) the warehouse.

Cached per week in data/projections/nfl_<season>_w<week>.json and
recomputed when the cache is from an earlier ET day.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

from backend.features.nfl_features import build_features
from backend.ingest.nfl_nflverse import current_season
from backend.ingest.util import logger
from backend.models.nfl_market_model import MARKETS, POS_CODES, Market, Spread, _oof_predictions, _predict, _train, fit_spread
from backend.nfl_db import DEFAULT_NFL_DB_PATH

ET = ZoneInfo("America/New_York")
CACHE_DIR = Path("data/projections")
MODEL_DIR = Path("data/models")
MODEL_MAX_AGE_DAYS = 7
CACHE_VERSION = 1
Q_LO, Q_HI = 0.25, 0.75
DEPTH_LIMITS = {"QB": 1, "RB": 3, "WR": 5, "TE": 2}
POSTSEASON_LABELS = {"WC": "Wild Card", "DIV": "Divisional", "CON": "Conference", "SB": "Super Bowl"}
POSITION_ORDER = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}


def _db_path() -> str:
    return os.environ.get("NFL_DUCKDB_PATH", DEFAULT_NFL_DB_PATH)


def _scratch() -> duckdb.DuckDBPyConnection:
    """In-memory DB with the warehouse attached read-only: reads resolve to
    the warehouse, CREATEs land in memory."""
    conn = duckdb.connect()
    conn.execute(f"ATTACH '{_db_path()}' AS w (READ_ONLY)")
    conn.execute("SET search_path = 'memory.main,w.main'")
    return conn


def weeks(conn: duckdb.DuckDBPyConnection, season: int) -> list[dict]:
    rows = conn.execute(
        """SELECT week, any_value(game_type) AS game_type, MIN(kickoff_utc) AS first, MAX(kickoff_utc) AS last,
                  COUNT(*) AS n_games, COUNT(home_score) AS n_final
           FROM nfl_schedule WHERE season = ? GROUP BY week ORDER BY week""",
        [season],
    ).fetchall()
    return [
        {
            "week": w,
            "label": f"Week {w}" if gt == "REG" else POSTSEASON_LABELS.get(gt, gt),
            "first_kickoff_utc": first.isoformat() if first else None,
            "last_kickoff_utc": last.isoformat() if last else None,
            "n_games": n,
            "n_final": nf,
        }
        for w, gt, first, last, n, nf in rows
    ]


def default_week(week_list: list[dict]) -> int | None:
    """The first week with an unplayed game; the last week once all are final."""
    for w in week_list:
        if w["n_final"] < w["n_games"]:
            return w["week"]
    return week_list[-1]["week"] if week_list else None


# ---- models -----------------------------------------------------------------

def _get_or_train(market: Market, train_through: int) -> tuple[lgb.Booster, Spread]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"nfl_{market.name}_thru{train_through}.txt"
    spread_path = MODEL_DIR / f"nfl_{market.name}_thru{train_through}_spread.pkl"
    fresh = model_path.exists() and spread_path.exists() and (time.time() - model_path.stat().st_mtime) / 86400 <= MODEL_MAX_AGE_DAYS
    if fresh:
        return lgb.Booster(model_file=str(model_path)), pickle.loads(spread_path.read_bytes())

    conn = duckdb.connect(_db_path(), read_only=True)
    try:
        df = conn.execute(f"SELECT * FROM nfl_features WHERE {market.universe} AND season <= ? ORDER BY kickoff_utc, game_id, player_id", [train_through]).fetchdf()
    finally:
        conn.close()
    df["pos_code"] = df["position"].map(POS_CODES)
    spread = fit_spread(df[market.target].to_numpy(float), _oof_predictions(df, market), market.family)
    model = _train(df, market)
    model.save_model(str(model_path))
    spread_path.write_bytes(pickle.dumps(spread))
    logger.info("nfl_projections: trained %s on seasons through %d", market.name, train_through)
    return model, spread


def quantiles(mu: np.ndarray, spread: Spread, q: float) -> np.ndarray:
    if spread.r is not None:
        mu = np.clip(mu, 1e-6, None)
        return stats.nbinom.ppf(q, spread.r, spread.r / (spread.r + mu))
    # Clipped at 0: a negative 25th percentile reads as nonsense on a yards line.
    return np.clip(mu + (spread.a + spread.b * mu) * np.quantile(spread.z_sorted, q), 0, None)


# ---- rows -------------------------------------------------------------------

def _pending_players(conn: duckdb.DuckDBPyConnection, season: int, week: int) -> pd.DataFrame:
    """Depth-chart players for this week's unplayed games, from the latest
    chart published before now."""
    limits = " OR ".join(f"(d.position = '{p}' AND d.depth_rank <= {n})" for p, n in DEPTH_LIMITS.items())
    return conn.execute(
        f"""
        WITH games AS (
            SELECT game_id, home_team AS team FROM nfl_schedule WHERE season = $1 AND week = $2 AND home_score IS NULL
            UNION ALL
            SELECT game_id, away_team FROM nfl_schedule WHERE season = $1 AND week = $2 AND home_score IS NULL
        ),
        latest AS (
            SELECT team, MAX(snapshot_at) AS snapshot_at FROM nfl_depth_charts
            WHERE season = $1 AND snapshot_at <= now() GROUP BY team
        )
        SELECT d.player_id, g.game_id, g.team, d.position, any_value(d.player_name) AS player_name
        FROM games g
        JOIN latest l ON l.team = g.team
        JOIN nfl_depth_charts d ON d.team = l.team AND d.snapshot_at = l.snapshot_at AND ({limits})
        WHERE d.player_id NOT IN (
            SELECT player_id FROM nfl_injuries WHERE season = $1 AND week = $2 AND team = g.team AND report_status = 'Out'
        )
        GROUP BY d.player_id, g.game_id, g.team, d.position
        """,
        [season, week],
    ).fetchdf()


def compute(season: int, week: int) -> dict:
    train_through = season - 1
    models = {m.name: _get_or_train(m, train_through) for m in MARKETS}

    conn = _scratch()
    try:
        pending = _pending_players(conn, season, week)
        build_features(conn, pending)
        sched = conn.execute(
            """SELECT s.*, wx.temperature_c, wx.wind_kph, wx.precip_mm, wx.source AS weather_source
               FROM nfl_schedule s LEFT JOIN nfl_weather wx USING (game_id)
               WHERE s.season = ? AND s.week = ? ORDER BY s.kickoff_utc, s.game_id""",
            [season, week],
        ).fetchdf()
        rows = conn.execute("SELECT * FROM memory.main.nfl_features WHERE season = ? AND week = ?", [season, week]).fetchdf()
        injuries_posted = conn.execute(
            "SELECT COUNT(*) > 0 FROM nfl_injuries WHERE season = ? AND week = ?", [season, week]
        ).fetchone()[0]
        week_list = weeks(conn, season)
    finally:
        conn.close()

    rows["pos_code"] = rows["position"].map(POS_CODES)
    final_games = set(sched.loc[sched["home_score"].notna(), "game_id"])
    proj: dict[str, pd.DataFrame] = {}
    for m in MARKETS:
        sub = rows[rows[m.universe].astype("boolean").fillna(False).to_numpy(bool)].copy()
        if sub.empty:
            continue
        model, spread = models[m.name]
        mu = _predict(model, sub)
        sub["mean"], sub["lo"], sub["hi"] = mu, quantiles(mu, spread, Q_LO), quantiles(mu, spread, Q_HI)
        proj[m.name] = sub.set_index(["player_id", "game_id"])

    games = []
    for g in sched.itertuples():
        final = g.game_id in final_games
        teams = {}
        for side, team, opp in (("away", g.away_team, g.home_team), ("home", g.home_team, g.away_team)):
            players = []
            team_rows = rows[(rows["game_id"] == g.game_id) & (rows["team"] == team)]
            for r in team_rows.itertuples():
                markets = {}
                for m in MARKETS:
                    key = (r.player_id, r.game_id)
                    if m.name in proj and key in proj[m.name].index:
                        p = proj[m.name].loc[key]
                        markets[m.name] = {
                            "mean": round(float(p["mean"]), 1),
                            "lo": round(float(p["lo"]), 1),
                            "hi": round(float(p["hi"]), 1),
                            "actual": float(getattr(r, m.target)) if final else None,
                        }
                if markets:
                    players.append({
                        "player_id": r.player_id,
                        "player_name": r.player_name or r.player_id,
                        "position": r.position,
                        "depth_rank": None if pd.isna(r.depth_rank) else int(r.depth_rank),
                        "questionable": bool(r.questionable_flag == 1),
                        "projections": markets,
                    })
            players.sort(key=lambda p: (POSITION_ORDER.get(p["position"], 9), p["depth_rank"] or 99))
            teams[side] = {"team": team, "opponent": opp, "players": players}
        indoor = g.roof not in ("outdoors", "open")
        games.append({
            "game_id": g.game_id,
            "kickoff_utc": pd.Timestamp(g.kickoff_utc).isoformat() if pd.notna(g.kickoff_utc) else None,
            "home": g.home_team,
            "away": g.away_team,
            "final": final,
            "home_score": None if pd.isna(g.home_score) else int(g.home_score),
            "away_score": None if pd.isna(g.away_score) else int(g.away_score),
            "stadium": g.stadium,
            "roof": g.roof,
            "indoor": indoor,
            "weather": None if indoor or pd.isna(g.wind_kph) else {
                "temperature_c": round(float(g.temperature_c), 1),
                "wind_kph": round(float(g.wind_kph), 1),
                "precip_mm": round(float(g.precip_mm), 1),
                "source": g.weather_source,
            },
            "rest": {"home": None if pd.isna(g.home_rest) else int(g.home_rest), "away": None if pd.isna(g.away_rest) else int(g.away_rest)},
            "teams": teams,
        })

    return {
        "version": CACHE_VERSION,
        "league": "nfl",
        "season": season,
        "week": week,
        "weeks": week_list,
        "computed_at": datetime.now(ET).isoformat(),
        "models_trained_through": train_through,
        "injuries_posted": bool(injuries_posted),
        "games": games,
    }


def get(week: int | None = None, season: int | None = None) -> dict:
    season = season or current_season()
    if week is None:
        conn = duckdb.connect(_db_path(), read_only=True)
        try:
            week = default_week(weeks(conn, season))
        finally:
            conn.close()
        if week is None:
            return {"league": "nfl", "season": season, "week": None, "weeks": [], "games": []}

    path = CACHE_DIR / f"nfl_{season}_w{week:02d}.json"
    if path.exists():
        cached = json.loads(path.read_text())
        if cached.get("version") == CACHE_VERSION and datetime.fromisoformat(cached["computed_at"]).date() == datetime.now(ET).date():
            return cached
    data = compute(season, week)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Warm the NFL projection cache for a week")
    parser.add_argument("--week", type=int, default=None)
    args = parser.parse_args()
    d = get(args.week)
    n = sum(len(t["players"]) for g in d["games"] for t in g["teams"].values())
    print(f"NFL {d['season']} week {d['week']}: {len(d['games'])} games, {n} players")
