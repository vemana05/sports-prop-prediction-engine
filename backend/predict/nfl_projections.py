"""Tentative NFL projections for any week of the current season (the
Schedule page's NFL week picker, spec Section 10.4 H / Phase 6).

Per game, every relevant player gets a projected passing/rushing/receiving
yards and receptions line (only the markets whose pre-game universe he's in;
see nfl_features.py) with a 25th-75th percentile range from that market's
own distribution (nfl_market_model.Spread).

Projection-only: NFL odds collection is not enabled (shared 500-credit Odds
API budget, spec 3.5), and this module never calls The Odds API.

Week N is projected by the newest saved model trained strictly before it:
completed seasons plus this season through week N-1 (weekly walk-forward,
see the models section), falling back to the completed-seasons model. So
every projection, past weeks included, is out-of-sample, and played games
show the projection that was knowable before kickoff next to the actual
result, labelled with the model version that made it.

Unplayed games: each team's players come from its latest depth chart
(QB1, RB1-3, WR1-5, TE1-2), minus anyone listed Out on that week's injury
report if one has posted. Their features are built by the same SQL as
training, on a scratch in-memory database that attaches the NFL warehouse
read-only, so serving never writes to (or locks) the warehouse.

Cached per week in data/projections/nfl_<season>_w<week>.json and
recomputed when the cache is from an earlier ET day or a newer model
version for that week has been saved. The API checks
ensure_models() first so a missing model trains in the background instead
of inside a request.
"""

from __future__ import annotations

import json
import os
import pickle
import threading
import time
from dataclasses import dataclass
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
CACHE_VERSION = 2
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
#
# Two kinds of saved model, both versioned by what they were trained on:
#   nfl_<market>_thru2025       every completed season (the base)
#   nfl_<market>_thru2026w04    ... plus the current season through week 4
#                               (weekly walk-forward, trained by the Tuesday job)
# Week N of a season is always projected by the newest version trained
# strictly before week N (thru week N-1 if it exists, else an earlier week,
# else the base), so a played week's projection never saw that week. Old
# versions are kept: they're what past weeks were projected with.
#
# Weekly retraining is OFF unless NFL_WEEKLY_RETRAIN=1: on 2026 weeks 1-4 it
# didn't beat the base model (reports/nfl_weekly_retrain_eval.json; MAE and
# 25-75 coverage within noise, passing yards slightly worse). Off, every
# week uses the base and no weekly versions are trained or served.
#
# A saved version's data never changes, so it's valid for as long as it
# loads (no age rule). Training only happens when a version is missing; one
# lock per version makes concurrent callers share that run, files are
# written atomically, and the API trains in a background thread
# (ensure_models), never inside a request.


@dataclass(frozen=True, order=True)
class Through:
    season: int
    week: int | None = None  # None: the whole season

    @property
    def tag(self) -> str:
        return f"thru{self.season}" if self.week is None else f"thru{self.season}w{self.week:02d}"

    @property
    def label(self) -> str:
        return f"seasons through {self.season}" if self.week is None else f"seasons through {self.season - 1} + {self.season} week{'s 1-' if self.week > 1 else ' '}{self.week}"


_MODELS: dict[str, tuple[lgb.Booster, Spread]] = {}
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()
_WARM_LOCK = threading.Lock()
_warm_thread: threading.Thread | None = None
_warm_error: tuple[float, str] | None = None  # (when, message) of the last failed run
WARM_RETRY_S = 300


def weekly_retrain_enabled() -> bool:
    return os.environ.get("NFL_WEEKLY_RETRAIN") == "1"


def _paths(market: Market, through: Through) -> tuple[Path, Path]:
    stem = f"nfl_{market.name}_{through.tag}"
    return MODEL_DIR / f"{stem}.txt", MODEL_DIR / f"{stem}_spread.pkl"


def _lock_for(key: str) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.Lock())


def _load(model_path: Path, spread_path: Path) -> tuple[lgb.Booster, Spread] | None:
    if not (model_path.exists() and spread_path.exists()):
        return None
    try:
        return lgb.Booster(model_file=str(model_path)), pickle.loads(spread_path.read_bytes())
    except Exception as e:  # truncated/corrupt file: retrain rather than fail every request
        logger.warning("nfl_projections: can't load %s (%s); retraining", model_path.name, e)
        return None


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _where(through: Through) -> tuple[str, list]:
    if through.week is None:
        return "season <= ?", [through.season]
    return "(season < ? OR (season = ? AND week <= ?))", [through.season, through.season, through.week]


def _training_frame(market: Market, through: Through) -> pd.DataFrame:
    where, params = _where(through)
    conn = duckdb.connect(_db_path(), read_only=True)
    try:
        return conn.execute(
            f"SELECT * FROM nfl_features WHERE {market.universe} AND {where} ORDER BY kickoff_utc, game_id, player_id",
            params,
        ).fetchdf()
    finally:
        conn.close()


def complete(through: Through) -> bool:
    """True when every game the version would train on is final, i.e. it
    can't absorb a partial week. Completed seasons only for the base."""
    if through.week is None:
        return through.season < current_season()
    conn = duckdb.connect(_db_path(), read_only=True)
    try:
        n, open_, featured = conn.execute(
            """SELECT COUNT(*), COUNT(*) - COUNT(home_score),
                      (SELECT COUNT(*) FROM nfl_features WHERE season = $1 AND week = $2)
               FROM nfl_schedule WHERE season = $1 AND week <= $2""",
            [through.season, through.week],
        ).fetchone()
    finally:
        conn.close()
    return n > 0 and open_ == 0 and featured > 0


def _get_or_train(market: Market, through: Through) -> tuple[lgb.Booster, Spread]:
    model_path, spread_path = _paths(market, through)
    key = str(model_path)
    if key in _MODELS:
        return _MODELS[key]
    with _lock_for(key):
        if key in _MODELS:  # another thread finished while we waited
            return _MODELS[key]
        loaded = _load(model_path, spread_path)
        if loaded is None:
            if not complete(through):
                raise ValueError(f"NFL {through.tag} would train on games that aren't final")
            df = _training_frame(market, through)
            df["pos_code"] = df["position"].map(POS_CODES)
            spread = fit_spread(df[market.target].to_numpy(float), _oof_predictions(df, market), market.family)
            model = _train(df, market)
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            _write_atomic(spread_path, pickle.dumps(spread))
            _write_atomic(model_path, model.model_to_string().encode())
            logger.info("nfl_projections: trained %s on %s (%s)", market.name, through.label, through.tag)
            loaded = (model, spread)
        _MODELS[key] = loaded
        return loaded


def models_ready(through: Through) -> bool:
    """True when every market's version is in memory or saved on disk, i.e.
    serving it won't train."""
    return all(str(p) in _MODELS or (p.exists() and s.exists()) for p, s in (_paths(m, through) for m in MARKETS))


def warm_models(through: Through) -> None:
    for m in MARKETS:
        _get_or_train(m, through)


def target_version(season: int, week: int) -> Through:
    """The version week N should use: thru week N-1, or the base for week 1."""
    return Through(season - 1) if week <= 1 else Through(season, week - 1)


def version_for(season: int, week: int) -> Through:
    """The newest version saved before week N: thru N-1, else the latest
    earlier week, else the base (thru last season). Never one that saw week N."""
    if not weekly_retrain_enabled():
        return Through(season - 1)
    for w in range(week - 1, 0, -1):
        if models_ready(Through(season, w)):
            return Through(season, w)
    return Through(season - 1)


def ensure_models(season: int | None = None, week: int | None = None) -> dict | None:
    """Non-blocking: None when serving won't train; otherwise starts (at
    most one) background training run and returns a "warming" payload for
    the API to send instead of blocking. Trains the base if it's missing,
    and week N's thru-(N-1) version once that week's results are final;
    until then week N is served by an earlier version."""
    global _warm_thread, _warm_error
    season = season or current_season()
    if week is None:
        conn = duckdb.connect(_db_path(), read_only=True)
        try:
            week = default_week(weeks(conn, season))
        finally:
            conn.close()
    needed = [Through(season - 1)]
    if week is not None and week > 1 and weekly_retrain_enabled():
        needed.append(target_version(season, week))
    missing = [t for t in needed if not models_ready(t)]
    if missing and missing[-1].week is not None and not complete(missing[-1]):
        missing.pop()  # week N-1 not final yet: serve the fallback
    if not missing:
        return None
    with _WARM_LOCK:
        idle = _warm_thread is None or not _warm_thread.is_alive()
        if idle and _warm_error and time.time() - _warm_error[0] < WARM_RETRY_S:
            return {"status": "error", "league": "nfl", "message": f"NFL model training failed: {_warm_error[1]}"}
        if idle:
            def run() -> None:
                global _warm_error
                try:
                    for t in missing:
                        warm_models(t)
                    _warm_error = None
                except Exception as e:
                    _warm_error = (time.time(), str(e))
                    logger.exception("nfl_projections: model warm-up failed")
            _warm_thread = threading.Thread(target=run, name="nfl-model-warm", daemon=True)
            _warm_thread.start()
    return {"status": "warming", "league": "nfl", "message": f"Training NFL models on {missing[0].label}; this takes a few minutes."}


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
    # Loads only: ensure_models / the Tuesday job do the training.
    version = version_for(season, week)
    models = {m.name: _get_or_train(m, version) for m in MARKETS}

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
        "models_trained_through": season - 1,
        "model_version": version.tag,
        "model_label": version.label,
        "injuries_posted": bool(injuries_posted),
        "games": games,
    }


def _read_cache(path: Path, season: int, week: int) -> dict | None:
    """Today's cache, unless a newer model version for this week has been
    saved since (e.g. the Tuesday job trained thru week N-1)."""
    if not path.exists():
        return None
    cached = json.loads(path.read_text())
    if (cached.get("version") == CACHE_VERSION
            and datetime.fromisoformat(cached["computed_at"]).date() == datetime.now(ET).date()
            and cached.get("model_version") == version_for(season, week).tag):
        return cached
    return None


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
    cached = _read_cache(path, season, week)
    if cached:
        return cached
    with _lock_for(str(path)):  # concurrent requests for one week share a compute
        fresh = _read_cache(path, season, week)
        if fresh:
            return fresh
        data = compute(season, week)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _write_atomic(path, json.dumps(data).encode())
        return data


def train_weekly(season: int | None = None) -> list[str]:
    """Tuesday job: train every weekly version of this season whose weeks
    are all final and that isn't saved yet (normally just last week's).
    Older versions are kept."""
    season = season or current_season()
    if not weekly_retrain_enabled():
        return []
    conn = duckdb.connect(_db_path(), read_only=True)
    try:
        played = [w for (w,) in conn.execute("SELECT DISTINCT week FROM nfl_features WHERE season = ? ORDER BY 1", [season]).fetchall()]
    finally:
        conn.close()
    trained = []
    for w in played:
        t = Through(season, w)
        if not models_ready(t) and complete(t):
            warm_models(t)
            trained.append(t.tag)
    return trained


if __name__ == "__main__":
    import argparse

    # Tuesday results job: --retrain trains last week's walk-forward version
    # (thru<season>w<NN>), then the week's projection cache is built with it.
    parser = argparse.ArgumentParser(description="Warm the NFL models and projection cache for a week")
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--retrain", action="store_true", help="train missing weekly walk-forward versions first")
    args = parser.parse_args()
    warm_models(Through(current_season() - 1))
    if args.retrain:
        if not weekly_retrain_enabled():
            print("weekly retrain off (set NFL_WEEKLY_RETRAIN=1 in .env to enable)")
        else:
            print(f"weekly versions trained: {train_weekly() or 'none (up to date)'}")
    d = get(args.week)
    n = sum(len(t["players"]) for g in d["games"] for t in g["teams"].values())
    print(f"NFL {d['season']} week {d['week']} ({d['model_version']}): {len(d['games'])} games, {n} players")
