"""Tentative per-player projections for any NBA game day (the Schedule &
projections page, spec Section 10.4 H).

For a chosen date: every game in `nba_schedule`, each team's rest going in
(days since its previous scheduled game, back-to-back, season opener), and
a projected line per player — minutes, points, rebounds, assists, 3PM —
from the same cached models the leg pool uses, with a 25th-75th percentile
range from each market's own distribution:

    points          Normal(mean, sigma)            sigma from the points model
    reb / ast / 3PM Negative Binomial(mean, r)     r from that market's model
    minutes         Normal(mean, MINUTES_SIGMA)    no fitted spread exists; see below

3PM's mean is the naive last-10 average, same as everywhere else live
(`predictor_choice` — it beat the model in walk-forward).

Inputs are `build_current_form` as of TODAY (games strictly before today),
not as of the game date: for a game a week out, the games in between
haven't happened. Two consequences, both shown in the UI:
- Rest is taken from the schedule (the team's previous scheduled game),
  not from each player's last logged game.
- Teammates-out features are 0 (nobody out) because injury reports only
  post the day before a game. The UI labels this "unknown until injury
  reports post" rather than pretending to know.

Projections are cached per game date in `data/projections/<date>.json`
with `computed_at`, and recomputed when the cache is from an earlier day
(ET) — so every projection is at most one day stale. The daily job also
runs this module to warm the next game day. Never calls The Odds API.

Roster caveat (same as nba_current_form): a player appears under the team
he last played for, so offseason moves don't show until he plays a game.
"""

from __future__ import annotations

import json
import math
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd
from nba_api.stats.static import teams as nba_static_teams
from scipy import stats

from backend.db import DEFAULT_DB_PATH
from backend.features.nba_current_form import build_current_form
from backend.ingest.util import logger
from backend.models import nba_minutes_model
from backend.models.model_cache import get_or_train_booster
from backend.models.nba_minutes_model import FEATURE_COLUMNS
from backend.parlay.predictor_choice import use_model_for
from backend.predict.log_predictions import MODEL_MAX_AGE_DAYS, _train_market, _train_points

ET = ZoneInfo("America/New_York")
CACHE_DIR = Path("data/projections")
CACHE_VERSION = 1
# Minutes has no fitted spread. For a Normal, MAE = sigma * sqrt(2/pi), so the
# minutes model's walk-forward MAE (4.957, predictor_choice.py) implies this sigma.
MINUTES_SIGMA = 4.957 * math.sqrt(math.pi / 2)
MIN_PROJECTED_MINUTES = 8.0  # below this a player is end-of-bench noise
MAX_PLAYERS_PER_TEAM = 13
RECENT_DAYS = 365  # only players with a logged game in the last year
Q_LO, Q_HI = 0.25, 0.75

_MARKETS = [  # (market, model name, train fn args, last-10 column for the naive path)
    ("rebounds", ("reb", "reb_roll10"), "reb_roll10"),
    ("assists", ("ast", "ast_roll10"), "ast_roll10"),
    ("fg3m", ("fg3m", "fg3m_roll10"), "fg3m_roll10"),
]


def today_et() -> date:
    return datetime.now(ET).date()


def _abbr() -> dict[str, str]:
    return {str(t["id"]): t["abbreviation"] for t in nba_static_teams.get_teams()}


def _connect(db_path: str | None = None) -> duckdb.DuckDBPyConnection:
    # Read-only so this page never takes the warehouse write lock. If this
    # process already has the file open read-write (the API's slate build),
    # DuckDB refuses a second config; reuse the process's existing mode then.
    path = db_path or os.environ.get("DUCKDB_PATH", DEFAULT_DB_PATH)
    try:
        return duckdb.connect(path, read_only=True)
    except duckdb.ConnectionException:
        return duckdb.connect(path)


def game_days(conn: duckdb.DuckDBPyConnection) -> list[date]:
    return [r[0] for r in conn.execute("SELECT DISTINCT game_date FROM nba_schedule ORDER BY 1").fetchall()]


def next_game_day(conn: duckdb.DuckDBPyConnection, on_or_after: date | None = None) -> date | None:
    row = conn.execute(
        "SELECT min(game_date) FROM nba_schedule WHERE game_date >= ?", [on_or_after or today_et()]
    ).fetchone()
    return row[0]


def _games(conn: duckdb.DuckDBPyConnection, game_date: date) -> pd.DataFrame:
    return conn.execute(
        """
        WITH g AS (
            SELECT game_id, season, game_date, home_team_id, away_team_id, tip_utc FROM nba_schedule WHERE game_date = ?
        ),
        team_games AS (
            SELECT home_team_id AS team_id, game_date, season FROM nba_schedule
            UNION ALL SELECT away_team_id, game_date, season FROM nba_schedule
        )
        SELECT g.*,
            (SELECT max(t.game_date) FROM team_games t WHERE t.team_id = g.home_team_id AND t.game_date < g.game_date) AS home_prev,
            (SELECT max(t.game_date) FROM team_games t WHERE t.team_id = g.home_team_id AND t.game_date < g.game_date AND t.season = g.season) AS home_prev_same_season,
            (SELECT max(t.game_date) FROM team_games t WHERE t.team_id = g.away_team_id AND t.game_date < g.game_date) AS away_prev,
            (SELECT max(t.game_date) FROM team_games t WHERE t.team_id = g.away_team_id AND t.game_date < g.game_date AND t.season = g.season) AS away_prev_same_season
        FROM g ORDER BY tip_utc, game_id
        """,
        [game_date],
    ).fetchdf()


def _rest(game_date: date, prev, prev_same_season) -> dict:
    """Days since the team's previous scheduled game; a team with no earlier
    game this season is on its opener (rest is then the offseason)."""
    prev = prev.date() if hasattr(prev, "date") else prev
    opener = pd.isna(prev_same_season)
    days = (game_date - prev).days if prev is not None and not pd.isna(prev) else None
    return {"rest_days": days, "back_to_back": days == 1, "season_opener": bool(opener)}


def _load_models(conn: duckdb.DuckDBPyConnection) -> dict:
    minutes, _ = get_or_train_booster("minutes", lambda: (nba_minutes_model.train_final_model(conn), {}), MODEL_MAX_AGE_DAYS)
    points, pextra = get_or_train_booster("points", lambda: _train_points(conn), MODEL_MAX_AGE_DAYS)
    models = {"minutes": (minutes, None), "points": (points, pextra["sigma"])}
    for market, (target, line_col), _ in _MARKETS:
        m, extra = get_or_train_booster(
            market, lambda t=target, l=line_col: _train_market(conn, t, l), MODEL_MAX_AGE_DAYS
        )
        models[market] = (m, extra["r"])
    return models


def _normal_range(mean: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    z = stats.norm.ppf(Q_HI)
    return np.clip(mean - z * sigma, 0, None), mean + z * sigma


def _nb_range(mean: np.ndarray, r: float) -> tuple[np.ndarray, np.ndarray]:
    mu = np.clip(mean, 1e-6, None)
    p = r / (r + mu)
    return stats.nbinom.ppf(Q_LO, r, p), stats.nbinom.ppf(Q_HI, r, p)


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def compute(game_date: date, db_path: str | None = None) -> dict:
    conn = _connect(db_path)
    try:
        games = _games(conn, game_date)
        days = game_days(conn)
        base = {
            "version": CACHE_VERSION,
            "date": game_date.isoformat(),
            "computed_at": datetime.now(ET).isoformat(timespec="seconds"),
            "form_as_of": today_et().isoformat(),
            "prev_game_day": next((d.isoformat() for d in reversed(days) if d < game_date), None),
            "next_game_day": next((d.isoformat() for d in days if d > game_date), None),
            "games": [],
        }
        if games.empty:
            return base

        abbr = _abbr()
        opponent_by_team = {}
        for g in games.itertuples():
            h, a = abbr.get(str(g.home_team_id)), abbr.get(str(g.away_team_id))
            opponent_by_team[h], opponent_by_team[a] = a, h

        form = build_current_form(conn, as_of=today_et(), opponent_by_team=opponent_by_team)
        form = form[pd.to_datetime(form["last_game_date"]) >= pd.Timestamp(today_et() - timedelta(days=RECENT_DAYS))]
        pace = form.groupby("team")["team_pace_roll10"].first().dropna().sort_values(ascending=False)
        pace_rank = {t: i + 1 for i, t in enumerate(pace.index)}
        models = _load_models(conn)

        for g in games.itertuples():
            teams = {"home": abbr.get(str(g.home_team_id)), "away": abbr.get(str(g.away_team_id))}
            rest = {
                "home": _rest(game_date, g.home_prev, g.home_prev_same_season),
                "away": _rest(game_date, g.away_prev, g.away_prev_same_season),
            }
            out = {
                "game_id": g.game_id,
                "tip_utc": pd.Timestamp(g.tip_utc).tz_convert("UTC").isoformat(),
                "home": teams["home"],
                "away": teams["away"],
                "rest": rest,
                "teams": {},
            }
            for side, team in teams.items():
                opp = teams["away" if side == "home" else "home"]
                roster = form[form["team"] == team].copy()
                out["teams"][side] = {
                    "team": team,
                    "opponent": opp,
                    "opp_pace": _round(pace.get(opp)),
                    "opp_pace_rank": pace_rank.get(opp),
                    "n_teams_ranked": len(pace_rank),
                    "players": _project(roster, rest[side], side == "home", models) if len(roster) else [],
                }
            base["games"].append(out)
        return base
    finally:
        conn.close()


def _round(x, nd=1):
    return None if x is None or pd.isna(x) else round(float(x), nd)


def _project(roster: pd.DataFrame, team_rest: dict, is_home: bool, models: dict) -> list[dict]:
    X = roster.reindex(columns=FEATURE_COLUMNS).copy()
    X["home_flag"] = 1.0 if is_home else 0.0
    # Rest comes from the schedule, not the player's last logged game (see module docstring).
    if team_rest["rest_days"] is not None:
        X["rest_days"] = float(team_rest["rest_days"])
        X["is_back_to_back"] = 1.0 if team_rest["back_to_back"] else 0.0
    X["is_back_to_back"] = X["is_back_to_back"].astype(float)
    # Unknown until injury reports post: modeled as nobody out.
    X[["teammates_out_count", "teammates_out_pts_l10", "teammates_out_min_l10"]] = 0.0

    proj: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    m_min = np.clip(models["minutes"][0].predict(X), 0, 48)
    proj["minutes"] = (m_min, *_normal_range(m_min, MINUTES_SIGMA))
    pmodel, sigma = models["points"]
    m_pts = np.clip(pmodel.predict(X), 0, None)
    proj["points"] = (m_pts, *_normal_range(m_pts, sigma))
    for market, _, naive_col in _MARKETS:
        model, r = models[market]
        mean = model.predict(X) if use_model_for(market) else roster[naive_col].to_numpy(dtype=float)
        mean = np.clip(np.nan_to_num(mean, nan=0.0), 0, None)
        proj[market] = (mean, *_nb_range(mean, r))

    players = []
    for i, p in enumerate(roster.itertuples()):
        if proj["minutes"][0][i] < MIN_PROJECTED_MINUTES:
            continue
        players.append({
            "player_id": p.player_id_nba,
            "player_name": p.player_name,
            "last_game_date": str(p.last_game_date),
            "opp_def_vs_position_pts": _round(p.opp_def_vs_position_pts_roll10),
            "projections": {
                m: {"mean": round(float(v[0][i]), 1), "lo": round(float(v[1][i]), 1), "hi": round(float(v[2][i]), 1)}
                for m, v in proj.items()
            },
        })
    players.sort(key=lambda p: -p["projections"]["minutes"]["mean"])
    return players[:MAX_PLAYERS_PER_TEAM]


def games_with_lines(game_ids: list[str], db_path: str | None = None) -> set[str]:
    """Which of these games already have player-prop lines in odds_snapshots.
    Read live (not cached) since lines post during the day. No API call."""
    if not game_ids:
        return set()
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT DISTINCT game_id_nba FROM odds_snapshots WHERE game_id_nba IN (SELECT unnest(?))", [game_ids]
        ).fetchall()
    finally:
        conn.close()
    return {r[0] for r in rows}


def _cache_path(game_date: date) -> Path:
    return CACHE_DIR / f"{game_date.isoformat()}.json"


def get(game_date: date, db_path: str | None = None, force: bool = False) -> dict:
    """Cached projections for `game_date`; recomputed if the cache was made
    on an earlier day (ET) than today, or with an older cache format."""
    path = _cache_path(game_date)
    if not force and path.exists():
        try:
            data = json.loads(path.read_text())
            if data.get("version") == CACHE_VERSION and data.get("form_as_of") == today_et().isoformat():
                return data
        except (ValueError, OSError):
            pass
    data = compute(game_date, db_path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


if __name__ == "__main__":
    # Daily warm-up: the next game day on or after today.
    conn = _connect()
    try:
        d = next_game_day(conn)
    finally:
        conn.close()
    if d is None:
        print("schedule_projections: no upcoming games in nba_schedule")
    else:
        data = get(d, force=True)
        n = sum(len(t["players"]) for g in data["games"] for t in g["teams"].values())
        logger.info("schedule_projections: %s — %d games, %d player projections", d, len(data["games"]), n)
        print(f"schedule_projections complete: {d}, {len(data['games'])} games, {n} players")
