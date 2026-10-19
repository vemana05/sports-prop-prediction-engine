"""NFL leg pool for the Builder: over-threshold ("60+ rushing yards") legs at
common sportsbook ladder rungs, priced by the model alone. No odds are
collected for NFL and nothing here calls The Odds API; a user can type a
book's line and price per leg in the slip (nfl_search.apply_price).

Pipeline per leg, same shape as NBA (spec Section 2):
  NFL projection (nfl_projections, models trained on earlier seasons only)
  -> P(over rung) from that market's own distribution (nfl_market_model.Spread:
     empirical standardized residuals for yards, Negative Binomial for receptions)
  -> isotonic calibration, fit out of sample on the walk-forward replay
  -> correlation-aware Monte Carlo (monte_carlo.py, via NflLeg.ppf)

Calibrators are fit on every walk-forward fold's held-out rows
(nfl_walk_forward_replay) at every ladder rung, using that fold's own spread
(SPREADS_PATH, saved by nfl_replay.build) — the NFL analogue of
models/calibration.py. Correlations are pooled the same way as NBA's
(correlations._pairwise_corr) from the same replay residuals.

Players listed Out never reach the pool: nfl_projections only projects
depth-chart players not on that week's Out list. Questionable players stay,
marked "Q". `out_players` lists the Out ones so "Build my own" can show them
disabled; `make_leg` builds a leg at any line or side for that picker.
"""

from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
from scipy import stats
from sklearn.isotonic import IsotonicRegression

from backend.backtest.nfl_replay import REPLAY_TABLE, SPREADS_PATH
from backend.ingest.util import logger
from backend.models.nfl_market_model import MARKETS, Spread, p_over
from backend.parlay.correlations import _pairwise_corr
from backend.parlay.monte_carlo import Leg
from backend.predict import nfl_projections

LADDERS: dict[str, list[int]] = {
    "passing_yards": [170, 190, 210, 220, 240, 260],
    "rushing_yards": [25, 30, 40, 50, 60, 70, 80],
    "receiving_yards": [30, 40, 50, 60, 70, 80],
    "receptions": [3, 4, 5, 6, 7, 8],
}
MARKET_ABBR = {"passing_yards": "pass yds", "rushing_yards": "rush yds", "receiving_yards": "rec yds", "receptions": "receptions"}
MARKET_NOUN = {"passing_yards": "Passing-yard", "rushing_yards": "Rushing-yard", "receiving_yards": "Receiving-yard", "receptions": "Receptions"}
NFL_MARKETS = [m.name for m in MARKETS]
P_RANGE = (0.03, 0.97)  # rungs outside this are near-certain or near-hopeless; not worth a leg
NARROW_COVERAGE = 0.40  # last week's 25th-75th coverage below this -> "ranges ran narrow" warning
MODELS_DIR = Path("data/models")

UNTESTED_NOTE = (
    "These probabilities come from projections checked only against actual results (see Replay), never against "
    "real NFL sportsbook lines, and the ladder rungs are typical book values, not posted lines."
)


@dataclass
class NflLeg(Leg):
    """A Leg whose marginal is an NFL market's own distribution. Everything
    downstream (calibration, Monte Carlo, shrink, sanity guard) goes
    through the same Leg methods; only the distribution differs."""
    spread: Spread | None = None
    position: str = ""
    opponent: str = ""
    is_home: bool | None = None
    kickoff_utc: str | None = None
    questionable: bool = False
    rung: int | None = None  # ladder value ("60+"); None once a user-typed book line replaces it
    base_id: str | None = None  # the pool leg this was derived from (price/line override)
    warnings: tuple[str, ...] = ()

    @property
    def distribution(self) -> str:
        return "negbinom" if self.spread is not None and self.spread.r is not None else "yards"

    def raw_p_over(self) -> float:
        return float(p_over(np.array([self.predicted_mean]), np.array([self.line]), self.spread)[0])

    def ppf(self, u: np.ndarray) -> np.ndarray:
        mu = max(self.predicted_mean, 1e-6)
        if self.distribution == "negbinom":
            r = self.spread.r
            return stats.nbinom.ppf(u, r, r / (r + mu))
        return mu + (self.spread.a + self.spread.b * mu) * np.quantile(self.spread.z_sorted, u)

    @property
    def own_price(self) -> float | None:
        """The typed price for this leg's own side."""
        return self.over_price if self.side == "over" else self.under_price

    def no_vig_p_hit(self) -> float | None:
        """Both sides typed: the book's no-vig probability. Only this leg's
        side typed: its implied probability, vig included (a slightly
        conservative stand-in, flagged in the UI)."""
        if self.over_price and self.under_price:
            return super().no_vig_p_hit()
        if self.own_price:
            return 1 / self.own_price
        return None

    @property
    def priced(self) -> bool:
        return bool(self.own_price)


def rung_line(rung: int) -> float:
    """"60+" clears at 60 or more, i.e. over 59.5."""
    return rung - 0.5


def _db_path() -> str:
    return os.environ.get("NFL_DUCKDB_PATH", nfl_projections.DEFAULT_NFL_DB_PATH)


def _replay_frame():
    conn = duckdb.connect(_db_path(), read_only=True)
    try:
        return conn.execute(
            f"SELECT player_id AS player_id_nba, game_id, team, market, season, week, actual, predicted, lo, hi, "
            f"actual - predicted AS residual FROM {REPLAY_TABLE}"
        ).fetchdf()
    finally:
        conn.close()


# ---- calibration ------------------------------------------------------------------

def _fit_calibrator(df, spreads: dict[int, Spread], market: str) -> IsotonicRegression:
    raw, hit = [], []
    rows = df[df["market"] == market]
    for season, spread in spreads.items():
        s = rows[rows["season"] == season]
        if s.empty:
            continue
        mu = s["predicted"].to_numpy(float)
        for rung in LADDERS[market]:
            line = np.full(len(s), rung_line(rung))
            raw.append(p_over(mu, line, spread))
            hit.append((s["actual"].to_numpy(float) > line).astype(int))
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.concatenate(raw), np.concatenate(hit))
    return iso


def calibrators() -> dict[str, IsotonicRegression | None]:
    """One ladder calibrator per market, cached next to the models and
    refit whenever the replay's spreads file changes. None (uncalibrated)
    if the replay hasn't been built with spreads yet."""
    if not SPREADS_PATH.exists():
        logger.warning("nfl_legs: %s missing — run python -m backend.backtest.nfl_replay; legs are uncalibrated", SPREADS_PATH)
        return {m: None for m in NFL_MARKETS}
    stamp = SPREADS_PATH.stat().st_mtime
    cache = MODELS_DIR / "nfl_ladder_calibration.pkl"
    if cache.exists():
        cached = pickle.loads(cache.read_bytes())
        if cached.get("stamp") == stamp:
            return cached["calibrators"]
    spreads = pickle.loads(SPREADS_PATH.read_bytes())
    df = _replay_frame()
    cals = {m: _fit_calibrator(df, spreads[m], m) for m in NFL_MARKETS}
    cache.write_bytes(pickle.dumps({"stamp": stamp, "calibrators": cals}))
    logger.info("nfl_legs: fit ladder calibrators for %s", ", ".join(cals))
    return cals


# ---- correlations -------------------------------------------------------------------

def correlation_matrices() -> dict[str, dict]:
    """Pooled same-player / teammate / opponent residual correlations for the
    four NFL markets, same estimator as NBA (correlations.py)."""
    df = _replay_frame()
    return {rel: _pairwise_corr(df, rel, NFL_MARKETS) for rel in ("same_player", "teammate", "opponent")}


# ---- recent-week diagnostics ------------------------------------------------------------

def narrow_range_warnings() -> dict[str, str]:
    """Per market: a warning when the latest finished week's 25th-75th ranges
    caught too few actuals (the model was overconfident that week)."""
    df = _replay_frame()
    if df.empty:
        return {}
    season, week = df[["season", "week"]].sort_values(["season", "week"]).iloc[-1]
    last = df[(df["season"] == season) & (df["week"] == week)]
    out = {}
    for market, g in last.groupby("market"):
        coverage = float(((g["actual"] >= g["lo"]) & (g["actual"] <= g["hi"])).mean())
        if coverage < NARROW_COVERAGE:
            out[market] = (
                f"{MARKET_NOUN[market]} ranges ran narrow in Week {week}: only {coverage:.0%} of actuals landed inside the "
                f"25th–75th range (about 50% expected), so these probabilities may be overconfident."
            )
    return out


# ---- pool ----------------------------------------------------------------------------

def _unstarted(game: dict, now: datetime) -> bool:
    if game["final"] or not game["kickoff_utc"]:
        return False
    return datetime.fromisoformat(game["kickoff_utc"]).astimezone(timezone.utc) > now


def default_week(week_list: list[dict], min_games: int = 3) -> int | None:
    """First week with at least `min_games` games still to kick off (so a
    lone Monday-night game doesn't become the whole slate), else the first
    with any."""
    now = datetime.now(timezone.utc)
    upcoming = [w for w in week_list if w["last_kickoff_utc"] and datetime.fromisoformat(w["last_kickoff_utc"]).astimezone(timezone.utc) > now]
    for w in upcoming:
        if w["n_games"] - w["n_final"] >= min_games and datetime.fromisoformat(w["first_kickoff_utc"]).astimezone(timezone.utc) > now:
            return w["week"]
    return upcoming[0]["week"] if upcoming else None


def build_pool(week: int | None = None) -> dict:
    """{legs, games, season, week, weeks, injuries_posted, warnings} for one
    week's not-yet-started games."""
    if week is None:
        week = default_week(nfl_projections.get()["weeks"])
    proj = nfl_projections.get(week)
    season = proj["season"]
    version = nfl_projections.version_for(season, proj["week"])  # the version that made these projections
    spreads = {m.name: nfl_projections._get_or_train(m, version)[1] for m in MARKETS}
    cals = calibrators()
    narrow = narrow_range_warnings()
    now = datetime.now(timezone.utc)

    legs: list[NflLeg] = []
    games = {}
    players: dict[str, dict] = {}
    for g in proj["games"]:
        if not _unstarted(g, now):
            continue
        games[g["game_id"]] = {"game_id": g["game_id"], "home": g["home"], "away": g["away"], "kickoff_utc": g["kickoff_utc"]}
        for side in ("away", "home"):
            t = g["teams"][side]
            for p in t["players"]:
                player = {
                    "player_id": p["player_id"], "player_name": p["player_name"], "position": p["position"],
                    "team": t["team"], "opponent": t["opponent"], "is_home": side == "home", "game_id": g["game_id"],
                    "kickoff_utc": g["kickoff_utc"], "questionable": p["questionable"],
                    "means": {m: v["mean"] for m, v in p["projections"].items()},
                }
                players[p["player_id"]] = player
                for market in player["means"]:
                    for rung in LADDERS[market]:
                        leg = make_leg(player, market, "over", rung_line(rung), spreads, cals, narrow, rung=rung)
                        if P_RANGE[0] <= leg.p_hit() <= P_RANGE[1]:
                            legs.append(leg)

    logger.info("nfl_legs: week %s: %d legs from %d unstarted games", week, len(legs), len(games))
    return {
        "legs": legs, "games": games, "season": season, "week": week, "weeks": proj["weeks"],
        "injuries_posted": proj["injuries_posted"], "computed_at": proj["computed_at"],
        "calibrated": all(c is not None for c in cals.values()),
        "players": players, "out": out_players(season, week, {t for g in games.values() for t in (g["home"], g["away"])}),
        "spreads": spreads, "calibrators": cals, "narrow": narrow,
    }


def make_leg(player: dict, market: str, side: str, line: float, spreads: dict, cals: dict, narrow: dict,
             rung: int | None = None) -> NflLeg:
    """One leg for a projected player (a `build_pool` players entry) at any
    line, over or under. `rung` marks an over at a ladder value ("60+")."""
    return NflLeg(
        player_id_nba=player["player_id"], player_name=player["player_name"], game_id=player["game_id"],
        team=player["team"], market=market, side=side, line=line,
        predicted_mean=player["means"][market], dispersion=spreads[market].r or 0.0, decimal_odds=0.0,
        calibrator=cals[market], spread=spreads[market], position=player["position"],
        opponent=player["opponent"], is_home=player["is_home"], kickoff_utc=player["kickoff_utc"],
        questionable=player["questionable"], rung=rung if side == "over" else None,
        warnings=(narrow[market],) if market in narrow else (),
    )


def out_players(season: int, week: int, teams: set[str]) -> list[dict]:
    """Skill-position players listed Out this week on these teams."""
    if not teams:
        return []
    conn = duckdb.connect(_db_path(), read_only=True)
    try:
        rows = conn.execute(
            "SELECT player_id, any_value(player_name), team, any_value(position) FROM nfl_injuries "
            "WHERE season = ? AND week = ? AND report_status = 'Out' AND position IN ('QB', 'RB', 'WR', 'TE') "
            f"AND team IN ({', '.join('?' * len(teams))}) GROUP BY player_id, team ORDER BY team, 2",
            [season, week, *sorted(teams)],
        ).fetchall()
    finally:
        conn.close()
    return [{"player_id": r[0], "player_name": r[1] or r[0], "team": r[2], "position": r[3]} for r in rows]
