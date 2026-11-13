"""Model vs. lines for NBA points: the logged predictions
(model_predictions_log) against real sportsbook lines (odds_snapshots) and
the actual points (nba_player_games), plus a replay of the parlays the
Builder would have suggested each day. Reads the warehouse only; makes no
API calls and writes nothing but the JSON report.

Points only: odds_snapshots collects player_points and nothing else, so the
other logged markets have no lines to grade against.

One row per logged points prediction:
- Line: ONE book's last snapshot pulled before tip-off. DraftKings when it
  has one with both prices, else FanDuel (BOOK_PRIORITY); every row records
  its book. Odds names are resolved the way the Builder resolves them
  (leg_pool._resolve_player_names), and an event is tied to the scheduled
  game by player and ET date (a player plays at most one game a day).
- Features: only predictions whose features_through is dated before the
  game. Rows logged before that column existed (NULL) can't be checked and
  are left out, not guessed.
- Finished: the game is final in nba_game_results, its box score is loaded,
  or it tipped more than FINAL_AFTER ago. A finished game without a box
  score is "no box score yet"; a loaded box score without the player is
  "did not play", which books void.
- Probabilities: the model's P(over) is the Builder's own (Normal with the
  logged mean and sigma, through the points isotonic calibrator). On a
  whole-number line the push probability is taken out first, so it's
  P(over | no push). The book's is the no-vig over from the same snapshot.
  Pushes are left out of hit rates, Brier and calibration.

Builder replay: for each slate day, the legs the Builder would have had at
its build (the last odds pull before that day's first tip, same book rule),
priced with the predictions logged by then, run through the Builder's own
search (backend/parlay/search.run_search: tiers, 0.30 shrink toward the
book, sanity guard, Monte Carlo, fixed seed). Each tier's top
DEFAULT_RESULTS_PER_TIER parlays are graded. Parlays logged live at build
time (nba_suggested_parlays_log) are graded the same way.

Recommendation: a shrink toward the book's no-vig probability and a cap on
the model-vs-book gap, p' = book + clip((1 - w) * (model - book), -cap, cap),
chosen by Brier score over a grid. It is reported, never applied; the
replay is re-run with it to show what the suggested parlays would have done.

    docker compose run --rm --no-deps api python -m backend.backtest.nba_model_vs_lines
"""

from __future__ import annotations

import json
import math
import os
import pickle
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from backend.api.slate import REFERENCE_BANKROLL, SEARCH_SEED, load_tiers
from backend.ingest.util import logger
from backend.parlay.monte_carlo import DEFAULT_N_SIMULATIONS, Leg
from backend.parlay.search import DEFAULT_RESULTS_PER_TIER, run_search
from backend.parlay.suggestion_log import TABLE as SUGGESTIONS_TABLE
from backend.predict.schedule_projections import ET

MARKET = "points"
ODDS_MARKET = "player_points"
BOOK_PRIORITY = ("draftkings", "fanduel")
BOOK_LABELS = {"draftkings": "DraftKings", "fanduel": "FanDuel"}
MIN_SAMPLE = 300
MIN_RECOMMEND = 50  # graded props before a shrink/cap is suggested at all
FINAL_AFTER = timedelta(hours=4)
BUCKET_EDGES = np.linspace(0, 1, 11)
GAP_BUCKETS = ((0, 5), (5, 10), (10, 15), (15, None))  # probability points
STAR_COUNT = 30
STAR_MIN_GAMES = 40
SHRINK_GRID = tuple(round(float(w), 1) for w in np.arange(0, 1.01, 0.1))
CAP_GRID = (0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, None)
CALIBRATOR_PATH = Path("data/models/points_calibration.pkl")
REPORT_PATH = Path("reports/nba_model_vs_lines.json")

BOOK_RULE = ("One book per prop: the last DraftKings snapshot pulled before tip-off with both prices, "
             "otherwise FanDuel's. No cross-book consensus, so every price is one a bettor could have taken.")
FEATURES_RULE = ("Only predictions whose features were built from game logs dated before the game. Predictions logged "
                 "before the app recorded that date can't be checked and are left out.")
STAR_RULE = (f"Stars: the top {STAR_COUNT} scorers by points per game the season before "
             f"(at least {STAR_MIN_GAMES} games).")
REPLAY_RULE = ("Each day's Builder run is rebuilt from what it would have had: the last odds pull before the day's first "
               "tip (DraftKings, else FanDuel), the predictions logged by then, the Builder's own search, tiers, "
               "0.30 shrink and sanity guard. The current points calibrator stands in for the one in use then.")

STATUS_ORDER = ("graded", "push", "no_line", "no_features_date", "features_not_before_game", "not_finished",
                "no_box_score", "did_not_play")


# ---- loading ------------------------------------------------------------------

def _naive_utc(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert("UTC").dt.tz_localize(None) if s.dt.tz is not None else s


def load_predictions(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Every logged points prediction with its game, result state and actual points."""
    df = conn.execute(
        """
        WITH loaded AS (SELECT DISTINCT game_id FROM nba_player_games)
        SELECT p.game_id, p.player_id_nba, p.predicted_at, p.predicted_mean, p.dispersion_param AS sigma,
               p.model_version, p.features_through, s.game_date, s.tip_utc, s.season, s.home_team_id, s.away_team_id,
               r.game_status, l.game_id IS NOT NULL AS box_loaded,
               g.pts AS actual, g.team AS box_team, g.player_name AS box_name
        FROM model_predictions_log p
        JOIN nba_schedule s USING (game_id)
        LEFT JOIN nba_game_results r USING (game_id)
        LEFT JOIN loaded l USING (game_id)
        LEFT JOIN nba_player_games g ON g.game_id = p.game_id AND g.player_id_nba = p.player_id_nba
        WHERE p.market = ?
        """,
        [MARKET],
    ).fetchdf()
    df["tip"] = _naive_utc(df["tip_utc"])
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    return df


def load_odds(conn: duckdb.DuckDBPyConnection, resolve=None) -> tuple[pd.DataFrame, list[str]]:
    """(snapshots from BOOK_PRIORITY books with both prices, each with its
    player_id_nba and ET date; names that didn't resolve)."""
    odds = conn.execute(
        f"""SELECT pulled_at, event_id, commence_time, player_name_raw, bookmaker, line, over_price, under_price
            FROM odds_snapshots
            WHERE market = ? AND bookmaker IN ({', '.join('?' * len(BOOK_PRIORITY))})
              AND line IS NOT NULL AND over_price > 1 AND under_price > 1""",
        [ODDS_MARKET, *BOOK_PRIORITY],
    ).fetchdf()
    if odds.empty:
        return odds.assign(player_id_nba=pd.Series(dtype=str), game_date=pd.Series(dtype=object)), []
    if resolve is None:
        from backend.parlay.leg_pool import _resolve_player_names as resolve
    names = odds["player_name_raw"].unique().tolist()
    name_map = resolve(conn, names)
    odds["player_id_nba"] = odds["player_name_raw"].map(name_map)
    odds["commence"] = _naive_utc(odds["commence_time"])
    odds["game_date"] = pd.to_datetime(odds["commence_time"]).dt.tz_convert(ET).dt.date
    return odds.dropna(subset=["player_id_nba"]), sorted(n for n in names if n not in name_map)


def pick_lines(odds: pd.DataFrame, games: pd.DataFrame, cutoff: str = "tip") -> pd.DataFrame:
    """One snapshot per (game_id, player): the last pulled before
    games[cutoff] (and before the event's own start), from the first book
    in BOOK_PRIORITY that has one. `games` has game_id, player_id_nba,
    game_date and the cutoff column."""
    cols = ["game_id", "player_id_nba", "bookmaker", "line", "over_price", "under_price", "pulled_at", "event_id"]
    if odds.empty or games.empty:
        return pd.DataFrame(columns=cols)
    m = games[["game_id", "player_id_nba", "game_date", cutoff]].merge(odds, on=["player_id_nba", "game_date"])
    m = m[(m["pulled_at"] < m[cutoff]) & (m["pulled_at"] < m["commence"])]
    if m.empty:
        return pd.DataFrame(columns=cols)
    m = m.sort_values("pulled_at").groupby(["game_id", "player_id_nba", "bookmaker"], as_index=False).last()
    m["rank"] = m["bookmaker"].map({b: i for i, b in enumerate(BOOK_PRIORITY)})
    return m.sort_values("rank").drop_duplicates(["game_id", "player_id_nba"])[cols]


# ---- probabilities ------------------------------------------------------------

def no_vig_over(over_price, under_price):
    io, iu = 1 / np.asarray(over_price, float), 1 / np.asarray(under_price, float)
    return io / (io + iu)


def model_p_over(mean, sigma, line, calibrator=None) -> tuple[np.ndarray, np.ndarray]:
    """(calibrated, raw) P(over), the Builder's Normal; on a whole-number
    line, P(over | no push) with points as whole numbers."""
    mean, sigma, line = (np.asarray(x, float) for x in (mean, sigma, line))
    whole = np.isclose(line, np.round(line))
    raw = 1 - stats.norm.cdf(line, mean, sigma)
    over = 1 - stats.norm.cdf(line + 0.5, mean, sigma)
    push = stats.norm.cdf(line + 0.5, mean, sigma) - stats.norm.cdf(line - 0.5, mean, sigma)
    raw = np.where(whole, over / np.maximum(1 - push, 1e-9), raw)
    raw = np.clip(raw, 0.0, 1.0)
    cal = np.asarray(calibrator.predict(raw), float) if calibrator is not None and len(raw) else raw
    return cal, raw


def adjust(p_model, p_book, shrink: float, cap: float | None):
    """The recommended fix: shrink the model's gap to the book by `shrink`,
    then cap it at `cap` probability."""
    gap = (1 - shrink) * (np.asarray(p_model, float) - np.asarray(p_book, float))
    if cap is not None:
        gap = np.clip(gap, -cap, cap)
    return np.asarray(p_book, float) + gap


# ---- grading ------------------------------------------------------------------

def grade(preds: pd.DataFrame, lines: pd.DataFrame, now: datetime, calibrator=None) -> pd.DataFrame:
    """preds (load_predictions) + their closing line (pick_lines) -> one row
    per prediction with status, p_model, p_book, outcome (1 over, 0 under,
    NaN push/ungraded) and push."""
    df = preds.merge(lines, how="left", on=["game_id", "player_id_nba"])
    now = now.astimezone(timezone.utc).replace(tzinfo=None) if now.tzinfo else now
    has_line = df["line"].notna()
    ft = pd.to_datetime(df["features_through"])
    gd = pd.to_datetime(df["game_date"])
    finished = (df["game_status"] == 3) | df["box_loaded"] | (df["tip"] + FINAL_AFTER <= now)
    status = np.select(
        [~has_line, ft.isna(), ft >= gd, ~finished, ~df["box_loaded"], df["actual"].isna()],
        ["no_line", "no_features_date", "features_not_before_game", "not_finished", "no_box_score", "did_not_play"],
        default="graded",
    )
    df["push"] = (status == "graded") & (df["actual"] == df["line"])
    df["status"] = np.where(df["push"], "push", status)
    df["p_model"] = df["p_model_raw"] = df["p_book"] = np.nan
    if has_line.any():
        cal, raw = model_p_over(df.loc[has_line, "predicted_mean"], df.loc[has_line, "sigma"], df.loc[has_line, "line"], calibrator)
        df.loc[has_line, "p_model"], df.loc[has_line, "p_model_raw"] = cal, raw
        df.loc[has_line, "p_book"] = no_vig_over(df.loc[has_line, "over_price"], df.loc[has_line, "under_price"])
    df["outcome"] = np.where(df["status"] == "graded", (df["actual"] > df["line"]).astype(float), np.nan)
    return df


def star_ids(conn: duckdb.DuckDBPyConnection, seasons) -> dict[str, set[str]]:
    """{season: player ids of the top STAR_COUNT scorers the season before}."""
    out = {}
    for season in seasons:
        y = int(str(season)[:4]) - 1
        prev = f"{y}-{str(y + 1)[2:]}"
        out[season] = {pid for (pid,) in conn.execute(
            """SELECT player_id_nba FROM nba_player_games WHERE season = ? GROUP BY 1 HAVING count(*) >= ?
               ORDER BY avg(pts) DESC LIMIT ?""", [prev, STAR_MIN_GAMES, STAR_COUNT]).fetchall()}
    return out


# ---- metrics ------------------------------------------------------------------

def _r(x, nd=4):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)


def buckets(p, y) -> list[dict]:
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.clip(np.digitize(p, BUCKET_EDGES[1:-1]), 0, len(BUCKET_EDGES) - 2)
    out = []
    for i in range(len(BUCKET_EDGES) - 1):
        m = idx == i
        if m.any():
            out.append({"lo": _r(BUCKET_EDGES[i], 1), "hi": _r(BUCKET_EDGES[i + 1], 1), "n": int(m.sum()),
                        "mean_p": _r(p[m].mean()), "hit_rate": _r(y[m].mean())})
    return out


def calibration_error(p, y) -> float | None:
    """Bucket-weighted mean |predicted - actual| (expected calibration error)."""
    b = buckets(p, y)
    n = sum(x["n"] for x in b)
    return sum(x["n"] * abs(x["mean_p"] - x["hit_rate"]) for x in b) / n if n else None


def _brier(p, y) -> float | None:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2)) if len(y) else None


def _side(gp: pd.DataFrame, side: str) -> dict:
    over = side == "over"
    pm = gp["p_model"] if over else 1 - gp["p_model"]
    pb = gp["p_book"] if over else 1 - gp["p_book"]
    y = gp["outcome"] if over else 1 - gp["outcome"]
    n = len(gp)
    return {
        "n": n,
        "hit_rate": _r(y.mean()) if n else None,
        "mean_model": _r(pm.mean()) if n else None,
        "mean_book": _r(pb.mean()) if n else None,
        "calibration_model": buckets(pm, y) if n else [],
        "calibration_book": buckets(pb, y) if n else [],
        "ece_model": _r(calibration_error(pm, y)) if n else None,
        "ece_book": _r(calibration_error(pb, y)) if n else None,
    }


def _model_side(gp: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(picked, hit): the model's side is over when it rates the over above
    the book's no-vig price; props where they agree exactly have no side."""
    picked = gp["p_model"] != gp["p_book"]
    over = gp["p_model"] > gp["p_book"]
    hit = (over & (gp["outcome"] == 1)) | (~over & (gp["outcome"] == 0))
    return picked, hit


def gap_table(gp: pd.DataFrame) -> list[dict]:
    gap = (gp["p_model"] - gp["p_book"]).abs() * 100
    picked, hit = _model_side(gp)
    over = gp["p_model"] > gp["p_book"]
    out = []
    for lo, hi in GAP_BUCKETS:
        m = picked & (gap >= lo) & ((gap < hi) if hi is not None else True)
        n = int(m.sum())
        out.append({"lo": lo, "hi": hi, "n": n, "hits": int((hit & m).sum()), "hit_rate": _r(hit[m].mean()) if n else None,
                    "n_over": int((m & over).sum()), "n_under": int((m & ~over).sum()),
                    "over_hit_rate": _r(hit[m & over].mean()) if (m & over).any() else None,
                    "under_hit_rate": _r(hit[m & ~over].mean()) if (m & ~over).any() else None})
    return out


def _group(g: pd.DataFrame) -> dict:
    """Hit rates and error for one player group (graded rows incl. pushes)."""
    gp = g[g["status"] == "graded"]
    picked, hit = _model_side(gp)
    err = g["predicted_mean"] - g["actual"]
    return {
        "n": len(gp), "players": int(g["player_id_nba"].nunique()),
        "model_side_hit_rate": _r(hit[picked].mean()) if picked.any() else None,
        "over_rate": _r(gp["outcome"].mean()) if len(gp) else None,
        "brier_model": _r(_brier(gp["p_model"], gp["outcome"])),
        "brier_book": _r(_brier(gp["p_book"], gp["outcome"])),
        "n_error": len(g),
        "mean_error": _r(err.mean(), 2) if len(g) else None,  # model minus actual: positive = projected too high
        "mae": _r(err.abs().mean(), 2) if len(g) else None,
        "mean_line_minus_actual": _r((g["line"] - g["actual"]).mean(), 2) if len(g) else None,
    }


def summarize(df: pd.DataFrame) -> dict:
    g = df[df["status"].isin(["graded", "push"])]
    gp = g[g["status"] == "graded"]
    picked, hit = _model_side(gp)
    return {
        "graded": len(gp),
        "pushes": int((g["status"] == "push").sum()),
        "enough_sample": len(gp) >= MIN_SAMPLE,
        "brier_model": _r(_brier(gp["p_model"], gp["outcome"])),
        "brier_model_raw": _r(_brier(gp["p_model_raw"], gp["outcome"])),
        "brier_book": _r(_brier(gp["p_book"], gp["outcome"])),
        "model_side_hit_rate": _r(hit[picked].mean()) if picked.any() else None,
        "model_side_n": int(picked.sum()),
        "over": _side(gp, "over"),
        "under": _side(gp, "under"),
        "gap": gap_table(gp),
        "stars": _group(g[g["star"]]),
        "others": _group(g[~g["star"]]),
    }


def recommend(gp: pd.DataFrame) -> dict:
    """Grid-search the shrink toward the book and the gap cap by Brier
    score; ties go to the smaller change (less shrink, looser cap)."""
    n = len(gp)
    if n < MIN_RECOMMEND:
        return {"available": False, "n": n,
                "reason": f"Needs at least {MIN_RECOMMEND} graded props to suggest anything; there are {n}."}
    pm, pb, y = gp["p_model"].to_numpy(), gp["p_book"].to_numpy(), gp["outcome"].to_numpy()
    grid = []
    for w in SHRINK_GRID:
        for cap in CAP_GRID:
            p = adjust(pm, pb, w, cap)
            grid.append((round(_brier(p, y), 6), w, -(cap if cap is not None else 1.0), cap))
    grid.sort()
    _, w, _, cap = grid[0]
    p = adjust(pm, pb, w, cap)
    return {
        "available": True, "n": n, "enough_sample": n >= MIN_SAMPLE, "shrink": w, "cap": cap,
        "brier_before": _r(_brier(pm, y)), "brier_after": _r(_brier(p, y)), "brier_book": _r(_brier(pb, y)),
        "ece_before": _r(calibration_error(pm, y)), "ece_after": _r(calibration_error(p, y)),
        "ece_book": _r(calibration_error(pb, y)),
        "calibration_after": buckets(p, y),
        "runner_up": [{"shrink": g[1], "cap": g[3], "brier": g[0]} for g in grid[1:4]],
    }


# ---- Builder replay ---------------------------------------------------------------

class AdjustedCalibrator:
    """Applies `adjust` after the points calibrator, so a Leg (and the Monte
    Carlo, which reads the calibrated probability) sees the adjusted P(over)."""

    def __init__(self, base, p_book_over: float, shrink: float, cap: float | None):
        self.base, self.p_book, self.shrink, self.cap = base, p_book_over, shrink, cap

    def predict(self, raw):
        p = np.asarray(self.base.predict(raw), float) if self.base is not None else np.asarray(raw, float)
        return adjust(p, np.full_like(p, self.p_book), self.shrink, self.cap)


def _abbr() -> dict[str, str]:
    from nba_api.stats.static import teams

    return {str(t["id"]): t["abbreviation"] for t in teams.get_teams()}


def last_teams(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Every (player, game_date, team) in the game logs, for placing a player
    on one of a game's two teams as of that day."""
    return conn.execute("SELECT player_id_nba, game_date, team FROM nba_player_games ORDER BY game_date").fetchdf()


def place(rows: pd.DataFrame, history: pd.DataFrame, abbr: dict[str, str]) -> pd.DataFrame:
    """Adds team / opponent / is_home: the player's team in this game's box
    score, else their latest logged team before the game, kept only when
    it's one of the game's two teams."""
    rows = rows.copy()
    rows["home"] = rows["home_team_id"].astype(str).map(abbr)
    rows["away"] = rows["away_team_id"].astype(str).map(abbr)
    hist = history.assign(game_date=pd.to_datetime(history["game_date"]).astype("datetime64[ns]")).sort_values("game_date")
    left = rows.assign(_d=(pd.to_datetime(rows["game_date"]) - timedelta(days=1)).astype("datetime64[ns]")).sort_values("_d")
    prior = pd.merge_asof(left, hist.rename(columns={"game_date": "_d", "team": "_prior"}), on="_d", by="player_id_nba")
    prior = prior.set_index(["game_id", "player_id_nba"])["_prior"]
    prior_team = pd.Series([prior.get((g, p)) for g, p in zip(rows["game_id"], rows["player_id_nba"])], index=rows.index)
    team = rows["box_team"].where(rows["box_team"].notna(), prior_team)
    is_home = team == rows["home"]
    ok = is_home | (team == rows["away"])
    rows["team"], rows["is_home"] = team.where(ok), is_home.where(ok)
    rows["opponent"] = np.where(is_home, rows["away"], rows["home"])
    return rows


def day_legs(rows: pd.DataFrame, calibrator, adjustment: dict | None = None) -> list[Leg]:
    legs = []
    for r in rows.itertuples():
        cal = calibrator
        if adjustment:
            cal = AdjustedCalibrator(calibrator, float(no_vig_over(r.over_price, r.under_price)), adjustment["shrink"], adjustment["cap"])
        base = dict(player_id_nba=r.player_id_nba, player_name=r.box_name if isinstance(r.box_name, str) else r.player_name_raw,
                    game_id=r.game_id, team=r.team, opponent=r.opponent, is_home=bool(r.is_home), market=MARKET,
                    line=float(r.line), predicted_mean=float(r.predicted_mean), dispersion=float(r.sigma),
                    over_price=float(r.over_price), under_price=float(r.under_price), calibrator=cal)
        legs.append(Leg(side="over", decimal_odds=float(r.over_price), **base))
        legs.append(Leg(side="under", decimal_odds=float(r.under_price), **base))
    return legs


@dataclass
class ReplayDay:
    day: date
    build_at: datetime
    rows: pd.DataFrame  # one per player prop available at build time, placed on a team
    early_season: bool


def replay_days(conn, preds: pd.DataFrame, odds: pd.DataFrame, history: pd.DataFrame, abbr: dict, early_fn) -> list[ReplayDay]:
    """The Builder's inputs on each slate day: predictions with features
    dated before the game, the last odds pull before the day's first tip,
    and only predictions logged by then."""
    ok = preds[pd.to_datetime(preds["features_through"]) < pd.to_datetime(preds["game_date"])]
    out = []
    for day, g in ok.groupby("game_date"):
        first_tip = g["tip"].min()
        day_odds = odds[(odds["game_date"] == day) & (odds["pulled_at"] < first_tip)]
        if day_odds.empty:
            continue
        build_at = day_odds["pulled_at"].max()
        g = g[g["predicted_at"] <= build_at].assign(cutoff=build_at + timedelta(microseconds=1))
        lines = pick_lines(day_odds, g, cutoff="cutoff")
        rows = g.merge(lines, on=["game_id", "player_id_nba"])
        if rows.empty:
            continue
        names = day_odds.drop_duplicates("player_id_nba").set_index("player_id_nba")["player_name_raw"]
        rows["player_name_raw"] = rows["player_id_nba"].map(names)
        rows = place(rows, history, abbr).dropna(subset=["team"])
        out.append(ReplayDay(day, build_at, rows, bool(early_fn(day))))
    return out


def grade_leg(actual: float | None, line: float, side: str, state: str) -> str:
    """hit | miss | push | void (did not play) | pending."""
    if state == "pending":
        return "pending"
    if state == "did_not_play" or actual is None or (isinstance(actual, float) and math.isnan(actual)):
        return "void"
    if actual == line:
        return "push"
    return "hit" if (actual > line) == (side == "over") else "miss"


def grade_parlay(results: list[str]) -> str:
    """lost if any leg missed; pending while any leg is; won if every leg
    hit; reduced if the rest hit around a push or void (books pay those at
    shorter odds, so they're left out of the hit-rate comparison); void if
    no leg stood."""
    if "miss" in results:
        return "lost"
    if "pending" in results:
        return "pending"
    if all(r == "hit" for r in results):
        return "won"
    return "reduced" if "hit" in results else "void"


def leg_state(results: pd.DataFrame) -> dict:
    """{(game_id, player_id_nba): (actual, pending | did_not_play | final)}
    from grade()'s statuses; also keyed by (game_date, player_id_nba)."""
    out = {}
    for r in results.itertuples():
        if r.status in ("not_finished", "no_box_score"):
            state = "pending"
        elif r.status == "did_not_play" or (r.box_loaded and pd.isna(r.actual)):
            state = "did_not_play"
        elif r.box_loaded:
            state = "final"
        else:
            state = "pending"
        actual = None if pd.isna(r.actual) else float(r.actual)
        out[(r.game_id, r.player_id_nba)] = out[(str(r.game_date), r.player_id_nba)] = (actual, state)
    return out


def run_replay(days: list[ReplayDay], calibrator, corr: dict, tiers: dict, states: dict,
               adjustment: dict | None = None, n_simulations: int = DEFAULT_N_SIMULATIONS) -> list[dict]:
    out = []
    for d in days:
        legs = day_legs(d.rows, calibrator, adjustment)
        for tier, cfg in tiers.items():
            results, _ = run_search(legs, tier, cfg, corr, REFERENCE_BANKROLL, n_simulations=n_simulations,
                                    seed=SEARCH_SEED, early_season=d.early_season)
            for rank, c in enumerate(results[:DEFAULT_RESULTS_PER_TIER], start=1):
                leg_rows = []
                for leg in c.legs:
                    actual, state = states.get((leg.game_id, leg.player_id_nba), (None, "pending"))
                    leg_rows.append({"player_name": leg.player_name, "side": leg.side, "line": leg.line,
                                     "decimal_odds": leg.decimal_odds, "p_hit": _r(leg.p_hit()), "no_vig_p": _r(leg.no_vig_p_hit()),
                                     "actual": actual, "result": grade_leg(actual, leg.line, leg.side, state)})
                out.append({"date": d.day.isoformat(), "tier": tier, "rank": rank, "legs": leg_rows,
                            "p_hit": _r(c.p_hit), "p_sized": _r(c.p_sized), "no_vig_p": _r(c.no_vig_p),
                            "decimal_odds": _r(c.combined_decimal_odds, 3), "flagged": c.flagged,
                            "result": grade_parlay([l["result"] for l in leg_rows])})
    return out


def grade_logged(conn: duckdb.DuckDBPyConnection, states: dict) -> list[dict]:
    """Parlays the live Builder logged (first appearance of each per tier)."""
    try:
        df = conn.execute(
            f"""SELECT * FROM {SUGGESTIONS_TABLE}
                QUALIFY row_number() OVER (PARTITION BY tier, parlay_key ORDER BY built_at) = 1 ORDER BY built_at, tier, rank"""
        ).fetchdf()
    except duckdb.CatalogException:
        return []
    out = []
    for r in df.itertuples():
        leg_rows = []
        for leg in json.loads(r.legs):
            key = (leg.get("nba_game_id"), leg["player_id_nba"])
            actual, state = states.get(key) or states.get((leg.get("game_date"), leg["player_id_nba"]), (None, "pending"))
            leg_rows.append({"player_name": leg["player_name"], "side": leg["side"], "line": leg["line"],
                             "decimal_odds": leg["decimal_odds"], "p_hit": _r(leg["p_hit"]), "no_vig_p": _r(leg["no_vig_p"]),
                             "actual": actual, "result": grade_leg(actual, leg["line"], leg["side"], state)})
        out.append({"date": pd.Timestamp(r.built_at).tz_localize("UTC").tz_convert(ET).date().isoformat(), "tier": r.tier,
                    "rank": int(r.rank), "legs": leg_rows, "p_hit": _r(r.p_hit), "p_sized": _r(r.p_sized),
                    "no_vig_p": _r(r.no_vig_p), "decimal_odds": _r(r.decimal_odds, 3), "flagged": bool(r.flagged),
                    "result": grade_parlay([l["result"] for l in leg_rows])})
    return out


def parlay_summary(parlays: list[dict], tiers) -> dict:
    """Per tier: claimed hit chance vs. how often they actually hit, over
    parlays settled won or lost; flat one-unit profit at the parlay price."""
    out = {}
    for tier in tiers:
        ps = [p for p in parlays if p["tier"] == tier]
        settled = [p for p in ps if p["result"] in ("won", "lost")]
        clean = [p for p in settled if not p["flagged"]]
        won = [p for p in settled if p["result"] == "won"]
        profit = sum(p["decimal_odds"] - 1 for p in won) - (len(settled) - len(won))
        out[tier] = {
            "suggested": len(ps), "settled": len(settled), "won": len(won),
            "pending": sum(p["result"] == "pending" for p in ps),
            "reduced_or_void": sum(p["result"] in ("reduced", "void") for p in ps),
            "flagged": sum(p["flagged"] for p in ps),
            "claimed": _r(np.mean([p["p_hit"] for p in settled])) if settled else None,
            "claimed_sized": _r(np.mean([p["p_sized"] for p in settled])) if settled else None,
            "book": _r(np.mean([p["no_vig_p"] for p in settled if p["no_vig_p"] is not None])) if settled else None,
            "actual": _r(len(won) / len(settled)) if settled else None,
            "profit": _r(profit, 2) if settled else None,
            "clean_settled": len(clean),
            "clean_claimed": _r(np.mean([p["p_hit"] for p in clean])) if clean else None,
            "clean_actual": _r(sum(p["result"] == "won" for p in clean) / len(clean)) if clean else None,
        }
    return out


# ---- report -------------------------------------------------------------------------

def _pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def plain_summary(r: dict) -> list[str]:
    o, n, cov = r["overall"], r["overall"]["graded"], r["coverage"]
    out = []
    with_line = sum(v for k, v in cov.items() if k != "no_line")
    if n == 0:
        waiting = cov.get("not_finished", 0) + cov.get("no_box_score", 0)
        out.append(
            f"No points props can be graded yet. {with_line} logged predictions have a DraftKings or FanDuel line: "
            f"{cov.get('no_features_date', 0)} were logged before the app recorded the features date, and {waiting} are "
            f"waiting on a box score (game logs are loaded through {r['box_scores_through'] or 'no date'}). "
            "There is nothing to conclude.")
    elif not o["enough_sample"]:
        out.append(f"Only {n} graded props, under the {MIN_SAMPLE} needed: too small a sample to conclude anything. "
                   "Treat every number below as noise for now.")
    if n:
        bm, bb = o["brier_model"], o["brier_book"]
        out.append(f"The model's probabilities scored {bm:.4f} (Brier, lower is better) against the book's no-vig "
                   f"{bb:.4f}: {'better' if bm < bb else 'worse' if bm > bb else 'the same'} than the line.")
        for side in ("over", "under"):
            s = o[side]
            out.append(f"{side.capitalize()}s hit {_pct(s['hit_rate'])} of {s['n']} props; the model gave them "
                       f"{_pct(s['mean_model'])} on average and the book {_pct(s['mean_book'])}.")
        big = next((g for g in reversed(o["gap"]) if g["n"]), None)
        if big:
            label = f"{big['lo']}+" if big["hi"] is None else f"{big['lo']}-{big['hi']}"
            out.append(f"When the model and the book disagreed most ({label} points), the model's side hit "
                       f"{_pct(big['hit_rate'])} of {big['n']} props.")
        st = o["stars"]
        if st["n_error"]:
            out.append(f"For stars the model was off by {st['mean_error']:+.1f} points on average "
                       f"({'projected too high' if st['mean_error'] > 0 else 'projected too low'}) across {st['n_error']} props.")
    b = r["builder"]["tiers"]
    settled = sum(t["settled"] for t in b.values())
    suggested = sum(t["suggested"] for t in b.values())
    if suggested:
        msg = f"Replaying the Builder over {len(r['builder']['days'])} days gives {suggested} suggested parlays; {settled} have settled."
        for tier, t in b.items():
            if t["settled"]:
                msg += f" {tier.capitalize()}: claimed {_pct(t['claimed'])}, hit {_pct(t['actual'])} of {t['settled']}."
        out.append(msg)
    rec = r["recommendation"]
    if rec["available"]:
        cap = "no cap" if rec["cap"] is None else f"a {rec['cap'] * 100:.0f}-point cap on the gap"
        out.append(f"Suggested fix (not applied): shrink the model {rec['shrink'] * 100:.0f}% toward the book with {cap}; "
                   f"Brier {rec['brier_before']:.4f} -> {rec['brier_after']:.4f} on these props"
                   f"{'' if rec['enough_sample'] else ', on too small a sample to trust'}.")
    else:
        out.append(f"No shrink or cap suggested yet: {rec['reason']}")
    return out


def build_report(graded: pd.DataFrame, unresolved: list[str], n_odds_names: int, days: list[ReplayDay],
                 replay: list[dict], adjusted: list[dict] | None, logged: list[dict], tiers, box_through,
                 now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    gp = graded[graded["status"] == "graded"]
    counts = graded["status"].value_counts().to_dict()
    rec = recommend(gp)
    report = {
        "as_of": now.isoformat(timespec="seconds"),
        "market": MARKET,
        "book_rule": BOOK_RULE, "features_rule": FEATURES_RULE, "star_rule": STAR_RULE, "replay_rule": REPLAY_RULE,
        "min_sample": MIN_SAMPLE,
        "box_scores_through": str(box_through) if box_through else None,
        "books_used": {BOOK_LABELS.get(b, b): int(c) for b, c in gp["bookmaker"].value_counts().items()},
        "names": {"total": n_odds_names, "resolved": n_odds_names - len(unresolved), "unresolved": unresolved},
        "coverage": {k: int(counts.get(k, 0)) for k in STATUS_ORDER},
        "overall": summarize(graded),
        "recommendation": rec,
        "builder": {
            "days": [{"date": d.day.isoformat(), "build_at": d.build_at.isoformat(), "props": len(d.rows),
                      "early_season": d.early_season} for d in days],
            "tiers": parlay_summary(replay, tiers),
            "parlays": replay,
            "adjusted": {"tiers": parlay_summary(adjusted, tiers), "parlays": adjusted} if adjusted is not None else None,
            "logged": {"tiers": parlay_summary(logged, tiers), "parlays": logged},
        },
        "other_markets": "Rebounds, assists and 3-pointers are logged too, but no lines are collected for them yet.",
    }
    report["summary"] = plain_summary(report)
    return report


def _load_calibrator():
    return pickle.loads(CALIBRATOR_PATH.read_bytes()) if CALIBRATOR_PATH.exists() else None


def run(db_path: str | None = None, out: Path = REPORT_PATH, now: datetime | None = None,
        n_simulations: int = DEFAULT_N_SIMULATIONS) -> dict:
    from backend.parlay import correlations
    from backend.parlay.residuals import RESIDUALS_TABLE
    from backend.predict import early_season

    now = now or datetime.now(timezone.utc)
    calibrator = _load_calibrator()
    tiers = load_tiers()
    conn = duckdb.connect(db_path or os.environ.get("DUCKDB_PATH", "data/warehouse.duckdb"), read_only=True)
    try:
        preds = load_predictions(conn)
        odds, unresolved = load_odds(conn)
        n_names = odds["player_name_raw"].nunique() + len(unresolved)
        graded = grade(preds, pick_lines(odds, preds), now, calibrator)
        stars = star_ids(conn, graded["season"].unique())
        graded["star"] = [pid in stars.get(s, set()) for pid, s in zip(graded["player_id_nba"], graded["season"])]
        box_through = conn.execute("SELECT max(game_date) FROM nba_player_games").fetchone()[0]
        states = leg_state(graded)
        days = replay_days(conn, preds, odds, last_teams(conn), _abbr(),
                           lambda d: early_season.status(as_of=d, conn=conn).get("active"))
        # Same matrices as correlations.build_correlation_matrices, read from
        # the residuals cache without its build-if-missing step (read-only).
        residuals = conn.execute(f"SELECT * FROM {RESIDUALS_TABLE}").fetchdf()
        corr = {rel: correlations._pairwise_corr(residuals, rel) for rel in ("same_player", "teammate", "opponent")}
        logged = grade_logged(conn, states)
    finally:
        conn.close()
    replay = run_replay(days, calibrator, corr, tiers, states, n_simulations=n_simulations)
    rec = recommend(graded[graded["status"] == "graded"])
    adjusted = run_replay(days, calibrator, corr, tiers, states, rec, n_simulations) if rec["available"] else None
    report = build_report(graded, unresolved, n_names, days, replay, adjusted, logged, tiers, box_through, now)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str))
    logger.info("nba_model_vs_lines: %d points predictions, %d graded, %d replay parlays", len(graded),
                report["overall"]["graded"], len(replay))
    return report


if __name__ == "__main__":
    r = run()
    print(f"nba_model_vs_lines: {r['overall']['graded']} graded props; coverage {r['coverage']}; "
          f"{len(r['builder']['parlays'])} replayed parlays -> {REPORT_PATH}")
