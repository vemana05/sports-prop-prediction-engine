"""Model vs. lines for NFL: the logged projections (nfl_predictions_log)
against real closing prop lines (OddsPapi, nfl_prop_lines) and the actual
stat (nfl_player_weekly). Reads the NFL warehouse only; makes no API calls.

One row per (game, player, market):
- Line: ONE book's closing main line (the collector already keeps, per
  book, the line priced closest to 50/50). FanDuel when it has one, else
  DraftKings, else Hard Rock Bet (BOOK_PRIORITY); every row records its book.
  No cross-book consensus, so each price is one a bettor could have taken.
- Projection: the earliest logged one (the log is write-once per key).
- Actual: the player's stat from nfl_player_weekly. A player with no stat
  row and no offensive snaps didn't play: books void those, so they're
  counted as "did not play", not graded.

Names: OddsPapi players are matched to nflverse ids against the two teams'
rosters that season (weeks up to the game's), by normalized name
("Last, First" flipped, accents, punctuation and Jr/Sr/II-V removed), then,
only if that finds nobody, by the same last name (either half of a
hyphenated one) plus first initial, so "Cameron Ward" finds "Cam Ward".
Several candidates: only offensive skill players are kept. Names still
ambiguous or missing are listed, never dropped.

Probabilities:
- Model: P(over) from that market's own distribution (the Spread saved
  with the logged model version) through the market's isotonic ladder
  calibrator (parlay/nfl_legs.calibrators). On a whole-number line the push
  probability is taken out first, so it's P(over | not a push).
- Book: no-vig over probability from the closing over and under prices.
- Pushes (actual == line) are left out of Brier, calibration and hit rate,
  and return the stake in ROI.

ROI: one unit on the model's side at the closing price, for props where the
model and the book's no-vig differ by at least each margin in MARGINS.
CLV: the model's side at the OPENING line (same book) vs. where that line
closed: a better closing number, or the same number with the side's no-vig
probability higher at close, means the opening bet beat the close.

    docker compose run --rm --no-deps api python -m backend.backtest.nfl_model_vs_lines
"""

from __future__ import annotations

import json
import math
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from backend.ingest.probe_oddspapi import _norm
from backend.ingest.util import logger
from backend.models.nfl_market_model import MARKETS, Spread, p_over

SKILL_POSITIONS = ("QB", "RB", "FB", "WR", "TE")
BOOK_PRIORITY = ("fanduel", "draftkings", "hardrockbet")
BOOK_LABELS = {"fanduel": "FanDuel", "draftkings": "DraftKings", "hardrockbet": "Hard Rock Bet"}
MARGINS = (0.0, 0.03, 0.05, 0.08, 0.10)
MIN_SAMPLE = 300
BUCKET_EDGES = np.linspace(0, 1, 11)
MODELS_DIR = Path("data/models")
REPORT_PATH = Path("reports/nfl_model_vs_lines.json")
LABELS = {m.name: m.label for m in MARKETS}
FAMILY = {m.name: m.family for m in MARKETS}
TARGET = {m.name: m.target for m in MARKETS}

BOOK_RULE = ("One book's closing main line per player and market: FanDuel when it has one, otherwise DraftKings, "
             "otherwise Hard Rock Bet. The main line is the one priced closest to 50/50.")
MATCH_RULE = ("OddsPapi names matched to nflverse rosters of the game's two teams by normalized name (\"Last, First\" "
              "flipped, accents, punctuation and Jr/Sr/II-V removed), then by last name + first initial; ties go to the offensive skill player.")


# ---- name matching ------------------------------------------------------------

def _last_parts(norm: str) -> set[str]:
    last = norm.split()[-1] if norm else ""
    return {last, *last.split("-")} - {""}


def _skill_tiebreak(cand: pd.DataFrame) -> np.ndarray:
    ids = cand["player_id"].unique()
    if len(ids) > 1:
        skill = cand.loc[cand["position"].isin(SKILL_POSITIONS), "player_id"].unique()
        if len(skill) >= 1:
            return skill
    return ids


def match_names(players: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """players: fixture_id, op_player_id, player_name, season, week,
    home_team, away_team. roster: season, week, team, player_id,
    player_name, position. Adds player_id, match_method (exact | initial
    | None) and unmatched_reason. When a rule finds several players, only
    offensive skill players (the only ones with props) are kept, so a
    cornerback can't block his namesake quarterback."""
    roster = roster.dropna(subset=["player_name"]).assign(norm=lambda d: d["player_name"].map(_norm))
    out = []
    for p in players.itertuples(index=False):
        cand = roster[(roster["season"] == p.season) & (roster["week"] <= p.week)
                      & roster["team"].isin([p.home_team, p.away_team])].drop_duplicates("player_id")
        n = _norm(p.player_name)
        exact = _skill_tiebreak(cand[cand["norm"] == n])
        pid, method, reason = None, None, None
        if len(exact) == 1:
            pid, method = exact[0], "exact"
        elif len(exact) > 1:
            reason = f"{len(exact)} roster players share this name"
        else:
            parts = _last_parts(n)
            loose = _skill_tiebreak(cand[cand["norm"].map(lambda r: bool(r) and r[0] == n[:1] and bool(_last_parts(r) & parts))])
            if len(loose) == 1:
                pid, method = loose[0], "initial"
            elif len(loose) > 1:
                reason = f"{len(loose)} roster players share last name and first initial"
            else:
                reason = "no roster player on either team with this name"
        out.append({**p._asdict(), "player_id": pid, "match_method": method, "unmatched_reason": reason})
    return pd.DataFrame(out, columns=[*players.columns, "player_id", "match_method", "unmatched_reason"])


# ---- probabilities ------------------------------------------------------------

def no_vig_over(over_price: float, under_price: float) -> float:
    io, iu = 1 / over_price, 1 / under_price
    return io / (io + iu)


def model_p_over(mu: float, line: float, spread: Spread, calibrator=None) -> tuple[float, float]:
    """(calibrated P(over | no push), raw P(over | no push))."""
    m, whole = np.array([mu]), float(line).is_integer()
    if not whole:
        raw = float(p_over(m, np.array([line]), spread)[0])
    else:
        if spread.r is not None:  # counts: P(X > line) and P(X == line)
            over = float(p_over(m, np.array([line]), spread)[0])
            r = spread.r
            push = float(stats.nbinom.pmf(int(line), r, r / (r + max(mu, 1e-6))))
        else:  # yards: the stat is whole yards, so a push is anything rounding to the line
            over = float(p_over(m, np.array([line + 0.5]), spread)[0])
            push = float(p_over(m, np.array([line - 0.5]), spread)[0]) - over
        raw = over / (1 - push) if push < 1 else 0.5
    raw = min(max(raw, 0.0), 1.0)
    cal = float(calibrator.predict([raw])[0]) if calibrator is not None else raw
    return cal, raw


# ---- loading ------------------------------------------------------------------

def _load_spread(version: str, market: str, cache: dict) -> Spread | None:
    key = (version, market)
    if key not in cache:
        path = MODELS_DIR / f"nfl_{market}_{version}_spread.pkl"
        cache[key] = pickle.loads(path.read_bytes()) if path.exists() else None
    return cache[key]


def load_rows(conn: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(props, names): one closing prop per (fixture, market, OddsPapi
    player), with its open line from the same book, and the name match for
    every OddsPapi player who has a closing line."""
    lines = conn.execute(
        """SELECT l.fixture_id, coalesce(l.game_id, f.game_id) AS game_id, l.bookmaker, l.market, l.op_player_id,
                  l.kind, l.line, l.over_price, l.under_price, f.season, f.week,
                  s.home_team, s.away_team, s.home_score IS NOT NULL AS final
           FROM nfl_prop_lines l JOIN nfl_oddspapi_fixtures f USING (fixture_id)
           LEFT JOIN nfl_schedule s ON s.game_id = coalesce(l.game_id, f.game_id)"""
    ).fetchdf()
    close = lines[(lines["kind"] == "close") & lines["bookmaker"].isin(BOOK_PRIORITY)
                  & lines["over_price"].notna() & lines["under_price"].notna()].copy()
    close["rank"] = close["bookmaker"].map({b: i for i, b in enumerate(BOOK_PRIORITY)})
    close = close.sort_values("rank").drop_duplicates(["fixture_id", "market", "op_player_id"]).drop(columns=["rank", "kind"])
    opens = lines.loc[lines["kind"] == "open", ["fixture_id", "bookmaker", "market", "op_player_id", "line", "over_price", "under_price"]]
    opens = opens.rename(columns={"line": "open_line", "over_price": "open_over", "under_price": "open_under"})
    props = close.merge(opens, how="left", on=["fixture_id", "bookmaker", "market", "op_player_id"])

    op_names = conn.execute("SELECT fixture_id, op_player_id, player_name FROM nfl_oddspapi_players").fetchdf()
    players = props[["fixture_id", "op_player_id", "season", "week", "home_team", "away_team", "game_id"]].drop_duplicates(["fixture_id", "op_player_id"])
    players = players.merge(op_names, how="left", on=["fixture_id", "op_player_id"])
    players["player_name"] = players["player_name"].fillna("")
    roster = conn.execute("SELECT season, week, team, player_id, player_name, position FROM nfl_rosters_weekly").fetchdf()
    names = match_names(players[["fixture_id", "op_player_id", "player_name", "season", "week", "home_team", "away_team", "game_id"]], roster)
    props = props.merge(names[["fixture_id", "op_player_id", "player_name", "player_id", "match_method", "unmatched_reason"]],
                        how="left", on=["fixture_id", "op_player_id"])

    preds = conn.execute(
        """SELECT game_id, player_id, market, mean, model_version FROM nfl_predictions_log
           QUALIFY row_number() OVER (PARTITION BY game_id, player_id, market ORDER BY logged_at) = 1"""
    ).fetchdf()
    props = props.merge(preds, how="left", on=["game_id", "player_id", "market"])
    stats_ = conn.execute(
        "SELECT player_id, game_id, passing_yards, rushing_yards, receiving_yards, receptions FROM nfl_player_weekly"
    ).fetchdf()
    snaps = conn.execute(
        "SELECT player_id, game_id, max(offense_snaps) AS offense_snaps FROM nfl_snap_counts WHERE player_id IS NOT NULL GROUP BY ALL"
    ).fetchdf()
    props = props.merge(stats_, how="left", on=["player_id", "game_id"]).merge(snaps, how="left", on=["player_id", "game_id"])
    props["has_stat_row"] = props[list(TARGET.values())].notna().any(axis=1)
    # No stat row but offensive snaps: played and recorded nothing (0).
    # Neither: didn't play (NaN), which books void.
    played = props["has_stat_row"] | (props["offense_snaps"].fillna(0) > 0)
    stat = pd.Series([r[TARGET[m]] for m, r in zip(props["market"], props.to_dict("records"))], index=props.index, dtype=float)
    props["actual"] = stat.fillna(0.0).where(played)
    return props, names


def score_rows(props: pd.DataFrame, calibrators: dict, spread_for) -> pd.DataFrame:
    """Adds status, p_model, p_model_raw, p_book, outcome, push and the
    opening-line model probability. spread_for(version, market) -> Spread | None."""
    rows = []
    for r in props.to_dict("records"):
        r["p_model"] = r["p_model_raw"] = r["p_model_open"] = r["outcome"] = np.nan
        r["push"] = False
        r["p_book"] = no_vig_over(r["over_price"], r["under_price"])
        if pd.isna(r.get("player_id")):
            r["status"] = "unmatched_name"
        elif pd.isna(r.get("mean")):
            r["status"] = "no_projection"
        elif not r.get("final"):
            r["status"] = "game_not_final"
        elif pd.isna(r.get("actual")):
            r["status"] = "did_not_play"
        else:
            spread = spread_for(r["model_version"], r["market"])
            if spread is None:
                r["status"] = "no_spread"
            else:
                cal = calibrators.get(r["market"])
                r["p_model"], r["p_model_raw"] = model_p_over(r["mean"], r["line"], spread, cal)
                if pd.notna(r.get("open_line")):
                    r["p_model_open"] = model_p_over(r["mean"], r["open_line"], spread, cal)[0]
                r["push"] = bool(r["actual"] == r["line"])
                r["outcome"] = np.nan if r["push"] else float(r["actual"] > r["line"])
                r["status"] = "graded"
        rows.append(r)
    return pd.DataFrame(rows)


# ---- metrics ------------------------------------------------------------------

def _r(x, nd=4):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)


def buckets(p: pd.Series, y: pd.Series) -> list[dict]:
    idx = np.clip(np.digitize(p, BUCKET_EDGES[1:-1]), 0, len(BUCKET_EDGES) - 2)
    out = []
    for i in range(len(BUCKET_EDGES) - 1):
        m = idx == i
        if m.any():
            out.append({"lo": _r(BUCKET_EDGES[i], 1), "hi": _r(BUCKET_EDGES[i + 1], 1), "n": int(m.sum()),
                        "mean_p": _r(p[m].mean()), "hit_rate": _r(y[m].mean())})
    return out


def roi_table(df: pd.DataFrame) -> list[dict]:
    """Flat one-unit bets on the model's side at the closing price."""
    edge = df["p_model"] - df["p_book"]
    over = edge > 0
    price = np.where(over, df["over_price"], df["under_price"])
    won = np.where(over, df["outcome"] == 1, df["outcome"] == 0)
    profit = np.where(df["push"], 0.0, np.where(won, price - 1, -1.0))
    out = []
    for margin in MARGINS:
        m = ((edge.abs() > 0) if margin == 0 else (edge.abs() >= margin - 1e-9)).to_numpy()
        n = int(m.sum())
        out.append({"margin": margin, "bets": n, "wins": int((won & ~df["push"].to_numpy() & m).sum()),
                    "pushes": int((df["push"].to_numpy() & m).sum()), "profit": _r(profit[m].sum(), 2),
                    "roi": _r(profit[m].sum() / n) if n else None})
    return out


def clv(df: pd.DataFrame) -> dict:
    d = df[df["open_line"].notna() & df["p_model_open"].notna()]
    if d.empty:
        return {"n": 0, "beat": 0, "same": 0, "worse": 0, "beat_rate": None, "mean_line_move": None}
    p_open_book = [no_vig_over(o, u) for o, u in zip(d["open_over"], d["open_under"])]
    over = (d["p_model_open"] - p_open_book).to_numpy() >= 0
    sign = np.where(over, 1.0, -1.0)
    move = (d["line"] - d["open_line"]).to_numpy() * sign
    p_side_close = np.where(over, d["p_book"], 1 - d["p_book"])
    p_side_open = np.where(over, p_open_book, 1 - np.asarray(p_open_book))
    prob_move = p_side_close - p_side_open
    beat = (move > 0) | ((move == 0) & (prob_move > 1e-9))
    worse = (move < 0) | ((move == 0) & (prob_move < -1e-9))
    nb, nw = int(beat.sum()), int(worse.sum())
    return {"n": len(d), "beat": nb, "same": len(d) - nb - nw, "worse": nw,
            "beat_rate": _r(nb / (nb + nw)) if nb + nw else None, "mean_line_move": _r(move.mean(), 2)}


def summarize(df: pd.DataFrame) -> dict:
    g = df[df["status"] == "graded"].astype({"push": bool})
    gp = g[~g["push"]]
    y = gp["outcome"].astype(float)
    side_over = gp["p_model"] > gp["p_book"]
    picked = gp["p_model"] != gp["p_book"]
    hits = (side_over & (y == 1)) | (~side_over & (y == 0))
    return {
        "props": len(df),
        "graded": len(gp),
        "pushes": int(g["push"].sum()),
        "enough_sample": len(gp) >= MIN_SAMPLE,
        "brier_model": _r(((gp["p_model"] - y) ** 2).mean()) if len(gp) else None,
        "brier_model_raw": _r(((gp["p_model_raw"] - y) ** 2).mean()) if len(gp) else None,
        "brier_book": _r(((gp["p_book"] - y) ** 2).mean()) if len(gp) else None,
        "over_rate": _r(y.mean()) if len(gp) else None,
        "side_hit_rate": _r(hits[picked].mean()) if picked.any() else None,
        "side_n": int(picked.sum()),
        "calibration_model": buckets(gp["p_model"].to_numpy(), y.to_numpy()) if len(gp) else [],
        "calibration_book": buckets(gp["p_book"].to_numpy(), y.to_numpy()) if len(gp) else [],
        "roi": roi_table(g) if len(g) else [],
        "clv": clv(g),
    }


def plain_summary(report: dict) -> list[str]:
    o, n = report["overall"], report["overall"]["graded"]
    out = []
    if n == 0:
        out.append("No props have been graded yet: closing lines arrive with the Tuesday results job, after the week's games. "
                   "There is nothing to conclude.")
    elif not o["enough_sample"]:
        out.append(f"Only {n} graded props so far, under the {MIN_SAMPLE} needed: the sample is too small to conclude anything. "
                   "Treat every number below as noise for now.")
    if n:
        bm, bb = o["brier_model"], o["brier_book"]
        better = "better" if bm < bb else "worse" if bm > bb else "the same"
        out.append(f"The model's probabilities scored {bm:.4f} (Brier, lower is better) against the book's no-vig {bb:.4f}: "
                   f"{better} than the closing line.")
        if o["side_hit_rate"] is not None:
            out.append(f"Siding with the model against the book won {o['side_hit_rate'] * 100:.1f}% of {o['side_n']} props "
                       "(about 52.4% is needed to break even at typical -110 prices).")
        r0 = next((r for r in o["roi"] if r["margin"] == 0.05 and r["bets"]), None)
        if r0:
            out.append(f"Betting one unit at the closing price whenever the model disagreed by 5 points or more: "
                       f"{r0['bets']} bets, ROI {r0['roi'] * 100:+.1f}%.")
        if o["clv"]["beat_rate"] is not None:
            out.append(f"Lines moved toward the model's opening-line side {o['clv']['beat_rate'] * 100:.0f}% of the time "
                       f"({o['clv']['beat']} of {o['clv']['beat'] + o['clv']['worse']} that moved).")
    nm = report["names"]
    if nm["total"]:
        out.append(f"{nm['matched']} of {nm['total']} sportsbook player names matched an nflverse player "
                   f"({nm['match_rate'] * 100:.0f}%); the unmatched ones are listed below.")
    return out


def build_report(props: pd.DataFrame, names: pd.DataFrame, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    matched = names["player_id"].notna()
    status_counts = props["status"].value_counts().to_dict() if len(props) else {}
    report = {
        "as_of": now.isoformat(timespec="seconds"),
        "book_rule": BOOK_RULE,
        "match_rule": MATCH_RULE,
        "min_sample": MIN_SAMPLE,
        "margins": list(MARGINS),
        "books_used": {BOOK_LABELS.get(b, b): int(c) for b, c in props.loc[props["status"] == "graded", "bookmaker"].value_counts().items()} if len(props) else {},
        "names": {
            "total": len(names),
            "matched": int(matched.sum()),
            "match_rate": _r(matched.mean()) if len(names) else None,
            "by_method": {k: int(v) for k, v in names["match_method"].value_counts().items()},
            "unmatched": [{"name": r.player_name or f"(no name for OddsPapi id {r.op_player_id})", "game_id": r.game_id,
                           "teams": f"{r.away_team} @ {r.home_team}", "reason": r.unmatched_reason}
                          for r in names[~matched].itertuples()],
        },
        "coverage": {k: int(v) for k, v in status_counts.items()},
        "not_graded": [{"name": r.player_name, "player_id": r.player_id, "game_id": r.game_id, "market": r.market, "status": r.status}
                       for r in props[~props["status"].isin(["graded", "unmatched_name"])].itertuples()] if len(props) else [],
        "overall": summarize(props) if len(props) else summarize(_empty()),
        "markets": {},
    }
    for m in MARKETS:
        sub = props[props["market"] == m.name] if len(props) else _empty()
        report["markets"][m.name] = {"label": m.label, **summarize(sub)}
    report["summary"] = plain_summary(report)
    return report


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["status", "push", "outcome", "p_model", "p_model_raw", "p_book", "over_price", "under_price",
                                 "open_line", "open_over", "open_under", "line", "p_model_open", "market", "bookmaker"])


def run(db_path: str | None = None, out: Path = REPORT_PATH) -> dict:
    from backend.parlay import nfl_legs  # loads models/replay machinery; keep the module import light

    calibrators = nfl_legs.calibrators()  # before opening our own connection: it opens the warehouse too
    path = db_path or os.environ.get("NFL_DUCKDB_PATH", "data/nfl_warehouse.duckdb")
    conn = duckdb.connect(path, read_only=True)
    try:
        props, names = load_rows(conn)
    finally:
        conn.close()
    cache: dict = {}
    scored = score_rows(props, calibrators, lambda v, m: _load_spread(v, m, cache)) if len(props) else _empty()
    report = build_report(scored, names)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str))
    o = report["overall"]
    logger.info("nfl_model_vs_lines: %d props, %d graded, names matched %s/%s", o["props"], o["graded"],
                report["names"]["matched"], report["names"]["total"])
    return report


if __name__ == "__main__":
    r = run()
    print(f"nfl_model_vs_lines: {r['overall']['graded']} graded props; coverage {r['coverage']}; "
          f"names {r['names']['matched']}/{r['names']['total']} matched -> {REPORT_PATH}")
