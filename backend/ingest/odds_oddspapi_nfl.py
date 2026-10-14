"""Forward collector for NFL player-prop lines from OddsPapi's free tier
(250 requests/month), FanDuel, DraftKings and Hard Rock Bet only, and only
the four markets the NFL model projects (passing / rushing / receiving
yards, receptions). Writes to data/nfl_warehouse.duckdb, never to
warehouse.duckdb. Lines are kept for evaluating projections against the
market; feature builders never read them (USE_MARKET_FEATURES is false).

Three steps, one per scheduled job:

  wednesday  GET /fixtures for the coming week's games, then GET /odds per
             game: saves the playerId -> name map (a finished game's /odds
             no longer has names and /historical-odds never does) and the
             line as of Wednesday ('open'). A game whose /odds has no
             modeled prop lines yet (books post Sunday props late in the
             week; a probe six days out found none) is neither cached nor
             marked collected, so Saturday retries it.
  saturday   GET /odds again only for this week's not-yet-kicked-off games
             that Wednesday found without prop lines. Runs inside the
             Saturday prep job; spends nothing (not even /account) when
             there's nothing to retry.
  tuesday    GET /historical-odds per collected game that has since kicked
             off: rebuilds each player's closing line per market and book
             ('close'). Runs inside the Tuesday results job; spends nothing
             when there's nothing to fetch.

Raw responses are filtered to the three books and four markets, gzipped,
and cached under data/raw/oddspapi/nfl/<season>/w<week>/, so reruns are
free. Expected spend in a 14-16 game week: ~16-19 on Wednesday (account,
/markets every other week, fixtures, one /odds per game), 0 to ~15 on
Saturday (account plus each game Wednesday found without props), ~15-17 on
Tuesday (account, one per game): ~32-50 a week. `--dry-run` prints the
plan per week from the schedule and simulates the budget guard, without
any request or write.

Request budget (same idea as odds_theoddsapi's credit guard): every run
first reads request_count from GET /account (which itself costs one
request), then counts each request it makes on top of that and stops before
the running total would reach ODDSPAPI_MONTHLY_REQUEST_BUDGET (default 200
of the free tier's 250). If the last logged count this month is already at
the budget, the run doesn't even call /account. Only request_count and
request_limit are kept from /account; its body is never written to disk.

Spends nothing unless both ODDSPAPI_API_KEY is set and
ODDSPAPI_COLLECT_ENABLED=1 (off by default, so wiring the step into the
scheduled jobs doesn't start live calls on its own).

    docker compose run --rm --no-deps api python -m backend.ingest.odds_oddspapi_nfl wednesday
    docker compose run --rm --no-deps api python -m backend.ingest.odds_oddspapi_nfl saturday
    docker compose run --rm --no-deps api python -m backend.ingest.odds_oddspapi_nfl tuesday
    docker compose run --rm --no-deps api python -m backend.ingest.odds_oddspapi_nfl --dry-run
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import duckdb
import requests

from backend.ingest.probe_oddspapi import _norm
from backend.ingest.util import logger
from backend.nfl_db import get_nfl_connection

BASE_URL = "https://api.oddspapi.io/v4"
CACHE_ROOT = Path("data/raw/oddspapi/nfl")
NFL_SPORT_ID = 14
NFL_TOURNAMENT_ID = 31
BOOKMAKERS = ("fanduel", "draftkings", "hardrockbet")

# OddsPapi marketType -> the NFL model's market key (nfl_market_model.MARKETS).
MARKET_TYPES = {
    "playertotals-passyards": "passing_yards",
    "playertotals-rushyards": "rushing_yards",
    "playertotals-receivingyards": "receiving_yards",
    "playertotals-receptions": "receptions",
}

# Per-endpoint cooldowns from the docs (429 RATE_LIMITED otherwise), plus margin.
COOLDOWN_S = {"/account": 1.2, "/markets": 1.2, "/fixtures": 2.5, "/odds": 1.0, "/historical-odds": 5.5}
MARKETS_TTL = timedelta(days=8)  # Wednesday refreshes weekly; Tuesday reuses it
HIST_DELAY_AFTER_KICKOFF = timedelta(hours=5)  # game over, closing snapshots final


class BudgetExhausted(Exception):
    pass


def _api_key() -> str | None:
    return os.environ.get("ODDSPAPI_API_KEY") or None


def _enabled() -> bool:
    return os.environ.get("ODDSPAPI_COLLECT_ENABLED") == "1"


def _budget() -> int:
    return int(os.environ.get("ODDSPAPI_MONTHLY_REQUEST_BUDGET", "200"))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---- request budget -----------------------------------------------------------

def last_known_count(conn: duckdb.DuckDBPyConnection, now: datetime) -> int:
    """Latest logged request_count this calendar month (0 if none)."""
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).replace(tzinfo=None)
    row = conn.execute(
        "SELECT request_count FROM oddspapi_usage_log WHERE request_count IS NOT NULL AND called_at >= ? "
        "ORDER BY called_at DESC LIMIT 1",
        [month_start],
    ).fetchone()
    return row[0] if row else 0


class Client:
    """All OddsPapi traffic goes through here: one budget check per request,
    cooldowns, and a usage-log row per request. `http_get` and `sleep` are
    injectable so tests never touch the network."""

    def __init__(self, conn: duckdb.DuckDBPyConnection, key: str, budget: int | None = None,
                 http_get: Callable[..., requests.Response] = requests.get,
                 sleep: Callable[[float], None] = time.sleep, now: Callable[[], datetime] = _utcnow):
        self.conn, self.key, self.budget = conn, key, budget if budget is not None else _budget()
        self.http_get, self.sleep, self.now = http_get, sleep, now
        self.count: int | None = None  # account request_count, plus what this run spent since
        self.limit: int | None = None
        self.spent = 0
        self._last_call: dict[str, float] = {}

    def start(self) -> None:
        """Read the account's request_count. Skips even that call when the
        last logged count this month already reached the budget."""
        known = last_known_count(self.conn, self.now())
        if known >= self.budget:
            raise BudgetExhausted(f"last logged count {known} >= budget {self.budget}; /account not called")
        body = self._raw("/account", {})
        try:
            self.count = int(body["request_count"])
            self.limit = int(body["request_limit"]) if body.get("request_limit") is not None else None
        except (KeyError, TypeError, ValueError):
            raise BudgetExhausted("/account response had no request_count; stopping to be safe")
        # Whether /account counts itself in request_count isn't documented; assume not.
        self.count += 1
        self._log("/account", "account", "success")  # only the two counts are kept
        if self.count >= self.budget:
            raise BudgetExhausted(f"account at {self.count}/{self.limit} requests >= budget {self.budget}")

    def get(self, path: str, **params) -> object:
        if self.count is None:
            raise RuntimeError("Client.start() must read /account first")
        if self.count + 1 > self.budget:
            raise BudgetExhausted(f"next request would pass budget ({self.count}/{self.budget})")
        body = self._raw(path, params)
        self.count += 1
        self._log(path, "estimate", "success", note=json.dumps(params)[:200])
        return body

    def _raw(self, path: str, params: dict) -> object:
        wait = COOLDOWN_S.get(path, 1.2) - (time.monotonic() - self._last_call.get(path, -1e9))
        if wait > 0:
            self.sleep(wait)
        self.spent += 1
        resp = self.http_get(f"{BASE_URL}{path}", params={**params, "apiKey": self.key}, timeout=90)
        self._last_call[path] = time.monotonic()
        if resp.status_code != 200:
            if self.count is not None:
                self.count += 1  # assume an error still counts
            self._log(path, "estimate", "error", note=f"{resp.status_code} {resp.text[:200]}")
            raise RuntimeError(f"OddsPapi {path} -> {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    def _log(self, endpoint: str, source: str, status: str, note: str | None = None) -> None:
        self.conn.execute(
            "INSERT INTO oddspapi_usage_log VALUES (?, ?, ?, ?, ?, ?, ?)",
            [self.now().replace(tzinfo=None), endpoint, self.count, self.limit, source, status, note],
        )


# ---- raw cache ----------------------------------------------------------------

def _write_gz(path: Path, body: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        json.dump(body, f, separators=(",", ":"))
    tmp.replace(path)


def _read_gz(path: Path) -> object | None:
    if not path.is_file():
        return None
    with gzip.open(path, "rt") as f:
        return json.load(f)


def _week_dir(season: int, week: int) -> Path:
    return CACHE_ROOT / str(season) / f"w{week:02d}"


def load_markets(client: Client | None, now: datetime) -> dict[int, dict]:
    """marketId -> {marketType, handicap, side_of: {outcomeId: over|under}}
    for the four modeled markets, full-game only. Cached for MARKETS_TTL."""
    path = CACHE_ROOT / "markets.json.gz"
    cached = _read_gz(path)
    fresh = cached and now - datetime.fromisoformat(cached["fetched_at"]) < MARKETS_TTL
    if not fresh and client is not None:
        body = client.get("/markets", sportId=NFL_SPORT_ID)
        cached = {"fetched_at": now.isoformat(), "markets": filter_markets(body)}
        _write_gz(path, cached)
    if not cached:
        raise RuntimeError("no cached /markets and no client to fetch it")
    return {int(k): v for k, v in cached["markets"].items()}


def filter_markets(markets: list[dict]) -> dict[str, dict]:
    out = {}
    for m in markets:
        if m.get("marketType") in MARKET_TYPES and m.get("period", "result") == "result":
            out[str(m["marketId"])] = {
                "market": MARKET_TYPES[m["marketType"]],
                "handicap": float(m["handicap"]),
                "side_of": {str(o["outcomeId"]): o["outcomeName"].lower() for o in m["outcomes"]},
            }
    return out


def filter_bookmaker_markets(books: dict, markets: dict[int, dict]) -> dict:
    """Keep only our three books and the modeled prop markets (shared shape
    of /odds' bookmakerOdds and /historical-odds' bookmakers)."""
    out = {}
    for book in BOOKMAKERS:
        bd = (books or {}).get(book)
        if not bd:
            continue
        kept = {mid: md for mid, md in (bd.get("markets") or {}).items() if int(mid) in markets}
        out[book] = {**{k: v for k, v in bd.items() if k != "markets"}, "markets": kept}
    return out


# ---- line extraction ----------------------------------------------------------

def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def player_names(odds: dict) -> dict[str, str]:
    names = {}
    for bd in (odds.get("bookmakerOdds") or {}).values():
        for md in (bd.get("markets") or {}).values():
            for od in (md.get("outcomes") or {}).values():
                for pid, p in (od.get("players") or {}).items():
                    if pid != "0" and p and p.get("playerName"):
                        names[pid] = p["playerName"]
    return names


def _pick(candidates: list[dict]) -> dict | None:
    """Main line: the bookmaker-flagged one if any, else prices closest to even."""
    if not candidates:
        return None
    return min(candidates, key=lambda c: (not c.get("main", False), abs(c["over_price"] - c["under_price"]), c["line"]))


def open_lines(odds: dict, markets: dict[int, dict]) -> list[dict]:
    """(book, market, player) main lines from a /odds response, two-sided and active."""
    groups: dict[tuple, dict[float, dict]] = {}
    for book, bd in (odds.get("bookmakerOdds") or {}).items():
        if book not in BOOKMAKERS:
            continue
        for mid, md in (bd.get("markets") or {}).items():
            meta = markets.get(int(mid))
            if not meta or md.get("marketActive") is False:
                continue
            for oid, od in (md.get("outcomes") or {}).items():
                side = meta["side_of"].get(oid)
                for pid, p in (od.get("players") or {}).items():
                    if pid == "0" or not p or not p.get("active") or p.get("price") is None or side not in ("over", "under"):
                        continue
                    c = groups.setdefault((book, meta["market"], pid), {}).setdefault(meta["handicap"], {"line": meta["handicap"]})
                    c[f"{side}_price"] = float(p["price"])
                    c["main"] = c.get("main", False) or bool(p.get("mainLine"))
                    t = p.get("changedAt")
                    if t and (c.get("observed_at") is None or _ts(t) > c["observed_at"]):
                        c["observed_at"] = _ts(t)
    rows = []
    for (book, market, pid), by_line in groups.items():
        best = _pick([c for c in by_line.values() if "over_price" in c and "under_price" in c])
        if best:
            rows.append({"bookmaker": book, "market": market, "op_player_id": pid, **best})
    return rows


def close_lines(hist: dict, markets: dict[int, dict], kickoff: datetime) -> list[dict]:
    """Closing main line per (book, market, player) from /historical-odds:
    for each line, the last snapshot at or before kickoff on each side;
    keep lines where both are active; pick the one closest to even."""
    groups: dict[tuple, dict[float, dict]] = {}
    for book, bd in (hist.get("bookmakers") or {}).items():
        if book not in BOOKMAKERS:
            continue
        for mid, md in (bd.get("markets") or {}).items():
            meta = markets.get(int(mid))
            if not meta:
                continue
            for oid, od in (md.get("outcomes") or {}).items():
                side = meta["side_of"].get(oid)
                if side not in ("over", "under"):
                    continue
                for pid, snaps in (od.get("players") or {}).items():
                    pre = [s for s in snaps or [] if _ts(s["createdAt"]) <= kickoff]
                    if pid == "0" or not pre:
                        continue
                    last = max(pre, key=lambda s: s["createdAt"])
                    if not last.get("active") or last.get("price") is None:
                        continue
                    c = groups.setdefault((book, meta["market"], pid), {}).setdefault(meta["handicap"], {"line": meta["handicap"]})
                    c[f"{side}_price"] = float(last["price"])
                    t = _ts(last["createdAt"])
                    c["observed_at"] = max(c.get("observed_at") or t, t)
    rows = []
    for (book, market, pid), by_line in groups.items():
        best = _pick([c for c in by_line.values() if "over_price" in c and "under_price" in c])
        if best:
            rows.append({"bookmaker": book, "market": market, "op_player_id": pid, **best})
    return rows


def _save_lines(conn: duckdb.DuckDBPyConnection, fixture_id: str, game_id: str | None, kind: str,
                rows: list[dict], fetched_at: datetime) -> None:
    conn.execute("DELETE FROM nfl_prop_lines WHERE fixture_id = ? AND kind = ?", [fixture_id, kind])
    if rows:
        conn.executemany(
            "INSERT INTO nfl_prop_lines VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [[fixture_id, game_id, r["bookmaker"], r["market"], r["op_player_id"], kind, r["line"],
              r["over_price"], r["under_price"], r.get("observed_at"), fetched_at.replace(tzinfo=None)] for r in rows],
        )


# ---- fixtures and names --------------------------------------------------------

def match_fixtures(fixtures: list[dict], games: list[tuple]) -> list[dict]:
    """OddsPapi fixtures -> nflverse games by team pair and kickoff (within
    36 h). games: (game_id, season, week, kickoff_utc, home_team, away_team)."""
    out = []
    for f in fixtures:
        if f.get("tournamentId", NFL_TOURNAMENT_ID) != NFL_TOURNAMENT_ID or f.get("statusId") == 3:
            continue
        pair = {f.get("participant1Abbr"), f.get("participant2Abbr")}
        start = _ts(f["startTime"])
        for game_id, season, week, kickoff, home, away in games:
            if pair == {home, away} and abs(start - kickoff) < timedelta(hours=36):
                out.append({"fixture_id": f["fixtureId"], "game_id": game_id, "season": season, "week": week,
                            "kickoff_utc": start, "home_team": home, "away_team": away})
                break
    return out


def _roster_ids(conn: duckdb.DuckDBPyConnection, season: int, week: int, teams: tuple[str, str]) -> dict[str, str | None]:
    """normalized name -> gsis id for the two teams' latest roster week
    (None when two players share a normalized name)."""
    rows = conn.execute(
        """SELECT player_id, player_name FROM nfl_rosters_weekly
           WHERE season = ? AND team IN (?, ?) AND week = (
               SELECT max(week) FROM nfl_rosters_weekly WHERE season = ? AND week <= ?)""",
        [season, *teams, season, week],
    ).fetchall()
    out: dict[str, str | None] = {}
    for pid, name in rows:
        if name:
            k = _norm(name)
            out[k] = None if k in out and out[k] != pid else pid
    return out


def _save_names(conn: duckdb.DuckDBPyConnection, fx: dict, names: dict[str, str]) -> None:
    ids = _roster_ids(conn, fx["season"], fx["week"], (fx["home_team"], fx["away_team"]))
    conn.execute("DELETE FROM nfl_oddspapi_players WHERE fixture_id = ?", [fx["fixture_id"]])
    if names:
        conn.executemany(
            "INSERT INTO nfl_oddspapi_players VALUES (?, ?, ?, ?)",
            [[fx["fixture_id"], pid, name, ids.get(_norm(name))] for pid, name in names.items()],
        )


# ---- the three steps -----------------------------------------------------------

def _collect_open(conn: duckdb.DuckDBPyConnection, client: Client, fx: dict, markets: dict[int, dict],
                  now: datetime) -> int:
    """/odds for one fixture: names and open lines. Only a response with
    prop lines is cached and marks the fixture collected; an empty one is
    retried by a later run (Saturday)."""
    path = _week_dir(fx["season"], fx["week"]) / f"{fx['fixture_id']}_odds.json.gz"
    odds = _read_gz(path)
    fresh = odds is None
    if fresh:
        body = client.get("/odds", fixtureId=fx["fixture_id"], bookmakers=",".join(BOOKMAKERS), oddsFormat="decimal")
        odds = {**{k: v for k, v in body.items() if k != "bookmakerOdds"},
                "bookmakerOdds": filter_bookmaker_markets(body.get("bookmakerOdds"), markets)}
    rows = open_lines(odds, markets)
    names = player_names(odds)
    if names:
        _save_names(conn, fx, names)
    if not rows:
        return 0
    if fresh:
        _write_gz(path, odds)
    _save_lines(conn, fx["fixture_id"], fx["game_id"], "open", rows, now)
    conn.execute("UPDATE nfl_oddspapi_fixtures SET odds_fetched_at = ? WHERE fixture_id = ?",
                 [now.replace(tzinfo=None), fx["fixture_id"]])
    return len(rows)


def run_wednesday(conn: duckdb.DuckDBPyConnection, client: Client, now: datetime) -> dict:
    """Coming week (now .. now+7d): fixtures, then /odds per game not yet collected."""
    end = now + timedelta(days=7)
    games = conn.execute(
        "SELECT game_id, season, week, kickoff_utc, home_team, away_team FROM nfl_schedule "
        "WHERE kickoff_utc > ? AND kickoff_utc <= ? ORDER BY kickoff_utc",
        [now, end],
    ).fetchall()
    if not games:
        return {"games": 0, "requests": client.spent}
    games = [(g, s, w, k if k.tzinfo else k.replace(tzinfo=timezone.utc), h, a) for g, s, w, k, h, a in games]
    season, week = games[0][1], games[0][2]
    wdir = _week_dir(season, week)

    markets = load_markets(client, now)
    fx_path = wdir / f"fixtures_{now:%Y%m%d}.json.gz"
    fixtures = _read_gz(fx_path)
    if fixtures is None:
        fixtures = client.get("/fixtures", tournamentId=NFL_TOURNAMENT_ID,
                              **{"from": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "to": end.strftime("%Y-%m-%dT%H:%M:%SZ")})
        _write_gz(fx_path, fixtures)
    matched = match_fixtures(fixtures, games)

    collected = lines = 0
    for fx in matched:
        conn.execute(
            "INSERT INTO nfl_oddspapi_fixtures VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL) ON CONFLICT (fixture_id) DO UPDATE "
            "SET game_id = excluded.game_id, kickoff_utc = excluded.kickoff_utc",
            [fx["fixture_id"], fx["game_id"], fx["season"], fx["week"], fx["kickoff_utc"], fx["home_team"], fx["away_team"]],
        )
        n = _collect_open(conn, client, fx, markets, now)
        collected += n > 0
        lines += n
    return {"games": len(games), "matched": len(matched), "collected": collected,
            "no_props_yet": len(matched) - collected, "open_lines": lines, "requests": client.spent}


def _saturday_todo(conn: duckdb.DuckDBPyConnection, now: datetime) -> list[tuple]:
    return conn.execute(
        "SELECT fixture_id, game_id, season, week, kickoff_utc, home_team, away_team FROM nfl_oddspapi_fixtures "
        "WHERE odds_fetched_at IS NULL AND kickoff_utc > ? AND kickoff_utc <= ? ORDER BY kickoff_utc",
        [now, now + timedelta(days=4)],
    ).fetchall()


def run_saturday(conn: duckdb.DuckDBPyConnection, client: Client, now: datetime) -> dict:
    """Retry /odds for this week's games Wednesday found without prop lines."""
    todo = _saturday_todo(conn, now)
    if not todo:
        return {"games": 0, "requests": client.spent}
    markets = load_markets(client, now)
    collected = lines = 0
    for fixture_id, game_id, season, week, kickoff, home, away in todo:
        fx = {"fixture_id": fixture_id, "game_id": game_id, "season": season, "week": week,
              "home_team": home, "away_team": away}
        n = _collect_open(conn, client, fx, markets, now)
        collected += n > 0
        lines += n
    return {"games": len(todo), "collected": collected, "no_props_yet": len(todo) - collected,
            "open_lines": lines, "requests": client.spent}


def _tuesday_todo(conn: duckdb.DuckDBPyConnection, now: datetime) -> list[tuple]:
    return conn.execute(
        "SELECT fixture_id, game_id, season, week, kickoff_utc FROM nfl_oddspapi_fixtures "
        "WHERE hist_fetched_at IS NULL AND odds_fetched_at IS NOT NULL AND kickoff_utc <= ? AND kickoff_utc >= ? "
        "ORDER BY kickoff_utc",
        [now - HIST_DELAY_AFTER_KICKOFF, now - timedelta(days=10)],
    ).fetchall()


def run_tuesday(conn: duckdb.DuckDBPyConnection, client: Client, now: datetime) -> dict:
    """/historical-odds for collected games that finished in the last 10 days."""
    todo = _tuesday_todo(conn, now)
    if not todo:
        return {"games": 0, "requests": client.spent}
    markets = load_markets(client, now)
    lines = 0
    for fixture_id, game_id, season, week, kickoff in todo:
        kickoff = kickoff if kickoff.tzinfo else kickoff.replace(tzinfo=timezone.utc)
        path = _week_dir(season, week) / f"{fixture_id}_hist.json.gz"
        hist = _read_gz(path)
        if hist is None:
            body = client.get("/historical-odds", fixtureId=fixture_id, bookmakers=",".join(BOOKMAKERS))
            hist = {**{k: v for k, v in body.items() if k != "bookmakers"},
                    "bookmakers": filter_bookmaker_markets(body.get("bookmakers"), markets)}
            _write_gz(path, hist)
        rows = close_lines(hist, markets, kickoff)
        _save_lines(conn, fixture_id, game_id, "close", rows, now)
        conn.execute("UPDATE nfl_oddspapi_fixtures SET hist_fetched_at = ? WHERE fixture_id = ?",
                     [now.replace(tzinfo=None), fixture_id])
        lines += len(rows)
    return {"games": len(todo), "close_lines": lines, "requests": client.spent}


STEPS = {"wednesday": (run_wednesday, None), "saturday": (run_saturday, _saturday_todo),
         "tuesday": (run_tuesday, _tuesday_todo)}


def run(step: str, conn: duckdb.DuckDBPyConnection | None = None, **client_kwargs) -> dict:
    key = _api_key()
    if not key:
        return {"skipped": "ODDSPAPI_API_KEY not set; nothing spent"}
    if not _enabled():
        return {"skipped": "ODDSPAPI_COLLECT_ENABLED is not 1; nothing spent"}
    conn = conn or get_nfl_connection()
    client = Client(conn, key, **client_kwargs)
    now = client.now()
    step_fn, todo = STEPS[step]
    if todo is not None and not todo(conn, now):
        return {"games": 0, "requests": 0, "account_count": None}  # nothing to do: skip even /account
    try:
        client.start()
        result = step_fn(conn, client, now)
    except BudgetExhausted as e:
        logger.warning("odds_oddspapi_nfl: %s", e)
        result = {"stopped": str(e), "requests": client.spent}
    except Exception as e:  # noqa: BLE001 — report, keep the requests-spent count
        logger.error("odds_oddspapi_nfl: %s", e)
        result = {"error": str(e), "requests": client.spent}
    result["account_count"] = client.count
    return result


# ---- dry run -------------------------------------------------------------------

ET = ZoneInfo("America/New_York")


def _at_et(day, hour: int = 9) -> datetime:
    return datetime(day.year, day.month, day.day, hour, tzinfo=ET).astimezone(timezone.utc)


def _guard(count: int, budget: int, wanted: int) -> tuple[int, int]:
    """What the budget guard lets one run spend: (requests made, new count).
    wanted includes the /account call."""
    if wanted == 0 or count >= budget:
        return 0, count
    made, count = 1, count + 1  # /account
    if count >= budget:
        return made, count
    extra = min(wanted - 1, budget - count)
    return made + extra, count + extra


def plan(conn: duckdb.DuckDBPyConnection, now: datetime, budget: int | None = None) -> dict:
    """Requests each scheduled run would make for the rest of the season,
    from nfl_schedule alone, with the budget guard simulated per calendar
    month (worst case: Saturday retries every game after it). No HTTP, no
    writes. The live guard starts from GET /account's real count instead
    of the last logged one."""
    budget = budget if budget is not None else _budget()
    games = conn.execute(
        "SELECT season, week, kickoff_utc FROM nfl_schedule WHERE kickoff_utc > ? ORDER BY kickoff_utc", [now]
    ).fetchall()
    cached = _read_gz(CACHE_ROOT / "markets.json.gz")
    markets_at = datetime.fromisoformat(cached["fetched_at"]) if cached else None

    # Wednesday runs: today if it's a Wednesday past 9 AM ET (a manual catch-up), then every Wednesday 9 AM ET.
    local = now.astimezone(ET)
    first = local.date() + timedelta(days=(2 - local.weekday()) % 7)
    weds = ([now] if local.weekday() == 2 and local.hour >= 9 else []) + \
        [_at_et(first + timedelta(weeks=i)) for i in range(0, 30) if _at_et(first + timedelta(weeks=i)) > now]
    kicks = [(s, w, k if k.tzinfo else k.replace(tzinfo=timezone.utc)) for s, w, k in games]

    counts: dict[str, int] = {f"{now:%Y-%m}": last_known_count(conn, now)}
    runs, weeks = [], []
    for wed in weds:
        wk = [g for g in kicks if wed < g[2] <= wed + timedelta(days=7)]
        if not wk:
            continue
        markets = 0 if markets_at and wed - markets_at < MARKETS_TTL else 1
        markets_at = wed if markets else markets_at
        sat = _at_et(wed.astimezone(ET).date() + timedelta(days=3))
        after_sat = sum(1 for g in wk if g[2] > sat)
        tue = _at_et(wed.astimezone(ET).date() + timedelta(days=6))
        wed_n = 2 + markets + len(wk)  # account, fixtures, one /odds per game
        tue_n = 1 + len(wk)
        week = {"season": wk[0][0], "week": wk[0][1], "games": len(wk),
                "wednesday": f"{wed.astimezone(ET):%a %b %d %H:%M} ET", "wednesday_requests": wed_n,
                "saturday_requests_max": 1 + after_sat if after_sat else 0, "tuesday_requests": tue_n}
        weeks.append(week)
        runs += [(wed, wed_n), (sat, week["saturday_requests_max"]), (tue, tue_n)]

    months: dict[str, dict] = {}
    for when, wanted in sorted(runs):
        m = f"{when.astimezone(ET):%Y-%m}"
        made, counts[m] = _guard(counts.get(m, 0), budget, wanted)
        mo = months.setdefault(m, {"start_count": counts[m] - made, "wanted": 0, "made": 0})
        mo["wanted"] += wanted
        mo["made"] += made
        mo["end_count"] = counts[m]
        mo["guard_trims"] = mo["wanted"] - mo["made"]
    return {"budget": budget, "bookmakers": list(BOOKMAKERS), "markets": sorted(MARKET_TYPES),
            "weeks": weeks, "months_worst_case": months}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", nargs="?", choices=list(STEPS))
    ap.add_argument("--dry-run", action="store_true", help="print the planned requests per week; no HTTP, no writes")
    args = ap.parse_args()
    if args.dry_run:
        conn = get_nfl_connection(read_only=True)
        try:
            print(json.dumps(plan(conn, _utcnow()), indent=1, default=str))
        finally:
            conn.close()
        return
    if not args.step:
        ap.error("step is required unless --dry-run")
    result = run(args.step)
    print(json.dumps(result, default=str))
    sys.exit(1 if "error" in result else 0)


if __name__ == "__main__":
    main()
