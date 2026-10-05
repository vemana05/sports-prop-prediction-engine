"""Read-only probe of OddsPapi's free tier (not a source; nothing is written
to any warehouse). For one completed NFL regular-season game (and one NBA
game, if OddsPapi lists any) it answers:

  - do /historical-odds carry player-prop lines (passing / rushing /
    receiving yards, receptions) with line values and prices?
  - do the snapshots reach back to an opening line and up to a closing one?
  - do the player names match nflverse / nba_api names?

API shape (from the docs and the cached responses):
  - each line value is its own market (marketId), the line is that
    market's `handicap` in /markets, `playerProp` flags props;
  - /historical-odds needs `bookmakers` (max 3) and returns, per
    bookmaker > market > outcome > playerId, a list of {createdAt, price}
    snapshots with no player name, so the game's /odds response (one more
    request) supplies playerId -> playerName.

Budget: hard cap MAX_REQUESTS; stops at the first non-200. Every response
is cached under data/raw/oddspapi/probe/, so reruns replay for free.

    docker compose run --rm api python -m backend.ingest.probe_oddspapi
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.oddspapi.io/v4"
CACHE_DIR = Path("data/raw/oddspapi/probe")
MAX_REQUESTS = 15
BOOKMAKERS = "fanduel,draftkings,hardrockbet"
HISTORICAL_COOLDOWN_S = 5.5  # docs: 5000 ms cooldown on /historical-odds

NFL_PROP_TYPES = {
    "playertotals-passyards": "Passing yards",
    "playertotals-rushyards": "Rushing yards",
    "playertotals-receivingyards": "Receiving yards",
    "playertotals-receptions": "Receptions",
}
NBA_PROP_TYPES = {"playertotals-points": "Points", "playertotals-rebounds": "Rebounds", "playertotals-assists": "Assists"}

_spent = 0
_last_historical = 0.0
_quota_seen: dict[str, str] = {}


def _get(path: str, **params) -> object:
    """GET with an on-disk cache keyed by path+params (never the key)."""
    global _spent, _last_historical
    key = hashlib.sha1(json.dumps([path, sorted(params.items())]).encode()).hexdigest()[:16]
    cache = CACHE_DIR / f"{path.strip('/').replace('/', '_')}_{key}.json"
    if cache.is_file():
        return json.loads(cache.read_text())["body"]

    if _spent >= MAX_REQUESTS:
        _done(f"stopping: request cap {MAX_REQUESTS} reached")
    if path == "/historical-odds":
        time.sleep(max(0.0, HISTORICAL_COOLDOWN_S - (time.time() - _last_historical)))
    time.sleep(1.2)  # every endpoint has a ~1 s per-endpoint cooldown (429 RATE_LIMITED otherwise)
    _spent += 1
    resp = requests.get(f"{BASE_URL}{path}", params={**params, "apiKey": os.environ["ODDSPAPI_API_KEY"]}, timeout=60)
    if path == "/historical-odds":
        _last_historical = time.time()
    quota = {k: v for k, v in resp.headers.items() if any(t in k.lower() for t in ("limit", "remaining", "quota", "usage", "credit"))}
    _quota_seen.update(quota)
    print(f"  [request {_spent}] {path} {params} -> {resp.status_code}")
    if resp.status_code != 200:
        _done(f"stopping on error: {resp.status_code} {resp.text[:400]}")
    body = resp.json()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"status": resp.status_code, "headers": dict(resp.headers), "body": body}))
    return body


def _done(msg: str | None = None) -> None:
    if msg:
        print(f"\n{msg}")
    print(f"\nrequests spent this run: {_spent} (cap {MAX_REQUESTS}; cached responses are free)")
    print(f"quota reported by the API: {_quota_seen or 'none (no quota/limit headers on any response)'}")
    sys.exit(0 if msg is None else 1)


def _norm(name: str) -> str:
    """'Vele, Devaughn' and 'Devaughn Vele' -> 'devaughn vele'."""
    if "," in name:
        last, first = name.split(",", 1)
        name = f"{first} {last}"
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    name = re.sub(r"[.'’]", "", name)
    name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", name)
    return " ".join(name.split())


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


# ---- fixture choice -----------------------------------------------------------

def _nfl_fixture(fixtures: list[dict]) -> dict:
    """A finished 2026 regular-season game from nflverse's Week 4, matched
    to OddsPapi by team names and kickoff (no API call)."""
    sched = pd.read_parquet("data/raw/nfl/schedules/2026.parquet")
    wk4 = sched[(sched["game_type"] == "REG") & (sched["week"] == 4) & sched["home_score"].notna()]
    print(f"  nflverse: {len(wk4)} completed Week 4 regular-season games")

    season_ids = {f.get("seasonId") for f in fixtures if f["startTime"].startswith("2026")}
    print(f"  OddsPapi tournament fixtures: {len(fixtures)} total; 2026 kickoffs carry seasonId(s) {season_ids}")
    teams = {t for row in wk4.itertuples() for t in (row.home_team, row.away_team)}
    for f in sorted(fixtures, key=lambda f: f["startTime"]):
        start = _ts(f["startTime"])
        if f.get("statusName") != "Finished" or not (datetime(2026, 10, 1, tzinfo=timezone.utc) <= start <= datetime(2026, 10, 6, tzinfo=timezone.utc)):
            continue
        abbrs = {f.get("participant1Abbr"), f.get("participant2Abbr")}
        if abbrs <= teams:
            print(f"  picked {f['participant1Name']} vs {f['participant2Name']} ({f['fixtureId']}), kickoff {start:%Y-%m-%d %H:%M}Z, "
                  f"status {f['statusName']}, seasonId {f.get('seasonId')} — teams match nflverse Week 4")
            return f
    _done("no finished Week 4 fixture matched nflverse teams")


# ---- prop analysis -------------------------------------------------------------

def _player_names(odds: dict) -> dict[str, str]:
    names = {}
    for book in (odds.get("bookmakerOdds") or {}).values():
        for market in (book.get("markets") or {}).values():
            for outcome in (market.get("outcomes") or {}).values():
                for pid, p in (outcome.get("players") or {}).items():
                    if p and p.get("playerName"):
                        names[pid] = p["playerName"]
    return names


def _analyze(label: str, hist: dict, odds: dict, markets: dict[int, dict], prop_types: dict[str, str],
             kickoff: datetime, known_names: set[str]) -> None:
    names = _player_names(odds)
    books = hist.get("bookmakers") or {}
    print(f"  historical bookmakers returned: {sorted(books) or 'none'}; player names known for {len(names)} playerIds (from /odds)")

    # (book, prop type, player, line) -> {"over": snaps, "under": snaps}
    props: dict[tuple, dict[str, list]] = {}
    for book, bd in books.items():
        for mid, md in (bd.get("markets") or {}).items():
            meta = markets.get(int(mid))
            if not meta or meta.get("marketType") not in prop_types:
                continue
            side_of = {str(o["outcomeId"]): o["outcomeName"].lower() for o in meta["outcomes"]}
            for oid, od in (md.get("outcomes") or {}).items():
                for pid, snaps in (od.get("players") or {}).items():
                    if snaps:
                        key = (book, prop_types[meta["marketType"]], pid, meta["handicap"])
                        props.setdefault(key, {})[side_of.get(oid, oid)] = sorted(snaps, key=lambda s: s["createdAt"])

    if not props:
        print(f"  {label}: NO player-prop lines in historical odds for these bookmakers")
        return
    by_type: dict[str, set] = {}
    for book, t, pid, line in props:
        by_type.setdefault(t, set()).add(pid)
    print("  prop lines found (players per market): " + ", ".join(f"{t}: {len(p)}" for t, p in sorted(by_type.items())))
    for t in prop_types.values():
        if t not in by_type:
            print(f"    {t}: none")

    # Opening / closing coverage across all prop snapshots.
    all_snaps = [s for sides in props.values() for snaps in sides.values() for s in snaps]
    first, last = min(_ts(s["createdAt"]) for s in all_snaps), max(_ts(s["createdAt"]) for s in all_snaps)
    pre = [_ts(s["createdAt"]) for s in all_snaps if _ts(s["createdAt"]) <= kickoff]
    print(f"  snapshots: {len(all_snaps)} total; earliest {first:%Y-%m-%d %H:%M}Z ({(kickoff - first) / timedelta(hours=1):.0f} h before kickoff), "
          f"latest pre-kickoff {max(pre):%Y-%m-%d %H:%M}Z ({(kickoff - max(pre)) / timedelta(minutes=1):.0f} min before)" if pre else
          f"  snapshots: {len(all_snaps)}, none before kickoff")

    # Three examples: two-sided props, one per market type where possible,
    # on the line whose closing prices are closest to even (the main line).
    def closing(snaps):
        pre_s = [s for s in snaps if _ts(s["createdAt"]) <= kickoff]
        return (pre_s or snaps)[-1]

    two_sided = {k: v for k, v in props.items() if "over" in v and "under" in v}
    examples, used_types = [], set()
    for key, sides in sorted(two_sided.items(), key=lambda kv: abs(closing(kv[1]["over"])["price"] - closing(kv[1]["under"])["price"])):
        if key[1] not in used_types and key[2] in names:
            examples.append((key, sides))
            used_types.add(key[1])
        if len(examples) == 3:
            break
    for key, sides in sorted(two_sided.items(), key=lambda kv: abs(closing(kv[1]["over"])["price"] - closing(kv[1]["under"])["price"])):
        if len(examples) == 3:
            break
        if key[2] in names and (key, sides) not in examples:
            examples.append((key, sides))

    matched = 0
    print("  examples (decimal prices; open = first snapshot, close = last before kickoff):")
    for (book, t, pid, line), sides in examples:
        name = names[pid]
        hit = _norm(name) in known_names
        matched += hit
        o, u = sides["over"], sides["under"]
        print(f"    {name:<24} {t:<16} {line:>6}  over {o[0]['price']:.2f}->{closing(o)['price']:.2f}  "
              f"under {u[0]['price']:.2f}->{closing(u)['price']:.2f}  [{book}, {len(o)}+{len(u)} snapshots]  "
              f"{'matches' if hit else 'NO match in'} reference names")
    all_pids = {k[2] for k in props}
    named = [names[p] for p in all_pids if p in names]
    all_hits = sum(_norm(n) in known_names for n in named)
    print(f"  name match: {matched}/{len(examples)} examples; {all_hits}/{len(named)} of all prop players with a name "
          f"({len(all_pids) - len(named)} playerIds had no name in /odds)")


# ---- main ----------------------------------------------------------------------

def _nfl_names() -> set[str]:
    df = pd.read_parquet("data/raw/nfl/rosters_weekly/2026.parquet", columns=["full_name"])
    return {_norm(n) for n in df["full_name"].dropna().unique()}


def _nba_names() -> set[str]:
    from nba_api.stats.static import players  # bundled list, no network
    return {_norm(p["full_name"]) for p in players.get_players()}


def main() -> None:
    if not os.environ.get("ODDSPAPI_API_KEY"):
        sys.exit("ODDSPAPI_API_KEY is not set in .env — nothing spent.")

    print("== NFL ==")
    tournaments = _get("/tournaments", sportId=14)
    nfl = next(t for t in tournaments if t["tournamentSlug"] == "nfl")
    others = [t["tournamentName"] for t in tournaments if "nfl" in t["tournamentSlug"] and t is not nfl]
    print(f"  tournament {nfl['tournamentId']} '{nfl['tournamentName']}' ({nfl['categoryName']}); separate: {others}")
    fixture = _nfl_fixture(_get("/fixtures", tournamentId=nfl["tournamentId"]))
    markets = {m["marketId"]: m for m in _get("/markets", sportId=14)}
    hist = _get("/historical-odds", fixtureId=fixture["fixtureId"], bookmakers=BOOKMAKERS)
    odds = _get("/odds", fixtureId=fixture["fixtureId"])
    _analyze("NFL", hist, odds, markets, NFL_PROP_TYPES, _ts(fixture.get("trueStartTime") or fixture["startTime"]), _nfl_names())

    print("\n== NBA ==")
    tournaments = _get("/tournaments", sportId=11)
    nba = [t for t in tournaments if t["tournamentSlug"] in ("nba", "nba-preseason")]
    print("  " + ", ".join(f"{t['tournamentId']} '{t['tournamentName']}' (future {t.get('futureFixtures')}, upcoming {t.get('upcomingFixtures')})" for t in nba))
    finished = []
    for t in nba:
        if t.get("futureFixtures") or t.get("upcomingFixtures"):
            finished += [f for f in _get("/fixtures", tournamentId=t["tournamentId"])
                         if f.get("statusName") == "Finished" and f["startTime"] >= "2026-09"]
    if not finished:
        print("  no finished NBA fixtures this season yet (preseason/regular season not started or not listed) — skipped")
        _done()
    f = max(finished, key=lambda f: f["startTime"])
    print(f"  picked {f['participant1Name']} vs {f['participant2Name']} ({f['fixtureId']}), {f['startTime']}, tournament {f['tournamentId']}")
    nba_markets = {m["marketId"]: m for m in _get("/markets", sportId=11)}
    types = {m["marketType"] for m in nba_markets.values() if m.get("playerProp")}
    prop_types = {t: lbl for t, lbl in NBA_PROP_TYPES.items() if t in types}
    hist = _get("/historical-odds", fixtureId=f["fixtureId"], bookmakers=BOOKMAKERS)
    odds = _get("/odds", fixtureId=f["fixtureId"])
    _analyze("NBA", hist, odds, nba_markets, prop_types, _ts(f.get("trueStartTime") or f["startTime"]), _nba_names())
    _done()


if __name__ == "__main__":
    main()
