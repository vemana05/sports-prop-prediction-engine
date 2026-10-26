"""Free-tier NBA player points props via The Odds API (spec Section 3.5 /
5.7). Hard-capped at the free tier: 500 credits/month, no historical data.
This module never assumes it's safe to call — it checks the account's own
most recently logged `x-requests-used` header against
`ODDS_API_MONTHLY_CREDIT_BUDGET` before every paid call, and stops the
moment another call would risk exceeding that budget. The header, not a
locally-computed estimate, is the source of truth: The Odds API tracks
usage server-side per key per billing month, so reading it back is more
accurate than predicting cost from "markets x regions" ourselves.

Collects only `player_points`, region `us` — one credit per event per
pull, the cheapest real NBA prop data obtainable, so a modest daily
collection schedule can run the whole season without exhausting the free
tier. Decimal odds (matches Section 8's EV formula, which uses decimal
odds `d`), not American.

Meant to run once a day during the season (cron, or the Phase 7 scheduler
once it exists) — see `backend/predict/log_predictions.py`, which should
run just before this each day so a prediction is always on record before
the matching odds snapshot.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import duckdb
import requests
from nba_api.stats.static import teams as nba_static_teams

from backend.db import get_connection
from backend.ingest.util import IngestResult, logger, retry, utcnow

API_BASE = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"
MARKET = "player_points"
REGION = "us"
LOOKAHEAD_HOURS = 36  # only pull events starting within this window


def _api_key() -> str | None:
    return os.environ.get("ODDS_API_KEY") or None


def _budget() -> int:
    return int(os.environ.get("ODDS_API_MONTHLY_CREDIT_BUDGET", "450"))


def _last_known_usage(conn: duckdb.DuckDBPyConnection) -> int:
    row = conn.execute(
        "SELECT requests_used FROM odds_api_usage_log WHERE requests_used IS NOT NULL ORDER BY called_at DESC LIMIT 1"
    ).fetchone()
    return row[0] if row and row[0] is not None else 0


def _budget_ok(conn: duckdb.DuckDBPyConnection) -> bool:
    used = _last_known_usage(conn)
    budget = _budget()
    if used >= budget:
        logger.warning("odds_theoddsapi: credit budget reached (%d/%d used this cycle) — skipping", used, budget)
        return False
    return True


def _log_usage(conn: duckdb.DuckDBPyConnection, endpoint: str, resp: requests.Response, status: str, note: str | None = None) -> None:
    used = resp.headers.get("x-requests-used")
    remaining = resp.headers.get("x-requests-remaining")
    conn.execute(
        "INSERT INTO odds_api_usage_log VALUES (?, ?, ?, ?, ?, ?)",
        [utcnow(), endpoint, int(used) if used else None, int(remaining) if remaining else None, status, note],
    )


def fetch_upcoming_events(conn: duckdb.DuckDBPyConnection) -> list[dict]:
    """The events-list endpoint (no odds) is free on The Odds API, but we
    log it in odds_api_usage_log anyway for a complete usage trail."""
    key = _api_key()
    url = f"{API_BASE}/sports/{SPORT}/events"
    resp = retry(lambda: requests.get(url, params={"apiKey": key}, timeout=20), what="events list")
    resp.raise_for_status()
    _log_usage(conn, "events", resp, "success")
    events = resp.json()
    cutoff = datetime.now(timezone.utc) + timedelta(hours=LOOKAHEAD_HOURS)
    return [e for e in events if datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")) <= cutoff]


def fetch_event_player_points(conn: duckdb.DuckDBPyConnection, event_id: str) -> dict | None:
    key = _api_key()
    url = f"{API_BASE}/sports/{SPORT}/events/{event_id}/odds"
    params = {"apiKey": key, "regions": REGION, "markets": MARKET, "oddsFormat": "decimal"}
    resp = retry(lambda: requests.get(url, params=params, timeout=20), what=f"event odds {event_id}")
    if resp.status_code != 200:
        _log_usage(conn, f"event_odds:{event_id}", resp, "error", note=resp.text[:200])
        return None
    _log_usage(conn, f"event_odds:{event_id}", resp, "success")
    return resp.json()


def _write_event_teams(conn: duckdb.DuckDBPyConnection, event: dict, now) -> None:
    """The event's two teams, as abbreviations, into odds_events."""
    abbr = {t["full_name"]: t["abbreviation"] for t in nba_static_teams.get_teams()}
    home, away = abbr.get(event.get("home_team")), abbr.get(event.get("away_team"))
    if not (event.get("id") and home and away):
        logger.warning("odds_theoddsapi: event %s has unrecognized teams %r @ %r",
                       event.get("id"), event.get("away_team"), event.get("home_team"))
        return
    conn.execute(
        """
        INSERT INTO odds_events VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (event_id) DO UPDATE SET commence_time = excluded.commence_time,
            home_team = excluded.home_team, away_team = excluded.away_team, updated_at = excluded.updated_at
        """,
        [event["id"], event.get("commence_time"), home, away, now],
    )


def _write_snapshot(conn: duckdb.DuckDBPyConnection, event: dict) -> int:
    now = utcnow()
    _write_event_teams(conn, event, now)
    rows = []
    event_id = event.get("id")
    commence_time = event.get("commence_time")
    for bk in event.get("bookmakers", []):
        bookmaker = bk.get("key")
        for market in bk.get("markets", []):
            if market.get("key") != MARKET:
                continue
            by_player: dict[str, dict] = {}
            for outcome in market.get("outcomes", []):
                player = outcome.get("description")
                side = outcome.get("name")  # "Over" / "Under"
                if player:
                    by_player.setdefault(player, {})[side] = outcome
            for player, sides in by_player.items():
                over, under = sides.get("Over"), sides.get("Under")
                line = (over or under or {}).get("point")
                rows.append(
                    (now, event_id, None, commence_time, player, MARKET, bookmaker,
                     line, (over or {}).get("price"), (under or {}).get("price"))
                )
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT INTO odds_snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (pulled_at, event_id, player_name_raw, market, bookmaker) DO NOTHING
        """,
        rows,
    )
    return len(rows)


def run(conn: duckdb.DuckDBPyConnection | None = None) -> IngestResult:
    own_conn = conn is None
    conn = conn or get_connection()
    try:
        if not _api_key():
            logger.warning(
                "odds_theoddsapi: ODDS_API_KEY not set — nothing to do. "
                "Sign up at the-odds-api.com (free tier, 500 credits/month) and set it in .env."
            )
            return IngestResult(rows_written=0, notes={"error": "no API key"})

        if not _budget_ok(conn):
            return IngestResult(rows_written=0, notes={"error": "budget reached"})

        events = fetch_upcoming_events(conn)
        logger.info("odds_theoddsapi: %d events in the next %dh", len(events), LOOKAHEAD_HOURS)

        total_rows, events_pulled = 0, 0
        for event in events:
            if not _budget_ok(conn):
                logger.warning("odds_theoddsapi: stopping mid-run after %d/%d events — budget reached", events_pulled, len(events))
                break
            data = fetch_event_player_points(conn, event["id"])
            events_pulled += 1
            if data:
                total_rows += _write_snapshot(conn, data)

        logger.info("odds_theoddsapi: wrote %d prop line rows across %d/%d events", total_rows, events_pulled, len(events))
        return IngestResult(rows_written=total_rows, notes={"events_found": len(events), "events_pulled": events_pulled})
    finally:
        if own_conn:
            conn.close()


if __name__ == "__main__":
    result = run()
    print(f"odds_theoddsapi run complete: {result.rows_written} rows ({result.notes})")
