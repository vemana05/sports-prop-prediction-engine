"""Sleeper player dump (api.sleeper.app) — status, depth chart, trending.

Public API, no key. The full `/v1/players/nba` dump is ~5MB and Sleeper asks
that it be pulled at most once a day, so `run()` is meant to be called on
that cadence (see `refresh: "hourly"` in sources.yaml — hourly is the ceiling
imposed by the ingestion runner's schedule, not a requirement to actually
hit the endpoint that often; callers should still rate-limit themselves to
roughly daily for this specific endpoint).
"""

from __future__ import annotations

import requests

import duckdb

from backend.db import get_connection
from backend.ingest.util import IngestResult, logger, retry, utcnow

PLAYERS_URL = "https://api.sleeper.app/v1/players/nba"
REQUEST_TIMEOUT = 30


def fetch_players() -> dict:
    def _call():
        resp = requests.get(PLAYERS_URL, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    return retry(_call, what="sleeper players dump")


def _write(conn: duckdb.DuckDBPyConnection, players: dict) -> int:
    now = utcnow()
    rows = []
    for sleeper_id, p in players.items():
        if not p:
            continue
        rows.append(
            (
                sleeper_id,
                p.get("full_name") or " ".join(filter(None, [p.get("first_name"), p.get("last_name")])),
                p.get("team"),
                p.get("position"),
                p.get("status"),
                p.get("injury_status"),
                p.get("depth_chart_order"),
                p.get("search_rank"),
                now,
            )
        )
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT INTO sleeper_players
            (sleeper_id, full_name, team, position, status, injury_status, depth_chart_order, search_rank, ingested_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (sleeper_id) DO UPDATE SET
            full_name = excluded.full_name, team = excluded.team, position = excluded.position,
            status = excluded.status, injury_status = excluded.injury_status,
            depth_chart_order = excluded.depth_chart_order, search_rank = excluded.search_rank,
            ingested_at = excluded.ingested_at
        """,
        rows,
    )
    return len(rows)


def run(conn: duckdb.DuckDBPyConnection | None = None) -> IngestResult:
    own_conn = conn is None
    conn = conn or get_connection()
    try:
        players = fetch_players()
        rows = _write(conn, players)
        logger.info("sleeper: wrote %d players", rows)
        return IngestResult(rows_written=rows)
    finally:
        if own_conn:
            conn.close()


if __name__ == "__main__":
    result = run()
    print(f"Sleeper ingest complete: {result.rows_written} players")
