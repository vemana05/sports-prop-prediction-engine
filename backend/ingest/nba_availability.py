"""Derive the player-availability table (Section 5.2 teammate-availability
features read this) from three already-ingested sources:

- `nba_box_status` (backend.ingest.nba_box_status): full per-game roster,
  played or not, from BoxScoreTraditionalV3 — the ground truth for
  "played" vs. "did not play this game".
- `nba_injuries` (backend.ingest.injuries_nba_official, resolved to
  player_id_nba by backend.xref.match_injury_names): the official
  pre-game injury report.
- `nba_schedule` (backend.ingest.nba_schedule): tip-off times, so the
  injury-report lookup is genuinely point-in-time (see
  injuries_nba_official.find_games_missing_pretip_report for why "same
  calendar day" isn't good enough).

Three-way classification per (game_id, player_id_nba):
  - "played": has a real minutes value in nba_box_status.
  - "dnp_listed_injury_report": no minutes, AND the injury report
    published before this game's tip-off (matched on the report row's own
    `game_date`) lists this player.
  - "dnp_other": no minutes, not on that injury report — a healthy
    scratch, coach's decision, or two-way/G-League assignment (whatever
    BoxScoreTraditionalV3's own `comment` says, carried through as detail).

Done in Python/pandas rather than SQL: the point-in-time injury lookup is a
per-(player, game_date) "latest report before this specific game's
tip-off" search, which is simplest as a sorted-list lookup instead of
fighting correlated-subquery performance in SQL for ~160k rows.
"""

from __future__ import annotations

import bisect
from collections import defaultdict

import duckdb
import pandas as pd

from backend.db import get_connection
from backend.ingest.util import IngestResult, logger, utcnow


def _load_schedule(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return conn.execute("SELECT game_id, season, game_date, tip_utc FROM nba_schedule").fetchdf()


def _load_injury_index(conn: duckdb.DuckDBPyConnection) -> dict[tuple, list[tuple]]:
    """Returns {(player_id_nba, game_date): [(report_time_utc, status, reason), ...]} sorted by report_time."""
    rows = conn.execute(
        """
        SELECT player_id_nba, game_date,
               (report_time AT TIME ZONE 'America/New_York') AS report_time_utc,
               status, reason
        FROM nba_injuries
        WHERE player_id_nba IS NOT NULL
        """
    ).fetchall()
    index: dict[tuple, list[tuple]] = defaultdict(list)
    for player_id, game_date, report_time_utc, status, reason in rows:
        index[(player_id, game_date)].append((report_time_utc, status, reason))
    for key in index:
        index[key].sort(key=lambda t: t[0])
    return index


def _latest_before(entries: list[tuple], tip_utc) -> tuple | None:
    times = [e[0] for e in entries]
    i = bisect.bisect_left(times, tip_utc)
    return entries[i - 1] if i > 0 else None


def build(conn: duckdb.DuckDBPyConnection, game_ids: list[str] | None = None) -> IngestResult:
    schedule = _load_schedule(conn)
    schedule_by_game = {
        # .date() matters: fetchdf() returns game_date as a pandas Timestamp,
        # but _load_injury_index's game_date keys come from plain .fetchall()
        # (Python datetime.date) — without normalizing to the same type here,
        # every injury_index lookup below silently misses (different key
        # types never compare equal, even for "the same day").
        r.game_id: (r.season, r.game_date.date(), r.tip_utc) for r in schedule.itertuples()
    }
    injury_index = _load_injury_index(conn)

    query = "SELECT game_id, team_id, player_id_nba, minutes_raw, comment FROM nba_box_status"
    params: list = []
    if game_ids is not None:
        placeholders = ",".join(["?"] * len(game_ids))
        query += f" WHERE game_id IN ({placeholders})"
        params = game_ids
    box = conn.execute(query, params).fetchall()

    now = utcnow()
    rows = []
    counts = {"played": 0, "dnp_listed_injury_report": 0, "dnp_other": 0}
    skipped_no_schedule = 0

    for game_id, team_id, player_id_nba, minutes_raw, comment in box:
        sched = schedule_by_game.get(game_id)
        if sched is None:
            skipped_no_schedule += 1
            continue
        season, game_date, tip_utc = sched

        if minutes_raw not in (None, ""):
            status, detail = "played", None
        else:
            entry = _latest_before(injury_index.get((player_id_nba, game_date), []), tip_utc)
            if entry is not None:
                _, inj_status, inj_reason = entry
                status = "dnp_listed_injury_report"
                detail = f"{inj_status}: {inj_reason}" if inj_reason else inj_status
            else:
                status, detail = "dnp_other", comment

        counts[status] += 1
        rows.append((game_id, season, team_id, player_id_nba, status, detail, now))

    if skipped_no_schedule:
        logger.warning("nba_availability: %d box_status rows had no matching nba_schedule row", skipped_no_schedule)

    if rows:
        conn.executemany(
            """
            INSERT INTO nba_player_availability (game_id, season, team_id, player_id_nba, status, detail, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (game_id, player_id_nba) DO UPDATE SET
                status = excluded.status, detail = excluded.detail, ingested_at = excluded.ingested_at
            """,
            rows,
        )

    logger.info(
        "nba_availability: %d rows (played=%d, dnp_listed=%d, dnp_other=%d)",
        len(rows), counts["played"], counts["dnp_listed_injury_report"], counts["dnp_other"],
    )
    return IngestResult(rows_written=len(rows), notes=counts)


def run(conn: duckdb.DuckDBPyConnection | None = None) -> IngestResult:
    own_conn = conn is None
    conn = conn or get_connection()
    try:
        return build(conn)
    finally:
        if own_conn:
            conn.close()


if __name__ == "__main__":
    result = run()
    print(f"nba_availability build complete: {result.rows_written} rows ({result.notes})")
