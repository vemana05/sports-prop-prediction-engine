"""Official NBA injury report snapshots via `nbainjuries`.

`nbainjuries.injury.get_reportdata(timestamp, return_df=True)` parses the NBA's
PDF injury report published at that moment into a DataFrame with columns
"Game Date", "Game Time", "Matchup", "Team", "Player Name" (format
"Last, First"), "Current Status", "Reason". `check_reportvalid(timestamp)`
tells you whether a report exists at that timestamp. Historical snapshots go
back to the 2021-22 season.

Importing `nbainjuries` starts a JVM immediately (via jpype/tabula-py), so
this module only runs where a JRE is installed — the backend Docker image,
not necessarily the host.

Note on validation timing: `run()`'s single-report validation pull validates
against a known in-season timestamp from the 2025-26 season rather than
scanning back from "now", since today falls in the NBA offseason between the
2025-26 season and the 2026-27 tip-off — there's no *current* report to find,
and scanning back from "now" could take thousands of 30-minute steps to
reach the last live report.

`backfill_game_days()` is the real historical pull: one pre-game snapshot
per game day, for every day in `nba_player_games` (all 5 backfilled
seasons) — what the Phase 2 walk-forward backtest actually needs. It does
NOT pull the full 15-minute-snapshot archive (`backfill_range()` does that,
exhaustively, for a narrower date range if ever needed) — one report per day
is enough to know who was hurt before that day's games.

Raw PDF caching: `fetch_report` downloads the report PDF itself (via
`injury.gen_url`) and saves it to `data/raw/injuries/<same filename NBA
uses>.pdf` before ever asking `nbainjuries` to parse it. A re-run (or a
from-scratch rebuild after dropping warehouse.duckdb) replays from that
cached PDF via `nbainjuries`' own built-in `local=True` mode instead of
re-downloading — `gen_url(timestamp)`'s basename is exactly the filename
`get_reportdata(..., local=True, localdir=...)` looks for, so no separate
naming logic is needed here.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import requests
from nbainjuries import injury

from backend.db import get_connection, next_log_id
from backend.ingest.util import IngestResult, logger, retry, utcnow

# Used only to download the raw PDF for caching (nbainjuries does its own
# validation/parsing requests separately with its own default headers) — a
# plain desktop user-agent is enough, the NBA's static PDF host needs no auth.
_DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

# nbainjuries compares the timestamp you pass against its own naive (no
# tzinfo) reference datetimes internally, so timestamps here must stay naive
# too — they are implicitly US/Eastern, matching how the NBA times its
# reports, but must NOT be made timezone-aware or those comparisons raise.
#
# A Tuesday well inside the completed 2025-26 regular season — used to
# validate the ingestion pipeline end-to-end. Not "now", since the league is
# in its offseason as of this writing.
VALIDATION_TIMESTAMP = datetime(2026, 2, 3, 17, 30)

_COLUMN_MAP = {
    "Game Date": "game_date",
    "Team": "team",
    "Player Name": "player_name_raw",
    "Current Status": "status",
    "Reason": "reason",
}


def _find_valid_timestamp(start: datetime, max_days_back: int = 14) -> datetime | None:
    candidate = start
    for _ in range(max_days_back + 1):
        if retry(lambda: injury.check_reportvalid(candidate), what=f"check_reportvalid({candidate})"):
            return candidate
        candidate -= timedelta(days=1)
    return None


def _cache_dir() -> Path:
    return Path(os.environ.get("NBA_INJURIES_CACHE_DIR", "data/raw/injuries"))


def _cache_path(timestamp: datetime) -> Path:
    # Same basename nbainjuries' own local-file lookup (_gen_filepath) expects —
    # gen_url()'s URL and the local filename share the identical date+time stem.
    return _cache_dir() / injury.gen_url(timestamp).rsplit("/", 1)[-1]


def _ensure_cached(timestamp: datetime) -> None:
    cache_path = _cache_path(timestamp)
    if cache_path.is_file():
        return
    url = injury.gen_url(timestamp)

    def _call():
        resp = requests.get(url, headers=_DOWNLOAD_HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.content

    pdf_bytes = retry(_call, what=f"download injury report PDF {url}")
    _cache_dir().mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(pdf_bytes)


def fetch_report(timestamp: datetime) -> pd.DataFrame:
    _ensure_cached(timestamp)
    df = retry(
        lambda: injury.get_reportdata(timestamp, local=True, localdir=str(_cache_dir()), return_df=True),
        what=f"get_reportdata({timestamp}) [cached]",
    )
    if df is None or df.empty:
        return pd.DataFrame(columns=list(_COLUMN_MAP.values()))
    out = df.rename(columns=_COLUMN_MAP)[list(_COLUMN_MAP.values())]
    # PDF table extraction occasionally yields blank rows (header/separator
    # lines with no player) — drop anything without a player name rather than
    # let it violate the NOT NULL constraint.
    out = out[out["player_name_raw"].notna() & (out["player_name_raw"].str.strip() != "")]
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.date
    # PDF extraction also occasionally yields two rows for the same player,
    # e.g. a status line split across pages — (player_name_raw, team) is the
    # PK's grain per report, and a duplicate within one report is a parsing
    # artifact, not two real facts, so keep the last (most-processed) one.
    out = out.drop_duplicates(subset=["player_name_raw", "team"], keep="last")
    return out


def _write(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame, report_time: datetime) -> int:
    if df.empty:
        return 0
    now = utcnow()
    df = df.copy()
    df["report_time"] = report_time
    df["source_published_at"] = report_time
    df["ingested_at"] = now
    conn.register("df_injuries", df)
    conn.execute(
        """
        INSERT INTO nba_injuries
            (player_name_raw, team, game_date, report_time, status, reason, source_published_at, ingested_at)
        SELECT player_name_raw, team, game_date, report_time, status, reason, source_published_at, ingested_at
        FROM df_injuries
        ON CONFLICT (player_name_raw, team, report_time) DO UPDATE SET
            status = excluded.status, reason = excluded.reason, ingested_at = excluded.ingested_at
        """
    )
    conn.unregister("df_injuries")
    return len(df)


def run(conn: duckdb.DuckDBPyConnection | None = None) -> IngestResult:
    own_conn = conn is None
    conn = conn or get_connection()
    try:
        timestamp = _find_valid_timestamp(VALIDATION_TIMESTAMP)
        if timestamp is None:
            logger.warning("nbainjuries: no valid report found near %s", VALIDATION_TIMESTAMP)
            return IngestResult(rows_written=0, notes={"error": "no valid report found"})
        logger.info("nbainjuries: fetching report for %s", timestamp)
        df = fetch_report(timestamp)
        rows = _write(conn, df, timestamp)
        logger.info("nbainjuries: wrote %d rows for report %s", rows, timestamp)
        return IngestResult(rows_written=rows, notes={"report_time": timestamp.isoformat()})
    finally:
        if own_conn:
            conn.close()


# Reports are typically published around these times (US/Eastern, naive —
# see the note above `VALIDATION_TIMESTAMP`). We only need one valid,
# pre-game snapshot per game day for the walk-forward backtest, not every
# slot, so try the closest-to-tipoff time first and fall back earlier in
# the day rather than scanning every 15/30-minute step.
_CANDIDATE_REPORT_TIMES = [(17, 30), (12, 0), (8, 30), (20, 30)]


def game_days(conn: duckdb.DuckDBPyConnection, seasons: list[str]) -> list:
    placeholders = ",".join(["?"] * len(seasons))
    rows = conn.execute(
        f"SELECT DISTINCT game_date FROM nba_player_games WHERE season IN ({placeholders}) ORDER BY game_date",
        seasons,
    ).fetchall()
    return [r[0] for r in rows]


def backfill_game_days(seasons: list[str] | None = None) -> IngestResult:
    """Pull at least one pre-game injury report snapshot for every day an
    NBA game was played in `seasons` (default: every season already in
    nba_player_games). This is what the Phase 2 walk-forward backtest
    actually needs — a point-in-time view of who was hurt before that day's
    games — and is far cheaper than `backfill_range`'s exhaustive
    every-15-minutes scan.
    """
    conn = get_connection()
    started = utcnow()
    try:
        if seasons is None:
            seasons = [r[0] for r in conn.execute("SELECT DISTINCT season FROM nba_player_games ORDER BY season").fetchall()]
        days = game_days(conn, seasons)
        season_by_date = dict(conn.execute("SELECT DISTINCT game_date, season FROM nba_player_games").fetchall())
        logger.info("nbainjuries backfill: %d game days across seasons %s", len(days), seasons)

        rows_by_season: dict[str, int] = {s: 0 for s in seasons}
        found_days = 0
        missing_days = []
        total_rows = 0

        for i, d in enumerate(days):
            timestamp = None
            for h, m in _CANDIDATE_REPORT_TIMES:
                candidate = datetime(d.year, d.month, d.day, h, m)
                if retry(lambda c=candidate: injury.check_reportvalid(c), what=f"check_reportvalid({candidate})"):
                    timestamp = candidate
                    break
            if timestamp is None:
                missing_days.append(d)
                continue

            df = fetch_report(timestamp)
            rows = _write(conn, df, timestamp)
            total_rows += rows
            found_days += 1
            season = season_by_date.get(d)
            if season:
                rows_by_season[season] = rows_by_season.get(season, 0) + rows

            if (i + 1) % 25 == 0:
                logger.info(
                    "nbainjuries backfill: %d/%d days processed (%d found, %d rows so far)",
                    i + 1, len(days), found_days, total_rows,
                )

        logger.info(
            "nbainjuries backfill complete: %d/%d days had a report (%d missing), %d total rows",
            found_days, len(days), len(missing_days), total_rows,
        )
        # This bypasses the runner's per-source _run_source() logging (it's a
        # standalone, explicitly-triggered backfill, not a routine ingestion
        # pass), so log it here directly — otherwise the freshness check has
        # no idea this data is current and reports "never_run".
        conn.execute(
            "INSERT INTO ingestion_log VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [next_log_id(conn), "nbainjuries", "success", total_rows, started, utcnow(), None, None],
        )
        return IngestResult(
            rows_written=total_rows,
            notes={
                "days_total": len(days),
                "days_with_report": found_days,
                "days_missing": len(missing_days),
                "missing_sample": [str(d) for d in missing_days[:15]],
                "rows_by_season": rows_by_season,
            },
        )
    finally:
        conn.close()


def find_games_missing_pretip_report(conn: duckdb.DuckDBPyConnection) -> list[tuple]:
    """Games where no ingested injury report row has a matching `game_date`
    AND a `report_time` before that game's actual tip-off (from
    nba_schedule). A report fetched on day D commonly already lists games
    for day D+1 (its own `game_date` field says so), so this is NOT simply
    "one report per day" — it's genuine per-game point-in-time correctness.

    Scoped to games already in `nba_player_games` (i.e. already played) —
    `nba_schedule` also holds the upcoming season's future, unplayed games
    (for forward prediction logging), which obviously have no injury report
    yet and aren't a correctness problem.
    """
    return conn.execute(
        """
        WITH game_reports AS (
            SELECT s.game_id, s.season, s.game_date, s.tip_utc,
                   (SELECT MAX(i.report_time) FROM nba_injuries i
                    WHERE i.game_date = s.game_date
                      AND (i.report_time AT TIME ZONE 'America/New_York') < s.tip_utc) AS ok_report
            FROM nba_schedule s
            WHERE s.game_id IN (SELECT DISTINCT game_id FROM nba_player_games)
        )
        SELECT game_id, season, game_date, tip_utc FROM game_reports WHERE ok_report IS NULL ORDER BY game_date
        """
    ).fetchall()


def fix_late_reports(step_minutes: int = 30, max_hours_back: int = 96) -> IngestResult:
    """For every game with no valid pre-tip-off report (see
    `find_games_missing_pretip_report`), scan backward from tip-off in
    `step_minutes` steps for a report that validates, and ingest it. Grouped
    by (season, game_date) since multiple games on the same date (e.g. a
    3-game Christmas slate) share the fix.
    """
    conn = get_connection()
    try:
        missing = find_games_missing_pretip_report(conn)
        if not missing:
            return IngestResult(rows_written=0, notes={"games_fixed": 0, "games_still_missing": 0})

        by_date: dict[tuple, datetime] = {}
        for game_id, season, game_date, tip_utc in missing:
            key = (season, game_date)
            if key not in by_date or tip_utc < by_date[key]:
                by_date[key] = tip_utc  # earliest tip-off among that date's bad games

        logger.info("nbainjuries: %d games across %d dates need an earlier report", len(missing), len(by_date))

        total_rows = 0
        fixed_dates = 0
        for (season, game_date), earliest_tip_utc in sorted(by_date.items()):
            # earliest tip-off, converted to naive ET wall-clock (nbainjuries needs naive timestamps)
            et_tip = conn.execute(
                "SELECT (?::TIMESTAMPTZ AT TIME ZONE 'America/New_York')", [earliest_tip_utc]
            ).fetchone()[0]
            candidate = et_tip.replace(tzinfo=None) - timedelta(minutes=step_minutes)
            found = None
            steps = int(max_hours_back * 60 / step_minutes)
            for _ in range(steps):
                if retry(lambda c=candidate: injury.check_reportvalid(c), what=f"check_reportvalid({candidate})"):
                    found = candidate
                    break
                candidate -= timedelta(minutes=step_minutes)
            if found is None:
                logger.warning("nbainjuries: no earlier report found for %s %s within %dh", season, game_date, max_hours_back)
                continue
            df = fetch_report(found)
            rows = _write(conn, df, found)
            total_rows += rows
            fixed_dates += 1
            logger.info("nbainjuries: fetched %s for %s %s (%d rows)", found, season, game_date, rows)

        still_missing = find_games_missing_pretip_report(conn)
        logger.info(
            "nbainjuries fix complete: %d/%d dates fixed, %d rows added, %d games still missing",
            fixed_dates, len(by_date), total_rows, len(still_missing),
        )
        conn.execute(
            "INSERT INTO ingestion_log VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [next_log_id(conn), "nbainjuries", "success", total_rows, utcnow(), utcnow(), None, None],
        )
        return IngestResult(
            rows_written=total_rows,
            notes={"dates_fixed": fixed_dates, "dates_total": len(by_date), "games_still_missing": len(still_missing)},
        )
    finally:
        conn.close()


def backfill_range(start: datetime, end: datetime, step_minutes: int = 30) -> IngestResult:
    """Pull every valid report in [start, end]. Meant for the Phase 2
    backtest's point-in-time injury history, not routine ingestion."""
    conn = get_connection()
    total = 0
    reports_found = 0
    try:
        t = start
        while t <= end:
            if retry(lambda: injury.check_reportvalid(t), what=f"check_reportvalid({t})"):
                df = fetch_report(t)
                total += _write(conn, df, t)
                reports_found += 1
            t += timedelta(minutes=step_minutes)
        return IngestResult(rows_written=total, notes={"reports_found": reports_found, "start": start.isoformat(), "end": end.isoformat()})
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Official NBA injury report ingestion")
    parser.add_argument("--backfill-game-days", action="store_true", help="One pre-game snapshot per game day across all ingested seasons")
    parser.add_argument("--fix-late-reports", action="store_true", help="Fetch an earlier report for games whose day-level snapshot is after tip-off")
    args = parser.parse_args()

    if args.fix_late_reports:
        result = fix_late_reports()
        print(f"nbainjuries late-report fix complete: {result.rows_written} rows ({result.notes})")
    elif args.backfill_game_days:
        result = backfill_game_days()
        print(f"nbainjuries game-day backfill complete: {result.rows_written} rows ({result.notes})")
    else:
        result = run()
        print(f"nbainjuries validation ingest complete: {result.rows_written} rows ({result.notes})")
