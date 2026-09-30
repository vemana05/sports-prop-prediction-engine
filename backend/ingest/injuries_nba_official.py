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

Note on validation timing: today (see module-level TODAY) falls in the NBA
offseason between the 2025-26 season and the 2026-27 tip-off, so there is no
*current* report to fetch. `run()` therefore validates against a known
in-season timestamp from the most recently completed season rather than
scanning back from "now" (which could require thousands of 30-minute steps
to reach the last live report). Pulling the full historical archive of
15-minute snapshots back to 2021-22 is a separate, much larger job for when
the walk-forward backtest (Phase 2) needs point-in-time injury history —
not part of Phase 1 validation.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import duckdb
import pandas as pd
from nbainjuries import injury

from backend.db import get_connection
from backend.ingest.util import IngestResult, logger, retry, utcnow

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


def fetch_report(timestamp: datetime) -> pd.DataFrame:
    df = retry(lambda: injury.get_reportdata(timestamp, return_df=True), what=f"get_reportdata({timestamp})")
    if df is None or df.empty:
        return pd.DataFrame(columns=list(_COLUMN_MAP.values()))
    out = df.rename(columns=_COLUMN_MAP)[list(_COLUMN_MAP.values())]
    # PDF table extraction occasionally yields blank rows (header/separator
    # lines with no player) — drop anything without a player name rather than
    # let it violate the NOT NULL constraint.
    out = out[out["player_name_raw"].notna() & (out["player_name_raw"].str.strip() != "")]
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.date
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
    result = run()
    print(f"nbainjuries validation ingest complete: {result.rows_written} rows ({result.notes})")
