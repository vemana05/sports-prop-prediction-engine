"""NBA game schedule (tip-off times) via nba_api's ScheduleLeagueV2.

One call per season returns the entire season's schedule (preseason,
regular season, playoffs) with `gameDateTimeUTC` — the actual tip-off
timestamp. This is what makes point-in-time injury-report correctness
checkable: a report published after a game's tip-off is stale for that
game, even if it's the same calendar day (see backend/ingest/
injuries_nba_official.py's `fix_late_reports()`).
"""

from __future__ import annotations

import duckdb
import pandas as pd
from nba_api.stats.endpoints import scheduleleaguev2

from backend.db import get_connection
from backend.ingest.util import IngestResult, logger, retry, utcnow

SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]


def fetch_season_schedule(season: str) -> pd.DataFrame:
    def _call():
        ep = scheduleleaguev2.ScheduleLeagueV2(season=season, timeout=30)
        return ep.get_data_frames()[0]

    df = retry(_call, what=f"ScheduleLeagueV2({season})")
    df = df[df["gameId"].str.startswith("0022")].copy()  # regular season only
    df["season"] = season
    df["tip_utc"] = pd.to_datetime(df["gameDateTimeUTC"], utc=True)
    df["game_date"] = pd.to_datetime(df["gameDate"]).dt.date
    return df[["gameId", "season", "game_date", "homeTeam_teamId", "awayTeam_teamId", "tip_utc"]].rename(
        columns={"gameId": "game_id", "homeTeam_teamId": "home_team_id", "awayTeam_teamId": "away_team_id"}
    )


def _write(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    now = utcnow()
    df = df.copy()
    df["home_team_id"] = df["home_team_id"].astype(str)
    df["away_team_id"] = df["away_team_id"].astype(str)
    df["ingested_at"] = now
    conn.register("df_schedule", df)
    conn.execute(
        """
        INSERT INTO nba_schedule (game_id, season, game_date, home_team_id, away_team_id, tip_utc, ingested_at)
        SELECT game_id, season, game_date, home_team_id, away_team_id, tip_utc, ingested_at FROM df_schedule
        ON CONFLICT (game_id) DO UPDATE SET tip_utc = excluded.tip_utc, ingested_at = excluded.ingested_at
        """
    )
    conn.unregister("df_schedule")
    return len(df)


def run(conn: duckdb.DuckDBPyConnection | None = None) -> IngestResult:
    own_conn = conn is None
    conn = conn or get_connection()
    try:
        total = 0
        for season in SEASONS:
            df = fetch_season_schedule(season)
            rows = _write(conn, df)
            total += rows
            logger.info("nba_schedule: wrote %d games for %s", rows, season)
        return IngestResult(rows_written=total, notes={"seasons": SEASONS})
    finally:
        if own_conn:
            conn.close()


if __name__ == "__main__":
    result = run()
    print(f"nba_schedule ingest complete: {result.rows_written} rows ({result.notes})")
