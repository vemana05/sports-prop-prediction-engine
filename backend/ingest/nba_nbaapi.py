"""NBA player game logs via nba_api (stats.nba.com).

Uses LeagueGameLog (one call per season, player rows) rather than looping
PlayerGameLog per player — it's the same data, far fewer requests, and
stays well under stats.nba.com's informal rate limits.

`run()` (the runner's standard entrypoint) ingests only VALIDATION_SEASON,
so the default ingestion-runner pass stays fast. The full multi-season
backfill is a separate, explicit, long-running call — see `backfill()` —
meant to be kicked off in the background after `run()` has been validated.
"""

from __future__ import annotations

import os
import time

import duckdb
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

from backend.db import get_connection, next_log_id
from backend.ingest.util import IngestResult, logger, retry, utcnow

VALIDATION_SEASON = "2025-26"
BACKFILL_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]

_COLUMNS = {
    "PLAYER_ID": "player_id_nba",
    "PLAYER_NAME": "player_name",
    "TEAM_ABBREVIATION": "team",
    "GAME_ID": "game_id",
    "GAME_DATE": "game_date",
    "MIN": "minutes",
    "PTS": "pts",
    "REB": "reb",
    "AST": "ast",
    "FG3M": "fg3m",
    "STL": "stl",
    "BLK": "blk",
    "TOV": "tov",
}


def _request_delay() -> float:
    return float(os.environ.get("NBA_API_REQUEST_DELAY_SECONDS", "0.6"))


def _matchup_to_home_away_opponent(matchup: str) -> tuple[str, str]:
    # "LAL @ BOS" -> away, BOS ; "LAL vs. BOS" -> home, BOS
    if "@" in matchup:
        return "away", matchup.split("@")[-1].strip()
    return "home", matchup.split("vs.")[-1].strip()


def fetch_player_game_logs_season(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    def _call() -> pd.DataFrame:
        endpoint = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star=season_type,
            player_or_team_abbreviation="P",
            timeout=60,
        )
        return endpoint.get_data_frames()[0]

    df = retry(_call, what=f"LeagueGameLog({season})")
    if df.empty:
        return df

    home_away, opponent = zip(*df["MATCHUP"].map(_matchup_to_home_away_opponent))
    df = df.assign(HOME_AWAY=home_away, OPPONENT=opponent, SEASON=season)

    out = df.rename(columns=_COLUMNS)[
        list(_COLUMNS.values()) + ["HOME_AWAY", "OPPONENT", "SEASON"]
    ].rename(columns={"HOME_AWAY": "home_away", "OPPONENT": "opponent", "SEASON": "season"})
    out["game_date"] = pd.to_datetime(out["game_date"]).dt.date
    out["player_id_nba"] = out["player_id_nba"].astype(str)
    return out


def _write(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    now = utcnow()
    df = df.copy()
    df["source_published_at"] = df["game_date"]
    df["ingested_at"] = now
    conn.register("df_nba_games", df)
    conn.execute(
        """
        INSERT INTO nba_player_games
        SELECT player_id_nba, player_name, team, opponent, game_id, game_date, season,
               home_away, minutes, pts, reb, ast, fg3m, stl, blk, tov,
               source_published_at, ingested_at
        FROM df_nba_games
        ON CONFLICT (player_id_nba, game_id) DO UPDATE SET
            pts = excluded.pts, reb = excluded.reb, ast = excluded.ast,
            minutes = excluded.minutes, ingested_at = excluded.ingested_at
        """
    )
    conn.unregister("df_nba_games")
    return len(df)


def ingest_season(conn: duckdb.DuckDBPyConnection, season: str) -> int:
    logger.info("nba_api: fetching player game logs for %s", season)
    df = fetch_player_game_logs_season(season)
    rows = _write(conn, df)
    logger.info("nba_api: wrote %d rows for %s", rows, season)
    return rows


def run(conn: duckdb.DuckDBPyConnection | None = None) -> IngestResult:
    own_conn = conn is None
    conn = conn or get_connection()
    try:
        rows = ingest_season(conn, VALIDATION_SEASON)
        return IngestResult(rows_written=rows, notes={"season": VALIDATION_SEASON})
    finally:
        if own_conn:
            conn.close()


def backfill(seasons: list[str] | None = None) -> IngestResult:
    seasons = seasons or BACKFILL_SEASONS
    conn = get_connection()
    total = 0
    try:
        for i, season in enumerate(seasons):
            total += ingest_season(conn, season)
            if i < len(seasons) - 1:
                time.sleep(_request_delay())
        return IngestResult(rows_written=total, notes={"seasons": seasons})
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NBA player game log ingestion")
    parser.add_argument("--backfill", action="store_true", help="Ingest all BACKFILL_SEASONS")
    parser.add_argument("--seasons", nargs="*", help="Explicit season list, e.g. 2023-24 2024-25")
    args = parser.parse_args()

    if args.backfill or args.seasons:
        result = backfill(args.seasons)
        print(f"Backfill complete: {result.rows_written} rows across {result.notes['seasons']}")
    else:
        result = run()
        print(f"Validation ingest complete: {result.rows_written} rows for {result.notes['season']}")
