"""Daily incremental NBA game-log refresh (run by scripts/daily_odds_job.sh
before predictions are logged).

Steps, each skipping work already done so a re-run after a crash just picks
up where it stopped:
  1. Wait for stats.nba.com to resolve (the Mac can wake with no network).
  2. Refresh the current season's schedule (one ScheduleLeagueV2 call):
     tip times into nba_schedule, status and scores into nba_game_results.
  3. Player game logs (one LeagueGameLog call, from the earliest missing
     date) for finished games not yet in nba_player_games. A game the
     schedule says is final but stats.nba.com hasn't logged yet stays
     pending for the next run.
  4. Box status (one BoxScoreTraditionalV3 call per game, cached on disk)
     for this season's logged games not yet in nba_box_status, then
     nba_player_availability for those games.
  5. Rebuild nba_features if anything new was loaded or the features are
     behind the game logs. Models are NOT retrained here.

No API keys are involved (stats.nba.com is keyless); the job script still
pipes this step's output through its key redaction like every other step.
"""

from __future__ import annotations

import socket
import time
from datetime import date, datetime, timedelta, timezone
from typing import Callable
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd

from backend.db import get_connection
from backend.features import nba_features
from backend.ingest import nba_availability, nba_box_status, nba_nbaapi, nba_schedule
from backend.ingest.util import IngestResult, logger, utcnow

ET = ZoneInfo("America/New_York")
STATS_HOST = "stats.nba.com"
DNS_ATTEMPTS = 10
DNS_DELAY_S = 30.0
FINAL_STATUS = 3  # ScheduleLeagueV2 gameStatus: 1 scheduled, 2 live, 3 final
# A game with no status on record counts as finished this long after tip.
NO_STATUS_FINISHED_AFTER = timedelta(hours=4)


def wait_for_dns(
    host: str = STATS_HOST,
    attempts: int = DNS_ATTEMPTS,
    delay_s: float = DNS_DELAY_S,
    resolve: Callable = socket.getaddrinfo,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """True once `host` resolves; False after `attempts` failures."""
    for i in range(1, attempts + 1):
        try:
            resolve(host, 443)
            return True
        except OSError as exc:
            logger.warning("nba_daily_games: %s does not resolve (attempt %d/%d): %s", host, i, attempts, exc)
            if i < attempts:
                sleep(delay_s)
    return False


def season_for(day: date) -> str:
    """NBA season label for `day`: October onward is the new season."""
    start = day.year if day.month >= 10 else day.year - 1
    return f"{start}-{(start + 1) % 100:02d}"


def _write_results(conn: duckdb.DuckDBPyConnection, schedule: pd.DataFrame) -> int:
    if schedule.empty or "game_status" not in schedule.columns:
        return 0
    now = utcnow()
    rows = [
        (
            r.game_id, r.season,
            None if pd.isna(r.game_status) else int(r.game_status),
            None if pd.isna(getattr(r, "home_score", None)) else int(r.home_score),
            None if pd.isna(getattr(r, "away_score", None)) else int(r.away_score),
            now,
        )
        for r in schedule.itertuples()
    ]
    conn.executemany(
        """
        INSERT INTO nba_game_results VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (game_id) DO UPDATE SET game_status = excluded.game_status,
            home_score = excluded.home_score, away_score = excluded.away_score, updated_at = excluded.updated_at
        """,
        rows,
    )
    return len(rows)


def pending_log_games(conn: duckdb.DuckDBPyConnection, season: str, now: datetime) -> pd.DataFrame:
    """(game_id, game_date) of this season's finished games with no rows in
    nba_player_games. Finished = final per nba_game_results, or (no status
    on record) tipped more than NO_STATUS_FINISHED_AFTER ago."""
    return conn.execute(
        """
        SELECT s.game_id, s.game_date
        FROM nba_schedule s LEFT JOIN nba_game_results r USING (game_id)
        WHERE s.season = ?
          AND (r.game_status = ? OR (r.game_status IS NULL AND s.tip_utc < ?))
          AND s.game_id NOT IN (SELECT DISTINCT game_id FROM nba_player_games)
        ORDER BY s.game_date, s.game_id
        """,
        [season, FINAL_STATUS, now - NO_STATUS_FINISHED_AFTER],
    ).fetchdf()


def pending_box_games(conn: duckdb.DuckDBPyConnection, season: str) -> list[str]:
    """This season's logged games with no nba_box_status rows."""
    return [
        r[0]
        for r in conn.execute(
            """
            SELECT DISTINCT game_id FROM nba_player_games
            WHERE season = ? AND game_id NOT IN (SELECT DISTINCT game_id FROM nba_box_status)
            ORDER BY game_id
            """,
            [season],
        ).fetchall()
    ]


def features_behind(conn: duckdb.DuckDBPyConnection) -> bool:
    """True when nba_features is missing the latest logged game date."""
    logs, feats = conn.execute(
        f"SELECT (SELECT max(game_date) FROM nba_player_games), (SELECT max(game_date) FROM {nba_features.FEATURE_TABLE})"
    ).fetchone()
    return logs is not None and (feats is None or feats < logs)


def run(
    conn: duckdb.DuckDBPyConnection | None = None,
    now: datetime | None = None,
    wait_dns: bool = True,
) -> IngestResult:
    now = now or datetime.now(timezone.utc)
    season = season_for(now.astimezone(ET).date())
    if wait_dns and not wait_for_dns():
        logger.error("nba_daily_games: %s never resolved — nothing loaded", STATS_HOST)
        return IngestResult(rows_written=0, notes={"season": season, "errors": ["dns"]})

    own_conn = conn is None
    conn = conn or get_connection()
    errors: list[str] = []
    notes: dict = {"season": season}
    try:
        try:
            sched = nba_schedule.fetch_season_schedule(season)
            nba_schedule._write(conn, sched)
            notes["results"] = _write_results(conn, sched)
        except Exception as exc:  # noqa: BLE001 - fall back to the schedule already on disk
            errors.append("schedule")
            logger.error("nba_daily_games: schedule refresh failed, using the stored schedule: %s", exc)

        pending = pending_log_games(conn, season, now)
        notes["log_games_pending"] = len(pending)
        log_rows = 0
        if not pending.empty:
            date_from = pd.Timestamp(pending["game_date"].min()).date()
            try:
                logs = nba_nbaapi.fetch_player_game_logs_season(season, date_from=date_from)
                if not logs.empty:
                    logs = logs[logs["game_id"].isin(set(pending["game_id"]))]
                log_rows = nba_nbaapi._write(conn, logs)
            except Exception as exc:  # noqa: BLE001 - games stay pending for the next run
                errors.append("player_games")
                logger.error("nba_daily_games: LeagueGameLog failed: %s", exc)
        still = pending_log_games(conn, season, now)
        notes["log_rows"], notes["log_games_still_pending"] = log_rows, len(still)
        logger.info(
            "nba_daily_games: %d player-game rows across %d/%d finished games (%d not logged by stats.nba.com yet)",
            log_rows, len(pending) - len(still), len(pending), len(still),
        )

        box_ids = pending_box_games(conn, season)
        box_rows, failed = (0, [])
        if box_ids:
            box_rows, failed = nba_box_status._process_games(conn, box_ids, len(box_ids), "nba_daily_games box status")
            if failed:
                errors.append("box_status")
            done = [g for g in box_ids if g not in set(failed)]
            if done:
                nba_availability.build(conn, game_ids=done)
        notes["box_games"], notes["box_failed"] = len(box_ids) - len(failed), len(failed)

        if log_rows or box_rows or features_behind(conn):
            notes["features"] = nba_features.build(conn).rows_written
        else:
            logger.info("nba_daily_games: nothing new, nba_features left as is")
            notes["features"] = None

        notes["errors"] = errors
        return IngestResult(rows_written=log_rows + box_rows, notes=notes)
    finally:
        if own_conn:
            conn.close()


if __name__ == "__main__":
    result = run()
    print(f"nba_daily_games complete: {result.rows_written} rows ({result.notes})")
    raise SystemExit(1 if result.notes.get("errors") else 0)
