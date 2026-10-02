"""Full per-game team rosters (played AND DNP/inactive) via nba_api's
BoxScoreTraditionalV3 — one call per game, since there's no season-level
bulk equivalent that includes non-playing players.

This is the only source we have that lists players who did NOT play, with
a `comment` explaining why (e.g. "DNP - Coach's Decision", "NOT WITH TEAM",
or blank when the box score itself has no reason, e.g. for older seasons).
`nba_player_games` (from LeagueGameLog) silently omits these players
entirely — see its schema comment in backend/db.py.

Resumable: skips any game_id already present in nba_box_status, so an
interrupted run can just be re-started.

Raw response caching: every BoxScoreTraditionalV3 response is saved to
`data/raw/nba_box_status/<game_id>.json` before being parsed. A re-run (or
a from-scratch rebuild after dropping warehouse.duckdb) replays from that
cache instead of re-hitting stats.nba.com — `fetch_game_roster` builds the
endpoint object with `get_request=False` and feeds it the cached JSON
through the same response/parser classes nba_api itself uses, so the
output is identical to a live call.

Backfill isolation: the full historical backfill (~6,150 games, one API
call each) is a long-running, resumable job that would otherwise hold
open a writable connection to the live warehouse.duckdb for its entire
run. `run_backfill()` instead writes to its own separate DuckDB file
(BACKFILL_DB_PATH) — just the nba_box_status table, read-only-attaching
the main warehouse to know which games are still pending — and merges
into warehouse.duckdb only when that invocation finishes (or on demand via
`merge_into_warehouse()`). See PARLAY_BUILDER_SPEC.md's note on this
backfill being large enough to need multiple runs.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd
from nba_api.stats.endpoints import boxscoretraditionalv3
from nba_api.stats.library.http import NBAStatsResponse

from backend.db import DEFAULT_DB_PATH, NBA_BOX_STATUS_SCHEMA, get_connection
from backend.ingest.util import IngestResult, logger, retry, utcnow

_COLUMNS = {
    "gameId": "game_id",
    "teamId": "team_id",
    "personId": "player_id_nba",
    "minutes": "minutes_raw",
    "comment": "comment",
}

BACKFILL_DB_PATH = os.environ.get("BOX_STATUS_BACKFILL_DB_PATH", "data/box_status_backfill.duckdb")


def _escape_sql_literal(path: str) -> str:
    return path.replace("'", "''")


def _cache_dir() -> Path:
    return Path(os.environ.get("NBA_BOX_STATUS_CACHE_DIR", "data/raw/nba_box_status"))


def _cache_path(game_id: str) -> Path:
    return _cache_dir() / f"{game_id}.json"


def fetch_game_roster(game_id: str) -> pd.DataFrame:
    cache_path = _cache_path(game_id)
    if cache_path.is_file():
        bs = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id, get_request=False)
        bs.nba_response = NBAStatsResponse(cache_path.read_text(), 200, "")
        bs.load_response()
    else:
        def _call():
            return boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id, timeout=30)

        bs = retry(_call, what=f"BoxScoreTraditionalV3({game_id})")
        _cache_dir().mkdir(parents=True, exist_ok=True)
        cache_path.write_text(bs.get_json())

    df = bs.get_data_frames()[0]
    if df.empty:
        return df
    df = df.assign(player_name_raw=df["firstName"] + " " + df["familyName"])
    out = df.rename(columns=_COLUMNS)[list(_COLUMNS.values()) + ["player_name_raw"]]
    out["team_id"] = out["team_id"].astype(str)
    out["player_id_nba"] = out["player_id_nba"].astype(str)
    return out


def _write(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    now = utcnow()
    df = df.copy()
    df["ingested_at"] = now
    conn.register("df_box", df)
    conn.execute(
        """
        INSERT INTO nba_box_status (game_id, team_id, player_id_nba, player_name_raw, minutes_raw, comment, ingested_at)
        SELECT game_id, team_id, player_id_nba, player_name_raw, minutes_raw, comment, ingested_at FROM df_box
        ON CONFLICT (game_id, player_id_nba) DO UPDATE SET
            minutes_raw = excluded.minutes_raw, comment = excluded.comment, ingested_at = excluded.ingested_at
        """
    )
    conn.unregister("df_box")
    return len(df)


def pending_game_ids(conn: duckdb.DuckDBPyConnection) -> list[str]:
    return [
        r[0]
        for r in conn.execute(
            """
            SELECT game_id FROM nba_player_games
            WHERE game_id NOT IN (SELECT DISTINCT game_id FROM nba_box_status)
            GROUP BY game_id ORDER BY game_id
            """
        ).fetchall()
    ]


def _process_games(conn: duckdb.DuckDBPyConnection, game_ids: list[str], total_pending: int, log_prefix: str) -> tuple[int, list[str]]:
    total_rows = 0
    failed: list[str] = []
    for i, game_id in enumerate(game_ids):
        try:
            df = fetch_game_roster(game_id)
            total_rows += _write(conn, df)
        except Exception as exc:  # noqa: BLE001 - stats.nba.com is flaky under sustained load;
            # one persistently-failing game must not kill a 6000+ game run. It stays "pending"
            # (not written) so a later re-run retries it automatically.
            failed.append(game_id)
            logger.warning("%s: giving up on %s after retries: %s", log_prefix, game_id, exc)
        if (i + 1) % 100 == 0:
            logger.info(
                "%s: %d/%d games processed this run (%d failed so far)",
                log_prefix, i + 1, len(game_ids), len(failed),
            )
    logger.info(
        "%s: wrote %d roster rows across %d/%d games (%d failed, still pending for next run)",
        log_prefix, total_rows, len(game_ids) - len(failed), len(game_ids), len(failed),
    )
    return total_rows, failed


def run(conn: duckdb.DuckDBPyConnection | None = None, limit: int | None = None) -> IngestResult:
    own_conn = conn is None
    conn = conn or get_connection()
    try:
        game_ids = pending_game_ids(conn)
        total_pending = len(game_ids)
        if limit is not None:
            game_ids = game_ids[:limit]
        logger.info("nba_box_status: %d games pending (%d requested this run)", total_pending, len(game_ids))

        total_rows, failed = _process_games(conn, game_ids, total_pending, "nba_box_status")
        return IngestResult(
            rows_written=total_rows,
            notes={"games_processed": len(game_ids) - len(failed), "games_failed": len(failed), "games_pending_before": total_pending},
        )
    finally:
        if own_conn:
            conn.close()


# --- Backfill: separate DuckDB file, merged into warehouse.duckdb on completion ---


def _backfill_connection(path: str = BACKFILL_DB_PATH) -> duckdb.DuckDBPyConnection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(path)
    conn.execute(NBA_BOX_STATUS_SCHEMA)
    return conn


def backfill_pending_game_ids(backfill_conn: duckdb.DuckDBPyConnection, main_db_path: str) -> list[str]:
    """Games not yet in nba_box_status in EITHER the backfill db (progress
    from a prior, not-yet-merged backfill run) or the main warehouse
    (already merged) — read-only attaches the main db just to check
    nba_player_games/nba_box_status, never writes to it."""
    backfill_conn.execute(f"ATTACH '{_escape_sql_literal(main_db_path)}' AS main_db (READ_ONLY)")
    try:
        rows = backfill_conn.execute(
            """
            SELECT game_id FROM main_db.nba_player_games
            WHERE game_id NOT IN (SELECT DISTINCT game_id FROM nba_box_status)
              AND game_id NOT IN (SELECT DISTINCT game_id FROM main_db.nba_box_status)
            GROUP BY game_id ORDER BY game_id
            """
        ).fetchall()
    finally:
        backfill_conn.execute("DETACH main_db")
    return [r[0] for r in rows]


def merge_into_warehouse(backfill_db_path: str = BACKFILL_DB_PATH, main_db_path: str | None = None) -> int:
    """Upsert every row from the backfill db's nba_box_status into the main
    warehouse's nba_box_status. Safe to call repeatedly / on a partial
    backfill — it's the same ON CONFLICT upsert `_write` uses."""
    if not Path(backfill_db_path).is_file():
        logger.info("nba_box_status merge: %s does not exist, nothing to merge", backfill_db_path)
        return 0
    main_conn = get_connection(main_db_path)
    try:
        main_conn.execute(f"ATTACH '{_escape_sql_literal(backfill_db_path)}' AS backfill_db (READ_ONLY)")
        try:
            merged = main_conn.execute("SELECT COUNT(*) FROM backfill_db.nba_box_status").fetchone()[0]
            main_conn.execute(
                """
                INSERT INTO nba_box_status (game_id, team_id, player_id_nba, player_name_raw, minutes_raw, comment, ingested_at)
                SELECT game_id, team_id, player_id_nba, player_name_raw, minutes_raw, comment, ingested_at
                FROM backfill_db.nba_box_status
                ON CONFLICT (game_id, player_id_nba) DO UPDATE SET
                    minutes_raw = excluded.minutes_raw, comment = excluded.comment, ingested_at = excluded.ingested_at
                """
            )
        finally:
            main_conn.execute("DETACH backfill_db")
        logger.info("nba_box_status merge: merged %d rows from %s into warehouse", merged, backfill_db_path)
        return merged
    finally:
        main_conn.close()


def run_backfill(limit: int | None = None, merge: bool = True, main_db_path: str | None = None) -> IngestResult:
    main_db_path = main_db_path or os.environ.get("DUCKDB_PATH", DEFAULT_DB_PATH)
    conn = _backfill_connection()
    try:
        game_ids = backfill_pending_game_ids(conn, main_db_path)
        total_pending = len(game_ids)
        if limit is not None:
            game_ids = game_ids[:limit]
        logger.info(
            "nba_box_status backfill: %d games pending (%d requested this run), writing to %s",
            total_pending, len(game_ids), BACKFILL_DB_PATH,
        )

        total_rows, failed = _process_games(conn, game_ids, total_pending, "nba_box_status backfill")
    finally:
        conn.close()

    merged_rows = 0
    if merge:
        merged_rows = merge_into_warehouse(BACKFILL_DB_PATH, main_db_path)

    return IngestResult(
        rows_written=total_rows,
        notes={
            "games_processed": len(game_ids) - len(failed),
            "games_failed": len(failed),
            "games_pending_before": total_pending,
            "merged_rows": merged_rows if merge else None,
        },
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NBA per-game roster (played + DNP) ingestion")
    parser.add_argument("--limit", type=int, default=None, help="Only process this many pending games (omit for all)")
    parser.add_argument(
        "--backfill", action="store_true",
        help=f"Write to a separate DuckDB file ({BACKFILL_DB_PATH}) instead of warehouse.duckdb, "
             "merging into warehouse.duckdb when this run finishes (resumable across invocations)",
    )
    parser.add_argument(
        "--merge-only", action="store_true",
        help="Merge an existing backfill db into warehouse.duckdb without fetching anything",
    )
    args = parser.parse_args()

    if args.merge_only:
        rows = merge_into_warehouse()
        print(f"nba_box_status merge complete: {rows} rows merged into warehouse.duckdb")
    elif args.backfill:
        result = run_backfill(limit=args.limit)
        print(f"nba_box_status backfill complete: {result.rows_written} rows ({result.notes})")
    else:
        result = run(limit=args.limit)
        print(f"nba_box_status ingest complete: {result.rows_written} rows ({result.notes})")
