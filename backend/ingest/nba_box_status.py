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
"""

from __future__ import annotations

import duckdb
import pandas as pd
from nba_api.stats.endpoints import boxscoretraditionalv3

from backend.db import get_connection
from backend.ingest.util import IngestResult, logger, retry, utcnow

_COLUMNS = {
    "gameId": "game_id",
    "teamId": "team_id",
    "personId": "player_id_nba",
    "minutes": "minutes_raw",
    "comment": "comment",
}


def fetch_game_roster(game_id: str) -> pd.DataFrame:
    def _call():
        bs = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id, timeout=30)
        return bs.get_data_frames()[0]

    df = retry(_call, what=f"BoxScoreTraditionalV3({game_id})")
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


def run(conn: duckdb.DuckDBPyConnection | None = None, limit: int | None = None) -> IngestResult:
    own_conn = conn is None
    conn = conn or get_connection()
    try:
        game_ids = pending_game_ids(conn)
        total_pending = len(game_ids)
        if limit is not None:
            game_ids = game_ids[:limit]
        logger.info("nba_box_status: %d games pending (%d requested this run)", total_pending, len(game_ids))

        total_rows = 0
        failed: list[str] = []
        for i, game_id in enumerate(game_ids):
            try:
                df = fetch_game_roster(game_id)
                total_rows += _write(conn, df)
            except Exception as exc:  # noqa: BLE001 - stats.nba.com is flaky under sustained load;
                # one persistently-failing game must not kill a 6000+ game run. It stays "pending"
                # (not written) so a later re-run of run() retries it automatically.
                failed.append(game_id)
                logger.warning("nba_box_status: giving up on %s after retries: %s", game_id, exc)
            if (i + 1) % 100 == 0:
                logger.info(
                    "nba_box_status: %d/%d games processed this run (%d failed so far)",
                    i + 1, len(game_ids), len(failed),
                )

        logger.info(
            "nba_box_status: wrote %d roster rows across %d/%d games (%d failed, still pending for next run)",
            total_rows, len(game_ids) - len(failed), len(game_ids), len(failed),
        )
        return IngestResult(
            rows_written=total_rows,
            notes={"games_processed": len(game_ids) - len(failed), "games_failed": len(failed), "games_pending_before": total_pending},
        )
    finally:
        if own_conn:
            conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NBA per-game roster (played + DNP) ingestion")
    parser.add_argument("--limit", type=int, default=None, help="Only process this many pending games (omit for all)")
    args = parser.parse_args()

    result = run(limit=args.limit)
    print(f"nba_box_status ingest complete: {result.rows_written} rows ({result.notes})")
