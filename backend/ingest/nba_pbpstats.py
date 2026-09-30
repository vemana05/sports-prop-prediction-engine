"""NBA possessions/lineups via pbpstats.

pbpstats builds possessions with the on-court lineup for every play-by-play
event. This module was written and adjusted after live-testing all three of
pbpstats' data providers against real games — two are dead as of this
writing, which is worth recording so nobody "fixes" this back:

- `data_provider="stats_nba"` (playbyplayv2): the endpoint now returns `{}`
  for every game, old or new — NBA retired it.
- `data_provider="data_nba"` (data.nba.com): 403 Forbidden — that domain has
  been decommissioned.
- `data_provider="live"` (NBA's live-data S3 feed): works, and — despite the
  name — serves full historical games, not just in-progress ones. This is
  what we use. It does have a real bug in the installed pbpstats (1.3.11):
  `Possessions.player_stats` / `.lineup_stats` (the library's own
  aggregation helpers) crash on rebound events, because `LiveFieldGoal`'s
  inherited `shot_type` chain isn't fully implemented for the live provider.
  So we read possessions at a lower level instead (`game.possessions.items`
  and each possession's `.offense_team_id` / `.start_time` / `.end_time` /
  `.start_score_margin`, plus lineup ids straight off the first event) —
  verified safe for ~99% of possessions; the rare (~1%) possession where
  team-id resolution itself fails (`get_team_ids()` hitting a
  `LiveStartOfPeriod` event with no `team_id`) is skipped and logged rather
  than aborting the game.

Deriving on/off splits and possession-based pace from this table is Phase 2
feature-engineering work (aggregate on lineup_offense/lineup_defense), not
ingestion.

Caching: the live web loader writes `<dir>/pbp/live_<game_id>.json` on
success, but only if `<dir>/pbp` already exists — pbpstats doesn't create it
itself — and a "web" source never checks that cache before fetching. So
this module creates the directory up front and, like nba_api's game-log
resume flow, checks the cache file itself to choose "file" vs "web" so a
re-run never re-downloads a game it already has.

Game ids come from `nba_player_games` (already ingested via nba_api) rather
than pbpstats' own season/schedule lookup, which depends on the same two
dead endpoints above.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
from pbpstats.client import Client

from backend.db import get_connection
from backend.ingest.util import IngestResult, logger, retry, utcnow

VALIDATION_SEASON = "2025-26"
DEFAULT_VALIDATION_MAX_GAMES = 15


def _cache_dir() -> Path:
    return Path(os.environ.get("PBPSTATS_CACHE_DIR", "data/raw/pbpstats"))


def _client(source: str) -> Client:
    settings = {
        "dir": str(_cache_dir()),
        "Possessions": {"source": source, "data_provider": "live"},
    }
    return Client(settings)


def _is_cached(game_id: str) -> bool:
    return (_cache_dir() / "pbp" / f"live_{game_id}.json").is_file()


def season_game_ids(conn: duckdb.DuckDBPyConnection, season: str) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT game_id FROM nba_player_games WHERE season = ? ORDER BY game_id", [season]
    ).fetchall()
    return [r[0] for r in rows]


def _load_possessions(game_id: str) -> list:
    source = "file" if _is_cached(game_id) else "web"
    client = _client(source)

    def _call():
        return client.Game(game_id).possessions.items

    return retry(_call, what=f"pbpstats.Game({game_id})")


def _possession_rows(game_id: str, season: str, possessions: list) -> list[tuple]:
    now = utcnow()
    rows = []
    skipped = 0
    for i, p in enumerate(possessions):
        try:
            offense_team_id = p.offense_team_id
            defense_team_id = None
            lineup_offense = None
            lineup_defense = None
            try:
                team_ids = p.get_team_ids()
                others = [t for t in team_ids if t != offense_team_id]
                defense_team_id = others[0] if others else None
                lineup_ids = getattr(p.events[0], "lineup_ids", None)
                if lineup_ids:
                    lineup_offense = lineup_ids.get(offense_team_id)
                    lineup_defense = lineup_ids.get(defense_team_id) if defense_team_id else None
            except Exception:  # noqa: BLE001 - known edge case, see module docstring
                skipped += 1

            rows.append(
                (
                    game_id, season, i, p.period,
                    str(offense_team_id) if offense_team_id else None,
                    str(defense_team_id) if defense_team_id else None,
                    p.start_time, p.end_time, p.start_score_margin,
                    lineup_offense, lineup_defense, now,
                )
            )
        except Exception as exc:  # noqa: BLE001 - one bad possession shouldn't drop the game
            skipped += 1
            logger.warning("pbpstats: skipped possession %d in game %s: %s", i, game_id, exc)
    if skipped:
        logger.info("pbpstats: %d/%d possessions had partial/missing data in game %s", skipped, len(possessions), game_id)
    return rows


def _write(conn: duckdb.DuckDBPyConnection, rows: list[tuple]) -> int:
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT INTO nba_possessions
            (game_id, season, possession_number, period, offense_team_id, defense_team_id,
             start_time, end_time, start_score_margin, lineup_offense, lineup_defense, ingested_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (game_id, possession_number) DO UPDATE SET
            offense_team_id = excluded.offense_team_id, defense_team_id = excluded.defense_team_id,
            lineup_offense = excluded.lineup_offense, lineup_defense = excluded.lineup_defense,
            ingested_at = excluded.ingested_at
        """,
        rows,
    )
    return len(rows)


def ingest_season(conn: duckdb.DuckDBPyConnection, season: str, max_games: int | None = None) -> int:
    (_cache_dir() / "pbp").mkdir(parents=True, exist_ok=True)

    game_ids = season_game_ids(conn, season)
    if not game_ids:
        logger.warning("pbpstats: no game_ids found for %s in nba_player_games — run nba_nbaapi ingestion first", season)
        return 0
    if max_games is not None:
        game_ids = game_ids[:max_games]
    logger.info("pbpstats: processing %d games for %s", len(game_ids), season)

    total = 0
    for i, game_id in enumerate(game_ids):
        possessions = _load_possessions(game_id)
        total += _write(conn, _possession_rows(game_id, season, possessions))
        if (i + 1) % 10 == 0:
            logger.info("pbpstats: %d/%d games processed", i + 1, len(game_ids))
    logger.info("pbpstats: wrote %d possession rows for %s (%d games)", total, season, len(game_ids))
    return total


def run(conn: duckdb.DuckDBPyConnection | None = None) -> IngestResult:
    own_conn = conn is None
    conn = conn or get_connection()
    try:
        rows = ingest_season(conn, VALIDATION_SEASON, max_games=DEFAULT_VALIDATION_MAX_GAMES)
        return IngestResult(rows_written=rows, notes={"season": VALIDATION_SEASON, "max_games": DEFAULT_VALIDATION_MAX_GAMES})
    finally:
        if own_conn:
            conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NBA pbpstats possessions/lineups ingestion")
    parser.add_argument("--season", default=VALIDATION_SEASON)
    parser.add_argument("--max-games", type=int, default=None, help="Omit for the full season")
    args = parser.parse_args()

    conn = get_connection()
    try:
        rows = ingest_season(conn, args.season, max_games=args.max_games)
        print(f"pbpstats ingest complete: {rows} rows for {args.season}")
    finally:
        conn.close()
