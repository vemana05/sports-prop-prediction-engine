"""Phase 1 data summary: row counts per table, xref match rate, review queue
contents, and freshness status per source — the report Section 14 asks for
before Phase 2 starts.
"""

from __future__ import annotations

from backend.db import get_connection
from backend.validate.freshness import compute_freshness

TABLES = [
    "nba_player_games",
    "nba_possessions",
    "nba_injuries",
    "sleeper_players",
    "player_xref",
    "xref_review_queue",
    "ingestion_log",
]


def main() -> None:
    conn = get_connection()

    print("=== Row counts ===")
    for table in TABLES:
        n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {n}")

    print("\n=== nba_player_games coverage ===")
    row = conn.execute(
        "SELECT COUNT(DISTINCT season), COUNT(DISTINCT game_id), COUNT(DISTINCT player_id_nba), MIN(game_date), MAX(game_date) FROM nba_player_games"
    ).fetchone()
    print(f"  seasons: {row[0]}, games: {row[1]}, players: {row[2]}, date range: {row[3]} to {row[4]}")

    print("\n=== nba_injuries game-day coverage ===")
    rows = conn.execute(
        """
        WITH game_days AS (
            SELECT DISTINCT season, game_date FROM nba_player_games
        ), report_days AS (
            SELECT DISTINCT CAST(report_time AS DATE) AS report_date FROM nba_injuries
        )
        SELECT g.season, COUNT(DISTINCT g.game_date) AS game_days,
               COUNT(DISTINCT r.report_date) AS days_with_report
        FROM game_days g
        LEFT JOIN report_days r ON r.report_date = g.game_date
        GROUP BY g.season ORDER BY g.season
        """
    ).fetchall()
    for season, game_days, days_with_report in rows:
        print(f"  {season}: {days_with_report}/{game_days} game days have a pre-game injury snapshot")

    print("\n=== xref match rate ===")
    total_sleeper = conn.execute("SELECT COUNT(*) FROM sleeper_players").fetchone()[0]
    matched = conn.execute("SELECT COUNT(*) FROM player_xref WHERE sleeper_id IS NOT NULL").fetchone()[0]
    unmatched = conn.execute("SELECT COUNT(*) FROM xref_review_queue").fetchone()[0]
    rate = matched / total_sleeper * 100 if total_sleeper else 0
    print(f"  sleeper players: {total_sleeper}, matched to nba.com id: {matched} ({rate:.1f}%), in review queue: {unmatched}")

    print("\n=== xref review queue (sample, first 20) ===")
    rows = conn.execute(
        "SELECT source, raw_name, reason, best_candidate, best_score FROM xref_review_queue ORDER BY raw_name LIMIT 20"
    ).fetchall()
    for r in rows:
        print(f"  [{r[0]}] {r[1]!r} — {r[2]} (best guess: {r[3]!r} @ {r[4]})")

    print("\n=== Source freshness ===")
    freshness = compute_freshness(conn)
    for name, info in sorted(freshness.items()):
        print(f"  {name}: {info['status']} (last success: {info['last_success']}, expected within {info['expected_within_hours']}h)")

    print("\n=== Last ingestion_log entries ===")
    rows = conn.execute(
        "SELECT source, status, rows_written, started_at, error, fallback_from FROM ingestion_log ORDER BY id DESC LIMIT 10"
    ).fetchall()
    for r in rows:
        print(f"  {r}")

    conn.close()


if __name__ == "__main__":
    main()
