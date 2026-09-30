"""Reconciliation checks (Section 4.3): structural sanity on ingested data.
No duplicate player-games, stat ranges within physically plausible bounds.
"""


def test_no_duplicate_player_games(conn):
    dupes = conn.execute(
        """
        SELECT player_id_nba, game_id, COUNT(*) AS n
        FROM nba_player_games
        GROUP BY player_id_nba, game_id
        HAVING COUNT(*) > 1
        """
    ).fetchall()
    assert dupes == [], f"duplicate player-games found: {dupes[:10]}"


def test_minutes_within_plausible_range(conn):
    bad = conn.execute(
        "SELECT player_id_nba, game_id, minutes FROM nba_player_games "
        "WHERE minutes IS NOT NULL AND (minutes < 0 OR minutes > 60)"
    ).fetchall()
    assert bad == [], f"minutes out of [0, 60] range: {bad[:10]}"


def test_counting_stats_non_negative(conn):
    bad = conn.execute(
        "SELECT player_id_nba, game_id FROM nba_player_games "
        "WHERE pts < 0 OR reb < 0 OR ast < 0 OR fg3m < 0 OR stl < 0 OR blk < 0 OR tov < 0"
    ).fetchall()
    assert bad == [], f"negative counting stats found: {bad[:10]}"


def test_no_duplicate_injury_snapshots(conn):
    dupes = conn.execute(
        """
        SELECT player_name_raw, team, report_time, COUNT(*) AS n
        FROM nba_injuries
        GROUP BY player_name_raw, team, report_time
        HAVING COUNT(*) > 1
        """
    ).fetchall()
    assert dupes == [], f"duplicate injury snapshot rows found: {dupes[:10]}"


def test_xref_review_queue_entries_have_a_reason(conn):
    bad = conn.execute(
        "SELECT source, raw_name FROM xref_review_queue WHERE reason IS NULL OR reason = ''"
    ).fetchall()
    assert bad == [], f"review queue rows missing a reason (never silently drop unmatched names): {bad[:10]}"


def test_point_in_time_columns_have_no_nulls(conn):
    """Section 4.4: every fact row needs source_published_at + ingested_at
    so feature builders can query 'as of' a game's start time."""
    checks = [
        ("nba_player_games", "source_published_at"),
        ("nba_player_games", "ingested_at"),
        ("nba_possessions", "source_published_at"),
        ("nba_possessions", "ingested_at"),
        ("nba_injuries", "source_published_at"),
        ("nba_injuries", "ingested_at"),
        ("sleeper_players", "ingested_at"),
        ("player_xref", "created_at"),
        ("player_xref", "updated_at"),
    ]
    failures = []
    for table, col in checks:
        n = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} IS NULL").fetchone()[0]
        if n:
            failures.append((table, col, n))
    assert failures == [], f"NULL point-in-time timestamps found: {failures}"


def test_injury_reports_cover_almost_every_game_day(conn):
    """Section 3.3: the backtest needs the injury report as it looked
    before tip-off for essentially every game day, not just a sample."""
    row = conn.execute(
        """
        WITH game_days AS (SELECT DISTINCT game_date FROM nba_player_games),
             report_days AS (SELECT DISTINCT CAST(report_time AS DATE) AS d FROM nba_injuries)
        SELECT COUNT(*) AS total,
               COUNT(*) FILTER (WHERE d IS NOT NULL) AS covered
        FROM game_days g LEFT JOIN report_days r ON r.d = g.game_date
        """
    ).fetchone()
    total, covered = row
    rate = covered / total if total else 0
    assert rate >= 0.95, f"only {covered}/{total} game days ({rate:.1%}) have an injury snapshot"
