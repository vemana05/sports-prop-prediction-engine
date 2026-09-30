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
