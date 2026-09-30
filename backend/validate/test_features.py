"""Point-in-time correctness for nba_features (Section 4.4 / 5.2).

nba_features.py's rolling windows use `ROWS BETWEEN N PRECEDING AND 1
PRECEDING`, which structurally cannot include the current row — but this
test proves it empirically too: independently recompute each rolling
average in Python, using only games strictly before the target game's
(date, game_id), and require an exact match against what's stored.
"""

import random


def _recompute_roll(games: list[tuple], idx: int, n: int, stat_idx: int) -> float | None:
    """games: sorted list of (game_date, game_id, minutes, pts, ...) for one player.
    Average of the `n` games strictly before index `idx` (None if fewer than 1 exist)."""
    window = games[max(0, idx - n):idx]
    if not window:
        return None
    vals = [g[stat_idx] for g in window]
    return sum(vals) / len(vals)


def test_rolling_features_never_leak_the_current_game(conn):
    player_ids = [
        r[0]
        for r in conn.execute(
            "SELECT player_id_nba FROM nba_player_games GROUP BY player_id_nba HAVING COUNT(*) >= 15 ORDER BY random() LIMIT 20"
        ).fetchall()
    ]
    assert player_ids, "no players with enough games to test rolling windows"

    checked = 0
    for player_id in player_ids:
        games = conn.execute(
            """
            SELECT game_date, game_id, minutes, pts
            FROM nba_player_games WHERE player_id_nba = ?
            ORDER BY game_date, game_id
            """,
            [player_id],
        ).fetchall()
        features = {
            (r[0], r[1]): r
            for r in conn.execute(
                """
                SELECT game_date, game_id, min_roll3, min_roll5, pts_roll3
                FROM nba_features WHERE player_id_nba = ?
                """,
                [player_id],
            ).fetchall()
        }

        sample_idxs = random.sample(range(len(games)), min(5, len(games)))
        for idx in sample_idxs:
            game_date, game_id, minutes, pts = games[idx]
            expected_min3 = _recompute_roll(games, idx, 3, 2)
            expected_min5 = _recompute_roll(games, idx, 5, 2)
            expected_pts3 = _recompute_roll(games, idx, 3, 3)

            stored = features.get((game_date, game_id))
            assert stored is not None, f"missing nba_features row for {player_id} {game_id}"
            _, _, stored_min3, stored_min5, stored_pts3 = stored

            for expected, stored_val, label in [
                (expected_min3, stored_min3, "min_roll3"),
                (expected_min5, stored_min5, "min_roll5"),
                (expected_pts3, stored_pts3, "pts_roll3"),
            ]:
                if expected is None:
                    assert stored_val is None, f"{label} for {player_id} {game_id}: expected NULL, got {stored_val}"
                else:
                    assert stored_val is not None, f"{label} for {player_id} {game_id}: expected {expected}, got NULL"
                    assert abs(expected - stored_val) < 1e-6, (
                        f"{label} for {player_id} {game_id}: recomputed {expected} != stored {stored_val} "
                        "(rolling feature may be leaking the current game or using the wrong window)"
                    )
            checked += 1

    assert checked >= 20, f"only checked {checked} (player, game) pairs — sample too small"
