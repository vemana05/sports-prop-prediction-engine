"""Point-in-time correctness for nfl_features (Section 4.4, Phase 6).

Recomputes features independently in Python from the source tables, using
only information dated strictly before each sampled game's kickoff, and
requires a match against what nfl_features stored. A rolling window that
accidentally included the current game, a depth chart published after
kickoff, or an opponent-defense average that counted the game being
predicted would each fail here.
"""

import math
import random

import pytest

SAMPLE_PLAYERS = 25


def _close(a, b):
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


@pytest.fixture(scope="module")
def player_games(nfl_conn):
    players = [r[0] for r in nfl_conn.execute(
        "SELECT player_id FROM nfl_features GROUP BY 1 HAVING COUNT(*) >= 12 ORDER BY hash(player_id) LIMIT ?",
        [SAMPLE_PLAYERS],
    ).fetchall()]
    assert players, "no players with enough games"
    out = {}
    for pid in players:
        out[pid] = nfl_conn.execute(
            """SELECT game_id, season, kickoff_utc, receiving_yards, rushing_yards, passing_yards, receptions,
                      receiving_yards_roll3, rushing_yards_roll5, passing_yards_roll_season, receptions_roll5
               FROM nfl_features WHERE player_id = ? ORDER BY kickoff_utc, game_id""",
            [pid],
        ).fetchall()
    return out


def test_rolling_windows_exclude_the_current_game(player_games):
    random.seed(7)
    checked = 0
    for games in player_games.values():
        for idx in random.sample(range(len(games)), min(5, len(games))):
            g = games[idx]
            prior = games[:idx]
            same_season = [p for p in prior if p[1] == g[1]]

            def avg(rows, col):
                return sum(r[col] for r in rows) / len(rows) if rows else None

            assert _close(avg(prior[-3:], 3), g[7]), f"receiving_yards_roll3 leak/mismatch at {g[0]}"
            assert _close(avg(prior[-5:], 4), g[8]), f"rushing_yards_roll5 leak/mismatch at {g[0]}"
            assert _close(avg(same_season, 5), g[9]), f"passing_yards_roll_season leak/mismatch at {g[0]}"
            assert _close(avg(prior[-5:], 6), g[10]), f"receptions_roll5 leak/mismatch at {g[0]}"
            checked += 1
    assert checked >= 50


def test_rolling_features_are_unchanged_if_the_game_itself_changes(nfl_conn):
    """A direct statement of "no look-ahead": the first game of a career
    has no rolling history at all, whatever happened in it."""
    leaked = nfl_conn.execute(
        """SELECT COUNT(*) FROM nfl_features
           WHERE career_games_prior = 0
             AND (receiving_yards_roll3 IS NOT NULL OR passing_yards_roll5 IS NOT NULL OR rushing_yards_roll_season IS NOT NULL)"""
    ).fetchone()[0]
    assert leaked == 0


def test_depth_chart_snapshots_precede_kickoff(nfl_conn):
    """For dated (2025+) charts: every stored depth_rank must be reproducible
    from a snapshot published strictly before kickoff."""
    rows = nfl_conn.execute(
        """SELECT f.player_id, f.team, f.position, f.kickoff_utc, f.depth_rank_raw
           FROM nfl_features f WHERE f.season >= 2025 AND f.depth_rank_raw IS NOT NULL
           ORDER BY hash(f.player_id || f.game_id) LIMIT 200"""
    ).fetchall()
    assert rows
    for pid, team, pos, kickoff, rank in rows:
        snap = nfl_conn.execute(
            "SELECT MAX(snapshot_at) FROM nfl_depth_charts WHERE team = ? AND snapshot_at < ?", [team, kickoff]
        ).fetchone()[0]
        assert snap is not None and snap < kickoff
        expected = nfl_conn.execute(
            "SELECT MIN(depth_rank) FROM nfl_depth_charts WHERE team = ? AND snapshot_at = ? AND player_id = ? AND position = ?",
            [team, snap, pid, pos],
        ).fetchone()[0]
        assert expected == rank, f"{pid} {kickoff}: stored rank {rank}, pre-kickoff snapshot says {expected}"


def test_depth_rank_is_a_dense_ordinal_per_team_game(nfl_conn):
    bad = nfl_conn.execute(
        """SELECT COUNT(*) FROM (
               SELECT team, game_id, position, COUNT(*) n, MAX(depth_rank) mx, COUNT(DISTINCT depth_rank) d
               FROM nfl_features WHERE depth_rank IS NOT NULL GROUP BY ALL)
           WHERE mx <> n OR d <> n"""
    ).fetchone()[0]
    assert bad == 0


def test_opponent_defense_uses_only_earlier_games(nfl_conn):
    rows = nfl_conn.execute(
        """SELECT opponent, position, kickoff_utc, opp_allowed_rec_yds_roll6
           FROM nfl_features WHERE position = 'WR' AND opp_allowed_rec_yds_roll6 IS NOT NULL
           ORDER BY hash(player_id || game_id) LIMIT 40"""
    ).fetchall()
    for opp, pos, kickoff, stored in rows:
        allowed = nfl_conn.execute(
            """SELECT SUM(receiving_yards) FROM nfl_features
               WHERE opponent = ? AND position = ? AND kickoff_utc < ?
               GROUP BY game_id, kickoff_utc ORDER BY kickoff_utc DESC, game_id DESC LIMIT 6""",
            [opp, pos, kickoff],
        ).fetchall()
        expected = sum(r[0] for r in allowed) / len(allowed)
        assert _close(expected, stored), f"opp {opp} before {kickoff}: expected {expected}, stored {stored}"


def test_prior_season_features_read_only_the_previous_season(nfl_conn):
    bad = nfl_conn.execute(
        """WITH truth AS (
               SELECT player_id, season + 1 AS season, AVG(receiving_yards) AS v
               FROM nfl_features GROUP BY 1, 2)
           SELECT COUNT(*) FROM nfl_features f JOIN truth t USING (player_id, season)
           WHERE abs(f.prior_season_receiving_yards - t.v) > 1e-9"""
    ).fetchone()[0]
    assert bad == 0
