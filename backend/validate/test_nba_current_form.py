"""Regression test for a real bug found while building the forward-
prediction pipeline: `build_current_form` computed `rest_days` as
`as_of - last["game_date"]`, but `last["game_date"]` (read via
`fetchdf()`) is a pandas Timestamp, not the plain `datetime.date` that
`as_of` (from `date.today()`) is — subtracting them raised `TypeError`
immediately, on every call. Same bug class as nba_availability.py's
(see that module's own regression test); worth a dedicated one here since
it's a different function."""

from datetime import date, timedelta

import pandas as pd
import pytest

from backend.features.nba_current_form import build_current_form, teammates_out_from_availability
from backend.parlay.leg_pool import _opponent_by_team


def test_build_current_form_runs_without_a_date_type_error(conn):
    df = build_current_form(conn, as_of=date.today())
    assert isinstance(df, type(df))  # just: it returned, instead of raising TypeError
    if not df.empty:
        assert df["rest_days"].dtype.kind in "if"  # numeric, not an unresolved timedelta/object


def test_rest_days_is_plausible_for_a_known_recent_window(conn):
    """A loose sanity check: for an as_of date right after the last
    ingested game, rest_days for players who played recently should be
    small non-negative integers, not negative (which would mean the date
    arithmetic is backwards) or absurdly large for everyone."""
    row = conn.execute("SELECT MAX(game_date) FROM nba_player_games").fetchone()
    if row is None or row[0] is None:
        return
    last_ingested_date = row[0]
    as_of = last_ingested_date + timedelta(days=1)
    df = build_current_form(conn, as_of=as_of)
    if df.empty:
        return
    assert (df["rest_days"] >= 0).all(), "found negative rest_days — date subtraction is backwards"


# Columns that don't depend on `opponent_by_team`/a resolved current team —
# safe to compare for every player, trades included.
_TEAM_INDEPENDENT_COLUMNS = [
    "pts_roll3", "pts_roll5", "pts_roll10", "pts_roll_season", "pts_std10",
    "min_roll3", "min_roll5", "min_roll10", "min_roll_season", "min_std10",
    "reb_roll10", "ast_roll10", "fg3m_roll10", "stl_roll10", "blk_roll10", "tov_roll10",
    "usage_rate_roll10", "games_played_prior_season",
    "prior_season_games", "prior_season_min_avg", "prior_season_pts_avg",
    "prior_season_reb_avg", "prior_season_ast_avg", "prior_season_fg3m_avg",
]
# Depend on knowing the player's CURRENT team, which build_current_form can
# only infer from nba_player_games (see the module docstring's documented
# trade limitation) — only meaningful to compare when that inference landed
# on the same team nba_features' real game row did.
_TEAM_DEPENDENT_COLUMNS = [
    "team_pace_roll10", "opp_pace_roll10", "opp_def_vs_position_pts_roll10",
    "teammates_out_count", "teammates_out_pts_l10", "teammates_out_min_l10",
]


def test_season_opener_features_match_nba_features_point_in_time(conn):
    """Training-serving skew regression test for a real bug: build_current_form
    used to decide a player's "current season" from the season of their
    LAST PLAYED game, not the season of the game actually being predicted.
    That's invisible everywhere except exactly at a season boundary — asked
    to score a season's opening night itself (before any game of that
    season exists in nba_player_games), every player's last played game is
    still last season's finale, so games_played_prior_season came out as
    last season's full game count and pts_roll_season/min_roll_season as
    last season's full averages, instead of the 0/NULL nba_features.py
    computes for that exact row, point-in-time (see nba_current_form.py's
    module docstring for the full backstory).

    This directly recomputes build_current_form "as of" a real season's
    opening day — using only data strictly before it, same as a live
    leg_pool build would weeks before tip-off — and compares every
    feature against nba_features' own stored row for each opener's actual
    first game, which already exists for a past season in this warehouse.
    An exact mismatch here means the serving-time feature builder and the
    training-time feature builder have drifted apart."""
    seasons = conn.execute("SELECT DISTINCT season FROM nba_schedule ORDER BY season").fetchall()
    opener_date = None
    for (season,) in seasons:
        row = conn.execute(
            "SELECT MIN(game_date) FROM nba_features WHERE season = ? AND games_played_prior_season = 0", [season]
        ).fetchone()
        if not row or row[0] is None:
            continue
        candidate = row[0]
        # Needs at least one earlier season's games already ingested, or
        # there's no "current form" to carry over the offseason gap — same
        # reason walk_forward_splits() skips a first-season fold.
        earlier = conn.execute("SELECT COUNT(*) FROM nba_player_games WHERE game_date < ?", [candidate]).fetchone()[0]
        if earlier > 0:
            opener_date = candidate
            break
    if opener_date is None:
        pytest.skip("no season (with an earlier season already ingested) has opener rows in nba_features")

    opener_rows = conn.execute(
        "SELECT * FROM nba_features WHERE game_date = ? AND games_played_prior_season = 0", [opener_date]
    ).fetchdf()
    assert not opener_rows.empty

    opponent_by_team = _opponent_by_team(conn, opener_date)
    current_form = build_current_form(
        conn, as_of=opener_date, opponent_by_team=opponent_by_team,
        out_players_by_team=teammates_out_from_availability(conn, opener_date),
    )
    assert not current_form.empty

    merged = opener_rows.merge(current_form, on="player_id_nba", how="inner", suffixes=("_nf", "_cf"))
    assert len(merged) >= 20, f"only matched {len(merged)} players — sample too small to trust"

    traded = merged["team_nf"] != merged["team_cf"]  # see _TEAM_DEPENDENT_COLUMNS' docstring above

    checked = 0
    for col in _TEAM_INDEPENDENT_COLUMNS + _TEAM_DEPENDENT_COLUMNS:
        nf_col, cf_col = merged[f"{col}_nf"], merged[f"{col}_cf"]
        rows_to_check = merged.index if col in _TEAM_INDEPENDENT_COLUMNS else merged.index[~traded]
        for idx in rows_to_check:
            a, b = nf_col.loc[idx], cf_col.loc[idx]
            a_null, b_null = pd.isna(a), pd.isna(b)
            if a_null or b_null:
                assert a_null == b_null, (
                    f"{col} for player {merged.loc[idx, 'player_id_nba']}: "
                    f"nba_features={a!r} vs. build_current_form={b!r} (one NULL, one not)"
                )
            else:
                assert abs(a - b) < 1e-6, (
                    f"{col} for player {merged.loc[idx, 'player_id_nba']} on the season opener: "
                    f"nba_features={a} vs. build_current_form={b} — serving-time feature builder has "
                    "drifted from the training-time one"
                )
            checked += 1

    assert checked >= 20, f"only checked {checked} (player, column) pairs — sample too small"

    # The specific bug this test was written for: games_played_prior_season
    # and the season-scoped averages must be 0/NULL on an actual opener, in
    # BOTH feature builders — not just equal to each other.
    assert (merged["games_played_prior_season_cf"] == 0).all()
    assert merged["pts_roll_season_cf"].isna().all()
    assert merged["min_roll_season_cf"].isna().all()
