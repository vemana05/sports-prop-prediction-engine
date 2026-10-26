"""Early-season window: games-played countdown and the stricter sanity guard."""

from datetime import date

import duckdb

from backend.parlay.search import _sanity_flags, score_parlay
from backend.predict.early_season import EARLY_SEASON_GAMES, games_missing_logs, summarize, team_games_played
from tests.test_search import _leg, _zero_corr_matrices

TIER = {"max_stake_pct_bankroll": 0.05}


def _schedule(games: list[tuple[str, date, int, int]], logged: int | None = None) -> duckdb.DuckDBPyConnection:
    """nba_schedule with `games`, plus nba_player_games logs for the first
    `logged` of them (all by default) — one row per team per game."""
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE nba_schedule (game_id VARCHAR, season VARCHAR, game_date DATE, home_team_id BIGINT, away_team_id BIGINT)")
    conn.execute("CREATE TABLE nba_player_games (player_id_nba VARCHAR, team VARCHAR, game_id VARCHAR, game_date DATE, season VARCHAR)")
    conn.executemany("INSERT INTO nba_schedule VALUES (?, ?, ?, ?, ?)", [(f"g{i}", *g) for i, g in enumerate(games)])
    abbr = {1610612738: "BOS", 1610612752: "NYK", 1610612747: "LAL"}
    for i, (season, day, home, away) in enumerate(games[:logged]):
        for team in (home, away):
            conn.execute("INSERT INTO nba_player_games VALUES (?, ?, ?, ?, ?)", [f"p_{abbr[team]}", abbr[team], f"g{i}", day, season])
    return conn


def test_games_played_counts_this_season_before_today_and_counts_down():
    bos, nyk, lal = 1610612738, 1610612752, 1610612747
    conn = _schedule([
        ("2025-26", date(2026, 4, 12), bos, nyk),  # last season: ignored
        ("2026-27", date(2026, 10, 20), bos, nyk),
        ("2026-27", date(2026, 10, 22), lal, bos),
        ("2026-27", date(2026, 10, 25), nyk, lal),
    ])
    season, played = team_games_played(conn, date(2026, 10, 15))
    assert season == "2026-27" and played == {"BOS": 0, "NYK": 0, "LAL": 0}

    season, played = team_games_played(conn, date(2026, 10, 23))
    assert played == {"BOS": 2, "NYK": 1, "LAL": 1}
    s = summarize(season, played)
    assert s["active"] and s["min_games_played"] == 1 and s["games_remaining"] == EARLY_SEASON_GAMES - 1
    assert s["teams_below"] == ["BOS", "LAL", "NYK"]


def test_games_played_counts_only_games_whose_logs_are_loaded():
    """Same data as the model's season features: a game that tipped but
    whose logs aren't loaded yet doesn't count."""
    bos, nyk, lal = 1610612738, 1610612752, 1610612747
    conn = _schedule([
        ("2026-27", date(2026, 10, 20), bos, nyk),
        ("2026-27", date(2026, 10, 22), lal, bos),  # tipped, logs not loaded
    ], logged=1)
    season, played = team_games_played(conn, date(2026, 10, 23))
    assert played == {"BOS": 1, "NYK": 1, "LAL": 0}
    assert games_missing_logs(conn, season, date(2026, 10, 23)) == 1


def test_window_closes_once_every_team_reaches_the_threshold():
    s = summarize("2026-27", {"BOS": EARLY_SEASON_GAMES, "NYK": EARLY_SEASON_GAMES + 2})
    assert not s["active"] and s["games_remaining"] == 0 and s["teams_below"] == []
    assert summarize("2026-27", {"BOS": EARLY_SEASON_GAMES - 1, "NYK": 20})["active"]


def test_early_season_guard_is_stricter_and_flagged_parlays_stake_zero():
    # Each leg's model probability is ~12 points above the 50% no-vig price:
    # inside the normal 15-point limit, outside the early-season 10-point one.
    legs = [_leg("a", "points", 20.5, 22.3, 6.0), _leg("b", "points", 20.5, 22.3, 6.0, game_id="g2", team="B")]
    for leg in legs:
        assert 0.10 < leg.p_hit() - leg.no_vig_p_hit() < 0.15
    assert _sanity_flags(legs, model_ev=0.0) == []
    assert len(_sanity_flags(legs, model_ev=0.0, early_season=True)) == 2

    # Parlay EV between the early (0.20) and normal (0.30) limits.
    assert _sanity_flags([], model_ev=0.25) == []
    assert _sanity_flags([], model_ev=0.25, early_season=True)

    # One leg keeps parlay EV (~+0.18) under both EV limits, so only the
    # divergence check decides.
    normal = score_parlay(legs[:1], TIER, _zero_corr_matrices(), 1000.0, n_simulations=4000, seed=1)
    early = score_parlay(legs[:1], TIER, _zero_corr_matrices(), 1000.0, n_simulations=4000, seed=1, early_season=True)
    assert not normal.flagged and normal.stake.stake > 0
    assert early.flagged and early.stake.stake == 0.0 and "early-season limit" in early.stake.reason
