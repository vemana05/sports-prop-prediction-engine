"""Unit tests for pairwise correlation estimation — synthetic residuals,
no DB needed."""

import numpy as np
import pandas as pd

from backend.parlay.correlations import MARKETS, MIN_PAIRS_FOR_ESTIMATE, _pairwise_corr


def _make_residuals(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_same_player_diagonal_is_always_exactly_one():
    df = _make_residuals(
        [{"player_id_nba": "p1", "game_id": f"g{i}", "team": "A", "market": "points", "residual": float(i)} for i in range(5)]
    )
    corr = _pairwise_corr(df, "same_player")
    assert corr[("points", "points")][0] == 1.0


def test_perfectly_correlated_residuals_recover_correlation_near_one():
    rng = np.random.default_rng(0)
    n = MIN_PAIRS_FOR_ESTIMATE + 50
    base = rng.normal(size=n)
    rows = []
    for i in range(n):
        rows.append({"player_id_nba": "p1", "game_id": f"g{i}", "team": "A", "market": "points", "residual": base[i]})
        rows.append({"player_id_nba": "p1", "game_id": f"g{i}", "team": "A", "market": "rebounds", "residual": base[i] * 2})
    df = _make_residuals(rows)
    corr = _pairwise_corr(df, "same_player")
    value, n_pairs = corr[("points", "rebounds")]
    assert value > 0.99
    assert n_pairs == n


def test_uncorrelated_residuals_recover_correlation_near_zero():
    rng = np.random.default_rng(1)
    n = MIN_PAIRS_FOR_ESTIMATE + 500
    rows = []
    for i in range(n):
        rows.append({"player_id_nba": "p1", "game_id": f"g{i}", "team": "A", "market": "points", "residual": float(rng.normal())})
        rows.append({"player_id_nba": "p1", "game_id": f"g{i}", "team": "A", "market": "assists", "residual": float(rng.normal())})
    df = _make_residuals(rows)
    corr = _pairwise_corr(df, "same_player")
    value, _ = corr[("points", "assists")]
    assert abs(value) < 0.1


def test_teammate_relationship_only_pairs_different_players_on_the_same_team():
    rows = [
        {"player_id_nba": "p1", "game_id": "g1", "team": "A", "market": "points", "residual": 1.0},
        {"player_id_nba": "p2", "game_id": "g1", "team": "A", "market": "points", "residual": 1.0},
        {"player_id_nba": "p3", "game_id": "g1", "team": "B", "market": "points", "residual": -1.0},
    ]
    df = _make_residuals(rows)
    corr = _pairwise_corr(df, "teammate")
    # only p1/p2 qualify (same team, different players); p3 is a different team
    _, n_pairs = corr[("points", "points")]
    assert n_pairs < MIN_PAIRS_FOR_ESTIMATE  # too few real pairs -> falls back to default, but shouldn't crash
    assert ("points", "points") in corr


def test_opponent_relationship_only_pairs_players_on_different_teams():
    rows = [
        {"player_id_nba": "p1", "game_id": "g1", "team": "A", "market": "points", "residual": 1.0},
        {"player_id_nba": "p2", "game_id": "g1", "team": "A", "market": "points", "residual": 1.0},
        {"player_id_nba": "p3", "game_id": "g1", "team": "B", "market": "points", "residual": -1.0},
    ]
    df = _make_residuals(rows)
    corr = _pairwise_corr(df, "opponent")
    assert ("points", "points") in corr


def test_insufficient_pairs_falls_back_to_a_default_not_a_crash():
    df = _make_residuals(
        [{"player_id_nba": "p1", "game_id": "g1", "team": "A", "market": "points", "residual": 1.0}]
    )
    corr = _pairwise_corr(df, "teammate")
    value, n_pairs = corr[("points", "points")]
    assert n_pairs == 0
    assert isinstance(value, float)


def test_every_market_pair_is_covered_both_directions():
    df = _make_residuals(
        [{"player_id_nba": "p1", "game_id": f"g{i}", "team": "A", "market": m, "residual": float(i)} for i in range(3) for m in MARKETS]
    )
    corr = _pairwise_corr(df, "same_player")
    for a in MARKETS:
        for b in MARKETS:
            assert (a, b) in corr
