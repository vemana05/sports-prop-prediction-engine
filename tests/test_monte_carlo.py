"""Unit tests for the Gaussian-copula Monte Carlo parlay simulator — pure
math against synthetic legs, no DB needed."""

import numpy as np
import pytest

from backend.parlay.monte_carlo import Leg, nearest_psd, simulate_parlay_hit_probability


def _normal_leg(name, line, mean, sigma, side="over", game_id="g1", team="A", player_id=None):
    return Leg(
        player_id_nba=player_id or name, player_name=name, game_id=game_id, team=team,
        market="points", side=side, line=line, predicted_mean=mean, dispersion=sigma,
        decimal_odds=1.91, over_price=1.91, under_price=1.91,
    )


_NO_CORRELATION = {"same_player": {}, "teammate": {}, "opponent": {}}


def _const_corr_matrices(value: float) -> dict:
    """A correlation dict that returns `value` for every market pair, every
    relationship — EXCEPT same_player's own diagonal, which is always 1.0
    (the same stat for the same player is definitionally itself, same as
    the real correlations.py). For teammate/opponent, same-market-
    different-player is a real correlation, not a trivial 1.0, so it gets
    `value` like every other cell."""
    markets = ["points", "rebounds", "assists", "fg3m"]
    same_player_table = {(a, b): (1.0 if a == b else value, 10_000) for a in markets for b in markets}
    other_table = {(a, b): (value, 10_000) for a in markets for b in markets}
    return {"same_player": same_player_table, "teammate": other_table, "opponent": other_table}


def test_single_leg_parlay_returns_the_legs_own_probability_exactly():
    leg = _normal_leg("A", line=20.5, mean=22, sigma=5)
    result = simulate_parlay_hit_probability([leg], _const_corr_matrices(0.0), seed=1)
    assert result["n_simulations"] == 0
    assert result["p_hit"] == pytest.approx(leg.p_hit())


def test_independent_legs_converge_to_the_product_of_individual_probabilities():
    legs = [
        _normal_leg("A", line=20.5, mean=22, sigma=5, game_id="g1", player_id="pA"),
        _normal_leg("B", line=15.5, mean=14, sigma=4, side="under", game_id="g2", player_id="pB"),
    ]
    result = simulate_parlay_hit_probability(legs, _const_corr_matrices(0.0), n_simulations=50_000, seed=42)
    expected = legs[0].p_hit() * legs[1].p_hit()
    assert result["independent_p_hit"] == pytest.approx(expected, abs=1e-9)
    # Monte Carlo estimate should be close to the analytic independent value
    # when correlation is genuinely 0 (cross-game legs are always independent
    # regardless of the matrix passed in, since _relationship checks game_id first)
    assert result["p_hit"] == pytest.approx(expected, abs=0.01)


def test_cross_game_legs_are_independent_even_if_a_correlation_matrix_says_otherwise():
    # same player id, but DIFFERENT games -> must still be treated as independent
    legs = [
        _normal_leg("A", line=20.5, mean=22, sigma=5, game_id="g1", player_id="pA"),
        _normal_leg("A", line=20.5, mean=22, sigma=5, game_id="g2", player_id="pA"),
    ]
    result = simulate_parlay_hit_probability(legs, _const_corr_matrices(0.9), n_simulations=50_000, seed=1)
    expected = legs[0].p_hit() * legs[1].p_hit()
    assert result["p_hit"] == pytest.approx(expected, abs=0.015)


def test_positive_correlation_increases_joint_hit_probability_above_independence():
    legs = [
        _normal_leg("A", line=22, mean=22, sigma=5, game_id="g1", team="X", player_id="pA"),
        _normal_leg("B", line=22, mean=22, sigma=5, game_id="g1", team="X", player_id="pB"),
    ]
    result = simulate_parlay_hit_probability(legs, _const_corr_matrices(0.8), n_simulations=50_000, seed=7)
    assert result["p_hit"] > result["independent_p_hit"] + 0.05


def test_negative_correlation_decreases_joint_hit_probability_below_independence():
    legs = [
        _normal_leg("A", line=22, mean=22, sigma=5, game_id="g1", team="X", player_id="pA"),
        _normal_leg("B", line=22, mean=22, sigma=5, game_id="g1", team="X", player_id="pB"),
    ]
    result = simulate_parlay_hit_probability(legs, _const_corr_matrices(-0.8), n_simulations=50_000, seed=7)
    assert result["p_hit"] < result["independent_p_hit"] - 0.05


def test_nearest_psd_fixes_an_invalid_correlation_matrix():
    # a matrix that cannot arise from real correlations (triangle inequality violated)
    invalid = np.array([
        [1.0, 0.9, -0.9],
        [0.9, 1.0, 0.9],
        [-0.9, 0.9, 1.0],
    ])
    eigvals = np.linalg.eigvalsh(invalid)
    assert eigvals.min() < 0, "test setup should start from a genuinely invalid (non-PSD) matrix"

    fixed = nearest_psd(invalid)
    fixed_eigvals = np.linalg.eigvalsh(fixed)
    assert fixed_eigvals.min() >= -1e-6
    assert np.allclose(np.diag(fixed), 1.0)


def test_nearest_psd_leaves_an_already_valid_matrix_close_to_unchanged():
    valid = np.array([[1.0, 0.3], [0.3, 1.0]])
    fixed = nearest_psd(valid)
    assert np.allclose(fixed, valid, atol=1e-6)


def test_negbinom_leg_ppf_and_hits_are_consistent():
    leg = Leg(
        player_id_nba="p1", player_name="P", game_id="g1", team="A", market="rebounds",
        side="over", line=8.5, predicted_mean=9.0, dispersion=15.0, decimal_odds=1.9,
    )
    u = np.array([0.1, 0.5, 0.9])
    values = leg.ppf(u)
    assert (values >= 0).all(), "rebound counts can't be negative"
    assert np.all(np.diff(values) >= 0), "ppf should be non-decreasing in u"


class _ConstantCalibrator:
    """Stand-in for a fitted IsotonicRegression: maps every input to the
    same fixed output, so a test can pin exactly what calibration should
    do to p_hit()/p_over() without depending on sklearn or real data."""

    def __init__(self, output: float):
        self.output = output

    def predict(self, raw):
        return np.full(len(raw), self.output)


def test_calibrator_overrides_the_raw_model_probability():
    leg = _normal_leg("A", line=20.5, mean=22, sigma=5)  # raw p_over() would be well above 0.5
    leg.calibrator = _ConstantCalibrator(0.3)
    assert leg.raw_p_over() != pytest.approx(0.3)
    assert leg.p_over() == pytest.approx(0.3)
    assert leg.p_hit() == pytest.approx(0.3)  # side="over" by default


def test_calibrator_also_shifts_the_monte_carlo_hit_test_not_just_p_hit():
    # Calibrating a leg down to a near-certain miss (p_over=0.01) should
    # make it (almost) never hit in the simulator too — not just in the
    # scalar p_hit() figure — since spec Section 2 calibrates BEFORE
    # simulation, not just before shrinkage. Checked via a >1-leg run (the
    # single-leg path already returns leg.p_hit() exactly by construction —
    # see test_single_leg_parlay_returns_the_legs_own_probability_exactly).
    leg = _normal_leg("A", line=20.5, mean=22, sigma=5)
    leg.calibrator = _ConstantCalibrator(0.01)
    assert leg.p_hit() == pytest.approx(0.01)

    sim = simulate_parlay_hit_probability(
        [leg, _normal_leg("B", line=5, mean=14, sigma=4, side="under", game_id="g2", player_id="pB")],
        _const_corr_matrices(0.0), n_simulations=20_000, seed=3,
    )
    assert sim["leg_p_hit"][0] == pytest.approx(0.01, abs=0.01)
