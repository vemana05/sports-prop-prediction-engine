"""Correlation-aware parlay probability via a Gaussian copula (spec
Section 6): draw correlated standard-normal variables (Cholesky of the
legs' correlation matrix), map each through the standard normal CDF to a
uniform, then through that leg's own marginal distribution's inverse CDF
(Normal for points, Negative Binomial for the count markets) — giving
correlated draws from each leg's own real distribution, not just
correlated normals. P(parlay hits) = fraction of simulated replications
where every leg clears its line.

Cross-game legs are independent (correlation 0) — the leg-to-leg
correlation matrix is built from `correlations.py`'s three relationship
matrices, with 0 for any pair whose legs are in different games.

Isotonic calibration (spec Section 2's pipeline — "P(over line) ->
Calibration (isotonic) -> correlation-aware Monte Carlo") is applied to
each leg's raw model probability via its optional `calibrator`, BEFORE
shrinkage (`shrunk_p_hit`) and BEFORE simulation: `p_hit`/`p_over` return
the calibrated probability, and the simulation's own hit test is shifted
to the calibrated threshold (`_calibrated_line`) rather than the raw line,
so a leg's simulated marginal hit rate matches its calibrated probability
too — not just the independent-assumption numbers shown alongside it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from backend.bankroll.kelly import shrink_to_market
from backend.models import nba_market_model as mm
from backend.models.nba_points_model import p_over as normal_p_over
from backend.parlay.correlations import correlation_for

DEFAULT_N_SIMULATIONS = 20_000


@dataclass
class Leg:
    player_id_nba: str
    player_name: str
    game_id: str
    team: str
    market: str  # "points" | "rebounds" | "assists" | "fg3m"
    side: str  # "over" | "under"
    line: float
    predicted_mean: float
    dispersion: float  # sigma (Normal, points) or r (NegBinom, other markets)
    decimal_odds: float  # this leg's own side's price (over_price if side="over", else under_price)
    over_price: float | None = None
    under_price: float | None = None
    calibrator: Any | None = None  # fitted IsotonicRegression (backend.models.calibration), or None to skip calibration
    opponent: str | None = None  # the other team in this leg's odds event (backend.parlay.leg_pool)
    is_home: bool | None = None

    @property
    def distribution(self) -> str:
        return "normal" if self.market == "points" else "negbinom"

    def raw_p_over(self) -> float:
        """Uncalibrated model P(over line) — straight out of the
        probability distribution, before isotonic calibration."""
        if self.distribution == "normal":
            return float(normal_p_over(np.array([self.predicted_mean]), np.array([self.dispersion]), np.array([self.line]))[0])
        return float(mm.p_over_nb(np.array([self.predicted_mean]), np.array([self.dispersion]), np.array([self.line]))[0])

    def p_over(self) -> float:
        """Calibrated P(over line) — the raw model probability passed
        through this leg's isotonic calibrator, if one was supplied.
        Falls back to the raw probability otherwise (e.g. in tests that
        build a Leg directly without a calibrator)."""
        raw = self.raw_p_over()
        if self.calibrator is None:
            return raw
        return float(self.calibrator.predict([raw])[0])

    def p_hit(self) -> float:
        """Calibrated model probability — NOT shrunk toward the market.
        Used for the correlation-aware Monte Carlo draw, where the model's
        own (calibrated) probability is what matters."""
        p_over = self.p_over()
        return p_over if self.side == "over" else 1 - p_over

    def no_vig_p_hit(self) -> float | None:
        """The book's own no-vig probability for this leg's side, from its
        quoted over/under prices — None if we don't have both sides."""
        if not self.over_price or not self.under_price:
            return None
        implied_over, implied_under = 1 / self.over_price, 1 / self.under_price
        total = implied_over + implied_under
        if total <= 0:
            return None
        no_vig_over = implied_over / total
        return no_vig_over if self.side == "over" else 1 - no_vig_over

    def shrunk_p_hit(self) -> float:
        """Section 8: shrink the model's probability toward the book's
        no-vig probability by MODEL_SHRINK_TO_MARKET before sizing. Falls
        back to the raw model probability if there's no market price to
        shrink toward."""
        return shrink_to_market(self.p_hit(), self.no_vig_p_hit())

    def ppf(self, u: np.ndarray) -> np.ndarray:
        """Inverse CDF of this leg's marginal distribution, for copula sampling."""
        if self.distribution == "normal":
            return stats.norm.ppf(u, loc=self.predicted_mean, scale=self.dispersion)
        mu = max(self.predicted_mean, 1e-6)
        p = self.dispersion / (self.dispersion + mu)
        return stats.nbinom.ppf(u, self.dispersion, p)

    def _calibrated_line(self) -> float:
        """The line this leg's own (uncalibrated) distribution would need
        to produce the CALIBRATED P(over) as its raw probability — lets
        the simulation keep drawing real simulated values from the model's
        own distribution shape (`ppf`), while the hit test reflects the
        calibrated probability instead of the raw one. A no-op (returns
        `line` unchanged) when there's no calibrator."""
        if self.calibrator is None:
            return self.line
        threshold = float(self.ppf(np.array([1 - self.p_over()]))[0])
        if self.distribution == "negbinom":
            threshold += 0.5  # keep the "line is always .5, never a push" convention (see p_over_nb)
        return threshold

    def hits(self, simulated_value: np.ndarray) -> np.ndarray:
        line = self._calibrated_line()
        return simulated_value > line if self.side == "over" else simulated_value < line


def _relationship(leg_a: Leg, leg_b: Leg) -> str:
    if leg_a.game_id != leg_b.game_id:
        return "independent"
    if leg_a.player_id_nba == leg_b.player_id_nba:
        return "same_player"
    if leg_a.team == leg_b.team:
        return "teammate"
    return "opponent"


def build_correlation_matrix(legs: list[Leg], corr_matrices: dict[str, dict]) -> np.ndarray:
    n = len(legs)
    R = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            rel = _relationship(legs[i], legs[j])
            corr = correlation_for(corr_matrices, rel, legs[i].market, legs[j].market)
            R[i, j] = R[j, i] = corr
    return R


def nearest_psd(matrix: np.ndarray) -> np.ndarray:
    """Clips negative eigenvalues to a small positive floor and
    renormalizes to unit diagonal. Pooled correlations estimated per-pair
    (not jointly) aren't guaranteed to form a valid (positive
    semi-definite) correlation matrix for an arbitrary N-leg combination —
    this is the standard, cheap fix so Cholesky never fails."""
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals_clipped = np.clip(eigvals, 1e-8, None)
    fixed = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    d = np.sqrt(np.diag(fixed))
    fixed = fixed / np.outer(d, d)
    np.fill_diagonal(fixed, 1.0)
    return fixed


def simulate_parlay_hit_probability(
    legs: list[Leg], corr_matrices: dict[str, dict], n_simulations: int = DEFAULT_N_SIMULATIONS, seed: int | None = None
) -> dict:
    """Returns {"p_hit": ..., "leg_p_hit": [...], "independent_p_hit": ...,
    "correlation_matrix": ..., "n_simulations": ...}."""
    if not legs:
        raise ValueError("need at least one leg")

    rng = np.random.default_rng(seed)
    n = len(legs)

    if n == 1:
        p = legs[0].p_hit()
        return {
            "p_hit": p, "leg_p_hit": [p], "independent_p_hit": p,
            "correlation_matrix": np.array([[1.0]]), "n_simulations": 0,
        }

    R = build_correlation_matrix(legs, corr_matrices)
    R = nearest_psd(R)
    L = np.linalg.cholesky(R)

    z = rng.standard_normal((n_simulations, n)) @ L.T
    u = stats.norm.cdf(z)

    all_hit = np.ones(n_simulations, dtype=bool)
    leg_p_hit = []
    for i, leg in enumerate(legs):
        simulated_value = leg.ppf(u[:, i])
        hits = leg.hits(simulated_value)
        all_hit &= hits
        leg_p_hit.append(float(hits.mean()))

    independent_p_hit = float(np.prod([leg.p_hit() for leg in legs]))

    return {
        "p_hit": float(all_hit.mean()),
        "leg_p_hit": leg_p_hit,
        "independent_p_hit": independent_p_hit,
        "correlation_matrix": R,
        "n_simulations": n_simulations,
    }
