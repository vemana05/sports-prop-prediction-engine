"""Pairwise leg correlations from historical model residuals — spec
Section 6: "Estimate pairwise correlations from historical model residuals
(same player, teammates, opponents)."

Three pooled correlation matrices (market x market, symmetric), not a
separate estimate per specific player-pair — there isn't enough shared
history for any one pair of players to estimate their own correlation
reliably, so this pools across every instance of each *relationship type*:

- `same_player`: one player, two different markets, same game (e.g. his
  points and his rebounds in one night) — diagonal is 1.0 by definition
  (a market always correlates perfectly with itself for the same player).
- `teammate`: two different players on the same team, same game.
- `opponent`: two different players on opposing teams, same game.

Cross-game legs are independent (correlation 0) — handled by the caller
(`monte_carlo.py`), not here, since this module only ever sees same-game
pairs by construction (see the self-join below).
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from backend.db import get_connection
from backend.ingest.util import logger
from backend.parlay import residuals as residuals_mod

MARKETS = ["points", "rebounds", "assists", "fg3m"]
MIN_PAIRS_FOR_ESTIMATE = 200  # below this, fall back to a conservative default rather than a noisy estimate

# Conservative defaults when there isn't enough pooled data for a market
# pair: same-player legs on different stats are usage-linked (more usage
# tends to lift everything a bit) -> mild positive; teammate legs in the
# same stat are usage-rival (more shots for one is fewer for the other)
# -> mild negative; opponent legs move with game pace/script -> mild
# positive. These are priors, not findings — real pooled estimates below
# override them wherever there's enough data.
_DEFAULT_CORR = {"same_player": 0.15, "teammate": -0.05, "opponent": 0.10}


def _load_residuals(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    residuals_mod.run(conn)  # ensure the cache exists; no-op if already built
    return conn.execute(f"SELECT * FROM {residuals_mod.RESIDUALS_TABLE}").fetchdf()


def _pairwise_corr(df: pd.DataFrame, relationship: str, markets: list[str] = MARKETS) -> dict[tuple[str, str], tuple[float, int]]:
    """Returns {(market_a, market_b): (correlation, n_pairs)} for every
    market pair under the given relationship, built by self-joining the
    residuals table on game_id (and team, for teammate/opponent)."""
    out: dict[tuple[str, str], tuple[float, int]] = {}

    if relationship == "same_player":
        key_cols = ["player_id_nba", "game_id"]
        extra_filter = None
    elif relationship == "teammate":
        key_cols = ["game_id", "team"]
        extra_filter = "player_id_nba"  # must differ
    elif relationship == "opponent":
        key_cols = ["game_id"]
        extra_filter = "team"  # must differ (and player must differ, implied by different team)
    else:
        raise ValueError(relationship)

    for i, market_a in enumerate(markets):
        for market_b in markets[i:]:
            if market_a == market_b and relationship == "same_player":
                out[(market_a, market_b)] = (1.0, len(df))
                continue

            a = df[df["market"] == market_a]
            b = df[df["market"] == market_b]
            merged = a.merge(b, on=key_cols, suffixes=("_a", "_b"))

            if relationship == "same_player":
                pass  # key_cols already pins the same player
            elif relationship == "teammate":
                merged = merged[merged["player_id_nba_a"] != merged["player_id_nba_b"]]
            elif relationship == "opponent":
                merged = merged[merged["team_a"] != merged["team_b"]]

            if market_a == market_b and relationship != "same_player":
                # undirected pair double-counted (a,b) and (b,a) from the self-join; keep one
                merged = merged[merged["player_id_nba_a"] < merged["player_id_nba_b"]]

            n = len(merged)
            if n < MIN_PAIRS_FOR_ESTIMATE:
                corr = _DEFAULT_CORR[relationship]
                logger.info(
                    "correlations: %s %s/%s has only %d pairs (<%d) -> using default %.2f",
                    relationship, market_a, market_b, n, MIN_PAIRS_FOR_ESTIMATE, corr,
                )
            else:
                corr = float(np.corrcoef(merged["residual_a"], merged["residual_b"])[0, 1])
                if np.isnan(corr):
                    corr = _DEFAULT_CORR[relationship]
            out[(market_a, market_b)] = (corr, n)
            out[(market_b, market_a)] = (corr, n)

    return out


def build_correlation_matrices(conn: duckdb.DuckDBPyConnection | None = None) -> dict[str, dict]:
    """Returns {"same_player": {...}, "teammate": {...}, "opponent": {...}},
    each a {(market_a, market_b): (correlation, n_pairs)} dict covering
    every market pair both directions."""
    own_conn = conn is None
    conn = conn or get_connection()
    try:
        df = _load_residuals(conn)
        return {rel: _pairwise_corr(df, rel) for rel in ("same_player", "teammate", "opponent")}
    finally:
        if own_conn:
            conn.close()


def correlation_for(matrices: dict[str, dict], relationship: str, market_a: str, market_b: str) -> float:
    if relationship == "independent":
        return 0.0
    return matrices[relationship][(market_a, market_b)][0]


if __name__ == "__main__":
    matrices = build_correlation_matrices()
    for rel, table in matrices.items():
        print(f"\n=== {rel} ===")
        seen = set()
        for (a, b), (corr, n) in sorted(table.items()):
            if (b, a) in seen:
                continue
            seen.add((a, b))
            print(f"  {a:10s} x {b:10s}: {corr:+.3f}  (n={n})")
