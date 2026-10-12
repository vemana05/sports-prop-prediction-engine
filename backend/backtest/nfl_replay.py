"""NFL rows for the Replay page: every player-game-market's out-of-sample
projection, 25th-75th range, naive baseline and actual result.

Comes from nfl_market_model.walk_forward, so a game in season S is
projected by a model trained on seasons before S only, with the range from
that fold's own spread (fit on out-of-fold residuals inside those training
seasons) — the same thing the live NFL projections show. Baseline is the
player's last-5 average, as in nfl_market_model.baseline_report.

Weekly walk-forward: in the current season, week N is re-projected by the
same saved model the live page used for it (thru week N-1, see
nfl_projections.version_for) when one exists, with that version's own
spread. Every row records its `model_version`, and none saw its own week.

Each fold's spread is also pickled to SPREADS_PATH ({market: {season:
Spread}}) so the NFL ladder calibrators (parlay/nfl_legs.py) can turn
these rows into out-of-sample P(over rung) without retraining anything.

Cached to `nfl_walk_forward_replay` in the NFL warehouse because the walk
forward retrains every fold (a few minutes). Rebuild after new weeks are
ingested and nfl_features is rebuilt:

    python -m backend.backtest.nfl_replay
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from backend.ingest.util import IngestResult, logger, utcnow
from backend.models.nfl_market_model import MARKETS, _predict, load_frame, walk_forward
from backend.nfl_db import get_nfl_connection
from backend.predict import nfl_projections
from backend.predict.nfl_projections import Q_HI, Q_LO, quantiles

REPLAY_TABLE = "nfl_walk_forward_replay"
SPREADS_PATH = Path("data/models/nfl_replay_spreads.pkl")

_SCHEMA = f"""
CREATE OR REPLACE TABLE {REPLAY_TABLE} (
    player_id   VARCHAR NOT NULL,
    player_name VARCHAR,
    team        VARCHAR,
    opponent    VARCHAR,
    position    VARCHAR,
    game_id     VARCHAR NOT NULL,
    season      INTEGER NOT NULL,
    week        INTEGER NOT NULL,
    kickoff_utc TIMESTAMP,
    market      VARCHAR NOT NULL,
    actual      DOUBLE NOT NULL,
    predicted   DOUBLE NOT NULL,
    lo          DOUBLE NOT NULL,
    hi          DOUBLE NOT NULL,
    naive       DOUBLE,
    model_version VARCHAR NOT NULL,
    ingested_at TIMESTAMP NOT NULL,
    PRIMARY KEY (player_id, game_id, market)
);
"""


def _frames() -> tuple[pd.DataFrame, dict]:
    conn = get_nfl_connection(read_only=True)
    try:
        ids = conn.execute("SELECT player_id, game_id, team, opponent, kickoff_utc FROM nfl_features").fetchdf()
        frames, all_spreads = [], {}
        for market in MARKETS:
            results, spreads = walk_forward(conn, market)
            all_spreads[market.name] = spreads
            results = results.dropna(subset=[market.target])
            mu = results["predicted_mean"].to_numpy(float)
            lo, hi = mu.copy(), mu.copy()
            version = results["season"].map(lambda s: f"thru{s - 1}").to_numpy(object)
            for season, spread in spreads.items():
                m = (results["season"] == season).to_numpy()
                lo[m], hi[m] = quantiles(mu[m], spread, Q_LO), quantiles(mu[m], spread, Q_HI)
            mu, lo, hi = _weekly(conn, market, results, mu, lo, hi, version)
            frames.append(pd.DataFrame({
                "player_id": results["player_id"], "player_name": results["player_name"],
                "position": results["position"], "game_id": results["game_id"],
                "season": results["season"], "week": results["week"], "market": market.name,
                "actual": results[market.target], "predicted": mu, "lo": lo, "hi": hi,
                "naive": results[f"{market.target}_roll5"], "model_version": version,
            }))
            logger.info("nfl_replay: %s %d rows", market.name, len(results))
    finally:
        conn.close()
    return pd.concat(frames, ignore_index=True).merge(ids, on=["player_id", "game_id"], how="left"), all_spreads


def _weekly(conn, market, results, mu, lo, hi, version):
    """Current-season weeks that had a weekly model when served: re-project
    with that saved model (load only, never trains here)."""
    season = nfl_projections.current_season()
    cur = results["season"] == season
    if not cur.any():
        return mu, lo, hi
    feats = load_frame(conn, market)
    feats = feats[feats["season"] == season].set_index(["player_id", "game_id"])
    for week in sorted(results.loc[cur, "week"].unique()):
        v = nfl_projections.version_for(season, int(week))
        if v.week is None:
            continue
        model, spread = nfl_projections._get_or_train(market, v)
        m = (cur & (results["week"] == week)).to_numpy()
        rows = feats.loc[list(zip(results.loc[m, "player_id"], results.loc[m, "game_id"]))]
        mu[m] = _predict(model, rows)
        lo[m], hi[m] = quantiles(mu[m], spread, Q_LO), quantiles(mu[m], spread, Q_HI)
        version[m] = v.tag
    return mu, lo, hi


def build() -> IngestResult:
    df, spreads = _frames()
    df["ingested_at"] = utcnow()
    conn = get_nfl_connection()
    try:
        conn.execute(_SCHEMA)
        conn.register("replay_df", df)
        conn.execute(f"INSERT INTO {REPLAY_TABLE} BY NAME SELECT * FROM replay_df")
        conn.unregister("replay_df")
    finally:
        conn.close()
    SPREADS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPREADS_PATH.write_bytes(pickle.dumps(spreads))
    logger.info("nfl_replay: built %d rows", len(df))
    return IngestResult(rows_written=len(df))


if __name__ == "__main__":
    print(f"nfl_replay: {build().rows_written} rows")
