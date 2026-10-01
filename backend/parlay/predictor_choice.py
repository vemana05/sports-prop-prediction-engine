"""Which predictor to use per market: the trained LightGBM model, or the
naive last-10-game rolling average — whichever actually won the
walk-forward comparison (spec Section 5.7's naive-baseline reporting).

Measured results (backend/models/*_model.py walk-forward MAE, all seasons
pooled, after the usage/pace/opponent-defense features were added):

    market     model    naive(last10)   winner
    minutes    4.974    5.232 (roll3)   model
    points     4.592    4.669 (season)  model
    rebounds   1.920    1.947           model
    assists    1.345    1.358           model
    fg3m       0.890    0.887           naive  <- model does NOT win here

This is a static table, not re-derived at runtime, because re-deriving it
means re-running a full walk-forward backtest (minutes of LightGBM
training) just to answer a yes/no question whose answer only changes when
the underlying data or features change meaningfully. If a future feature
addition might flip fg3m back to favoring the model, re-run
`nba_fg3m_model.py` and update this table by hand — don't silently drift.
"""

from __future__ import annotations

USE_MODEL = {
    "minutes": True,
    "points": True,
    "rebounds": True,
    "assists": True,
    "fg3m": False,  # naive last-10 average wins — see module docstring
}


def use_model_for(market: str) -> bool:
    if market not in USE_MODEL:
        raise ValueError(f"unknown market {market!r}, expected one of {list(USE_MODEL)}")
    return USE_MODEL[market]
