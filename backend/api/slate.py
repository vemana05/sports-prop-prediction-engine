"""In-memory snapshot of the current slate for the API: the leg pool, the
correlation matrices, and per-game matchup info, built once in a background
thread and reused by every request.

Building the leg pool runs feature builders and model inference over the
whole warehouse, so the API never does it on the request path — requests
made while it's warming get `status: "warming"` and the UI shows a
skeleton. The DuckDB connection is closed as soon as the build finishes
(it needs write access only because correlations cache residuals to a
table), so the API never holds the warehouse lock the daily odds job needs.

Search results are cached too, per (tier, side preference, vetoed legs,
excluded games) — search doesn't depend on bankroll, only the stake does,
and that's recomputed per request in `main.py`.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime

import duckdb
import yaml
from nba_api.stats.static import teams as nba_static_teams

from backend.db import get_connection
from backend.ingest.util import logger
from backend.parlay.correlations import build_correlation_matrices
from backend.parlay.leg_pool import build_leg_pool
from backend.parlay.monte_carlo import Leg
from backend.parlay.search import ParlayCandidate, SearchReport, filter_pool, leg_id, run_search

TIERS_PATH = "config/tiers.yaml"
SEARCH_SEED = 7  # fixed so a tier's results don't reshuffle between requests
REFERENCE_BANKROLL = 100.0  # search ranks by EV, which is bankroll-independent


def load_tiers() -> dict[str, dict]:
    with open(TIERS_PATH) as f:
        return yaml.safe_load(f)


@dataclass
class Game:
    game_id: str
    home: str | None
    away: str | None
    tip_utc: datetime | None


@dataclass
class Slate:
    status: str = "idle"  # idle | warming | ready | error
    error: str | None = None
    legs: list[Leg] = field(default_factory=list)
    corr: dict = field(default_factory=dict)
    games: dict[str, Game] = field(default_factory=dict)
    odds_pulled_at: datetime | None = None
    built_at: float | None = None
    build_seconds: float | None = None
    legs_by_id: dict[str, Leg] = field(default_factory=dict)
    _search_cache: dict[tuple, tuple[list[ParlayCandidate], SearchReport]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def start_build(self) -> None:
        with self._lock:
            if self.status == "warming":
                return
            self.status = "warming"
        threading.Thread(target=self._build, daemon=True).start()

    def _build(self) -> None:
        t0 = time.time()
        try:
            conn = get_connection()
            try:
                legs = build_leg_pool(conn)
                corr = build_correlation_matrices(conn)
                games, pulled_at = _load_games(conn)
            finally:
                conn.close()
        except Exception as e:  # surfaced to the UI as a plain-language error
            logger.exception("slate build failed")
            self.status, self.error = "error", str(e)
            return

        self.legs, self.corr, self.games, self.odds_pulled_at = legs, corr, games, pulled_at
        self.legs_by_id = {leg_id(leg): leg for leg in legs}
        self._search_cache = {}
        self.built_at, self.build_seconds = time.time(), round(time.time() - t0, 1)
        self.status, self.error = "ready", None
        logger.info("slate ready: %d legs, %d games in %.1fs", len(legs), len(games), self.build_seconds)

    def pool(self, sides: str = "both", vetoed: frozenset = frozenset(), excluded_games: frozenset = frozenset()) -> list[Leg]:
        return filter_pool(self.legs, sides, vetoed, excluded_games)

    def search(
        self, tier: str, sides: str = "both", vetoed: frozenset = frozenset(), excluded_games: frozenset = frozenset()
    ) -> tuple[list[ParlayCandidate], SearchReport]:
        key = (tier, sides, vetoed, excluded_games)
        if key not in self._search_cache:
            self._search_cache[key] = run_search(
                self.pool(sides, vetoed, excluded_games), tier, load_tiers()[tier], self.corr,
                REFERENCE_BANKROLL, seed=SEARCH_SEED, sides=sides,
            )
        return self._search_cache[key]


def _load_games(conn: duckdb.DuckDBPyConnection) -> tuple[dict[str, Game], datetime | None]:
    """Matchup + tip time for every game in the current odds pool, keyed the
    same way `Leg.game_id` is (nba game_id when resolved, else the odds
    event_id)."""
    id_to_abbr = {str(t["id"]): t["abbreviation"] for t in nba_static_teams.get_teams()}
    rows = conn.execute(
        """
        SELECT DISTINCT COALESCE(o.game_id_nba, o.event_id), s.home_team_id, s.away_team_id,
               COALESCE(s.tip_utc, o.commence_time)
        FROM odds_snapshots o
        LEFT JOIN nba_schedule s ON s.game_id = o.game_id_nba
        WHERE o.market = 'player_points'
        """
    ).fetchall()
    games = {
        gid: Game(gid, id_to_abbr.get(home), id_to_abbr.get(away), tip)
        for gid, home, away, tip in rows
    }
    pulled_at = conn.execute("SELECT max(pulled_at) FROM odds_snapshots WHERE market = 'player_points'").fetchone()[0]
    return games, pulled_at


SLATE = Slate()
