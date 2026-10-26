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

Only games that haven't tipped are in the slate: the build drops events
whose commence_time has passed, and `pool()` drops games that tip while the
slate is live. `ensure_fresh()` rebuilds when the ET date rolls over or the
daily odds job finishes after the last build.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import yaml

from backend.db import get_connection
from backend.ingest.util import logger
from backend.parlay.correlations import build_correlation_matrices
from backend.parlay.leg_pool import Unplaced, _schedule_for_days, build_leg_pool, match_events_to_schedule
from backend.parlay.monte_carlo import Leg
from backend.predict import early_season
from backend.parlay.search import ParlayCandidate, SearchReport, filter_pool, leg_id, run_search
from backend.predict.schedule_projections import ET, today_et

TIERS_PATH = "config/tiers.yaml"
SEARCH_SEED = 7  # fixed so a tier's results don't reshuffle between requests
REFERENCE_BANKROLL = 100.0  # search ranks by EV, which is bankroll-independent
LOGS_DIR = Path("logs")
DAILY_JOB_LOG_GLOB = "daily_odds_job_*.log"
DAILY_JOB_FINISHED = "=== daily_odds_job finished"
FRESHNESS_CHECK_S = 60  # how often ensure_fresh() looks at the job logs


def load_tiers() -> dict[str, dict]:
    with open(TIERS_PATH) as f:
        return yaml.safe_load(f)


@dataclass
class Game:
    game_id: str
    home: str | None
    away: str | None
    tip_utc: datetime | None
    nba_game_id: str | None = None

    @property
    def label(self) -> str | None:
        return f"{self.away} @ {self.home}" if self.home and self.away else None


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
    early: dict = field(default_factory=dict)  # early_season.status() as of the build
    built_day: date | None = None  # ET date the build ran on
    unplaced: list[Unplaced] = field(default_factory=list)  # odds players not in their event's two teams
    _checked_at: float = 0.0
    _search_cache: dict[tuple, tuple[list[ParlayCandidate], SearchReport]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def start_build(self) -> None:
        with self._lock:
            if self.status == "warming":
                return
            self.status = "warming"
        threading.Thread(target=self._build, daemon=True).start()

    def ensure_fresh(self, now: float | None = None) -> None:
        """Rebuild on a new ET day, or once the daily odds job has finished
        since the last build. Cheap: a date compare, plus a look at the
        newest job log at most every FRESHNESS_CHECK_S."""
        if self.status not in ("ready", "error") or self.built_at is None:
            return
        now = now or time.time()
        if self.built_day != today_et():
            logger.info("slate: ET date rolled over, rebuilding")
            self.start_build()
            return
        if now - self._checked_at < FRESHNESS_CHECK_S:
            return
        self._checked_at = now
        finished = _daily_job_finished_at()
        if finished and finished > self.built_at:
            logger.info("slate: daily odds job finished after the last build, rebuilding")
            self.start_build()

    def _build(self) -> None:
        t0 = time.time()
        now = datetime.now(timezone.utc)
        try:
            conn = get_connection()
            try:
                unplaced: list[Unplaced] = []
                legs = build_leg_pool(conn, now=now, unplaced=unplaced)
                corr = build_correlation_matrices(conn)
                games, pulled_at = _load_games(conn, legs, now)
                early = early_season.status(conn=conn)
            finally:
                conn.close()
        except Exception as e:  # surfaced to the UI as a plain-language error
            logger.exception("slate build failed")
            self.status, self.error = "error", str(e)
            self.built_at, self.built_day = time.time(), now.astimezone(ET).date()
            return

        self.legs, self.corr, self.games, self.odds_pulled_at, self.early = legs, corr, games, pulled_at, early
        self.unplaced = unplaced
        self.legs_by_id = {leg_id(leg): leg for leg in legs}
        self._search_cache = {}
        self.built_at, self.build_seconds = time.time(), round(time.time() - t0, 1)
        self.built_day = now.astimezone(ET).date()
        self.status, self.error = "ready", None
        logger.info("slate ready: %d legs, %d games in %.1fs", len(legs), len(games), self.build_seconds)

    @property
    def early_season(self) -> bool:
        """Stricter sanity guard while any team is under EARLY_SEASON_GAMES."""
        return bool(self.early.get("active"))

    def started_games(self, now: datetime | None = None) -> frozenset:
        now = now or datetime.now(timezone.utc)
        return frozenset(gid for gid, g in self.games.items() if g.tip_utc and g.tip_utc <= now)

    def upcoming_games(self, now: datetime | None = None) -> dict[str, Game]:
        started = self.started_games(now)
        return {gid: g for gid, g in self.games.items() if gid not in started}

    @property
    def slate_date(self) -> date | None:
        """ET date of the earliest upcoming tip — what "Next slate" means."""
        tips = [g.tip_utc for g in self.upcoming_games().values() if g.tip_utc]
        return min(tips).astimezone(ET).date() if tips else None

    def pool(self, sides: str = "both", vetoed: frozenset = frozenset(), excluded_games: frozenset = frozenset()) -> list[Leg]:
        return filter_pool(self.legs, sides, vetoed, excluded_games | self.started_games())

    def search(
        self, tier: str, sides: str = "both", vetoed: frozenset = frozenset(), excluded_games: frozenset = frozenset()
    ) -> tuple[list[ParlayCandidate], SearchReport]:
        excluded_games = excluded_games | self.started_games()  # games that tipped since the build
        key = (tier, sides, vetoed, excluded_games)
        if key not in self._search_cache:
            self._search_cache[key] = run_search(
                filter_pool(self.legs, sides, vetoed, excluded_games), tier, load_tiers()[tier], self.corr,
                REFERENCE_BANKROLL, seed=SEARCH_SEED, sides=sides, early_season=self.early_season,
            )
        return self._search_cache[key]


def _daily_job_finished_at(logs_dir: Path | None = None) -> float | None:
    """mtime of the newest daily odds job log, if that run has finished."""
    logs = sorted((logs_dir or LOGS_DIR).glob(DAILY_JOB_LOG_GLOB))
    if not logs:
        return None
    try:
        if DAILY_JOB_FINISHED not in logs[-1].read_text(errors="replace"):
            return None
        return logs[-1].stat().st_mtime
    except OSError:
        return None


def _load_games(
    conn: duckdb.DuckDBPyConnection, legs: list[Leg], now: datetime
) -> tuple[dict[str, Game], datetime | None]:
    """Matchup + tip time for every upcoming game in the odds pool, keyed
    the same way `Leg.game_id` is (nba game_id when resolved, else the odds
    event_id). Odds events are rarely resolved to an nba game_id, so
    unresolved ones are matched to nba_schedule here (`_match_schedule`);
    an unmatched event still gets its teams from odds_events when known."""
    events = conn.execute(
        """
        SELECT COALESCE(game_id_nba, event_id), any_value(game_id_nba), min(commence_time)
        FROM odds_snapshots
        WHERE market = 'player_points' AND commence_time > ?
        GROUP BY 1
        """,
        [now],
    ).fetchall()
    days = sorted({tip.astimezone(ET).date() for _, _, tip in events if tip})
    schedule = _schedule_for_days(conn, days)
    matched = _match_schedule(events, schedule, legs)
    sched_by_id = {s[0]: s for s in schedule}
    event_teams = dict(
        (ev, (home, away))
        for ev, home, away in conn.execute("SELECT event_id, home_team, away_team FROM odds_events").fetchall()
    )
    games = {}
    for gid, _, tip in events:
        s = sched_by_id.get(matched.get(gid))
        if s:
            games[gid] = Game(gid, s[1], s[2], tip, s[0])
        else:
            home, away = event_teams.get(gid, (None, None))
            games[gid] = Game(gid, home, away, tip)
    pulled_at = conn.execute("SELECT max(pulled_at) FROM odds_snapshots WHERE market = 'player_points'").fetchone()[0]
    return games, pulled_at


def _match_schedule(events: list[tuple], schedule: list[tuple], legs: list[Leg]) -> dict[str, str]:
    """{event game key: nba game_id}, via `match_events_to_schedule` with
    each event's legs' teams. Leg teams are current (Sleeper) and already
    checked against the event, so they place the event reliably."""
    players: dict[str, dict[str, str]] = {}
    for leg in legs:
        if leg.team:
            players.setdefault(leg.game_id, {})[leg.player_id_nba] = leg.team
    return match_events_to_schedule(events, schedule, {gid: list(p.values()) for gid, p in players.items()})


SLATE = Slate()
