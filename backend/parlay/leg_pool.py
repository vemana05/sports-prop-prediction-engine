"""Builds today's leg pool: for every real odds line in `odds_snapshots`,
resolve the player, compute a fresh model prediction, attach this market's
isotonic calibrator (spec Section 2's pipeline — calibration happens here,
before the `Leg` ever reaches shrinkage or the Monte Carlo simulator), and
produce both the "over" and "under" `Leg`.

Deliberately does NOT read or write `model_predictions_log` — that table's
whole point is "log a prediction once, near game day, and never let it
drift" (see log_predictions.py). Computing a fresh prediction here instead
means the leg pool is always honest about "what would the model say right
now," including when (as during this build) that's weeks before tip-off,
without permanently logging a stale early snapshot that would block the
real, close-to-game-time prediction later.

Only `points` has real odds right now (odds_theoddsapi.py only collects
`player_points` — see its own docstring for the credit-budget reasoning);
rebounds/assists/fg3m legs would need those markets added to collection
first. The pool and everything downstream (correlation, Monte Carlo, beam
search) already supports all 4 markets — it's purely an odds-collection
scope decision, not a modeling limitation.

Team and opponent: a leg's team is the player's CURRENT team (Sleeper via
player_xref, else their last game log — `current_team_by_player`), and it
must be one of the two teams in the leg's odds event; the opponent is
always the event's other team, never derived from the player's old team.
The event's teams come from odds_events (The Odds API's own home/away),
else from the nba_schedule game it matches. A player who can't be placed in
their event's two teams (stale roster, unknown event) gets no leg and is
listed in `Unplaced` instead — `python -m backend.parlay.leg_pool --check`
prints them for the current slate without running any model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone

import duckdb
import pandas as pd
from nba_api.stats.static import teams as nba_static_teams
from rapidfuzz import fuzz, process

from backend.db import get_connection
from backend.features.nba_current_form import (
    build_current_form,
    current_team_by_player,
    teammates_out_from_injury_report,
)
from backend.ingest.util import logger
from backend.models.calibration import get_or_fit_calibrator
from backend.models.model_cache import get_or_train_booster
from backend.models.nba_minutes_model import FEATURE_COLUMNS
from backend.models.nba_points_model import train_final_model as train_points_final
from backend.parlay.monte_carlo import Leg
from backend.parlay.predictor_choice import use_model_for
from backend.predict.schedule_projections import ET
from backend.xref.build_xref import normalize_name

FUZZY_MATCH_THRESHOLD = 88.0


@dataclass(frozen=True)
class EventTeams:
    home: str
    away: str
    nba_game_id: str | None = None

    @property
    def label(self) -> str:
        return f"{self.away} @ {self.home}"


@dataclass(frozen=True)
class Unplaced:
    """A player with odds whose team isn't one of their event's two teams."""
    player_name: str
    player_id_nba: str
    event_id: str
    team: str | None
    event: str | None  # "AWAY @ HOME", or None when the event's teams are unknown


def _opponent_by_team(conn: duckdb.DuckDBPyConnection, game_date: date) -> dict[str, str]:
    """{team_abbr: opponent_abbr} for every game on `game_date`, from the
    real nba_schedule table — used to fill in opp_pace_roll10 /
    opp_def_vs_position_pts_roll10, which are about the player's NEXT game,
    not a historical one. nba_schedule stores nba.com numeric team ids;
    nba_api's static team list is the standard way to resolve those back to
    the same 3-letter abbreviations nba_player_games uses."""
    id_to_abbr = {str(t["id"]): t["abbreviation"] for t in nba_static_teams.get_teams()}
    games = conn.execute(
        "SELECT home_team_id, away_team_id FROM nba_schedule WHERE game_date = ?", [game_date]
    ).fetchall()
    out: dict[str, str] = {}
    for home_id, away_id in games:
        home, away = id_to_abbr.get(home_id), id_to_abbr.get(away_id)
        if home and away:
            out[home] = away
            out[away] = home
    return out


def _resolve_player_names(conn: duckdb.DuckDBPyConnection, raw_names: list[str]) -> dict[str, str]:
    """{raw_name: player_id_nba}, fuzzy-matched against player_xref — same
    approach as build_xref.py / match_injury_names.py."""
    xref = conn.execute("SELECT player_id, full_name FROM player_xref WHERE full_name IS NOT NULL").fetchall()
    index = {normalize_name(name): pid for pid, name in xref}
    choices = list(index.keys())

    out = {}
    for raw in raw_names:
        norm = normalize_name(raw)
        if norm in index:
            out[raw] = index[norm]
            continue
        result = process.extractOne(norm, choices, scorer=fuzz.WRatio)
        if result and result[1] >= FUZZY_MATCH_THRESHOLD:
            out[raw] = index[result[0]]
        else:
            logger.warning("leg_pool: could not resolve odds player name %r to a player_id", raw)
    return out


def _points_predictions(
    conn: duckdb.DuckDBPyConnection,
    player_ids: list[str],
    out_players_by_team: dict[str, list[str]],
    team_by_player: dict[str, str],
    opponent_by_player: dict[str, str],
) -> dict[str, tuple[float, float]]:
    """{player_id_nba: (predicted_mean, sigma)} for the points market,
    using whichever predictor won the walk-forward comparison."""
    current_form = build_current_form(
        conn, out_players_by_team=out_players_by_team,
        team_by_player=team_by_player, opponent_by_player=opponent_by_player,
    )
    current_form = current_form[current_form["player_id_nba"].isin(player_ids)]
    if current_form.empty:
        return {}

    if use_model_for("points"):
        model, extra = get_or_train_booster("points", lambda: _train_points(conn))
        sigma = extra["sigma"]
        X = current_form[[c for c in FEATURE_COLUMNS if c != "home_flag"]].copy()
        X["home_flag"] = 0  # home/away unknown without a specific game joined in; a small, acknowledged simplification
        X["is_back_to_back"] = X["is_back_to_back"].astype(float)
        preds = model.predict(X[FEATURE_COLUMNS])
        return {pid: (float(p), sigma) for pid, p in zip(current_form["player_id_nba"], preds)}
    else:
        sigma = current_form["pts_std10"].median()  # fallback dispersion for the naive path
        return {
            pid: (float(roll), float(sigma) if pd.notna(sigma) else 5.0)
            for pid, roll in zip(current_form["player_id_nba"], current_form["pts_roll10"])
        }


def _train_points(conn):
    model, sigma = train_points_final(conn)
    return model, {"sigma": sigma}


def match_events_to_schedule(
    events: list[tuple], schedule: list[tuple], teams_by_event: dict[str, list[str]]
) -> dict[str, str]:
    """{event key: nba game_id}. `events` are (key, game_id_nba, tip),
    `schedule` (game_id, home, away, tip). A resolved game_id_nba wins;
    otherwise the scheduled game on the same ET date whose two teams cover
    the most of the event's players' teams, ties broken by the closer tip
    time. Each scheduled game is used at most once."""
    out: dict[str, str] = {}
    scored = []
    for gid, nba_id, tip in events:
        if nba_id:
            out[gid] = nba_id
            continue
        if tip is None:
            continue
        teams = teams_by_event.get(gid, [])
        for sid, home, away, stip in schedule:
            if stip is None or stip.astimezone(ET).date() != tip.astimezone(ET).date():
                continue
            hits = sum(t in (home, away) for t in teams)
            if hits:
                scored.append((-hits, abs((stip - tip).total_seconds()), gid, sid))
    used = set(out.values())
    for _, _, gid, sid in sorted(scored):
        if gid not in out and sid not in used:
            out[gid] = sid
            used.add(sid)
    return out


def _schedule_for_days(conn: duckdb.DuckDBPyConnection, days: list[date]) -> list[tuple]:
    """[(game_id, home_abbr, away_abbr, tip_utc)] for every scheduled game on `days`."""
    if not days:
        return []
    id_to_abbr = {str(t["id"]): t["abbreviation"] for t in nba_static_teams.get_teams()}
    return [
        (gid, id_to_abbr.get(home), id_to_abbr.get(away), tip)
        for gid, home, away, tip in conn.execute(
            "SELECT game_id, home_team_id, away_team_id, tip_utc FROM nba_schedule WHERE game_date IN (SELECT unnest(?))",
            [days],
        ).fetchall()
    ]


def event_teams(
    conn: duckdb.DuckDBPyConnection, odds: pd.DataFrame, team_by_player: dict[str, str]
) -> dict[str, EventTeams]:
    """{event_id: EventTeams} for every event in `odds` whose teams are
    known: from odds_events when the odds job recorded them, else from the
    nba_schedule game the event matches (by ET date, tip time and players'
    CURRENT teams)."""
    event_ids = odds["event_id"].unique().tolist()
    out = {
        ev: EventTeams(home, away)
        for ev, home, away in conn.execute(
            "SELECT event_id, home_team, away_team FROM odds_events WHERE event_id IN (SELECT unnest(?))", [event_ids]
        ).fetchall()
        if home and away
    }
    rest = odds[~odds["event_id"].isin(out)]
    if rest.empty:
        return out
    events = (
        rest.groupby("event_id")
        .agg(game_id_nba=("game_id_nba", "first"), tip=("commence_time", "min"))
        .reset_index()
    )
    tips = [t.to_pydatetime() for t in events["tip"]]
    days = sorted({t.astimezone(ET).date() for t in tips})
    schedule = _schedule_for_days(conn, days)
    teams_by_event = {
        ev: [team_by_player[p] for p in g["player_id_nba"] if p in team_by_player]
        for ev, g in rest.groupby("event_id")
    }
    matched = match_events_to_schedule(
        [(ev, gid if isinstance(gid, str) else None, tip) for ev, gid, tip in zip(events["event_id"], events["game_id_nba"], tips)],
        schedule, teams_by_event,
    )
    by_id = {s[0]: s for s in schedule}
    for ev, sid in matched.items():
        s = by_id.get(sid)
        if s and s[1] and s[2]:
            out[ev] = EventTeams(s[1], s[2], s[0])
    return out


def place_players(
    odds: pd.DataFrame, teams: dict[str, EventTeams], team_by_player: dict[str, str]
) -> tuple[dict[tuple[str, str], tuple[str, str, bool]], list[Unplaced]]:
    """({(player_id_nba, event_id): (team, opponent, is_home)}, unplaced).
    A player is placed only when their current team is one of the event's
    two teams; the opponent is then the event's other team."""
    placed: dict[tuple[str, str], tuple[str, str, bool]] = {}
    unplaced: dict[tuple[str, str], Unplaced] = {}
    for row in odds[["player_id_nba", "player_name_raw", "event_id"]].drop_duplicates().itertuples():
        key = (row.player_id_nba, row.event_id)
        team, ev = team_by_player.get(row.player_id_nba), teams.get(row.event_id)
        if ev and team == ev.home:
            placed[key] = (team, ev.away, True)
        elif ev and team == ev.away:
            placed[key] = (team, ev.home, False)
        else:
            unplaced[key] = Unplaced(row.player_name_raw, row.player_id_nba, row.event_id, team, ev.label if ev else None)
    return placed, list(unplaced.values())


def _upcoming_odds(conn: duckdb.DuckDBPyConnection, now: datetime) -> pd.DataFrame:
    """Latest upcoming player_points line per (player, event), with the
    player resolved to a player_id_nba (unresolved names dropped)."""
    odds = conn.execute(
        """
        SELECT player_name_raw, event_id, game_id_nba, commence_time, bookmaker, line, over_price, under_price
        FROM odds_snapshots
        WHERE market = 'player_points' AND commence_time > ?
        QUALIFY ROW_NUMBER() OVER (PARTITION BY player_name_raw, event_id ORDER BY pulled_at DESC) = 1
        """,
        [now],
    ).fetchdf()
    if odds.empty:
        return odds
    name_map = _resolve_player_names(conn, odds["player_name_raw"].unique().tolist())
    odds["player_id_nba"] = odds["player_name_raw"].map(name_map)
    return odds.dropna(subset=["player_id_nba"])


def check_placement(conn: duckdb.DuckDBPyConnection, now: datetime | None = None) -> tuple[int, list[Unplaced]]:
    """(players with upcoming odds, those that can't be placed) — the leg
    pool's team check without any model work."""
    odds = _upcoming_odds(conn, now or datetime.now(timezone.utc))
    if odds.empty:
        return 0, []
    team_by_player = current_team_by_player(conn)
    placed, unplaced = place_players(odds, event_teams(conn, odds, team_by_player), team_by_player)
    return len(placed) + len(unplaced), unplaced


def build_leg_pool(
    conn: duckdb.DuckDBPyConnection | None = None,
    now: datetime | None = None,
    unplaced: list[Unplaced] | None = None,
) -> list[Leg]:
    """Legs for events that haven't started yet: odds_snapshots keeps every
    pull, so yesterday's games are still in it. Players that can't be
    placed in their event's two teams get no legs; pass `unplaced` (a list)
    to collect them."""
    own_conn = conn is None
    now = now or datetime.now(timezone.utc)
    conn = conn or get_connection()
    try:
        odds = _upcoming_odds(conn, now)
        if odds.empty:
            logger.warning("leg_pool: no upcoming player_points odds in odds_snapshots")
            return []

        team_by_player = current_team_by_player(conn)
        placed, dropped = place_players(odds, event_teams(conn, odds, team_by_player), team_by_player)
        for u in dropped:
            logger.warning("leg_pool: dropping %s (team %s) — not in event %s (%s)", u.player_name, u.team, u.event_id, u.event or "teams unknown")
        if unplaced is not None:
            unplaced.extend(dropped)
        team_by_player = {pid: team for (pid, _), (team, _, _) in placed.items()}
        opponent_by_player = {pid: opp for (pid, _), (_, opp, _) in placed.items()}

        out_players_by_team: dict[str, list[str]] = {}
        for game_date in odds["commence_time"].dt.date.unique():
            out_players_by_team.update(teammates_out_from_injury_report(conn, game_date))

        preds = _points_predictions(
            conn, list(team_by_player), out_players_by_team, team_by_player, opponent_by_player
        )
        points_calibrator = get_or_fit_calibrator(conn, "points")

        legs: list[Leg] = []
        for row in odds.itertuples():
            placement = placed.get((row.player_id_nba, row.event_id))
            if placement is None or row.player_id_nba not in preds:
                continue
            team, opponent, is_home = placement
            predicted_mean, sigma = preds[row.player_id_nba]
            base = dict(
                player_id_nba=row.player_id_nba,
                player_name=row.player_name_raw,
                game_id=row.game_id_nba or row.event_id,  # falls back to event_id if not yet resolved to an nba game_id
                team=team,
                opponent=opponent,
                is_home=is_home,
                market="points",
                line=row.line,
                predicted_mean=predicted_mean,
                dispersion=sigma,
                over_price=row.over_price,
                under_price=row.under_price,
                calibrator=points_calibrator,
            )
            if row.over_price:
                legs.append(Leg(side="over", decimal_odds=row.over_price, **base))
            if row.under_price:
                legs.append(Leg(side="under", decimal_odds=row.under_price, **base))

        logger.info(
            "leg_pool: built %d legs from %d odds rows (%d players resolved, %d unplaced)",
            len(legs), len(odds), len(preds), len(dropped),
        )
        return legs
    finally:
        if own_conn:
            conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NBA leg pool")
    parser.add_argument("--check", action="store_true", help="List upcoming-odds players that can't be placed in their event (no model work)")
    args = parser.parse_args()
    conn = get_connection()
    try:
        if args.check:
            n, bad = check_placement(conn)
            for u in bad:
                print(f"{u.event or u.event_id:<12} {u.player_name:<28} team={u.team}")
            print(f"{len(bad)} of {n} players can't be placed in their event")
        else:
            print(f"{len(build_leg_pool(conn))} legs")
    finally:
        conn.close()
