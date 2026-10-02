"""Bet tracker storage (spec Section 10.4 D): bets logged from the slip,
settled by hand as won / lost / push.

Kept in its own small SQLite file, not the DuckDB warehouse — the API
deliberately never holds the warehouse open (see slate.py), and a handful
of hand-logged bets don't need it. Paper bets live in the same table with
`paper = 1` so "paper trading" is a view, not a separate system.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("data/bets.sqlite")
RESULTS = ("pending", "won", "lost", "push")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    league TEXT NOT NULL,
    tier TEXT NOT NULL,
    name TEXT NOT NULL,
    legs_json TEXT NOT NULL,      -- [{player_name, label, american_odds}]
    decimal_odds REAL NOT NULL,
    american_odds INTEGER NOT NULL,
    model_p REAL NOT NULL,        -- simulated hit chance at the time the bet was logged
    stake REAL NOT NULL,
    paper INTEGER NOT NULL DEFAULT 0,
    result TEXT NOT NULL DEFAULT 'pending',
    settled_at TEXT
)
"""


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def profit(bet: dict) -> float:
    """Net result of a settled bet; 0 while pending or on a push."""
    if bet["result"] == "won":
        return round(bet["stake"] * (bet["decimal_odds"] - 1), 2)
    if bet["result"] == "lost":
        return -bet["stake"]
    return 0.0


def _row(r: sqlite3.Row) -> dict:
    bet = dict(r)
    bet["legs"] = json.loads(bet.pop("legs_json"))
    bet["paper"] = bool(bet["paper"])
    bet["profit"] = profit(bet)
    return bet


def list_bets() -> list[dict]:
    with _conn() as conn:
        return [_row(r) for r in conn.execute("SELECT * FROM bets ORDER BY created_at, id")]


def add_bet(bet: dict) -> dict:
    with _conn() as conn:
        cur = conn.execute(
            """INSERT INTO bets (created_at, league, tier, name, legs_json, decimal_odds, american_odds, model_p, stake, paper)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [_now(), bet["league"], bet["tier"], bet["name"], json.dumps(bet["legs"]), bet["decimal_odds"],
             bet["american_odds"], bet["model_p"], bet["stake"], int(bet["paper"])],
        )
        return _row(conn.execute("SELECT * FROM bets WHERE id = ?", [cur.lastrowid]).fetchone())


def settle_bet(bet_id: int, result: str) -> dict | None:
    if result not in RESULTS:
        raise ValueError(f"result must be one of {RESULTS}")
    with _conn() as conn:
        conn.execute(
            "UPDATE bets SET result = ?, settled_at = ? WHERE id = ?",
            [result, None if result == "pending" else _now(), bet_id],
        )
        r = conn.execute("SELECT * FROM bets WHERE id = ?", [bet_id]).fetchone()
        return _row(r) if r else None


def delete_bet(bet_id: int) -> bool:
    with _conn() as conn:
        return conn.execute("DELETE FROM bets WHERE id = ?", [bet_id]).rowcount > 0
