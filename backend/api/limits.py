"""Loss limits, take-a-break pauses and the Kelly fraction (spec 10.4 G),
stored next to the bets in data/bets.sqlite (tracker.py).

Loss counted against a limit = stakes of real bets settled as lost minus
winnings of real bets settled as won, both with `settled_at` inside the
window, floored at $0. Paper bets never count. Bonus bets are left out too
(a lost one cost nothing) and are reported on their own. Windows are
calendar periods in ET: the day from midnight, the week from Monday
midnight, the month from the 1st.

Lowering (or adding) a limit takes effect at once. Raising or removing one
waits `RAISE_DELAY` and shows as pending until then; it can be cancelled.
A pause blocks real-bet logging until it ends and can't be shortened;
paper logging is never blocked.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from backend.api import tracker
from backend.bankroll import kelly

ET = ZoneInfo("America/New_York")
PERIODS = ("daily", "weekly", "monthly")
RAISE_DELAY = timedelta(hours=24)
PAUSE_DAYS = (1, 7, 30)
WARN_AT = 0.8
KELLY_DEFAULT = 0.25

_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS loss_limits (
        period TEXT PRIMARY KEY,
        amount REAL,                 -- NULL = no limit
        pending_set INTEGER NOT NULL DEFAULT 0,
        pending_amount REAL,         -- NULL with pending_set = 1 means "remove"
        pending_at TEXT              -- UTC ISO time the pending change applies
    )""",
    "CREATE TABLE IF NOT EXISTS app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
]


def _conn():
    conn = tracker._conn()
    for stmt in _SCHEMA:
        conn.execute(stmt)
    return conn


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _parse(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# --- windows and loss math -------------------------------------------------

def window(period: str, now: datetime) -> tuple[datetime, datetime]:
    """[start, end) of the ET calendar period containing `now`."""
    local = now.astimezone(ET)
    midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "daily":
        start, end = midnight, midnight + timedelta(days=1)
    elif period == "weekly":
        start = midnight - timedelta(days=local.weekday())
        end = start + timedelta(days=7)
    elif period == "monthly":
        start = midnight.replace(day=1)
        end = (start + timedelta(days=32)).replace(day=1)
    else:
        raise ValueError(f"period must be one of {PERIODS}")
    # Rebuild from wall-clock dates so DST shifts inside the window don't skew it.
    start = datetime(start.year, start.month, start.day, tzinfo=ET)
    end = datetime(end.year, end.month, end.day, tzinfo=ET)
    return start, end


def _in_window(bet: dict, start: datetime, end: datetime) -> bool:
    return bet.get("settled_at") is not None and start <= _parse(bet["settled_at"]) < end


def window_totals(bets: list[dict], period: str, now: datetime) -> dict:
    """Real-money loss for the limit, plus bonus bets on their own."""
    start, end = window(period, now)
    settled = [b for b in bets if not b["paper"] and _in_window(b, start, end)]
    real = [b for b in settled if not b.get("bonus")]
    bonus = [b for b in settled if b.get("bonus")]
    lost = sum(b["stake"] for b in real if b["result"] == "lost")
    won = sum(b["stake"] * (b["decimal_odds"] - 1) for b in real if b["result"] == "won")
    return {
        "start": _iso(start),
        "end": _iso(end),
        "lost": round(lost, 2),
        "won": round(won, 2),
        "net_loss": round(max(lost - won, 0.0), 2),
        "bonus_bets": len(bonus),
        "bonus_winnings": round(sum(b["stake"] * (b["decimal_odds"] - 1) for b in bonus if b["result"] == "won"), 2),
    }


# --- limits ----------------------------------------------------------------

def _apply_due(conn, now: datetime) -> None:
    for period, at in conn.execute("SELECT period, pending_at FROM loss_limits WHERE pending_set = 1").fetchall():
        if _parse(at) <= now:
            conn.execute(
                "UPDATE loss_limits SET amount = pending_amount, pending_set = 0, pending_amount = NULL, pending_at = NULL WHERE period = ?",
                [period],
            )


def get_limits(now: datetime | None = None) -> dict[str, dict]:
    now = now or _utcnow()
    with _conn() as conn:
        _apply_due(conn, now)
        rows = {r["period"]: dict(r) for r in conn.execute("SELECT * FROM loss_limits")}
    out = {}
    for p in PERIODS:
        r = rows.get(p, {})
        pending = {"amount": r["pending_amount"], "effective_at": r["pending_at"]} if r.get("pending_set") else None
        out[p] = {"amount": r.get("amount"), "pending": pending}
    return out


def set_limit(period: str, amount: float | None, now: datetime | None = None) -> dict:
    """Lowering or adding applies now (and drops any pending raise);
    raising or removing is queued for RAISE_DELAY."""
    if period not in PERIODS:
        raise ValueError(f"period must be one of {PERIODS}")
    if amount is not None and amount <= 0:
        raise ValueError("A limit must be more than $0. Remove it instead.")
    now = now or _utcnow()
    current = get_limits(now)[period]["amount"]
    with _conn() as conn:
        conn.execute("INSERT OR IGNORE INTO loss_limits (period) VALUES (?)", [period])
        stricter = amount is not None and (current is None or amount <= current)
        if stricter:
            conn.execute(
                "UPDATE loss_limits SET amount = ?, pending_set = 0, pending_amount = NULL, pending_at = NULL WHERE period = ?",
                [amount, period],
            )
        elif current is None:  # removing a limit that doesn't exist
            conn.execute("UPDATE loss_limits SET pending_set = 0, pending_amount = NULL, pending_at = NULL WHERE period = ?", [period])
        else:
            conn.execute(
                "UPDATE loss_limits SET pending_set = 1, pending_amount = ?, pending_at = ? WHERE period = ?",
                [amount, _iso(now + RAISE_DELAY), period],
            )
    return get_limits(now)[period]


def cancel_pending(period: str) -> dict:
    if period not in PERIODS:
        raise ValueError(f"period must be one of {PERIODS}")
    with _conn() as conn:
        conn.execute("UPDATE loss_limits SET pending_set = 0, pending_amount = NULL, pending_at = NULL WHERE period = ?", [period])
    return get_limits()[period]


# --- pause -------------------------------------------------------------------

def _get(key: str) -> str | None:
    with _conn() as conn:
        r = conn.execute("SELECT value FROM app_settings WHERE key = ?", [key]).fetchone()
        return r[0] if r else None


def _put(key: str, value: str) -> None:
    with _conn() as conn:
        conn.execute("INSERT INTO app_settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", [key, value])


def paused_until(now: datetime | None = None) -> str | None:
    now = now or _utcnow()
    v = _get("pause_until")
    return v if v and _parse(v) > now else None


def pause(days: int, now: datetime | None = None) -> str:
    """Start (or extend) a break. Never shortens one already running."""
    if days not in PAUSE_DAYS:
        raise ValueError(f"days must be one of {PAUSE_DAYS}")
    now = now or _utcnow()
    until = now + timedelta(days=days)
    current = paused_until(now)
    if current and _parse(current) >= until:
        return current
    _put("pause_until", _iso(until))
    return _iso(until)


# --- status --------------------------------------------------------------------

def status(now: datetime | None = None, bets: list[dict] | None = None) -> dict:
    """Every limit with its window's loss, % used and level (ok/warn/hit)."""
    now = now or _utcnow()
    bets = tracker.list_bets() if bets is None else bets
    limits = get_limits(now)
    rows = []
    for p in PERIODS:
        t = window_totals(bets, p, now)
        amount = limits[p]["amount"]
        used = t["net_loss"] / amount if amount else None
        level = "none" if amount is None else "hit" if used >= 1 else "warn" if used >= WARN_AT else "ok"
        rows.append({"period": p, "limit": amount, "pending": limits[p]["pending"], "used": used, "level": level, **t})
    order = {"none": 0, "ok": 1, "warn": 2, "hit": 3}
    worst = max(rows, key=lambda r: order[r["level"]])
    return {
        "now": _iso(now),
        "limits": rows,
        "level": worst["level"] if worst["level"] in ("warn", "hit") else "ok",
        "hit": [r["period"] for r in rows if r["level"] == "hit"],
        "paused_until": paused_until(now),
        "warn_at": WARN_AT,
        "raise_delay_hours": RAISE_DELAY.total_seconds() / 3600,
        "pause_days": list(PAUSE_DAYS),
    }


# --- Kelly fraction ----------------------------------------------------------------

def get_kelly() -> float:
    v = _get("kelly_fraction")
    return float(v) if v else kelly.kelly_fraction(default_only=True)


def set_kelly(value: float) -> float:
    if not 0 < value <= 1:
        raise ValueError("Kelly fraction must be above 0 and at most 1.")
    _put("kelly_fraction", repr(float(value)))
    kelly.set_kelly_fraction(float(value))
    return value


def load_kelly() -> None:
    """Apply the stored Kelly fraction to stake sizing at API startup."""
    v = _get("kelly_fraction")
    if v:
        kelly.set_kelly_fraction(float(v))
