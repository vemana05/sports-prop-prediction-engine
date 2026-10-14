"""Data Sources & status page (spec 10.4 F): scheduled jobs from their logs,
API usage from the stored usage logs, warehouse table freshness, saved
model files, and the unmatched-player review queue. Read-only: opens both
warehouses read-only, makes no API calls, and never returns a key value
(every log line passes through `redact`).

Lateness:
- a job is late when its most recent scheduled time is more than LATE_GRACE
  ago and no run has started since then (or within EARLY_RUN before it: a
  manual run that morning did the scheduled run's work);
- a table is late when its newest row is older than its expected cadence
  (TABLE_CADENCE_DAYS) plus LATE_GRACE.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb

ET = ZoneInfo("America/New_York")
LATE_GRACE = timedelta(days=1)
RUNNING_WINDOW = timedelta(hours=3)  # an unfinished log younger than this is still running
EARLY_RUN = timedelta(hours=12)  # a manual run this soon before a scheduled time covers it
LOGS_DIR = Path("logs")
MODELS_DIR = Path("data/models")
WAREHOUSE_PATH = "data/warehouse.duckdb"
NFL_WAREHOUSE_PATH = "data/nfl_warehouse.duckdb"
ODDS_API_LIMIT = 500
ODDSPAPI_LIMIT = 250
QUEUE_LIMIT = 300


@dataclass(frozen=True)
class Job:
    key: str
    label: str
    log_prefix: str
    weekday: int | None  # Python weekday (Mon=0); None = every day
    hour: int
    script: str

    @property
    def schedule(self) -> str:
        day = "Daily" if self.weekday is None else ["Mondays", "Tuesdays", "Wednesdays", "Thursdays", "Fridays", "Saturdays", "Sundays"][self.weekday]
        return f"{day} {self.hour}:00 ET"


# Mirrors the launchd plists in scripts/.
JOBS = [
    Job("daily_nba", "Daily NBA", "daily_odds_job_", None, 10, "daily_odds_job.sh"),
    Job("nfl_prep", "NFL prep (Saturday)", "weekly_nfl_job_prep_", 5, 9, "weekly_nfl_job.sh prep"),
    Job("nfl_lines", "NFL lines (Wednesday)", "weekly_nfl_job_lines_", 2, 9, "weekly_nfl_job.sh lines"),
    Job("nfl_results", "NFL results (Tuesday)", "weekly_nfl_job_results_", 1, 9, "weekly_nfl_job.sh results"),
]

# Expected refresh cadence in days for tables a scheduled job writes; the
# rest are rebuilt on demand and never flagged.
TABLE_CADENCE_DAYS = {
    "warehouse": {"odds_snapshots": 1, "odds_api_usage_log": 1, "model_predictions_log": 1, "nba_injuries": 1, "sleeper_players": 1},
    "nfl_warehouse": {"nfl_schedule": 7, "nfl_player_weekly": 7, "nfl_snap_counts": 7, "nfl_injuries": 7,
                      "nfl_predictions_log": 7, "nfl_prop_lines": 7, "nfl_oddspapi_fixtures": 7, "oddspapi_usage_log": 7},
}
# First column found is the table's "last updated" time.
TIME_COLUMNS = ("ingested_at", "fetched_at", "pulled_at", "logged_at", "predicted_at", "called_at", "updated_at",
                "created_at", "finished_at", "snapshot_at", "odds_fetched_at")


# ---- redaction -------------------------------------------------------------------

_SECRET_ENV = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD)$", re.I)
_KV = re.compile(r"(?i)\b(api[_-]?key|apikey|access[_-]?token|token|secret|password)(\s*[=:]\s*)([^\s&\"',)]+)")
_BEARER = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+")


def redact(text: str) -> str:
    """Removes key values: key=value / key: value pairs, bearer tokens, and
    the literal value of every *KEY/*TOKEN/*SECRET/*PASSWORD env var."""
    text = _KV.sub(r"\1\2REDACTED", text)
    text = _BEARER.sub(r"\1REDACTED", text)
    for name, val in os.environ.items():
        if _SECRET_ENV.search(name) and val and len(val) >= 6:
            text = text.replace(val, "REDACTED")
    return text


# ---- jobs ------------------------------------------------------------------------

_STAMP = re.compile(r"_(\d{8}_\d{6})\.log$")
_FINISHED = re.compile(r"=== .*? finished (.+?) \((.*?)\) ===")
_EXIT = re.compile(r"exit (-?\d+)")
_ERRORISH = re.compile(r"\bERROR\b|Traceback|Exception\b|\w+Error\b|(?i:\bfailed\b)")


def _parse_date_cmd(s: str) -> datetime | None:
    """`date` output, e.g. 'Wed Oct 14 17:00:15 EDT 2026', as ET."""
    parts = s.split()
    if len(parts) == 6:
        parts.pop(4)  # zone abbreviation; the Mac runs on Eastern time
    try:
        return datetime.strptime(" ".join(parts), "%a %b %d %H:%M:%S %Y").replace(tzinfo=ET)
    except ValueError:
        return None


def parse_log(path: Path, now: datetime) -> dict:
    m = _STAMP.search(path.name)
    started = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").replace(tzinfo=ET) if m else None
    lines = path.read_text(errors="replace").splitlines()
    finished, exits = None, None
    for line in lines:
        f = _FINISHED.search(line)
        if f:
            finished, exits = _parse_date_cmd(f.group(1)), [int(x) for x in _EXIT.findall(f.group(2))]
    errors = [ln.strip() for ln in lines if _ERRORISH.search(ln)]
    if exits is not None:
        outcome = "success" if exits and all(e == 0 for e in exits) else "failed"
    elif any(ln.startswith("ERROR") for ln in lines):
        outcome = "failed"
    else:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        outcome = "running" if now - mtime < RUNNING_WINDOW else "failed"
        if outcome == "failed":
            errors.append("Run never logged a finish line (stopped or crashed).")
    return {
        "log": path.name,
        "started_at": started.isoformat() if started else None,
        "finished_at": finished.isoformat() if finished else None,
        "duration_s": int((finished - started).total_seconds()) if started and finished else None,
        "outcome": outcome,
        "last_error": redact(errors[-1])[:400] if errors else None,
    }


def last_scheduled(job: Job, now: datetime) -> datetime:
    local = now.astimezone(ET)
    t = local.replace(hour=job.hour, minute=0, second=0, microsecond=0)
    if job.weekday is None:
        return t if t <= local else t - timedelta(days=1)
    t -= timedelta(days=(local.weekday() - job.weekday) % 7)
    return t if t <= local else t - timedelta(days=7)


def job_status(job: Job, logs_dir: Path, now: datetime) -> dict:
    logs = sorted(logs_dir.glob(f"{job.log_prefix}*.log")) if logs_dir.exists() else []
    due = last_scheduled(job, now)
    nxt = due + (timedelta(days=1) if job.weekday is None else timedelta(days=7))
    out = {"key": job.key, "label": job.label, "schedule": job.schedule, "script": job.script,
           "last_scheduled": due.isoformat(), "next_scheduled": nxt.isoformat(), "runs_logged": len(logs)}
    if not logs:
        return {**out, "outcome": "never ran", "log": None, "started_at": None, "finished_at": None, "duration_s": None,
                "last_error": None, "late": now - due > LATE_GRACE}
    last = parse_log(logs[-1], now)
    started = datetime.fromisoformat(last["started_at"]) if last["started_at"] else None
    late = (started is None or started < due - EARLY_RUN) and now - due > LATE_GRACE
    return {**out, **last, "late": late}


# ---- warehouses ------------------------------------------------------------------

def _open(path: str) -> tuple[duckdb.DuckDBPyConnection | None, str | None]:
    if not Path(path).exists():
        return None, f"{path} not found"
    try:
        return duckdb.connect(path, read_only=True), None
    except duckdb.Error as e:
        if "different configuration" in str(e):
            # This process already has the file open read-write (the API's
            # slate build), so a second read-only handle is refused. Share that
            # instance instead; only SELECTs run on it.
            try:
                return duckdb.connect(path), None
            except duckdb.Error as e2:
                e = e2
        return None, f"{path} is busy (a job may be writing to it): {redact(str(e))[:200]}"


def _as_utc(ts) -> datetime | None:
    if ts is None:
        return None
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)


def table_status(conn: duckdb.DuckDBPyConnection, db: str, now: datetime) -> list[dict]:
    cols: dict[str, list[str]] = {}
    for t, c in conn.execute(
        "SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = 'main' ORDER BY table_name, ordinal_position"
    ).fetchall():
        cols.setdefault(t, []).append(c)
    out = []
    for table, columns in sorted(cols.items()):
        tcol = next((c for c in TIME_COLUMNS if c in columns), None)
        rows, last = conn.execute(f'SELECT count(*), {f"max({tcol})" if tcol else "NULL"} FROM "{table}"').fetchone()
        last = _as_utc(last)
        cadence = TABLE_CADENCE_DAYS.get(db, {}).get(table)
        late = bool(cadence) and (last is None or now - last > timedelta(days=cadence) + LATE_GRACE)
        out.append({"table": table, "rows": int(rows), "last_updated": last.isoformat() if last else None, "time_column": tcol,
                    "cadence_days": cadence, "late": late})
    return out


def odds_api_usage(conn: duckdb.DuckDBPyConnection | None, now: datetime) -> dict:
    from backend.ingest.odds_theoddsapi import _budget

    out = {"name": "The Odds API", "limit": ODDS_API_LIMIT, "budget": _budget(), "used": None, "last_call": None, "note": None}
    if conn is None:
        out["note"] = "warehouse unavailable"
        return out
    row = conn.execute(
        "SELECT requests_used, requests_remaining, called_at FROM odds_api_usage_log "
        "WHERE requests_used IS NOT NULL ORDER BY called_at DESC LIMIT 1"
    ).fetchone()
    if not row:
        out["used"], out["note"] = 0, "No calls logged yet."
        return out
    used, remaining, called = row
    called = _as_utc(called)
    out.update(used=int(used), last_call=called.isoformat(), limit=int(used + remaining) if remaining is not None else ODDS_API_LIMIT)
    out["note"] = "As reported by the API's own usage headers on the last call."
    if (called.year, called.month) != (now.year, now.month):
        out["note"] += " Last call was in an earlier month; the counter may have reset since."
    return out


def oddspapi_usage(conn: duckdb.DuckDBPyConnection | None, now: datetime) -> dict:
    from backend.ingest.odds_oddspapi_nfl import _budget, last_known_count

    out = {"name": "OddsPapi", "limit": ODDSPAPI_LIMIT, "budget": _budget(), "used": None, "last_call": None, "note": None}
    if conn is None:
        out["note"] = "NFL warehouse unavailable"
        return out
    out["used"] = int(last_known_count(conn, now))
    row = conn.execute(
        "SELECT called_at, request_limit, source FROM oddspapi_usage_log ORDER BY called_at DESC LIMIT 1"
    ).fetchone()
    if row:
        out["last_call"] = _as_utc(row[0]).isoformat()
        out["limit"] = int(row[1]) if row[1] else ODDSPAPI_LIMIT
        out["note"] = ("Last count read from the OddsPapi account." if row[2] == "account"
                       else "Running estimate: the last account reading plus each request made since.")
    else:
        out["note"] = "No requests logged yet."
    return out


def review_queue(wh: duckdb.DuckDBPyConnection | None, nfl: duckdb.DuckDBPyConnection | None) -> dict:
    rows = []
    if wh is not None:
        for source, name, raw_id, reason, cand, score, created in wh.execute(
            "SELECT source, raw_name, raw_id, reason, best_candidate, best_score, created_at FROM xref_review_queue "
            "WHERE NOT coalesce(resolved, false) ORDER BY created_at DESC, source, raw_name"
        ).fetchall():
            rows.append({"source": source, "name": name, "raw_id": raw_id, "reason": reason, "best_candidate": cand,
                         "best_score": None if score is None else round(float(score), 3),
                         "created_at": _as_utc(created).isoformat() if created else None, "context": None})
    if nfl is not None:
        for name, raw_id, game_id, away, home in nfl.execute(
            """SELECT p.player_name, p.op_player_id, f.game_id, f.away_team, f.home_team FROM nfl_oddspapi_players p
               JOIN nfl_oddspapi_fixtures f USING (fixture_id) WHERE p.player_id IS NULL ORDER BY f.kickoff_utc DESC, p.player_name"""
        ).fetchall():
            rows.append({"source": "oddspapi_nfl", "name": name, "raw_id": raw_id,
                         "reason": "no single roster player on either team with this name", "best_candidate": None,
                         "best_score": None, "created_at": None, "context": f"{away} @ {home} ({game_id})"})
    by_source: dict[str, int] = {}
    for r in rows:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    return {"total": len(rows), "by_source": by_source, "rows": rows[:QUEUE_LIMIT], "truncated": len(rows) > QUEUE_LIMIT}


# ---- models ----------------------------------------------------------------------

_NFL_FILE = re.compile(r"^nfl_(passing_yards|rushing_yards|receiving_yards|receptions)_(thru\w+?)(_spread\.pkl|\.txt)$")


def model_files(models_dir: Path) -> list[dict]:
    """NFL boosters grouped by version tag; every other model file on its own."""
    if not models_dir.exists():
        return []
    nfl: dict[str, dict] = {}
    other = []
    for p in sorted(models_dir.iterdir()):
        if not p.is_file() or p.name.startswith("."):
            continue
        mtime = datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)
        m = _NFL_FILE.match(p.name)
        if m:
            v = nfl.setdefault(m.group(2), {"league": "NFL", "name": "NFL market models", "version": m.group(2), "markets": set(), "modified": mtime})
            v["markets"].add(m.group(1))
            v["modified"] = max(v["modified"], mtime)
        else:
            other.append({"league": "NFL" if p.name.startswith("nfl_") else "NBA", "name": p.name, "version": None,
                          "markets": [], "modified": mtime})
    out = [{**v, "markets": sorted(v["markets"])} for v in nfl.values()] + other
    out.sort(key=lambda r: r["modified"], reverse=True)
    return [{**r, "modified": r["modified"].isoformat()} for r in out]


# ---- assembly --------------------------------------------------------------------

def build_status(now: datetime | None = None, logs_dir: Path = LOGS_DIR, warehouse_path: str | None = None,
                 nfl_warehouse_path: str | None = None, models_dir: Path = MODELS_DIR) -> dict:
    now = now or datetime.now(timezone.utc)
    wh_path = warehouse_path or os.environ.get("DUCKDB_PATH", WAREHOUSE_PATH)
    nfl_path = nfl_warehouse_path or os.environ.get("NFL_DUCKDB_PATH", NFL_WAREHOUSE_PATH)
    jobs = [job_status(j, logs_dir, now) for j in JOBS]

    wh, wh_err = _open(wh_path)
    nfl, nfl_err = _open(nfl_path)
    try:
        warehouses = [
            {"name": "warehouse", "path": wh_path, "error": wh_err, "tables": table_status(wh, "warehouse", now) if wh else []},
            {"name": "nfl_warehouse", "path": nfl_path, "error": nfl_err, "tables": table_status(nfl, "nfl_warehouse", now) if nfl else []},
        ]
        usage = [oddspapi_usage(nfl, now), odds_api_usage(wh, now)]
        queue = review_queue(wh, nfl)
    finally:
        for c in (wh, nfl):
            if c is not None:
                c.close()

    flags = []
    for j in jobs:
        if j["late"]:
            flags.append(f"{j['label']} is late: last scheduled {_fmt(j['last_scheduled'])}, "
                         + ("never ran." if j["outcome"] == "never ran" else f"last run {_fmt(j['started_at'])}."))
        if j["outcome"] == "failed":
            flags.append(f"{j['label']}: last run failed ({_fmt(j['started_at'])}).")
    for w in warehouses:
        if w["error"]:
            flags.append(f"{w['name']}: {w['error']}")
        for t in w["tables"]:
            if t["late"]:
                flags.append(f"{w['name']}.{t['table']} is stale: last updated {_fmt(t['last_updated']) if t['last_updated'] else 'never'}, "
                             f"expected every {t['cadence_days']} day{'s' if t['cadence_days'] != 1 else ''}.")
    for u in usage:
        if u["used"] is not None and u["used"] >= u["budget"]:
            flags.append(f"{u['name']}: {u['used']} used, at or over our {u['budget']} cap; collection is paused until the counter resets.")
    if queue["total"]:
        flags.append(f"{queue['total']} unmatched player name{'s' if queue['total'] != 1 else ''} waiting for review.")

    return {"as_of": now.isoformat(timespec="seconds"), "flags": flags, "jobs": jobs, "usage": usage,
            "warehouses": warehouses, "models": model_files(models_dir), "review_queue": queue}


def _fmt(iso: str | None) -> str:
    if not iso:
        return "unknown"
    return datetime.fromisoformat(iso).astimezone(ET).strftime("%a %b %-d %-I:%M %p ET")
