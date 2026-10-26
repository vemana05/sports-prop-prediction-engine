"""scripts/daily_odds_job.sh: the NBA game-log step runs after the DNS wait
and before anything is predicted, its output is key-redacted, and a failure
there is reported without stopping predictions or odds. Runs a copy of the
script against fake `docker` and `sleep` commands; nothing real is started."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "daily_odds_job.sh"
SECRET = "sk_live_should_not_leak"

FAKE_DOCKER = r"""#!/bin/bash
echo "$*" >> "$CALLS"
case "$*" in
  info) exit 0 ;;
  "compose ps"*) exit 0 ;;
  *socket.getaddrinfo*)
    n=$(grep -c getaddrinfo "$CALLS")
    [ "$n" -gt "${DNS_OK_AFTER:-0}" ] && exit 0 || exit 1 ;;
  *nba_daily_games*)
    echo "GET https://stats.nba.com/x?apiKey=$SECRET failed"
    exit "${GAMELOG_EXIT:-0}" ;;
  *) exit 0 ;;
esac
"""


@pytest.fixture
def job(tmp_path):
    (tmp_path / "scripts").mkdir()
    shutil.copy(SCRIPT, tmp_path / "scripts" / SCRIPT.name)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("docker", FAKE_DOCKER), ("sleep", "#!/bin/bash\necho \"sleep $*\" >> \"$CALLS\"\n")):
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    calls = tmp_path / "calls.txt"

    def run(gamelog_exit=0, dns_ok_after=0):
        calls.write_text("")
        env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "CALLS": str(calls), "SECRET": SECRET,
               "GAMELOG_EXIT": str(gamelog_exit), "DNS_OK_AFTER": str(dns_ok_after)}
        p = subprocess.run(["bash", str(tmp_path / "scripts" / SCRIPT.name)], env=env,
                           capture_output=True, text=True, timeout=60)
        log = "".join(f.read_text() for f in sorted((tmp_path / "logs").glob("daily_odds_job_*.log")))
        for f in (tmp_path / "logs").glob("*.log"):
            f.unlink()
        return p.returncode, calls.read_text().splitlines(), log

    return run


def _index(calls, needle):
    return next(i for i, ln in enumerate(calls) if needle in ln)


def test_game_logs_load_after_dns_and_before_predictions(job):
    code, calls, log = job(dns_ok_after=2)
    assert code == 0
    gamelogs = _index(calls, "backend.ingest.nba_daily_games")
    assert _index(calls, "getaddrinfo") < gamelogs
    assert sum("getaddrinfo" in ln for ln in calls[:gamelogs]) == 3  # waited out the DNS failures first
    assert gamelogs < _index(calls, "backend.predict.schedule_projections")
    assert gamelogs < _index(calls, "backend.predict.log_predictions")
    assert gamelogs < _index(calls, "backend.ingest.odds_theoddsapi")
    assert "(gamelogs exit 0, predict exit 0, odds exit 0)" in log


def test_game_log_output_is_redacted(job):
    _, _, log = job()
    assert SECRET not in log and "apiKey=REDACTED" in log


def test_game_log_failure_is_reported_but_predictions_still_run(job):
    code, calls, log = job(gamelog_exit=1)
    assert code == 1
    assert any("backend.predict.log_predictions" in ln for ln in calls)
    assert any("backend.ingest.odds_theoddsapi" in ln for ln in calls)
    assert "(gamelogs exit 1, predict exit 0, odds exit 0)" in log
